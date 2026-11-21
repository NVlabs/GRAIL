"""Optional CPU-only compiled traversal of immutable original triangle BVHs.

Only AABB candidate enumeration is compiled, with float64 inclusive comparisons,
no fastmath and no candidate limit. The existing Python/NumPy narrowphase owns
all original surface tests, completeness checks and canonical witnesses. Numba
and LLVM are imported only when this explicitly selected backend is constructed;
missing, disabled or failing compilation raises instead of selecting a fallback.
"""

from __future__ import annotations

import copy
import platform
from collections.abc import Mapping

import numpy as np

from .repair_geometry import TriangleMesh, _tolerance


def _candidate_batches(
    al,
    ah,
    ac1,
    ac2,
    ast,
    aen,
    ao,
    atl,
    ath,
    bl,
    bh,
    bc1,
    bc2,
    bst,
    ben,
    bo,
    btl,
    bth,
    tolerance,
    stack_capacity,
    batch_size=4096,
):
    stack_a = np.empty(stack_capacity, dtype=np.int64)
    stack_b = np.empty(stack_capacity, dtype=np.int64)
    stack_a[0] = 0
    stack_b[0] = 0
    size = 1
    out_a = np.empty(batch_size, dtype=np.int64)
    out_b = np.empty(batch_size, dtype=np.int64)
    count = 0
    while size:
        size -= 1
        ai = stack_a[size]
        bi = stack_b[size]
        if (
            ah[ai, 0] < bl[bi, 0] - tolerance
            or bh[bi, 0] < al[ai, 0] - tolerance
            or ah[ai, 1] < bl[bi, 1] - tolerance
            or bh[bi, 1] < al[ai, 1] - tolerance
            or ah[ai, 2] < bl[bi, 2] - tolerance
            or bh[bi, 2] < al[ai, 2] - tolerance
        ):
            continue
        aleaf = ast[ai] >= 0
        bleaf = bst[bi] >= 0
        if aleaf and bleaf:
            for ap in range(ast[ai], aen[ai]):
                a = ao[ap]
                for bp in range(bst[bi], ben[bi]):
                    b = bo[bp]
                    if (
                        ath[a, 0] >= btl[b, 0] - tolerance
                        and bth[b, 0] >= atl[a, 0] - tolerance
                        and ath[a, 1] >= btl[b, 1] - tolerance
                        and bth[b, 1] >= atl[a, 1] - tolerance
                        and ath[a, 2] >= btl[b, 2] - tolerance
                        and bth[b, 2] >= atl[a, 2] - tolerance
                    ):
                        out_a[count] = a
                        out_b[count] = b
                        count += 1
                        if count == batch_size:
                            yield out_a, out_b
                            out_a = np.empty(batch_size, dtype=np.int64)
                            out_b = np.empty(batch_size, dtype=np.int64)
                            count = 0
        else:
            # Only branch choice/order depends on extents; no bounds are changed.
            ae = max(
                ah[ai, 0] - al[ai, 0], ah[ai, 1] - al[ai, 1], ah[ai, 2] - al[ai, 2]
            )
            be = max(
                bh[bi, 0] - bl[bi, 0], bh[bi, 1] - bl[bi, 1], bh[bi, 2] - bl[bi, 2]
            )
            if size + 2 > stack_capacity:
                raise RuntimeError(
                    "BVH traversal stack bound violated; no clear result"
                )
            if bleaf or (not aleaf and ae >= be):
                stack_a[size] = ac1[ai]
                stack_b[size] = bi
                stack_a[size + 1] = ac2[ai]
                stack_b[size + 1] = bi
            else:
                stack_a[size] = ai
                stack_b[size] = bc1[bi]
                stack_a[size + 1] = ai
                stack_b[size + 1] = bc2[bi]
            size += 2
    if count:
        yield out_a[:count].copy(), out_b[:count].copy()


class NumbaMeshQueries:
    """Context-owned immutable topology packs; every query uses current bounds.

    Names bind strongly to the original mesh/topology. Posed meshes must come
    from that topology, so a same-sized replacement cannot reuse stale indices.
    No posed geometry/results or shared object-ID cache are retained.
    """

    def __init__(self, meshes: Mapping[str, TriangleMesh], *, device="cpu"):
        if device != "cpu":
            raise ValueError("Numba geometry backend requires device cpu")
        if not isinstance(meshes, Mapping) or any(
            not isinstance(name, str) or not isinstance(mesh, TriangleMesh)
            for name, mesh in meshes.items()
        ):
            raise ValueError(
                "Numba meshes must map names to original TriangleMesh instances"
            )
        self._meshes = dict(meshes)
        self._packed = {}
        for name, mesh in self._meshes.items():
            topology = mesh._topology
            count = len(topology.nodes)
            left = np.array([node[0] for node in topology.nodes], dtype=np.int64)
            right = np.array([node[1] for node in topology.nodes], dtype=np.int64)
            start = np.full(count, -1, dtype=np.int64)
            end = start.copy()
            for node, offset in zip(topology.leaf_nodes, topology.leaf_offsets):
                start[node] = offset
                end[node] = offset + len(topology.nodes[node][2])
            for array in (left, right, start, end):
                array.setflags(write=False)
            self._packed[name] = (left, right, start, end)
        try:
            import llvmlite
            import llvmlite.binding as llvm
            import numba
        except ImportError as exc:
            raise RuntimeError(
                "Numba audit requires the optional compiled-audit runtime in the external Python"
            ) from exc
        if numba.config.DISABLE_JIT:
            raise RuntimeError(
                "Numba audit requires JIT enabled; no Python fallback is used"
            )
        self._compiled = numba.njit(cache=True, fastmath=False, nogil=True)(
            _candidate_batches
        )
        self._metadata = {
            "backend": "numba",
            "device": "cpu",
            "numpy": np.__version__,
            "numba": numba.__version__,
            "llvmlite": llvmlite.__version__,
            "llvm": list(llvm.llvm_version_info),
            "python": platform.python_version(),
            "machine": platform.machine(),
            "host_cpu": llvm.get_host_cpu_name(),
            "host_cpu_features": llvm.get_host_cpu_features().flatten(),
            "configured_cpu": numba.config.CPU_NAME,
            "configured_cpu_features": numba.config.CPU_FEATURES,
            "fastmath": False,
            "parallel": False,
            "bounds_dtype": "float64",
            "batch_size": 4096,
            "candidate_limit": None,
            "candidate_provider_contract": (
                "All overlapping original float64 triangle AABB pairs exactly once; "
                "current posed bounds; original Python/NumPy narrowphase unchanged"
            ),
            "acceptance": False,
        }

    def metadata(self):
        return copy.deepcopy(self._metadata)

    def _data(self, name, mesh):
        if name not in self._meshes:
            raise ValueError("Numba mesh name is not bound to this context")
        original = self._meshes[name]
        if (
            not isinstance(mesh, TriangleMesh)
            or mesh._topology is not original._topology
        ):
            raise ValueError("Numba posed mesh must retain its bound original topology")
        return (
            mesh._node_lower,
            mesh._node_upper,
            *self._packed[name],
            mesh._topology.face_order,
            mesh._lower,
            mesh._upper,
        )

    def candidate_pairs(self, name_a, mesh_a, name_b, mesh_b, *, tolerance=1e-9):
        """Stream independent original face-ID batches without truncation.

        A DFS splits one branch at a time: pending siblings cannot exceed the
        two tree depths plus root. An explicit guard still fails the query if
        this invariant is broken, never returning an incomplete clear result.
        """
        tolerance = _tolerance(tolerance)
        data_a = self._data(name_a, mesh_a)
        data_b = self._data(name_b, mesh_b)
        if not mesh_a._tree or not mesh_b._tree:
            return
        capacity = len(mesh_a._topology.levels) + len(mesh_b._topology.levels) + 2
        yield from self._compiled(*data_a, *data_b, tolerance, capacity)
