"""Cached original-mesh intersection screening; never motion acceptance.

Reuse is exact: original meshes/pairs are bound at construction and each query is
keyed by all rigid-transform bytes and its stopping policy. Rejected proposal
screening may stop at the first confirmed crossing. Detailed witness/count
queries remain the responsibility of repair_geometry.mesh_intersections.
"""
from __future__ import annotations

import copy
import hashlib
from types import MappingProxyType
import numpy as np

from .repair_geometry import TriangleMesh, mesh_intersection_probe


class MeshScreen:
    def __init__(self, meshes, pairs, *, retained_pairs=(), backend="numba"):
        # Independent copies prevent callers from changing a bound surface.
        self.meshes = MappingProxyType({str(k): TriangleMesh(v.vertices, v.faces)
                       for k, v in meshes.items()})
        self.pairs = tuple(tuple(p) for p in pairs)
        if (len({tuple(sorted(p)) for p in self.pairs}) != len(self.pairs)
                or any(len(p) != 2 or p[0] == p[1] or any(n not in self.meshes for n in p)
                       for p in self.pairs)):
            raise ValueError("Expected unique pairs of bound mesh names")
        self.retained = frozenset(tuple(p) for p in retained_pairs)
        if not self.retained.issubset(self.pairs):
            raise ValueError("Retained pairs must be explicitly in the checked scope")
        if backend == "numba":
            from .repair_numba import NumbaMeshQueries
            self.backend = NumbaMeshQueries(self.meshes)
        elif backend == "cpu":
            self.backend = None
        else:
            raise ValueError("backend must be cpu or numba")
        self._cache = {}
        self.cache_hits = 0
        self.cache_misses = 0

    def query(self, transforms, *, stop_on_first=False):
        if set(transforms) != set(self.meshes):
            raise ValueError("Every bound mesh needs an explicit transform")
        matrices = {k: np.asarray(transforms[k], dtype=np.float64) for k in self.meshes}
        if any(m.shape != (4, 4) or not np.isfinite(m).all() for m in matrices.values()):
            raise ValueError("Expected finite 4x4 affine transforms")
        key = hashlib.sha256(bytes([bool(stop_on_first)]) + b"".join(
            m.tobytes() for m in matrices.values())).digest()
        if key in self._cache:
            self.cache_hits += 1
            return copy.deepcopy(self._cache[key])
        self.cache_misses += 1
        posed = {k: m.transformed(matrices[k]) for k, m in self.meshes.items()}
        bounds = {k: (v.vertices.min(0, initial=np.inf), v.vertices.max(0, initial=-np.inf)) for k, v in posed.items()}
        hits, unknown, checked = [], [], 0
        exhausted = True
        for a, b in self.pairs:
            checked += 1
            ma, mb = posed[a], posed[b]
            # Preserve the same tolerance and degenerate-geometry semantics.
            disjoint = (np.any(bounds[a][1] < bounds[b][0] - 1e-9)
                        or np.any(bounds[b][1] < bounds[a][0] - 1e-9))
            if disjoint:
                if not (ma.normal_valid.all() and mb.normal_valid.all()):
                    unknown.append([a, b])
                continue
            stream = None if self.backend is None else self.backend.candidate_pairs(a, ma, b, mb)
            r = mesh_intersection_probe(ma, mb, candidate_pairs=stream)
            if not r['complete']:
                unknown.append([a, b])
            if r['intersecting']:
                hits.append({'pair': [a, b], 'retained': (a, b) in self.retained,
                             'witness_face_ids': r['witness_face_ids']})
                if stop_on_first and (a, b) not in self.retained:
                    exhausted = checked == len(self.pairs)
                    break
        result = {'surface_hits': hits, 'unknown_pairs': unknown,
                  'checked_pairs': checked, 'configured_pairs': len(self.pairs),
                  'pair_scope_exhausted': exhausted,
                  'no_unretained_crossing': exhausted and not unknown and not any(
                      not x['retained'] for x in hits),
                  'motion_accepted': False,
                  'scope': 'Sampled original surface intersections only; retained crossings stay visible. No total triangle counts, containment or dynamics.'}
        self._cache[key] = copy.deepcopy(result)
        return result
