"""Frozen source interaction graphs for experimental grasp proposals.

Topology and row-normalized distance weights follow TopoRetarget Eq. 3--7
(https://arxiv.org/html/2606.16272v2). ``decay=0`` gives the uniform graph
coordinates of DexOAK (https://arxiv.org/html/2610.26762v1). The optional
deterministic kNN fallback follows DexOAK's graph construction convention.

These are objective primitives, not either paper's complete retargeter. The
caller supplies explicit equal-size anatomical point correspondence; three
robot fingers are an adaptation, not an officially supported embodiment.
Original object points are never resampled or changed here. The graph is not
a collision test, contact-normal target, or grasp-acceptance certificate.
"""

from dataclasses import dataclass

import numpy as np


def _points(value, name):
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 3 or array.shape[-1] != 3 or not all(array.shape[:2]):
        raise ValueError(f"{name} must have shape (batch, points, 3), nonempty")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain finite points")
    return array.copy()


def _readonly(value):
    value.setflags(write=False)
    return value


def _batch_selection(count, indices):
    if indices is None:
        return np.arange(count)
    result = np.asarray(indices)
    if (result.ndim != 1 or not len(result) or result.dtype.kind not in "iu"
            or np.any(result < 0) or np.any(result >= count)):
        raise ValueError("batch_indices must be a nonempty valid integer index vector")
    return result


@dataclass(frozen=True)
class InteractionGraph:
    """Immutable NumPy reference, before explicit device transfer."""

    source: np.ndarray
    laplacian: np.ndarray
    hand_count: int
    topology: tuple[str, ...]
    decay: float
    fallback_neighbors: int | None

    @property
    def metadata(self):
        return dict(hand_count=self.hand_count,
                    object_count=self.source.shape[1]-self.hand_count,
                    topology=list(self.topology), decay=self.decay,
                    fallback_neighbors=self.fallback_neighbors,
                    frozen_source_weights=True,
                    loss_reduction="mean_over_all_nodes_squared_3d_norm")

    def to_torch(self, device="cpu", *, dtype=None, batch_indices=None):
        import torch

        dtype = torch.float64 if dtype is None else dtype
        if dtype not in (torch.float32, torch.float64):
            raise ValueError("interaction graph requires float32 or float64")
        rows = _batch_selection(len(self.source), batch_indices)
        source = torch.tensor(self.source[rows], device=device, dtype=dtype)
        matrix = torch.tensor(self.laplacian[rows], device=device, dtype=dtype)
        return TorchInteractionGraph(matrix, matrix @ source,
                                     source[:, self.hand_count:], self.hand_count)


@dataclass(frozen=True)
class TorchInteractionGraph:
    """Device-resident constant graph; loss keeps only target autograd history."""

    laplacian: object
    source_coordinates: object
    source_object: object
    hand_count: int

    def loss(self, target_hand, target_object=None):
        """Return one Eq. 7 loss per candidate, with no implicit batch reduction."""
        import torch

        expected = (len(self.laplacian), self.hand_count, 3)
        if (not isinstance(target_hand, torch.Tensor) or target_hand.shape != expected
                or target_hand.dtype != self.laplacian.dtype
                or target_hand.device != self.laplacian.device):
            raise ValueError("target_hand must match graph batch, shape, dtype and device")
        obj = self.source_object if target_object is None else target_object
        if (not isinstance(obj, torch.Tensor) or obj.shape != self.source_object.shape
                or obj.dtype != self.laplacian.dtype or obj.device != self.laplacian.device):
            raise ValueError("target_object must match bound object batch, shape, dtype and device")
        points = torch.cat((target_hand, obj), dim=1)
        error = self.laplacian @ points-self.source_coordinates
        return error.square().sum(dim=-1).mean(dim=-1)


def build_interaction_graph(source_hand, source_object, *, decay=30., fallback_neighbors=20):
    """Build one source Delaunay graph per batch, with explicit recorded fallback.

    ``source_object`` is (P, 3) shared across candidates or (B, P, 3).
    Weights exp(-decay * distance) are normalized per source-node neighborhood.
    Set ``fallback_neighbors=None`` to fail on degenerate Delaunay geometry.
    No random jitter is added to original points. A fallback uses symmetric
    kNN edges, distance then node-index tie breaking, excluding the node itself.
    """
    from scipy.spatial import Delaunay, QhullError

    hand = _points(source_hand, "source_hand")
    obj = np.asarray(source_object, dtype=np.float64)
    if obj.ndim == 2:
        obj = np.broadcast_to(obj, (len(hand), *obj.shape))
    obj = _points(obj, "source_object")
    if len(hand) != len(obj):
        raise ValueError("source_hand and source_object batches must match")
    if isinstance(decay, bool) or not np.isscalar(decay) or not np.isfinite(decay) or decay < 0:
        raise ValueError("decay must be finite and nonnegative")
    if fallback_neighbors is not None and (
            isinstance(fallback_neighbors, bool)
            or not isinstance(fallback_neighbors, (int, np.integer)) or fallback_neighbors < 1):
        raise ValueError("fallback_neighbors must be a positive integer or None")
    source = np.concatenate((hand, obj), axis=1)
    count = source.shape[1]
    if count < 2:
        raise ValueError("interaction graph needs at least two nodes")
    matrices, modes = [], []
    for points in source:
        distance = np.linalg.norm(points[:, None]-points[None], axis=-1)
        adjacency = np.zeros((count, count), dtype=bool)
        try:
            simplices = Delaunay(points).simplices
            for simplex in simplices:
                adjacency[np.ix_(simplex, simplex)] = True
            np.fill_diagonal(adjacency, False)
            if not adjacency.any(axis=1).all():
                raise QhullError("Delaunay omitted an original node")
            mode = "delaunay"
        except QhullError:
            if fallback_neighbors is None:
                raise ValueError("degenerate Delaunay graph; fallback is disabled") from None
            k = min(fallback_neighbors, count-1)
            adjacency[:] = False
            for i in range(count):
                order = np.lexsort((np.arange(count), distance[i]))
                order = order[order != i][:k]
                adjacency[i, order] = True
                adjacency[order, i] = True
            mode = "symmetric_knn_fallback"
        # Subtract the minimum neighbor distance to avoid underflow; row
        # normalization makes this exactly equivalent to exp(-decay * d).
        nearest = np.where(adjacency, distance, np.inf).min(axis=1, keepdims=True)
        weights = np.zeros_like(distance)
        shifted = np.broadcast_to(nearest, distance.shape)
        weights[adjacency] = np.exp(-float(decay)*(distance[adjacency]-shifted[adjacency]))
        weights /= weights.sum(axis=1, keepdims=True)
        matrices.append(np.eye(count)-weights)
        modes.append(mode)
    return InteractionGraph(_readonly(source), _readonly(np.stack(matrices)),
                            hand.shape[1], tuple(modes), float(decay), fallback_neighbors)


@dataclass(frozen=True)
class BoneReference:
    """Source unit-bone differences; chains and optional local frame are explicit."""

    endpoints: np.ndarray
    adjacent: np.ndarray
    differences: np.ndarray
    hand_count: int

    def to_torch(self, device="cpu", *, dtype=None, batch_indices=None):
        import torch

        dtype = torch.float64 if dtype is None else dtype
        if dtype not in (torch.float32, torch.float64):
            raise ValueError("bone reference requires float32 or float64")
        rows = _batch_selection(len(self.differences), batch_indices)
        return TorchBoneReference(torch.tensor(self.endpoints.copy(), device=device),
                                  torch.tensor(self.adjacent.copy(), device=device),
                                  torch.tensor(self.differences[rows], device=device, dtype=dtype),
                                  self.hand_count)


@dataclass(frozen=True)
class TorchBoneReference:
    endpoints: object
    adjacent: object
    differences: object
    hand_count: int

    def loss(self, target_hand, *, target_frame=None):
        """TopoRetarget Eq. 1 summed over adjacent bones, separately per batch.

        ``target_frame`` is an optional (B, 3, 3) local-to-world wrist rotation;
        directions are transformed into that local frame. Target zero-length
        bones are epsilon-guarded for finite gradients, not certified feasible.
        """
        import torch

        if (not isinstance(target_hand, torch.Tensor)
                or target_hand.shape != (len(self.differences), self.hand_count, 3)
                or target_hand.device != self.differences.device
                or target_hand.dtype != self.differences.dtype):
            raise ValueError("target_hand must match bone reference batch, shape, dtype and device")
        delta = target_hand[:, self.endpoints[:, 1]]-target_hand[:, self.endpoints[:, 0]]
        direction = delta/torch.linalg.vector_norm(delta, dim=-1, keepdim=True).clamp_min(1e-12)
        if target_frame is not None:
            if (target_frame.shape != (len(target_hand), 3, 3)
                    or target_frame.dtype != target_hand.dtype or target_frame.device != target_hand.device):
                raise ValueError("target_frame must match target batch, dtype and device")
            direction = direction @ target_frame
        difference = direction[:, self.adjacent[:, 0]]-direction[:, self.adjacent[:, 1]]
        return (difference-self.differences).square().sum(dim=(-1, -2))


def build_bone_reference(source_hand, chains, *, source_frame=None):
    """Bind adjacent unit-bone differences from explicit anatomical chains.

    Each chain contains >=3 distinct point indices, e.g. wrist/MCP/PIP/tip.
    A local-to-world ``source_frame`` transforms vectors into the source wrist
    frame; use the corresponding robot wrist frame in the target loss.
    No five-to-three finger correspondence or bone-length scaling is inferred.
    """
    points = _points(source_hand, "source_hand")
    endpoints, adjacent = [], []
    for chain in chains:
        chain = np.asarray(chain)
        if (chain.ndim != 1 or len(chain) < 3 or chain.dtype.kind not in "iu"
                or len(set(chain.tolist())) != len(chain)
                or np.any(chain < 0) or np.any(chain >= points.shape[1])):
            raise ValueError("each bone chain needs at least three distinct valid point indices")
        first = len(endpoints)
        endpoints.extend(zip(chain[:-1], chain[1:]))
        adjacent.extend((first+i, first+i+1) for i in range(len(chain)-2))
    if not adjacent:
        raise ValueError("bone reference needs at least one valid chain")
    endpoints, adjacent = np.asarray(endpoints, dtype=np.int64), np.asarray(adjacent, dtype=np.int64)
    delta = points[:, endpoints[:, 1]]-points[:, endpoints[:, 0]]
    length = np.linalg.norm(delta, axis=-1, keepdims=True)
    if np.any(length <= 1e-12):
        raise ValueError("source bone length must be nonzero")
    direction = delta/length
    if source_frame is not None:
        frame = np.asarray(source_frame, dtype=np.float64)
        if (frame.shape != (len(points), 3, 3) or not np.isfinite(frame).all()
                or not np.allclose(np.swapaxes(frame, -1, -2) @ frame, np.eye(3), atol=1e-6)
                or not np.allclose(np.linalg.det(frame), 1, atol=1e-6)):
            raise ValueError("source_frame must be a batch of proper rotation matrices")
        direction = direction @ frame
    difference = direction[:, adjacent[:, 0]]-direction[:, adjacent[:, 1]]
    return BoneReference(_readonly(endpoints), _readonly(adjacent),
                         _readonly(difference), points.shape[1])
