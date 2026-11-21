"""Batched original-surface clearance proposals for robot, object and scene.

The caller supplies rigid transforms for every mesh on the same time grid.
Autograd follows both sides of moving pairs. This sampled winding objective is
not an intersection audit or an acceptance gate; full original geometry remains
the independent audit input, including explicitly omitted degenerate faces.
"""
import hashlib
import numpy as np

from .repair_geometry import TriangleMesh
from .repair_tensor import WarpSignedDistance


class BatchedClearance:
    def __init__(self, meshes, pairs, *, device="cpu", samples=128,
                 margin_m=.002, allow_degenerate_subset=()):
        import torch
        if type(samples) is not int or not 16 <= samples <= 4096:
            raise ValueError("samples must be an integer in [16, 4096]")
        if not np.isfinite(margin_m) or not 0 <= margin_m <= .02:
            raise ValueError("Invalid clearance margin")
        pairs = [tuple(p) for p in pairs]
        if (not pairs or len({tuple(sorted(p)) for p in pairs}) != len(pairs)
                or any(len(p) != 2 or p[0] == p[1] or not set(p) <= set(meshes) for p in pairs)):
            raise ValueError("Expected explicit unique pairs of bound mesh names")
        allowed = set(allow_degenerate_subset)
        if not allowed <= set(meshes):
            raise ValueError("Unknown explicitly permitted query subset")
        self.device = torch.device(device); self.margin = float(margin_m)
        self.pairs = pairs; self.points = {}; self.queries = {}; self.records = {}
        self.original_face_ids = {}; self.by_target = {}
        for a, b in pairs:
            self.by_target.setdefault(b, []).append(a)
            self.by_target.setdefault(a, []).append(b)
        for name in self.by_target:
            mesh = meshes[name]
            original = TriangleMesh(mesh.vertices, mesh.faces)
            rounded = TriangleMesh(original.vertices.astype(np.float32), original.faces)
            valid = original.normal_valid & rounded.normal_valid
            if not valid.all() and name not in allowed:
                raise ValueError(f"{name}: incomplete original query geometry requires explicit opt-in")
            if not valid.any():
                raise ValueError(f"{name}: no usable original faces")
            ids = np.flatnonzero(valid); faces = original.faces[ids]
            vertices = original.vertices[np.unique(faces)]
            nv, nf = min(samples//2, len(vertices)), min(samples-samples//2, len(faces))
            points = np.concatenate([vertices[np.linspace(0, len(vertices)-1, nv, dtype=int)],
                original.vertices[faces[np.linspace(0, len(faces)-1, nf, dtype=int)]].mean(1)])
            self.points[name] = torch.as_tensor(points, dtype=torch.float32, device=self.device)
            self.queries[name] = WarpSignedDistance(original.vertices, faces, device=device)
            self.original_face_ids[name] = ids
            self.records[name] = dict(original_faces=len(original.faces), query_faces=len(ids),
                omitted_faces=int((~valid).sum()), samples=len(points),
                original_sha256=hashlib.sha256(original.vertices.tobytes()+original.faces.tobytes()).hexdigest(),
                original_face_map_sha256=hashlib.sha256(ids.tobytes()).hexdigest(),
                coverage="COMPLETE_TRIANGLE_SET" if valid.all() else "INCOMPLETE_QUERY_SUBSET",
                query=self.queries[name].metadata())

    def loss(self, transforms):
        """Return one top-k squared clearance loss per sampled pose.

        Transforms map names to (rotation [B,3,3], translation [B,3]); all
        transforms must use the same floating dtype/device/batch. Empty or
        nonfinite mesh queries propagate failure rather than a false pass.
        """
        import torch
        if set(transforms) != set(self.points):
            raise ValueError("Every queried mesh needs an explicit rigid transform")
        batch = None; dtype = None; world = {}
        for name, (r, p) in transforms.items():
            if (r.ndim != 3 or r.shape[1:] != (3, 3) or p.shape != (len(r), 3)
                    or r.device != self.device or p.device != self.device or r.dtype != p.dtype
                    or r.dtype not in (torch.float32, torch.float64)):
                raise ValueError("Invalid batched rigid transforms")
            if batch is not None and (batch != len(r) or dtype != r.dtype):
                raise ValueError("Transforms must share batch and dtype")
            batch, dtype = len(r), r.dtype
            world[name] = self.points[name].to(dtype) @ r.transpose(1, 2)+p[:, None]
        violations = []
        for target, sources in self.by_target.items():
            r, p = transforms[target]
            local = (torch.cat([world[s] for s in sources], dim=1)-p[:, None]) @ r
            v = torch.relu(self.margin-self.queries[target].distance(local))
            violations.append(v.topk(min(12, v.shape[1]), dim=1).values)
        v = torch.cat(violations, dim=1)
        return v.topk(min(12, v.shape[1]), dim=1).values.square().mean(1)

    def metadata(self):
        return dict(pairs=self.pairs, margin_m=self.margin, meshes=self.records,
                    motion_accepted=False, solid_sign_certified=False,
                    scope="Finite bidirectional surface samples, both moving-body derivatives; independent original-triangle audit required.")
