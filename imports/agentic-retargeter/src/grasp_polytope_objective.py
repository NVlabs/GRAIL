"""Source-bound three-finger ObjRetarget-inspired proposal objective.

Uses original human landmarks, original object poses and native robot FK. It is
an explicitly adapted geometric prior, not the unavailable official pipeline.
"""

from hashlib import sha256
from numbers import Integral, Real
from pathlib import Path

import numpy as np


def _hand_frame(points):
    """Columns define a common semantic frame from wrist/thumb/index/middle."""
    import torch

    lateral = points[:, 2] - points[:, 1]
    forward = points[:, 3] - points[:, 0]
    x = lateral / torch.linalg.vector_norm(lateral, dim=-1, keepdim=True).clamp_min(1e-8)
    z = torch.linalg.cross(x, forward)
    z = z / torch.linalg.vector_norm(z, dim=-1, keepdim=True).clamp_min(1e-8)
    y = torch.linalg.cross(z, x)
    return torch.stack((x, y, z), dim=-1)


class PolytopeObjective:
    def __init__(self, ctx, hand_name, hand, frames, fk, options):
        import torch

        allowed = {"method", "source_arrays", "edge_weight", "relative_weight"}
        if not isinstance(options, dict) or set(options) - allowed:
            raise ValueError("Unexpected ObjRetarget objective options")
        if options.get("method", "objretarget") != "objretarget":
            raise ValueError("ObjRetarget objective method must be objretarget")
        if hand_name not in ("left", "right"):
            raise ValueError("Three-finger adaptation requires an explicit left or right hand")
        self.edge_weight = options.get("edge_weight", 1.)
        self.relative_weight = options.get("relative_weight", 1.)
        for name in ("edge_weight", "relative_weight"):
            value = getattr(self, name)
            if (isinstance(value, bool) or not isinstance(value, Real)
                    or not np.isfinite(value) or value < 0):
                raise ValueError(f"{name} must be finite and nonnegative")
        if self.edge_weight == 0 and self.relative_weight == 0:
            raise ValueError("At least one ObjRetarget weight must be positive")
        path = Path(options["source_arrays"]).resolve(strict=True)
        with np.load(path, allow_pickle=False) as arrays:
            required = {"human_joint_positions", "human_joint_names", "object_R",
                        "object_trans", "object_scale", "timestamps"}
            if not required.issubset(arrays.files):
                raise ValueError("Source arrays lack required human/object/timing bindings")
            positions = np.array(arrays["human_joint_positions"], dtype=float)
            names = arrays["human_joint_names"].tolist()
            object_R, object_t = np.array(arrays["object_R"]), np.array(arrays["object_trans"])
            scale, timestamps = np.array(arrays["object_scale"]), np.array(arrays["timestamps"])
        if (positions.ndim != 3 or positions.shape != (len(ctx.q), len(names), 3)
                or not np.isfinite(positions).all() or len(set(names)) != len(names)):
            raise ValueError("Source human positions/names must be finite, unique, frame-aligned")
        if (not np.array_equal(object_R, ctx.R) or not np.array_equal(object_t, ctx.t)
                or not np.array_equal(scale, np.asarray(ctx.native["object_scale"]))):
            raise ValueError("Source object poses/scale must exactly match native original arrays")
        if not np.array_equal(timestamps, np.asarray(ctx.native["timestamps"])):
            raise ValueError("Source timestamps must exactly match native original arrays")
        frames = list(frames)
        if not frames or any(isinstance(i, bool) or not isinstance(i, Integral)
                             or not 0 <= i < len(ctx.q) for i in frames):
            raise ValueError("frames must contain valid source frame indices")
        prefix = hand_name.capitalize() + "Hand"
        human_names = [prefix, prefix+"ThumbEnd", prefix+"IndexEnd", prefix+"MiddleEnd"]
        if any(name not in names for name in human_names):
            raise ValueError("Source lacks required wrist/thumb/index/middle landmarks")
        source = positions[frames][:, [names.index(name) for name in human_names]]
        source = np.einsum("bji,bnj->bni", ctx.R[frames], source-ctx.t[frames, None])
        lateral, forward = source[:, 2]-source[:, 1], source[:, 3]-source[:, 0]
        if (np.linalg.norm(lateral, axis=-1).min() < 1e-8
                or np.linalg.norm(np.cross(lateral, forward), axis=-1).min() < 1e-10):
            raise ValueError("Source hand landmarks cannot define a nondegenerate hand frame")
        self.wrist = hand["wrist"]
        tip_names = [f"{hand_name}_hand_{digit}_finger_tip"
                     for digit in ("thumb", "index", "middle")]
        self.tips = []
        for name in tip_names:
            try:
                body = ctx.model.body(name).id
            except (KeyError, ValueError) as exc:
                raise ValueError(f"Native robot lacks required tip body {name}") from exc
            ancestor = body
            while ancestor and ancestor != self.wrist:
                ancestor = int(ctx.model.body_parentid[ancestor])
            if ancestor != self.wrist:
                raise ValueError("Robot fingertip must be downstream of the configured wrist")
            self.tips.append(body)
        faces = np.asarray(hand["region_faces"])
        if (faces.ndim != 1 or faces.size == 0 or faces.dtype.kind not in "iu"
                or faces.min() < 0 or faces.max() >= len(ctx.object.faces)
                or len(np.unique(faces)) != len(faces)):
            raise ValueError("Eligible region must contain unique original face indices")
        centroid = ctx.object.triangles[faces].mean(axis=1).mean(axis=0)
        self.centroid = fk.t(centroid)
        self.batch_size = len(frames)
        reference = fk.t(source)
        self.pairs = [(1, 2), (1, 3), (2, 3)]
        self.edge_i = [0, 0, 0, 1, 1, 2]
        self.edge_j = [1, 2, 3, 2, 3, 3]
        center = self.centroid[None].expand(self.batch_size, -1)
        tetra = torch.stack([torch.stack((reference[:, 0], reference[:, i], reference[:, j],
                                         center), dim=1) for i, j in self.pairs], dim=1)
        self.reference_edges = (tetra[:, :, self.edge_i]-tetra[:, :, self.edge_j]).detach()
        source_frame = _hand_frame(reference)
        self.reference_relative = torch.einsum(
            "bji,bj->bi", source_frame, center-reference[:, 0]).detach()
        self.metadata = dict(
            method="objretarget_inspired_three_finger_adaptation", acceptance=False,
            source_paper="https://arxiv.org/html/2607.03828", official_reproduction=False,
            source_arrays=str(path), source_arrays_sha256=sha256(path.read_bytes()).hexdigest(),
            original_object_pose_scale_and_timestamps_exact=True, frames=[int(i) for i in frames],
            hand=hand_name, human_landmarks=human_names, robot_tip_bodies=tip_names,
            wrist_body=hand.get("wrist_body", str(self.wrist)),
            tetrahedra=[["wrist", human_names[i], human_names[j], "eligible_region_centroid"]
                        for i, j in self.pairs],
            region_original_face_count=len(faces),
            region_original_face_ids_sha256=sha256(faces.astype("<i8").tobytes()).hexdigest(),
            centroid_object_local=centroid.tolist(),
            centroid_policy="arithmetic mean of original eligible triangle centroids",
            centroid_is_contact_anchor=False,
            edge_formulation="sum squared edge-vector discrepancies in original object frame",
            relative_formulation="contact centroid relative to geometric wrist/thumb/index/middle frame",
            geometric_frame="x=normalize(index-thumb); z=normalize(x cross (middle-wrist)); y=z cross x",
            edge_weight=float(self.edge_weight), relative_weight=float(self.relative_weight),
            contact_normal_loss_enabled=False, geometry_scaled=False,
            limitations=["three-digit adaptation; human ring/pinky not mapped to extra robot digits",
                         "region centroid can lie off the physical surface; this is a geometric prior",
                         "not an object/palm/finger collision test or force-closure metric"])

    def evaluate(self, rr, pp, rots, poss):
        """Return [B] weighted loss for proposed object-frame wrist and native FK."""
        import torch

        if rr.shape != (self.batch_size, 3, 3) or pp.shape != (self.batch_size, 3):
            raise ValueError("Proposed wrist batch does not match bound source frames")
        local = torch.einsum("bji,bnj->bni", rots[:, self.wrist],
                             poss[:, self.tips]-poss[:, self.wrist, None])
        tips = torch.einsum("bij,bnj->bni", rr, local)+pp[:, None]
        points = torch.cat((pp[:, None], tips), dim=1)
        center = self.centroid[None].expand(self.batch_size, -1)
        tetra = torch.stack([torch.stack((pp, tips[:, i-1], tips[:, j-1], center), dim=1)
                             for i, j in self.pairs], dim=1)
        edges = tetra[:, :, self.edge_i]-tetra[:, :, self.edge_j]
        loss = self.edge_weight*(edges-self.reference_edges).square().sum((1, 2, 3))
        if self.relative_weight:
            relative = torch.einsum("bji,bj->bi", _hand_frame(points), center-pp)
            # Each of the three tetrahedra has the same palm/centroid pair.
            loss = loss+3*self.relative_weight*(relative-self.reference_relative).square().sum(1)
        return loss


def build_polytope_objective(ctx, hand_name, hand, frames, fk, options):
    return PolytopeObjective(ctx, hand_name, hand, frames, fk, options)


build_paper_objective = build_polytope_objective
