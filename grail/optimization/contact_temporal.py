"""Optional acceleration regularization of live arm joints and a rigid object.

Coordinates are calibrated world metres. `fps` is explicit source-video fps;
this module neither reads inherited motion metadata nor resamples frames.
Second differences are divided by dt**2. Returned component losses therefore
have units m**2/s**4 (mean squared acceleration magnitude), not contact m**2.
Any coefficient used to combine these with a square-metre contact objective
must account for these units and the fps**4 numerical factor.

The object surface integral uses exact area-weighted first/second moments of
original canonical triangles. This is algebraically equal to recomputing the
current posed surface at each of the three frames and integrating its squared
second difference. It needs no saved world targets or arbitrary sample count.
This is kinematic regularization, not a force/friction/dynamics test.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn


@dataclass(frozen=True)
class ContactTemporalConfig:
    fps: float
    enabled: bool = False
    joint_indices: tuple[int, ...] = (16, 17, 18, 19, 20, 21)
    joint_weights: tuple[float, ...] = (1.0, 1.0, 1.0, 1.0, 4.0, 4.0)
    human_weight: float = 1.0
    object_weight: float = 1.0


def _numpy(value):
    return value.detach().cpu().numpy() if torch.is_tensor(value) else np.asarray(value)


class ContactTemporalLoss(nn.Module):
    """Penalize acceleration without locking a hand to its object counterpart.

    Human inputs are live native body joints, with original joint IDs.
    Only body joints16..21 may be selected. Fingers are never used or modified.
    Object geometry must have its fixed scale baked exactly once; transforms
    must use the same uncentered calibrated-world convention as human joints.

    Optional frame confidences are finite values in [0,1], distinct from the
    geometric fields. Each three-frame stencil receives the minimum confidence
    of its frames once. Averages include inactive stencils, so weaker evidence
    does not cancel through normalization. Source frame gaps disable the
    crossing stencils; no artificial adjacency is assumed across a crop.
    """

    def __init__(self, object_local_vertices, object_faces, config):
        super().__init__()
        self.config = ContactTemporalConfig(**config) if isinstance(config, dict) else config
        cfg = self.config
        if not isinstance(cfg, ContactTemporalConfig):
            raise ValueError("Provide ContactTemporalConfig or a mapping with explicit source fps")
        if not math.isfinite(cfg.fps) or cfg.fps <= 0:
            raise ValueError("fps must be an explicit finite positive source-video rate")
        ids, weights = tuple(cfg.joint_indices), tuple(cfg.joint_weights)
        if not ids or len(set(ids)) != len(ids) or any(i not in range(16, 22) for i in ids):
            raise ValueError("Select unique original body joint IDs from16 through21 only")
        if (
            len(weights) != len(ids)
            or any(not math.isfinite(w) or w < 0 for w in weights)
            or sum(weights) <= 0
        ):
            raise ValueError("Joint weights must be finite, nonnegative and have positive sum")
        if any(not math.isfinite(w) or w < 0 for w in (cfg.human_weight, cfg.object_weight)):
            raise ValueError("Human and object weights must be finite and nonnegative")
        v, f = _numpy(object_local_vertices), _numpy(object_faces)
        if v.ndim != 2 or v.shape[1] != 3 or not np.isfinite(v).all():
            raise ValueError(
                "Canonical object vertices must be finite [vertices,3] already-scaled metres"
            )
        if (
            f.ndim != 2
            or f.shape[1] != 3
            or f.dtype.kind not in "iu"
            or not len(f)
            or np.any(f < 0)
            or np.any(f >= len(v))
        ):
            raise ValueError("Object triangles must use valid original canonical vertex IDs")
        tri = np.asarray(v, dtype=np.float64)[f]
        area = 0.5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=-1)
        if not np.isfinite(area).all() or area.sum() <= 0:
            raise ValueError("Canonical object surface must have positive finite area")
        mass = area / area.sum()
        center = (mass[:, None] * tri.mean(1)).sum(0)
        # For uniform points on a triangle, E[p p^T] equals
        # (sum_i v_i v_i^T + (sum_i v_i)(sum_i v_i)^T)/12.
        summed = tri.sum(1)
        second = (
            (np.einsum("tni,tnj->tij", tri, tri) + np.einsum("ti,tj->tij", summed, summed))
            * mass[:, None, None]
        ).sum(0) / 12
        covariance = second - np.outer(center, center)
        eig, basis = np.linalg.eigh((covariance + covariance.T) / 2)
        if eig.min() < -1e-10:
            raise ValueError("Canonical object surface covariance is not positive semidefinite")
        factor = basis * np.sqrt(np.maximum(eig, 0))[None]
        self.register_buffer("joint_indices", torch.tensor(ids, dtype=torch.long))
        self.register_buffer("joint_weights", torch.tensor(weights, dtype=torch.float64))
        self.register_buffer("object_surface_centroid", torch.tensor(center, dtype=torch.float64))
        self.register_buffer(
            "object_surface_covariance_factor", torch.tensor(factor, dtype=torch.float64)
        )
        self.surface_area_m2 = float(area.sum())
        self.surface_definition = "All supplied original canonical triangles, uniform represented surface area; overlap/internal faces are included unless explicitly excluded by caller."

    @staticmethod
    def _confidence(value, frames, columns, like, name):
        if value is None:
            return like.new_ones((frames, columns))
        value = torch.as_tensor(value, dtype=like.dtype, device=like.device)
        if value.shape == (frames,):
            value = value[:, None].expand(-1, columns)
        if (
            value.shape != (frames, columns)
            or not bool(torch.isfinite(value).all())
            or bool(((value < 0) | (value > 1)).any())
        ):
            raise ValueError(f"{name} must be [frames] or [frames,{columns}], finite and in[0,1]")
        return value

    def forward(
        self,
        human_body_joints,
        object_R,
        object_t,
        *,
        frame_indices=None,
        human_frame_confidence=None,
        object_frame_confidence=None,
    ):
        """Return weighted total and unweighted human/wrist/object components."""
        cfg = self.config
        if human_body_joints.ndim != 3 or human_body_joints.shape[-1] != 3:
            raise ValueError("Human body joints must be [frames,original_joint_count,3]")
        frames = len(human_body_joints)
        if object_R.shape != (frames, 3, 3) or object_t.shape != (frames, 3):
            raise ValueError("Object R/t must have shapes[frames,3,3] and[frames,3]")
        if int(self.joint_indices.max()) >= human_body_joints.shape[1]:
            raise ValueError("Human input lacks selected original body joints")
        if (
            object_R.device != human_body_joints.device
            or object_t.device != human_body_joints.device
        ):
            raise ValueError("Human and object poses must share one device")
        zero = human_body_joints.sum() * 0 + object_R.sum() * 0 + object_t.sum() * 0
        joint_zero = zero.expand(len(self.joint_indices))
        result = dict(
            total=zero,
            human_m2_s4=zero,
            wrist_m2_s4=zero,
            object_m2_s4=zero,
            per_joint_m2_s4=joint_zero,
            active_human_stencils=0,
            active_object_stencils=0,
            units="m^2/s^4",
            fps=float(cfg.fps),
        )
        if not cfg.enabled or frames < 3:
            return result
        like = human_body_joints
        if frame_indices is None:
            frame_indices = torch.arange(frames, device=like.device)
        else:
            frame_indices = torch.as_tensor(frame_indices, device=like.device)
            if frame_indices.shape != (frames,) or frame_indices.dtype not in (
                torch.int32,
                torch.int64,
            ):
                raise ValueError("frame_indices must be integer original source-frame IDs")
            if bool((frame_indices < 0).any()) or bool(
                (frame_indices[1:] <= frame_indices[:-1]).any()
            ):
                raise ValueError("frame_indices must be nonnegative and strictly increasing")
        consecutive = (frame_indices[1:-1] - frame_indices[:-2] == 1) & (
            frame_indices[2:] - frame_indices[1:-1] == 1
        )
        hc = self._confidence(
            human_frame_confidence, frames, len(self.joint_indices), like, "human_frame_confidence"
        )
        oc = self._confidence(object_frame_confidence, frames, 1, like, "object_frame_confidence")

        def stencil_confidence(confidence):
            return (
                torch.minimum(torch.minimum(confidence[:-2], confidence[1:-1]), confidence[2:])
                * consecutive[:, None]
            )

        hs, os = stencil_confidence(hc), stencil_confidence(oc).squeeze(1)

        def acceleration(sequence):
            return (sequence[2:] - 2 * sequence[1:-1] + sequence[:-2]) * float(cfg.fps) ** 2

        ids = self.joint_indices.to(like.device)
        ha = acceleration(human_body_joints[:, ids])
        per_joint = (ha.square().sum(-1) * hs).mean(0)
        weights = self.joint_weights.to(like)
        human = (per_joint * weights).sum() / weights.sum()
        wrist_mask = (ids == 20) | (ids == 21)
        wrist = per_joint[wrist_mask].mean() if bool(wrist_mask.any()) else zero
        ra, ta = acceleration(object_R), acceleration(object_t)
        center = self.object_surface_centroid.to(like)
        factor = self.object_surface_covariance_factor.to(like)
        center_acceleration = torch.einsum("fij,j->fi", ra, center) + ta
        centered_surface_acceleration = ra @ factor
        object_per_stencil = center_acceleration.square().sum(
            -1
        ) + centered_surface_acceleration.square().sum((-1, -2))
        obj = (object_per_stencil * os).mean()
        result.update(
            total=cfg.human_weight * human + cfg.object_weight * obj,
            human_m2_s4=human,
            wrist_m2_s4=wrist,
            object_m2_s4=obj,
            per_joint_m2_s4=per_joint,
            active_human_stencils=int((hs > 0).sum()),
            active_object_stencils=int((os > 0).sum()),
        )
        return result
