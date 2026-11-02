"""Native SOMA-X 0.3.3 model for the pinned public GEM-X estimator.

This is deliberately separate from the legacy MHR-backed ``soma`` adapter.
GEM-X predicts 128 native identity coefficients and [1, local56] bone scales.
Downstream fitted identities may declare one fixed native identity scale in a
hash-bound v2 manifest; raw prediction retains its exact unit global channel.
Human fingers remain human fingers; a G1-like identity changes body dimensions,
not the joint topology or the source pose.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from pytorch3d.transforms import axis_angle_to_matrix, matrix_to_axis_angle

from grail.models.human_model import SomaHumanModel
from grail.models.soma_model import (
    SOMA_BODY_POSE_INDICES,
    SOMA_JOINT_NAMES,
    SOMA_LEFT_HAND_JOINT_INDICES,
    SOMA_LEFT_HAND_POSE_INDICES,
    SOMA_RIGHT_HAND_JOINT_INDICES,
    SOMA_RIGHT_HAND_POSE_INDICES,
    SOMA_SEGMENT_JOINT_INDICES,
)

# SOMA has eyes but no exact public nose/ear joints. Their observations are
# disabled below; HeadEnd placeholders never participate in a fitting loss.
COCO17_NATIVE = [7, 9, 10, 7, 7, 12, 40, 13, 41, 14, 42, 67, 72, 68, 73, 69, 74]
SOMA03_SEGMENT_JOINT_INDICES = {
    **SOMA_SEGMENT_JOINT_INDICES,
    "L_Toe": SOMA_JOINT_NAMES.index("LeftToeBase"),
    "R_Toe": SOMA_JOINT_NAMES.index("RightToeBase"),
}
IDENTITY_SCHEMA_UNIT = "grail.soma03.identity128.fixed_global1.local56.v1"
IDENTITY_SCHEMA_FIXED_SCALE = "grail.soma03.identity128.fixed_identity_scale.local56.v2"


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class Soma03HumanModel(SomaHumanModel):
    """GRAIL's body/hand fitting interface backed by native SOMA03 FK and skinning."""

    def __init__(self, cfg: dict, device: str = "cpu"):
        cfg = cfg.get("human_model", cfg)
        self.device = device
        self.runtime_kind = cfg.get("soma_runtime", "gem_x")
        if self.runtime_kind != "gem_x" or "gep_gem_bundle" in cfg:
            raise ValueError(
                "The vendored GEP/GEM SOMA03 runtime was retired. Use soma_runtime=gem_x "
                "with the pinned SOMA-X assets, or the original archived checkout for "
                "a historical runtime; legacy geometry is never silently substituted."
            )
        from grail.models.soma_x_runtime import DEFAULT_SOMA_ASSETS, SomaXNativeLayer

        self.model = SomaXNativeLayer(cfg.get("gem_x_soma_assets") or DEFAULT_SOMA_ASSETS, device)
        self._fixed_identity_global_scale = None
        self.model_type = "soma03"
        self.BODY_POSE_INDICES = SOMA_BODY_POSE_INDICES
        self.LEFT_HAND_POSE_INDICES = SOMA_LEFT_HAND_POSE_INDICES
        self.RIGHT_HAND_POSE_INDICES = SOMA_RIGHT_HAND_POSE_INDICES
        self.LEFT_HAND_JOINT_INDICES = SOMA_LEFT_HAND_JOINT_INDICES
        self.RIGHT_HAND_JOINT_INDICES = SOMA_RIGHT_HAND_JOINT_INDICES
        self.BODY_JOINT_INDICES = COCO17_NATIVE
        self._segment_cache = {}
        # Fail before fitting if a release changes semantic joint ordering.
        public = tuple(self.model.soma.public_joint_names)
        if public[-77:] != tuple(SOMA_JOINT_NAMES):
            raise ValueError("SOMA03 public skeleton differs from the reviewed nvskel77 mapping")

    def _tensor(self, value):
        return torch.as_tensor(value, dtype=torch.float32, device=self.device)

    def _shape(self, identity, scales, batch):
        return self._shape_with_fixed_scale(
            identity, scales, batch, self._fixed_identity_global_scale
        )

    def _shape_with_fixed_scale(self, identity, scales, batch, fixed_global_scale):
        """Validate without changing authorization, including during manifest load."""
        identity, scales = self._tensor(identity), self._tensor(scales)
        if identity.ndim not in (1, 2) or identity.shape[-1] != 128:
            raise ValueError("Native SOMA03 requires identity128; legacy MHR is incompatible")
        if scales.ndim not in (1, 2) or scales.shape[-1] != 57:
            raise ValueError("Native SOMA03 requires scale57 in [global, local56] layout")
        identity, scales = identity.reshape(-1, 128), scales.reshape(-1, 57)
        if not torch.isfinite(identity).all() or not torch.isfinite(scales).all():
            raise ValueError("SOMA03 identity/scale parameters must be finite")
        global_channel = scales[:, 0]
        unit_global = torch.equal(global_channel, torch.ones_like(global_channel))
        fixed_global = fixed_global_scale is not None and torch.equal(
            global_channel, torch.full_like(global_channel, fixed_global_scale)
        )
        if not unit_global and not fixed_global:
            raise ValueError(
                "SOMA03 global scale must be exactly one for raw prediction or one "
                "constant fixed identity scale authorized by its validated v2 manifest"
            )
        if (scales[:, 1:] <= 0).any():
            raise ValueError("Native SOMA03 bone-scale ratios must be positive")
        if identity.shape[0] not in (1, batch) or scales.shape[0] not in (1, batch):
            raise ValueError("SOMA03 shape must be constant or supplied for every source frame")
        return identity.expand(batch, -1), scales.expand(batch, -1)

    def _forward(self, motion_data, *, require_grad=False, joints_only=False):
        poses = self._tensor(motion_data["poses"]).reshape(-1, 77, 3)
        trans = self._tensor(motion_data["trans"]).reshape(-1, 3)
        if (
            len(poses) != len(trans)
            or not torch.isfinite(poses).all()
            or not torch.isfinite(trans).all()
        ):
            raise ValueError("Invalid SOMA03 pose/translation sequence")
        scale = motion_data.get("scale", 1.0)
        if not torch.all(self._tensor(scale) == 1):
            raise ValueError("An extra mesh scale would invalidate SOMA03 native coordinates")
        identity, scales = self._shape(
            motion_data["identity_coeffs"], motion_data["scale_params"], len(poses)
        )
        with torch.set_grad_enabled(require_grad):
            return self.model.static_forward(
                poses,
                identity,
                scales,
                trans,
                return_joints_only=joints_only,
            )

    def generate_mesh(self, motion_data, *, output_joints=False, require_grad=False):
        result = self._forward(motion_data, require_grad=require_grad)
        faces = torch.as_tensor(self.model.faces, dtype=torch.long, device=self.device)
        if output_joints:
            return result["vertices"], faces, result["joints"]
        return result["vertices"], faces

    def get_native_joints(self, motion_data, *, require_grad=False):
        """All 77 public native FK joints, with actual Hips at index zero."""
        return self._forward(motion_data, require_grad=require_grad, joints_only=True)["joints"]

    def get_body_joints(self, motion_data, *, require_grad=False):
        return self._forward(motion_data, require_grad=require_grad, joints_only=True)["joints"][
            :, COCO17_NATIVE
        ]

    def get_hand_joints(self, motion_data, *, require_grad=False):
        indices = self.LEFT_HAND_JOINT_INDICES + self.RIGHT_HAND_JOINT_INDICES
        return self._forward(motion_data, require_grad=require_grad, joints_only=True)["joints"][
            :, indices
        ]

    def get_foot_joint_indices(self):
        return (15, 16)

    def get_segment_indices(self, segment_names):
        """Segment the actual LOD/expanded rig by semantic descendants and weights."""
        rig = self.model.soma.rig_data
        names = list(map(str, rig["joint_names"]))
        parents = np.asarray(rig["joint_parent_ids"])
        weights = self.model.soma.skinning_weights
        selected = []
        for name in segment_names:
            if name not in self._segment_cache:
                if name not in SOMA03_SEGMENT_JOINT_INDICES:
                    raise ValueError(f"Unknown SOMA03 body segment: {name}")
                root_name = SOMA_JOINT_NAMES[SOMA03_SEGMENT_JOINT_INDICES[name]]
                descendants = {names.index(root_name)}
                previous = -1
                while previous != len(descendants):
                    previous = len(descendants)
                    descendants.update(
                        i for i, parent in enumerate(parents) if int(parent) in descendants
                    )
                mask = weights[:, sorted(descendants)].sum(dim=-1) > 0.1
                ids = torch.where(mask)[0].detach().cpu().numpy()
                if not len(ids):
                    raise ValueError(f"SOMA03 segment {name} has no mesh vertices")
                self._segment_cache[name] = ids
            selected.append(self._segment_cache[name])
        return np.unique(np.concatenate(selected)) if selected else np.empty(0, dtype=np.int64)

    def get_verts_segment(self, human_verts, segment_names):
        return human_verts[:, self.get_segment_indices(segment_names)]

    def load_shape_params(self, cfg, character_name, device):
        """Validate a paired identity and authorize its fixed scale on this instance.

        Reloading a nonunit downstream motion requires this call with its paired
        identity first. Constructor configuration or motion metadata alone never
        authorizes a nonunit channel; raw GEM-X motion always remains unit-scale.
        """
        path = cfg.get("human_model", cfg).get("soma_identity_params_path")
        if not path:
            return None
        path = Path(path).expanduser().resolve()
        manifest_path = path.with_suffix(".json")
        manifest = json.loads(manifest_path.read_text())
        schema = manifest.get("schema")
        if schema not in (IDENTITY_SCHEMA_UNIT, IDENTITY_SCHEMA_FIXED_SCALE):
            raise ValueError("Unrecognized SOMA03 identity provenance schema")
        if manifest.get("artifacts", {}).get(path.name) != _sha256(path):
            raise ValueError("SOMA03 identity checksum differs from its provenance")
        # Compare against this installation, not stale absolute paths from a
        # different workstation. The fitted identity is bound to source/assets.
        from grail.models.soma_x_runtime import validate_identity_binding

        validate_identity_binding(manifest, self.model.compatibility)
        fixed_scale = None
        if schema == IDENTITY_SCHEMA_FIXED_SCALE:
            declared = manifest.get("global_scale")
            if (
                isinstance(declared, bool)
                or not isinstance(declared, (int, float))
                or not np.isfinite(declared)
                or declared <= 0
                or not np.isfinite(np.float32(declared))
                or np.float32(declared) <= 0
                or manifest.get("global_scale_policy") != "fixed_identity_parameter"
            ):
                raise ValueError("Invalid SOMA03 fixed identity global scale policy or value")
            fixed_scale = float(np.float32(declared))
        with np.load(path, allow_pickle=False) as data:
            if schema == IDENTITY_SCHEMA_FIXED_SCALE:
                if (
                    "schema" not in data
                    or data["schema"].shape != ()
                    or str(data["schema"]) != schema
                ):
                    raise ValueError("SOMA03 identity NPZ schema differs from its v2 manifest")
                # Require the declared channel, not merely either allowed runtime
                # value: a unit NPZ cannot masquerade as a nonunit fitted identity.
                channels = np.asarray(data["scale_params"])
                if channels.shape not in ((57,), (1, 57)) or not np.all(
                    channels.reshape(-1, 57)[:, 0] == fixed_scale
                ):
                    raise ValueError("SOMA03 identity NPZ global scale differs from its manifest")
            identity, scales = self._shape_with_fixed_scale(
                data["identity_coeffs"], data["scale_params"], 1, fixed_scale
            )
        # Commit authorization only after every hash, asset, parameter and device
        # check succeeds. Failed loads preserve any previously selected identity.
        identity, scales = identity.to(device), scales.to(device)
        provenance = {
            "path": str(path),
            "sha256": _sha256(path),
            "schema": schema,
            "global_scale": 1.0 if fixed_scale is None else fixed_scale,
            "global_scale_policy": (
                "raw_unit" if fixed_scale is None else "fixed_identity_parameter"
            ),
        }
        provenance["runtime_compatibility"] = self.model.compatibility["id"]
        provenance["runtime_convention"] = self.model.compatibility["convention"]
        self._fixed_identity_global_scale = fixed_scale
        self.model.validate_external_global_scale = fixed_scale is None
        self.identity_provenance = provenance
        return identity, scales

    def compute_tpose_height(self, shape_params):
        if isinstance(shape_params, dict):
            identity, scales = self.get_shape_from_motion_data(shape_params)
        else:
            identity, scales = shape_params
        identity, scales = self._tensor(identity), self._tensor(scales)
        count = max(identity.reshape(-1, 128).shape[0], scales.reshape(-1, 57).shape[0])
        identity, scales = self._shape(identity, scales, count)
        # GEM-X can output time-varying identities. Report the median of fixed,
        # deterministic samples rather than quietly averaging all identities.
        samples = sorted({0, count // 2, count - 1})
        motion = dict(
            poses=torch.zeros(len(samples), 77, 3, device=self.device),
            trans=torch.zeros(len(samples), 3, device=self.device),
            identity_coeffs=identity[samples],
            scale_params=scales[samples],
        )
        verts, _ = self.generate_mesh(motion)
        heights = verts[..., 1].amax(dim=1) - verts[..., 1].amin(dim=1)
        return float(heights.median())

    def extract_gt_keypoints(self, motion_data):
        body = self._tensor(motion_data["vitpose"]).clone()
        hands = self._tensor(motion_data["soma_hand_keypoints_2d"]).clone()
        if body.shape[1:] != (17, 3) or hands.shape[1:] != (50, 3):
            raise ValueError("SOMA03 requires mapped COCO17 and wrist-only native50 observations")
        body[:, [0, 3, 4], 2] = 0
        return body, hands

    def transform_global_motion(
        self,
        global_motion_data,
        incam_motion_data,
        *,
        cam_R,
        cam_t,
        align_frame=0,
        use_global=False,
        gt_shape_params=None,
        gt_height=None,
        device="cuda",
    ):
        """Place actual native root joints, then apply a fixed identity once.

        SOMA transl is not assumed to be the pelvis position. Identity replacement
        changes the bind-pose root offset; translation is solved by FK afterwards.
        The fixed identity sets metric height, including its manifest-authorized
        native identity scale when present. Raw prediction stays unit-scale. A
        render-character height never authorizes another mesh/global multiplier.
        """
        if incam_motion_data.get("soma03_world_transform_applied", False) or (
            use_global and global_motion_data.get("soma03_world_transform_applied", False)
        ):
            raise ValueError("SOMA03 world/identity transform cannot be applied twice")
        source = global_motion_data if use_global else incam_motion_data
        poses = self._tensor(source["poses"]).reshape(-1, 77, 3).clone()
        incam_poses = self._tensor(incam_motion_data["poses"]).reshape(-1, 77, 3)
        rotation, translation = self._tensor(cam_R).reshape(3, 3), self._tensor(cam_t).reshape(3)
        root = self._forward(source, joints_only=True)["joints"][:, 0]
        root_rot = axis_angle_to_matrix(poses[:, 0])
        if use_global:
            incam_root = self._forward(incam_motion_data, joints_only=True)["joints"][:, 0]
            incam_rot = axis_angle_to_matrix(incam_poses[align_frame, 0])
            align_rot = incam_rot @ root_rot[align_frame].T
            root = (root - root[align_frame]) @ align_rot.T + incam_root[align_frame]
            root_rot = align_rot.unsqueeze(0) @ root_rot
        desired_root = root @ rotation.T + translation
        poses[:, 0] = matrix_to_axis_angle(rotation.unsqueeze(0) @ root_rot)
        output = dict(incam_motion_data)
        output.update(poses=poses, scale=1.0, model="soma03", model_type="soma03")
        # Keep immutable source shape available for checking the estimator output.
        output["source_identity_coeffs"] = self._tensor(source["identity_coeffs"]).clone()
        output["source_scale_params"] = self._tensor(source["scale_params"]).clone()
        ratio = 1.0
        if gt_shape_params is not None:
            height = self.compute_tpose_height(gt_shape_params)
            source_height = float(incam_motion_data["predicted_body_height"])
            if source_height <= 0 or not np.isfinite(source_height):
                raise ValueError("Invalid original SOMA03 predicted body height")
            ratio = height / source_height
            desired_root = translation + ratio * (desired_root - translation)
            identity, scales = self._shape(*gt_shape_params, len(poses))
            output["identity_coeffs"], output["scale_params"] = identity.clone(), scales.clone()
        else:
            output["identity_coeffs"] = output["source_identity_coeffs"]
            output["scale_params"] = output["source_scale_params"]
        output["trans"] = torch.zeros(len(poses), 3, device=self.device)
        offset = self._forward(output, joints_only=True)["joints"][:, 0]
        output["trans"] = desired_root - offset
        output["soma03_world_transform_applied"] = True
        output["soma03_placement"] = {
            "use_global": bool(use_global),
            "align_frame": int(align_frame),
            "source_to_target_height_ratio": ratio,
            "render_character_height_diagnostic": None if gt_height is None else float(gt_height),
            "identity": getattr(self, "identity_provenance", None),
            "external_global_scale": float(output["scale_params"].reshape(-1, 57)[0, 0]),
            "global_scale_policy": (
                "fixed_identity_parameter"
                if gt_shape_params is not None and self._fixed_identity_global_scale is not None
                else "raw_unit"
            ),
            "root_convention": "actual SOMA Hips joint from native FK",
        }
        return output
