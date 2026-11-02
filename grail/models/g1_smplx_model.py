"""G1-proportioned SMPL-X model — `G1ProportionSMPLX`.

Loads a pre-baked T-pose mesh and reference scale from
`data/body_profiles/unitree_g1/smplx/` into the standard SMPL-X
model so its FK output is G1-proportioned without any post-hoc joint
adjustment. Used by:

  - `grail/retargeting/retarget.py` (sonic env, no pytorch3d) — drives the
    GMR IK target body
  - `grail/models/g1_smplx_human.py::G1SmplxHumanModel` (grail env) — wraps
    this model for the HOI optimizer pipeline

This file deliberately avoids importing `grail.models.human_model` (which
pulls pytorch3d) so it stays importable from the sonic env.
"""

import os

import numpy as np
import smplx
import torch


def _load_obj_verts(path: str) -> np.ndarray:
    """Load vertex positions from a Wavefront OBJ file."""
    verts = []
    with open(path) as f:
        for line in f:
            if line.startswith("v "):
                verts.append([float(x) for x in line.split()[1:4]])
    return np.array(verts, dtype=np.float32)


class G1ProportionSMPLX:
    """SMPLX model with G1 robot proportions loaded from a pre-baked mesh.

    This loads the already-baked T-pose vertices from an OBJ file and sets
    them as v_template. shapedirs / expr_dirs are zeroed (shape is baked in),
    and posedirs are scaled by the reference scale.
    """

    def __init__(
        self,
        smplx_model_path: str,
        tpose_mesh_path: str,
        scale: float,
    ):
        self.scale = scale

        self.model = smplx.create(
            model_path=smplx_model_path,
            model_type="smplx",
            use_pca=False,
            flat_hand_mean=True,
            num_betas=10,
            num_expression_coeffs=10,
        ).to("cpu")

        verts = _load_obj_verts(tpose_mesh_path)
        expected = self.model.v_template.shape[0]
        if verts.shape[0] != expected:
            raise ValueError(f"OBJ has {verts.shape[0]} verts, expected {expected}")

        with torch.no_grad():
            self.model.v_template.copy_(torch.tensor(verts, dtype=torch.float32))
            self.model.shapedirs.zero_()
            if hasattr(self.model, "expr_dirs"):
                self.model.expr_dirs.zero_()
            self.model.posedirs.mul_(self.scale)

        print(
            f"G1ProportionSMPLX: loaded v_template from {tpose_mesh_path} "
            f"({verts.shape[0]} verts, scale={self.scale:.4f})"
        )

    @classmethod
    def from_g1_smplx_dir(
        cls,
        smplx_model_path: str,
        g1_smplx_dir: str = "data/body_profiles/unitree_g1/smplx",
    ) -> "G1ProportionSMPLX":
        """Load pre-baked params and T-pose mesh from the G1 SMPL-X body profile.

        Reads `g1_smplx_param.npz` for the reference scale and
        `g1_smplx_tpose.obj` for the v_template.
        """
        param_path = os.path.join(g1_smplx_dir, "g1_smplx_param.npz")
        tpose_path = os.path.join(g1_smplx_dir, "g1_smplx_tpose.obj")
        params = np.load(param_path)
        scale = float(params["scale"][0])
        return cls(smplx_model_path=smplx_model_path, tpose_mesh_path=tpose_path, scale=scale)

    @property
    def faces(self):
        return self.model.faces

    def _forward_smplx(self, motion_data, return_mesh=False, require_grad=False):
        """Standard SMPLX forward with betas=0 (shape baked into v_template)."""
        poses = motion_data["poses"]
        betas = motion_data["betas"]
        trans = motion_data["trans"]
        left_hand = motion_data.get("left_hand_pose", None)
        right_hand = motion_data.get("right_hand_pose", None)

        device = next(self.model.parameters()).device

        T = poses.shape[0]
        poses = poses.reshape(T, -1).to(device)

        betas_batch = betas.reshape(1, 10).repeat(T, 1).to(device)
        global_orient = poses[:, :3]
        body_pose = poses[:, 3 : 3 + 63]
        trans = trans.to(device)

        if left_hand is None:
            left_hand = torch.zeros((T, 45), dtype=torch.float32, device=device)
        else:
            left_hand = left_hand.reshape(T, 45).to(device)
        if right_hand is None:
            right_hand = torch.zeros((T, 45), dtype=torch.float32, device=device)
        else:
            right_hand = right_hand.reshape(T, 45).to(device)

        kwargs = dict(
            betas=betas_batch,
            global_orient=global_orient,
            body_pose=body_pose,
            left_hand_pose=left_hand,
            right_hand_pose=right_hand,
            transl=trans,
            expression=torch.zeros((T, 10), device=device),
            jaw_pose=torch.zeros((T, 3), device=device),
            leye_pose=torch.zeros((T, 3), device=device),
            reye_pose=torch.zeros((T, 3), device=device),
            return_verts=return_mesh,
            return_full_pose=True,
        )

        if require_grad:
            output = self.model(**kwargs)
        else:
            with torch.no_grad():
                output = self.model(**kwargs)
        return output


# G1SmplxHumanModel (the HOI-optimizer wrapper) lives in
# grail/models/g1_smplx_human.py — kept separate so this file stays
# importable from the sonic env (no pytorch3d dep).
