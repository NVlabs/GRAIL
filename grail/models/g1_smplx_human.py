"""G1SmplxHumanModel — HumanModel wrapper around G1ProportionSMPLX.

Used by the HOI optimizer (`grail/optimization/`) to produce G1-proportioned
SMPL-X output during 4D reconstruction. Lives in its own file (separate from
`g1_smplx_model.py`) so importing the lighter `G1ProportionSMPLX` from the
`sonic` env (which doesn't ship pytorch3d) doesn't pull in the SmplxHumanModel
parent class transitively.
"""

import os

import numpy as np
import torch

from grail.models.g1_smplx_model import G1ProportionSMPLX
from grail.models.human_model import SmplxHumanModel


def _root_joint_at_rest(smplx_model, betas, device):
    """Return the root (pelvis) joint at rest for a given SMPLX model and betas.

    This is `J_regressor @ v_shaped[root]`, a rotation-invariant property that
    depends only on the shaped template (betas + v_template + shapedirs).
    """
    from grail.models.smplx_model import generate_smplx_mesh

    tpose = {
        "betas": betas.reshape(1, 10).to(device),
        "poses": torch.zeros(1, 165, device=device),
        "trans": torch.zeros(1, 3, device=device),
        "left_hand_pose": torch.zeros(1, 45, device=device),
        "right_hand_pose": torch.zeros(1, 45, device=device),
    }
    _, _, joints = generate_smplx_mesh(
        smplx_model,
        tpose,
        output_joints=True,
        require_grad=False,
        device=device,
    )
    return joints[0, 0, :].detach()  # (3,) pelvis at rest


class G1SmplxHumanModel(SmplxHumanModel):
    """SMPL-X with Unitree G1 robot proportions baked into v_template.

    Inherits everything from SmplxHumanModel.  Only overrides the methods
    where the baked-proportion model diverges:

      __init__          – creates G1ProportionSMPLX from pre-baked OBJ
      generate_mesh     – uses baked model, no post-hoc scale
      load_shape_params – returns shared Jason params instead of per-character
      transform_global_motion – uses g1 model, zeros betas
    """

    G1_SMPLX_DIR = "data/body_profiles/unitree_g1/smplx"
    G1_PARAM_FILE = "g1_smplx_param.npz"
    G1_TPOSE_FILE = "g1_smplx_tpose.obj"

    def __init__(self, cfg: dict, device: str):
        super().__init__(cfg, device)
        # Stash the raw SMPLX created by SmplxHumanModel.__init__ before
        # load_shape_params overwrites self.model with the G1-baked model.
        self._raw_smplx_model = self.model

        smplx_path = cfg.get("smplx_model_path", "")
        self.smplx_dir = os.path.dirname(os.path.dirname(smplx_path))

        self.load_shape_params(cfg, None, device)

        self.model_type = "g1_smplx"

    def generate_mesh(self, motion_data, *, output_joints=False, require_grad=False):
        joints, vertices = self._forward_g1_smplx(
            motion_data,
            require_grad=require_grad,
            return_mesh=True,
        )
        faces = torch.from_numpy(self.g1_smplx.model.faces.astype(np.int64)).to(self.device).long()

        if output_joints:
            return vertices, faces, joints
        return vertices, faces

    def load_shape_params(self, cfg, character_name, device):
        """Load G1 shape params and T-pose mesh from its canonical body profile."""
        from grail.models.smplx_model import load_smplx_beta

        hm_cfg = cfg.get("human_model", cfg)
        g1_dir = hm_cfg.get("g1_smplx_dir", self.G1_SMPLX_DIR)

        param_path = os.path.join(g1_dir, self.G1_PARAM_FILE)
        tpose_path = os.path.join(g1_dir, self.G1_TPOSE_FILE)

        if not os.path.exists(param_path):
            raise FileNotFoundError(f"G1 SMPLX params not found: {param_path}")
        if not os.path.exists(tpose_path):
            raise FileNotFoundError(f"G1 SMPLX T-pose mesh not found: {tpose_path}")

        betas, scale = load_smplx_beta(param_path, device=device)

        self.g1_smplx = G1ProportionSMPLX(
            smplx_model_path=self.smplx_dir,
            tpose_mesh_path=tpose_path,
            scale=scale,
        )
        self.g1_smplx.model = self.g1_smplx.model.to(device)
        self.model = self.g1_smplx.model
        # Keep reference to the baked-model T-pose height (used for height ratio
        # during transform_global_motion; scale is already baked in, so no
        # further multiplication needed).
        self.g1_tpose_height = self._compute_baked_tpose_height()

        # Rebake the COCO17 lite model so keypoint regression matches the G1
        # body, not the neutral-adult body the parent class loaded.
        self._rebake_coco17_model_for_g1()

        return betas, scale

    def _rebake_coco17_model_for_g1(self):
        """Apply the G1 bake to self.coco17_model's buffers in place.

        Mirrors G1ProportionSMPLX: replace v_template with the G1 T-pose (on
        the 132-vertex subset the lite model keeps), zero shapedirs, scale
        posedirs by the G1 scale, and refresh J_template so get_skeleton
        returns G1 rest joints when betas=0.

        Idempotent: load_shape_params gets called again from
        HOIOptimizer._load_motion, but the coco17 buffers are static geometry
        — re-running posedirs.mul_(scale) would square the scale factor.
        """
        if getattr(self, "_coco17_rebaked", False):
            return

        raw_v = self._raw_smplx_model.v_template  # (10475, 3)
        lite_v = self.coco17_model.v_template  # (132, 3)
        smplx_vids = torch.cdist(lite_v, raw_v).argmin(dim=1)
        assert torch.allclose(
            raw_v[smplx_vids], lite_v
        ), "coco17 lite v_template is not a clean subset of the raw SMPLX v_template"

        g1_full_v = self.g1_smplx.model.v_template  # (10475, 3), baked
        self.coco17_model.v_template = g1_full_v[smplx_vids].clone()
        self.coco17_model.shapedirs.zero_()
        self.coco17_model.posedirs.mul_(self.g1_smplx.scale)
        self.coco17_model.J_template = self.g1_smplx.model.J_regressor @ g1_full_v
        self.coco17_model.J_shapedirs.zero_()

        self._coco17_rebaked = True

    def _compute_baked_tpose_height(self):
        """T-pose height of the baked G1 model (scale already in v_template)."""
        from grail.models.smplx_model import get_tpose_human_height

        return get_tpose_human_height(
            self.g1_smplx.model,
            betas=torch.zeros(10, device=self.device),
            device=self.device,
        )

    def compute_tpose_height(self, shape_params):
        """Override: return the cached baked-G1 T-pose height."""
        return self.g1_tpose_height

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
        """G1-aware global motion transform.

        The rigid coordinate change runs against the RAW SMPLX model with
        HMR's betas so the pelvis-offset correction inside
        transform_global_motion_smplx uses J_rest_HMR[0] — the offset that
        matches the input `trans`. We then shift by (J_rest_HMR[0] -
        J_rest_G1[0]) so the G1 body's root joint lands at HMR's predicted
        root world position. Finally we apply the height-ratio scaling
        pivoted at cam_t.
        """
        from grail.models.smplx_model import (
            transform_global_motion as transform_global_motion_smplx,
        )

        # Step 1: rigid transform against the RAW SMPLX so `diff = J_rest_HMR[0]`.
        motion_data = transform_global_motion_smplx(
            self._raw_smplx_model,
            global_motion_data,
            incam_motion_data,
            cam_R=cam_R,
            cam_t=cam_t,
            align_frame=align_frame,
            use_global=use_global,
            gt_beta=None,
            gt_scale=1.0,
            device=device,
        )

        # Step 2: shift output-side pelvis from HMR's offset to G1's offset.
        hmr_betas = incam_motion_data["betas"].reshape(1, 10).to(device)
        j_hmr_0 = _root_joint_at_rest(self._raw_smplx_model, hmr_betas, device)
        if not hasattr(self, "_g1_root_offset"):
            self._g1_root_offset = _root_joint_at_rest(
                self.g1_smplx.model, torch.zeros(1, 10, device=device), device
            )
        motion_data["trans"] = motion_data["trans"] + (j_hmr_0 - self._g1_root_offset)

        # Step 3: height-ratio scaling (unchanged).
        pred_height = incam_motion_data.get("predicted_body_height", None)
        if pred_height is None:
            raise RuntimeError(
                "predicted_body_height missing from incam_motion_data. "
                "Regenerate HMR output (step 1 of run_4dhoi_recon.py) to populate this field."
            )
        human_scale_ratio = float(self.g1_tpose_height) / float(pred_height)

        motion_data["trans"] = (motion_data["trans"] - cam_t) * human_scale_ratio + cam_t

        motion_data["betas"] = torch.zeros(1, 10, device=device)
        motion_data["scale"] = 1.0
        return motion_data

    def _forward_g1_smplx(self, motion_data, *, require_grad=False, return_mesh=False):
        """Run G1ProportionSMPLX forward, returning torch tensors on device.

        Returns:
            (joints, vertices) – vertices is None when return_mesh is False.
        """

        def _to_tensor(x):
            if isinstance(x, torch.Tensor):
                return x.float() if require_grad else x.detach().float()
            return torch.tensor(x, dtype=torch.float32)

        data = {
            "poses": _to_tensor(motion_data["poses"]),
            "betas": torch.zeros(10, dtype=torch.float32),
            "trans": _to_tensor(motion_data["trans"]),
        }
        for key in ("left_hand_pose", "right_hand_pose"):
            if key in motion_data and motion_data[key] is not None:
                data[key] = _to_tensor(motion_data[key])

        output = self.g1_smplx._forward_smplx(
            data,
            return_mesh=return_mesh,
            require_grad=require_grad,
        )

        joints = output.joints[:, :55, :3].to(self.device)
        vertices = output.vertices.to(self.device) if return_mesh else None
        return joints, vertices
