#!/usr/bin/env python
# ruff: noqa: E501
"""Retarget GRAIL 4D HOI motions to a G1 robot via GMR.

Reads SMPL-X human pose + object trajectory from `grail.pipelines.recon_4dhoi` output
and produces a G1 robot motion library under `data/motion_lib/<output>/`:
`robot/*.pkl`, `objects/*.pkl`, `object_usd/*.usd`, `meta/*.pkl`.

The pipeline:
  1. (Optional) Bake G1 body proportions into the SMPL-X model so the
     IK solver targets G1-shaped joint positions
     (`grail.models.g1_smplx_model.G1ProportionSMPLX`).
  2. Run GMR's IK + temporal smoothing through `grail.adapters.gmr` (which
     applies GRAIL-specific monkey-patches to the public GMR package at
     import time).
  3. Compose per-motion object USDs from the source meshes + tracked poses.

Driven from `grail/retargeting/scripts/retarget.sh` /
`grail/retargeting/scripts/retarget_pipeline.sh`. Full docs:
`docs/source/retargeting.md`.
"""

# Pre-import mink to dodge an import-order bug on AMD Zen 4 CPUs (AVX-512):
# loading mink as part of the GMR import chain (gmr.py -> general_motion_retargeting
# -> motion_retarget -> mink) crashes Python with `SystemError: structseq.c:476`
# during PyStructSequence_New. Importing mink first sidesteps the corruption.
import mink  # noqa: F401

import argparse
import glob
import os
from pathlib import Path
import pickle
import shutil
import subprocess
import sys
import tempfile
import threading

from grail.adapters.gmr import (
    GMR,
    IK_CONFIG_DICT,
    ROBOT_BASE_DICT,
    ROBOT_XML_DICT,
    VIEWER_CAM_DISTANCE_DICT,
    RobotMotionViewer,
    draw_frame,
)
from grail.models.g1_smplx_model import G1ProportionSMPLX
from grail.retargeting.convert_collision_to_sdf import convert_to_sdf
import joblib
import mujoco as mj
import mujoco.viewer as mjv
import numpy as np
from pxr import Sdf, Usd
from rich import print
from scipy.spatial.transform import Rotation as R
import smplx
from smplx.joint_names import JOINT_NAMES
import torch
from tqdm import tqdm
import trimesh

keyboard = None  # lazy import: only loaded when viewer is active (needs X display)

_REPO_ROOT = Path(__file__).resolve().parents[2]
_GMR_IK_CONFIG = _REPO_ROOT / "data/body_profiles/unitree_g1/smplx/gmr_smplx_to_g1.json"
IK_CONFIG_DICT["smplx"] = {
    "unitree_g1": _GMR_IK_CONFIG,
    "unitree_g1_with_hands": _GMR_IK_CONFIG,
}


def _install_numpy_pickle_aliases() -> None:
    core = getattr(np, "_core", np.core)
    sys.modules.setdefault("numpy._core", core)
    sys.modules.setdefault("numpy._core.multiarray", np.core.multiarray)
    sys.modules.setdefault("numpy._core.numeric", np.core.numeric)
    if hasattr(np.core, "_multiarray_umath"):
        sys.modules.setdefault("numpy._core._multiarray_umath", np.core._multiarray_umath)


def _pickle_load_compat(file_path):
    with open(file_path, "rb") as f:
        try:
            return pickle.load(f)
        except ModuleNotFoundError as exc:
            if not (exc.name or "").startswith("numpy._core"):
                raise
            _install_numpy_pickle_aliases()
            f.seek(0)
            return pickle.load(f)


def add_mesh(spec, mesh, name):

    vertices = mesh.vertices
    faces = mesh.faces

    # Defaults
    main = spec.default
    main.mesh.scale = np.array([1] * 3, dtype=np.float64)
    main.geom.type = mj.mjtGeom.mjGEOM_MESH

    rgba = np.array([200, 150, 100, 255]) / 255.0

    # Create Body and add mesh to the Geom of the Body
    mesh = spec.add_mesh(name=name, uservert=vertices.flatten(), userface=faces.flatten())

    body = spec.worldbody.add_body(name=name)
    body.add_geom(meshname=name, rgba=rgba)
    body.add_freejoint()

    return spec


def _parse_csv(value: str) -> list[str]:
    return [x.strip() for x in value.split(",") if x.strip()]


def _height_at_xy_from_triangle(xy: np.ndarray, tri: np.ndarray, eps: float = 1e-7):
    p = np.asarray(xy, dtype=np.float64)
    a, b, c = tri[:, :2]
    v0 = b - a
    v1 = c - a
    v2 = p - a
    den = v0[0] * v1[1] - v1[0] * v0[1]
    if abs(den) < eps:
        return None
    u = (v2[0] * v1[1] - v1[0] * v2[1]) / den
    v = (v0[0] * v2[1] - v2[0] * v0[1]) / den
    w = 1.0 - u - v
    if u < -eps or v < -eps or w < -eps:
        return None
    return float(w * tri[0, 2] + u * tri[1, 2] + v * tri[2, 2])


def _terrain_height_near_xy(vertices: np.ndarray, faces: np.ndarray, xy: np.ndarray, radius: float):
    heights = []
    for face in faces:
        height = _height_at_xy_from_triangle(xy, vertices[face])
        if height is not None:
            heights.append(height)
    if heights:
        return max(heights)

    dists = np.linalg.norm(vertices[:, :2] - np.asarray(xy, dtype=np.float64)[None, :], axis=1)
    nearby = vertices[dists <= radius]
    if len(nearby) == 0:
        return None
    return float(np.max(nearby[:, 2]))


def _apply_downstairs_initial_height_correction(
    *,
    output_qpos_list,
    retarget,
    object_mesh,
    object_transl,
    object_rot_quats,
    foot_body_names,
    radius,
    clearance,
):
    if not output_qpos_list:
        return 0.0

    if isinstance(object_mesh, trimesh.Scene):
        object_mesh = object_mesh.dump(concatenate=True)
    vertices = np.asarray(object_mesh.vertices, dtype=np.float64)
    faces = np.asarray(object_mesh.faces, dtype=np.int64)
    if len(vertices) == 0 or len(faces) == 0:
        print("  WARNING: downstairs correction skipped because object mesh has no triangles")
        return 0.0

    rot = R.from_quat(object_rot_quats[0], scalar_first=True)
    world_vertices = rot.apply(vertices) + object_transl[0]

    data = retarget.configuration.data
    old_qpos = data.qpos.copy()
    try:
        data.qpos[:] = output_qpos_list[0]
        mj.mj_forward(retarget.model, data)

        required_shift = 0.0
        checked = 0
        for body_name in foot_body_names:
            body_id = mj.mj_name2id(retarget.model, mj.mjtObj.mjOBJ_BODY, body_name)
            if body_id < 0:
                print(f"  WARNING: downstairs correction foot body not found: {body_name}")
                continue
            foot_pos = data.xpos[body_id].copy()
            terrain_height = _terrain_height_near_xy(world_vertices, faces, foot_pos[:2], radius)
            if terrain_height is None:
                print(f"  WARNING: no terrain height found near {body_name}")
                continue
            checked += 1
            required_shift = max(required_shift, terrain_height + clearance - float(foot_pos[2]))
    finally:
        data.qpos[:] = old_qpos
        mj.mj_forward(retarget.model, data)

    if checked == 0 or required_shift <= 0.0:
        return 0.0

    for qpos in output_qpos_list:
        qpos[2] += required_shift
    print(
        f"  Downstairs initial z correction: +{required_shift:.4f}m "
        f"(feet={foot_body_names}, radius={radius:.3f}, clearance={clearance:.3f})"
    )
    return float(required_shift)


class ModifiedRobotMotionViewer(RobotMotionViewer):
    def __init__(
        self,
        robot_type,
        camera_follow=True,
        motion_fps=30,
        transparent_robot=0,
        # video recording
        record_video=False,
        video_path=None,
        video_width=640,
        video_height=480,
        keyboard_callback=None,
        obj_meshes=[],
    ):

        self.robot_type = robot_type
        self.xml_path = ROBOT_XML_DICT[robot_type]

        spec = mj.MjSpec.from_file(str(self.xml_path))
        for i, obj_mesh in enumerate(obj_meshes):
            spec = add_mesh(spec, obj_mesh, name=f"obj_{i}")

        self.model = spec.compile()
        self.data = mj.MjData(self.model)
        self.robot_base = ROBOT_BASE_DICT[robot_type]
        self.viewer_cam_distance = VIEWER_CAM_DISTANCE_DICT[robot_type]
        mj.mj_step(self.model, self.data)

        self.motion_fps = motion_fps
        self.camera_follow = camera_follow
        self.record_video = record_video

        self.viewer = mjv.launch_passive(
            model=self.model,
            data=self.data,
            show_left_ui=False,
            show_right_ui=False,
            key_callback=keyboard_callback,
        )

        self.viewer.opt.flags[mj.mjtVisFlag.mjVIS_TRANSPARENT] = transparent_robot

        # Initialize rigid body recording structures
        self.rigid_body_data = []  # List to store data for each timestep
        self.recording_enabled = False  # Flag to enable/disable recording

    def step(
        self,
        # robot data
        root_pos,
        root_rot,
        dof_pos,
        # human data
        human_motion_data=None,
        show_human_body_name=False,
        # scale for human point visualization
        human_point_scale=0.1,
        # human pos offset add for visualization
        human_pos_offset=np.array([0.0, 0.0, 0]),
        # rate limit
        rate_limit=True,
        follow_camera=True,
        obj_pose=None,
        all_joint_pos=None,
    ):
        """
        by default visualize robot motion.
        also support visualize human motion by providing human_motion_data, to compare with robot motion.

        human_motion_data is a dict of {"human body name": (3d global translation, 3d global rotation)}.

        if rate_limit is True, the motion will be visualized at the same rate as the motion data.
        else, the motion will be visualized as fast as possible.
        """

        self.data.qpos[:3] = root_pos
        self.data.qpos[3:7] = root_rot  # quat need to be scalar first! for mujoco
        self.data.qpos[7 : 7 + len(dof_pos)] = dof_pos
        if obj_pose is not None:
            self.data.qpos[7 + len(dof_pos) :] = obj_pose

        mj.mj_forward(self.model, self.data)

        # Record rigid body data if recording is enabled
        if self.recording_enabled:
            timestep_data = {}
            for body_id in range(self.model.nbody):
                body_name = self.model.body(body_id).name
                # Get world position
                pos = self.data.xpos[body_id].copy()
                # Get world orientation as rotation matrix, then convert to quaternion
                rot_mat = self.data.xmat[body_id].reshape(3, 3).copy()
                # Convert rotation matrix to quaternion (scalar-last format)
                quat = R.from_matrix(rot_mat).as_quat()[..., [3, 0, 1, 2]]

                timestep_data[body_name] = {
                    "position": pos,
                    "quaternion": quat,  # [x, y, z, w] format
                    "rotation_matrix": rot_mat,
                }
            self.rigid_body_data.append(timestep_data)

        if follow_camera:
            self.viewer.cam.lookat = self.data.xpos[self.model.body(self.robot_base).id]
            self.viewer.cam.distance = self.viewer_cam_distance
            self.viewer.cam.elevation = -10  # 正面视角，轻微向下看

        if human_motion_data is not None:
            # Clean custom geometry
            self.viewer.user_scn.ngeom = 0
            # Draw the task targets for reference
            for human_body_name, (pos, rot) in human_motion_data.items():
                draw_frame(
                    pos,
                    R.from_quat(rot, scalar_first=True).as_matrix(),
                    self.viewer,
                    human_point_scale,
                    pos_offset=human_pos_offset,
                    joint_name=human_body_name if show_human_body_name else None,
                )

        if all_joint_pos is not None:
            for pos, _ in all_joint_pos.values():
                geom = self.viewer.user_scn.geoms[self.viewer.user_scn.ngeom]
                mj.mjv_initGeom(
                    geom,
                    type=mj.mjtGeom.mjGEOM_SPHERE,
                    size=[0.006, 0.006, 0.006],
                    pos=pos,
                    mat=np.eye(3).flatten(),
                    rgba=[1, 1, 0, 1],
                )
                self.viewer.user_scn.ngeom += 1

        self.viewer.sync()


PAD_FRAMES = 10

space_was_pressed = False
_listener_started = False

np.set_printoptions(precision=3, suppress=True)


def on_press(key):
    global space_was_pressed
    from pynput import keyboard as _kb

    if key == _kb.Key.space:
        space_was_pressed = True


def _start_listener():
    from pynput import keyboard as _kb

    with _kb.Listener(on_press=on_press) as listener:
        listener.join()


def ensure_listener_started():
    """Start the keyboard listener thread if not already started."""
    global _listener_started
    if not _listener_started:
        _listener_started = True
        listener_thread = threading.Thread(target=_start_listener, daemon=True)
        listener_thread.start()


def space_pressed_since_last_call():
    global space_was_pressed
    if space_was_pressed:
        space_was_pressed = False
        return True
    return False


def forward_smplx(smplx_model, motion_data, return_mesh=False):
    poses = motion_data["poses"]
    betas = motion_data["betas"]
    trans = motion_data["trans"]
    left_hand_pose = motion_data.get("left_hand_pose", None)
    right_hand_pose = motion_data.get("right_hand_pose", None)

    frame_num = poses.shape[0]
    poses = poses.reshape(frame_num, -1)

    # Prepare SMPLX inputs (assuming poses, betas, trans are already torch tensors)
    betas_batch = betas.reshape(1, 10).repeat(frame_num, 1)
    global_orient = poses[:, :3]
    body_pose = poses[:, 3 : 3 + 21 * 3]
    transl = trans

    # Zero hand poses and facial expressions
    if left_hand_pose is None:
        left_hand_pose = torch.zeros((frame_num, 45), device="cpu").float()
    else:
        left_hand_pose = left_hand_pose.reshape(frame_num, 45)

    if right_hand_pose is None:
        right_hand_pose = torch.zeros((frame_num, 45), device="cpu").float()
    else:
        right_hand_pose = right_hand_pose.reshape(frame_num, 45)

    expression = torch.zeros((frame_num, 10), device="cpu").float()
    jaw_pose = torch.zeros((frame_num, 3), device="cpu").float()
    leye_pose = torch.zeros((frame_num, 3), device="cpu").float()
    reye_pose = torch.zeros((frame_num, 3), device="cpu").float()

    # Generate SMPLX mesh
    with torch.no_grad():
        smplx_output = smplx_model(
            betas=betas_batch,
            global_orient=global_orient,
            body_pose=body_pose,
            left_hand_pose=left_hand_pose,
            right_hand_pose=right_hand_pose,
            transl=transl,
            expression=expression,
            jaw_pose=jaw_pose,
            leye_pose=leye_pose,
            reye_pose=reye_pose,
            return_verts=return_mesh,
            return_full_pose=True,
        )

    return smplx_output


def tensor_to(args):
    if isinstance(args, torch.Tensor):
        args_new = args
        return args_new
    elif isinstance(args, np.ndarray):
        return tensor_to(torch.tensor(args))
    elif isinstance(args, list):
        return [tensor_to(x) for x in args]
    elif isinstance(args, dict):
        return {k: tensor_to(x) for k, x in args.items()}
    else:
        return args


def compute_smpl_height(beta: np.ndarray) -> float:
    """
    Compute SMPL human height from beta parameters.

    Args:
        beta: Shape parameters (16,) or (10,)

    Returns:
        Estimated human height in meters
    """
    if isinstance(beta, torch.Tensor):
        beta = beta.numpy()

    if len(beta.shape) == 1:
        return 1.66 + 0.1 * beta[0]
    else:
        return 1.66 + 0.1 * beta[0, 0]


def load_hoi_sequence(
    file_path,
    smplx_model_path,
    target_fps,
    scene_scale=1.0,
    height_offset=0.0,
    return_mesh=False,
    g1_smplx=None,
):
    """
    Load and preprocess HOI data.

    Args:
        file_path (str): Path to the .pt file.
        g1_smplx: Optional G1ProportionSMPLX instance. When provided, uses
                   its model (G1-proportioned v_template) instead of standard SMPLX.

    Returns:
        tuple: (human_joints, object_poses) - processed data.
    """
    hoi_data = _pickle_load_compat(file_path)

    if g1_smplx is not None:
        smplx_model = g1_smplx.model
    else:
        smplx_model = smplx.create(
            model_path=smplx_model_path, model_type="smplx", num_betas=10, num_pca_comps=45
        ).to("cpu")

    smplx_poses = hoi_data["human_data"]
    num_frames = smplx_poses["poses"].shape[0]

    # pad the smplx poses with some dummy frames first frame
    keys = ["poses", "trans", "left_hand_pose", "right_hand_pose"]
    for k in keys:
        pad_shape = list(smplx_poses[k].shape)
        pad_shape[0] = PAD_FRAMES

        interp_weights = np.linspace(0, 1, PAD_FRAMES).astype(smplx_poses[k].dtype)
        inter_weights_shape = [PAD_FRAMES] + [1] * (len(smplx_poses[k].shape) - 1)
        interp_weights = interp_weights.reshape(inter_weights_shape)

        pad = interp_weights * smplx_poses[k][[0]]

        smplx_poses[k] = np.concatenate([pad, smplx_poses[k]], axis=0)

    scale = smplx_poses.get("scale", 1.0)
    if g1_smplx is not None:
        scale = 1.0

    transl = smplx_poses["trans"][:, None]

    smplx_output = forward_smplx(smplx_model, tensor_to(smplx_poses), return_mesh=return_mesh)

    object_transl = hoi_data["obj_data"]["obj_t"]
    object_rot_quats = R.from_matrix(hoi_data["obj_data"]["obj_R"]).as_quat(scalar_first=True)
    joint_names = JOINT_NAMES[: len(smplx_model.parents)]
    parents = smplx_model.parents
    global_orient = smplx_output.global_orient.detach().cpu().numpy()
    smplx_joints = ((smplx_output.joints - transl) * scale + transl).detach().cpu().numpy()

    num_frames = global_orient.shape[0]

    full_body_pose = smplx_output.full_pose.reshape(num_frames, -1, 3)

    smplx_data_frames = []
    for curr_frame in range(len(global_orient)):
        result = {}
        single_global_orient = global_orient[curr_frame]
        single_full_body_pose = full_body_pose[curr_frame]
        single_joints = smplx_joints[curr_frame]
        joint_orientations = []
        for i, joint_name in enumerate(joint_names):
            if i == 0:
                rot = R.from_rotvec(single_global_orient)
            else:
                rot = joint_orientations[parents[i]] * R.from_rotvec(
                    single_full_body_pose[i].squeeze()
                )
            joint_orientations.append(rot)
            result[joint_name] = (single_joints[i], rot.as_quat(scalar_first=True))

        smplx_data_frames.append(result)

    # hard code the height scale to 1.0 for now
    height_scale = 1.0  # float(robot_height / human_height)
    height_scale *= scene_scale
    print(f"height_scale: {height_scale}")

    mesh_file = Path(file_path).parent.parent / "mesh_data" / "model.obj"

    object_mesh = trimesh.load(mesh_file)
    if isinstance(object_mesh, trimesh.Scene):
        object_mesh = object_mesh.dump(concatenate=True)
    object_mesh.vertices *= height_scale

    for curr_frame in range(len(global_orient)):
        for joint_name in joint_names:
            smplx_data_frames[curr_frame][joint_name] = (
                smplx_data_frames[curr_frame][joint_name][0] * height_scale,
                smplx_data_frames[curr_frame][joint_name][1],
            )

    object_transl *= height_scale

    # Apply height offset to shift everything vertically (e.g., -0.1365 to lower)
    if height_offset != 0.0:
        print(f"Applying height_offset: {height_offset}")
        for curr_frame in range(len(global_orient)):
            for joint_name in joint_names:
                pos, rot = smplx_data_frames[curr_frame][joint_name]
                pos = pos.copy()
                pos[2] += height_offset
                smplx_data_frames[curr_frame][joint_name] = (pos, rot)
        object_transl[:, 2] += height_offset

    # Extract object scale for data format compatibility with old pipeline
    obj_scale = hoi_data["obj_data"].get("obj_scale")

    ret = {
        "smplx_data_frames": smplx_data_frames,
        "transl": smplx_poses["trans"] * height_scale,
        "obj_data": {
            "object": {
                "mesh": object_mesh,
                "transl": object_transl,
                "rot_quats": object_rot_quats,
                "file_path": str(mesh_file),
            },
        },
        "height_scale": height_scale,
        "obj_scale": obj_scale,
    }

    if return_mesh:
        ret["mesh_verts"] = (
            ((smplx_output.vertices - transl) * scale + transl).detach().cpu().numpy()
        )
        ret["mesh_faces"] = smplx_model.faces

    if "obj_contact_points" in hoi_data["obj_data"]:
        ret["obj_data"]["object"]["obj_contact_points"] = hoi_data["obj_data"]["obj_contact_points"]
    else:
        print("no contact points found")

    # Extract scene/table data for GeniHOI metadata
    scene_key = (
        "scene_data" if "scene_data" in hoi_data else "scene" if "scene" in hoi_data else None
    )
    if scene_key:
        scene_info = hoi_data[scene_key]
        if scene_info is not None and "table" in scene_info:
            table_info = scene_info["table"]
            ret["scene_table"] = {}
            if "pos" in table_info:
                pos = table_info["pos"]
                ret["scene_table"]["table_pos"] = (
                    pos.tolist() if hasattr(pos, "tolist") else list(pos)
                )
            if "size" in table_info:
                size = table_info["size"]
                ret["scene_table"]["table_size"] = (
                    size.tolist() if hasattr(size, "tolist") else list(size)
                )
            # Identity quaternion for cuboid tables
            ret["scene_table"]["table_quat"] = [1.0, 0.0, 0.0, 0.0]

    return ret


def params2torch(params, dtype=torch.float32):
    return {k: torch.from_numpy(v).type(dtype) for k, v in params.items()}


def load_grab_sequence(
    file_path,
    smplx_model_path,
    target_fps,
    scene_scale=1.0,
    height_offset=0.0,
    return_mesh=False,
    g1_smplx=None,
):

    data_dir = str(Path(file_path).parent.parent)

    seq_data = np.load(file_path, allow_pickle=True)
    seq_data = {k: seq_data[k].item() for k in seq_data.files}

    framerate = int(seq_data["framerate"])

    downsample_factor = int(framerate / target_fps)
    if target_fps * downsample_factor != framerate:
        raise ValueError(f"Target FPS {target_fps} and framerate {framerate} are not compatible")

    # downsample the data
    for data in [
        seq_data["body"]["params"],
        seq_data["object"]["params"],
        seq_data["table"]["params"],
    ]:
        for k in data.keys():
            data[k] = data[k][::downsample_factor]

    # Pad the body parameters with a dummy first frame
    # \todo: interpolate
    for k in seq_data["body"]["params"].keys():
        interp_weights = np.linspace(0, 1, PAD_FRAMES).astype(seq_data["body"]["params"][k].dtype)[
            :, None
        ]
        pad = interp_weights * seq_data["body"]["params"][k][0][None,]
        seq_data["body"]["params"][k] = np.concatenate([pad, seq_data["body"]["params"][k]], axis=0)

    sbj_mesh = os.path.join(data_dir, seq_data["body"]["vtemp"])
    sbj_mesh = trimesh.load(sbj_mesh)

    sbj_vtemp = sbj_mesh.vertices
    T = seq_data["body"]["params"]["transl"].shape[0]
    n_comps = seq_data["n_comps"]
    gender = seq_data["gender"]

    if g1_smplx is not None:
        # Use G1-proportioned v_template and zeroed shapedirs
        sbj_m = smplx.create(
            model_path=smplx_model_path,
            model_type="smplx",
            gender=gender,
            num_betas=10,
            num_pca_comps=n_comps,
            batch_size=T,
        )
        with torch.no_grad():
            sbj_m.v_template.copy_(g1_smplx.model.v_template)
            sbj_m.shapedirs.zero_()
            if hasattr(sbj_m, "expr_dirs"):
                sbj_m.expr_dirs.zero_()
            sbj_m.posedirs.copy_(g1_smplx.model.posedirs)
    else:
        sbj_m = smplx.create(
            model_path=smplx_model_path,
            model_type="smplx",
            gender=gender,
            num_betas=10,
            num_pca_comps=n_comps,
            v_template=sbj_vtemp,
            batch_size=T,
        )

    sbj_parms = params2torch(seq_data["body"]["params"])
    sbj_parms["return_full_pose"] = True
    sbj_parms["return_verts"] = return_mesh
    smplx_output = sbj_m(**sbj_parms)

    num_frames = smplx_output.global_orient.shape[0]
    smplx_joints = (smplx_output.joints).detach().cpu().numpy()
    global_orient = smplx_output.global_orient.squeeze()
    full_body_pose = smplx_output.full_pose.reshape(num_frames, -1, 3)

    joint_names = JOINT_NAMES[: len(sbj_m.parents)]
    parents = sbj_m.parents

    object_data = seq_data["object"]
    object_rot_quats = R.from_rotvec(-object_data["params"]["global_orient"]).as_quat(
        scalar_first=True
    )
    object_transl = object_data["params"]["transl"]

    table_rot_quats = R.from_rotvec(-seq_data["table"]["params"]["global_orient"]).as_quat(
        scalar_first=True
    )
    table_transl = seq_data["table"]["params"]["transl"]

    smplx_data_frames = []
    for curr_frame in range(len(global_orient)):
        result = {}
        single_global_orient = global_orient[curr_frame]
        single_full_body_pose = full_body_pose[curr_frame]
        single_joints = smplx_joints[curr_frame]
        joint_orientations = []
        for i, joint_name in enumerate(joint_names):
            if i == 0:
                rot = R.from_rotvec(single_global_orient)
            else:
                rot = joint_orientations[parents[i]] * R.from_rotvec(
                    single_full_body_pose[i].squeeze()
                )
            joint_orientations.append(rot)
            result[joint_name] = (single_joints[i], rot.as_quat(scalar_first=True))

        smplx_data_frames.append(result)

    # scale the entire scene down to the robot height:
    if g1_smplx is not None:
        # G1 proportions already baked in — no height scaling needed
        height_scale = 1.0
    else:
        human_height = 0.93 * (sbj_vtemp.max(axis=0)[1] - sbj_vtemp.min(axis=0)[1])
        robot_height = 1.36
        height_scale = robot_height / human_height
    height_scale *= scene_scale
    print(f"height_scale: {height_scale}")

    object_mesh = trimesh.load(os.path.join(data_dir, object_data["object_mesh"]))
    object_mesh.vertices *= height_scale

    table_mesh = trimesh.load(os.path.join(data_dir, seq_data["table"]["table_mesh"]))
    table_mesh.vertices *= height_scale

    for curr_frame in range(len(global_orient)):
        for joint_name in joint_names:
            smplx_data_frames[curr_frame][joint_name] = (
                smplx_data_frames[curr_frame][joint_name][0] * height_scale,
                smplx_data_frames[curr_frame][joint_name][1],
            )

    object_transl *= height_scale
    table_transl *= height_scale

    # Apply height offset to shift human and object vertically (e.g., -0.1365 to lower)
    # Table is NOT offset — it stays at its original position
    if height_offset != 0.0:
        print(f"Applying height_offset: {height_offset}")
        for curr_frame in range(len(global_orient)):
            for joint_name in joint_names:
                pos, rot = smplx_data_frames[curr_frame][joint_name]
                pos = pos.copy()
                pos[2] += height_offset
                smplx_data_frames[curr_frame][joint_name] = (pos, rot)
        object_transl[:, 2] += height_offset

    ret = {
        "smplx_data_frames": smplx_data_frames,
        "transl": seq_data["body"]["params"]["transl"] * height_scale,
        "obj_data": {
            seq_data["obj_name"]: {
                "mesh": object_mesh,
                "transl": object_transl,
                "rot_quats": object_rot_quats,
            },
            "table": {
                "mesh": table_mesh,
                "transl": table_transl,
                "rot_quats": table_rot_quats,
            },
        },
        "height_scale": height_scale,
    }

    if return_mesh:
        ret["mesh_verts"] = smplx_output.vertices.detach().cpu().numpy() * height_scale
        ret["mesh_faces"] = sbj_m.faces

    return ret


def convert_gmr_data_to_motion_lib(gmr_data, output_pkl_path, mj_model, seq_name):
    """
    Convert GMR retargeted pkl data to motion_lib format.

    Args:
        input_pkl_path: Path to input pkl file with GMR retargeted data
        output_pkl_path: Path to output pkl file
        mj_model: mujoco model for robot
        seq_name: Name for the sequence (default: derived from input filename)
        fix_height: Whether to fix the height so feet touch ground (default: True)
    """

    # Extract data
    fps = gmr_data.get("fps", 30)
    root_pos = gmr_data["root_pos"]  # (N, 3)
    root_rot = gmr_data["root_rot"]  # (N, 4) - quaternion format
    dof_pos = gmr_data["dof_pos"]  # (N, num_dofs) - could be 43 or 29

    # Determine if we need to extract first 29 DOFs
    num_dof = mj_model.nu
    if dof_pos.shape[1] > num_dof:
        print(f"Extracting first {num_dof} DOFs from {dof_pos.shape[1]} DOFs")
        dof = dof_pos[:, :num_dof]
    else:
        dof = dof_pos

    # Ensure root_rot is in the correct format [x, y, z, w]
    # The GMR data might already be in [w, x, y, z] or [x, y, z, w] format
    # We'll convert to [x, y, z, w] for processing
    if root_rot.shape[1] == 4:
        # Assume it's already in quaternion format, might need to check convention
        root_rot_xyzw = root_rot  # (N, 4)
    else:
        raise ValueError(f"Unexpected root_rot shape: {root_rot.shape}")

    num_frames = dof.shape[0]

    # Convert DOF to pose_aa format
    def joint_to_body_name(joint_name: str) -> str:
        j = mj_model.joint(joint_name)  # joint view by name
        body_id = mj_model.jnt_bodyid[j.id]  # integer body index for that joint
        return mj_model.body(body_id).name

    # find local space axis for each actuator
    dof_names = []
    dof_body_names = []
    dof_axis = []
    for i in range(mj_model.nu):  # loop over actuators
        j_id = mj_model.actuator_trnid[i, 0]  # joint id this actuator drives
        axis = mj_model.jnt_axis[j_id].copy()  # local hinge axis (x,y,z), unit vector
        dof_axis.append(axis)

        name = mj_model.actuator(i).name
        dof_names.append(name)

        dof_body_name = joint_to_body_name(name)
        dof_body_names.append(dof_body_name)

    dof_axis = np.stack(dof_axis, axis=0)

    dof_aa = dof_axis[None,] * dof[..., None]
    pose_aa = np.zeros((num_frames, len(dof_body_names) + 1, 3))
    pose_aa[:, 1:] = dof_aa

    # Set root rotation as axis-angle
    pose_aa[:, 0:1, :] = R.from_quat(root_rot_xyzw).as_rotvec()[:, None, :]

    # Fix height if requested (make sure feet touch ground)
    root_trans_offset = root_pos.copy()

    # Convert root_rot to [w, x, y, z] format for motion_lib
    root_rot_wxyz = root_rot_xyzw[:, [3, 0, 1, 2]]

    # Create motion_lib entry
    entry_dict = {
        "root_trans_offset": root_trans_offset.astype(np.float32),
        "pose_aa": pose_aa.astype(np.float32),
        "dof": dof.astype(np.float32),
        "root_rot": root_rot_wxyz.astype(np.float32),
        "smpl_joints": np.zeros((num_frames, 24, 3)).astype(np.float32),  # Placeholder
        "fps": fps,
    }

    # Create motion_lib dictionary
    motion_lib_dict = {seq_name: entry_dict}

    # Save to output path
    os.makedirs(os.path.dirname(output_pkl_path), exist_ok=True)
    print(f"Saving motion_lib to {output_pkl_path}...")
    joblib.dump(motion_lib_dict, output_pkl_path, compress=True)
    print(f"Successfully converted {seq_name}: {num_frames} frames at {fps} fps")
    return motion_lib_dict


def copy_mesh_file(mesh_dir: str, object_name: str, seq_name: str, output_mesh_dir: str) -> bool:
    """
    Copy object mesh file (.usda) to output directory with sequence name.

    Args:
        mesh_dir: Source directory containing mesh files
        object_name: Name of the object (e.g., 'mug')
        seq_name: Sequence name to use for output filename
        output_mesh_dir: Output directory for mesh files

    Returns:
        True if copy successful, False otherwise
    """

    print(f"Copying mesh file: {object_name} to {output_mesh_dir}")
    output_path = Path(output_mesh_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Try .usda first, then .usd
    for ext in [".usda", ".usd"]:
        src_path = Path(mesh_dir) / f"{object_name}{ext}"
        if src_path.exists():
            dst_path = output_path / f"{seq_name}.usda"
            try:
                shutil.copy2(src_path, dst_path)
                print(f"  Copied mesh: {src_path} -> {dst_path}")
                return True
            except Exception as e:
                print(f"  Error copying mesh: {e}")
                return False

    print(f"  Warning: No mesh file found for object '{object_name}' in {mesh_dir}")
    return False


def prepare_unique_mesh(obj_path: Path, seq_name: str, tmp_dir: Path) -> Path:
    """Copy mesh_data to tmp_dir with uniquely-named textures.

    Copies all MTL files and texture images (including from images/ subdir),
    renaming images to <seq_name>_<filename> so the converted USD references
    unique texture filenames without collisions.
    Returns path to the OBJ in the tmp dir.
    """
    mesh_dir = obj_path.parent
    work_dir = tmp_dir / seq_name
    work_dir.mkdir(parents=True, exist_ok=True)

    shutil.copy2(mesh_dir / "model.obj", work_dir / "model.obj")

    rename_map = {}
    for ext in ("*.png", "*.jpg", "*.jpeg"):
        for tex in mesh_dir.glob(ext):
            new_name = f"{seq_name}_{tex.name}"
            rename_map[tex.name] = new_name
            shutil.copy2(tex, work_dir / new_name)
        for tex in mesh_dir.glob(f"images/{ext}"):
            new_name = f"{seq_name}_{tex.name}"
            rename_map[f"images/{tex.name}"] = new_name
            shutil.copy2(tex, work_dir / new_name)

    for mtl_path in mesh_dir.glob("*.mtl"):
        mtl_text = mtl_path.read_text()
        for old_ref, new_name in rename_map.items():
            mtl_text = mtl_text.replace(old_ref, new_name)
        (work_dir / mtl_path.name).write_text(mtl_text)

    return work_dir / "model.obj"


def fixup_texture_paths(usd_path: Path, seq_name: str) -> None:
    """Post-process USD to use relative per-object texture subdirs.

    Converts any absolute or flat texture reference to:
        textures/<seq_name>/<clean_filename>

    and moves the texture files to that subdir. This ensures portability
    when the USD is copied to a cluster with a different filesystem path.
    """
    usd_dir = usd_path.parent
    seq_tex_dir = usd_dir / "textures" / seq_name

    stage = Usd.Stage.Open(str(usd_path))

    modified = False
    for prim in stage.Traverse():
        for attr in prim.GetAttributes():
            val = attr.Get()
            if not isinstance(val, Sdf.AssetPath):
                continue

            path_str = val.resolvedPath or val.path
            if not path_str:
                continue
            if not any(ext in path_str.lower() for ext in [".jpg", ".png", ".jpeg"]):
                continue

            tex_file = Path(path_str)
            if not tex_file.is_absolute():
                tex_file = (usd_dir / path_str).resolve()

            if not tex_file.exists():
                continue

            fname = tex_file.name
            clean_name = fname[len(seq_name) + 1 :] if fname.startswith(f"{seq_name}_") else fname

            seq_tex_dir.mkdir(parents=True, exist_ok=True)
            dest = seq_tex_dir / clean_name
            if not dest.exists():
                shutil.copy2(tex_file, dest)

            new_rel = f"textures/{seq_name}/{clean_name}"
            attr.Set(Sdf.AssetPath(new_rel))
            modified = True

    if modified:
        stage.GetRootLayer().Save()


def ensure_physics_schemas(usd_path: Path, mass: float = 1.0) -> None:
    """Apply RigidBodyAPI / MassAPI / CollisionAPI if `convert_mesh.py` skipped them.

    IsaacLab's MeshConverter sometimes raises mid-conversion (after writing the
    Xform wrapper but before applying physics schemas). Kit swallows the exit
    code so retarget.py can't detect failure via subprocess. We re-apply the
    expected schemas here as a no-op-on-success fix-up.
    """
    from pxr import Usd, UsdPhysics, UsdGeom

    stage = Usd.Stage.Open(str(usd_path))
    root = stage.GetDefaultPrim()
    if not root:
        return

    modified = False
    applied = set(root.GetAppliedSchemas())
    if "PhysicsRigidBodyAPI" not in applied:
        UsdPhysics.RigidBodyAPI.Apply(root)
        modified = True
    if "PhysicsMassAPI" not in applied:
        mass_api = UsdPhysics.MassAPI.Apply(root)
        mass_api.CreateMassAttr().Set(mass)
        modified = True

    for prim in stage.Traverse():
        if prim.GetTypeName() == "Mesh":
            mesh_applied = set(prim.GetAppliedSchemas())
            if "PhysicsCollisionAPI" not in mesh_applied:
                UsdPhysics.CollisionAPI.Apply(prim)
                modified = True
            if "PhysicsMeshCollisionAPI" not in mesh_applied:
                UsdPhysics.MeshCollisionAPI.Apply(prim)
                modified = True

    if modified:
        stage.GetRootLayer().Save()


def retarget_single_sequence(
    file_path,
    robot,
    smplx_model_path,
    output_dir,
    transparent_robot,
    no_viewer,
    mesh_dir,
    scene_scale=1.0,
    target_fps=30,
    mesh_scale=1.0,
    zero_out_wrist=False,
    height_offset=0.0,
    visualize_smpl=False,
    pelvis_rotation=0.0,
    g1_smplx=None,
    downstairs_initial_height_correction=False,
    downstairs_height_radius=0.08,
    downstairs_height_clearance=0.015,
    downstairs_foot_body_names="left_toe_link,right_toe_link,left_ankle_roll_link,right_ankle_roll_link",
) -> None:

    print("--------------------------------")
    print(file_path)
    print("--------------------------------")

    if file_path.endswith(".npz"):
        seq_name = "_".join(Path(file_path).parts[-2:]).replace(".npz", "")
    elif file_path.endswith(".pkl"):
        seq_name = "_".join(Path(file_path).parts[-4:-2])
    else:
        raise ValueError(f"Unsupported file type: {file_path}")

    print("seq_name: ", seq_name)

    if output_dir is not None:
        output_dir_robot = os.path.join(output_dir, "robot")
        robot_output_file = os.path.join(output_dir_robot, f"{seq_name}.pkl")
        if os.path.exists(robot_output_file):
            print(f"Skipping {seq_name} because {robot_output_file} exists")
            return

    if file_path.endswith(".npz"):
        result = load_grab_sequence(
            file_path,
            smplx_model_path,
            target_fps,
            scene_scale=scene_scale,
            height_offset=height_offset,
            return_mesh=visualize_smpl,
            g1_smplx=g1_smplx,
        )
    elif file_path.endswith(".pkl"):
        result = load_hoi_sequence(
            file_path,
            smplx_model_path,
            target_fps,
            scene_scale=scene_scale,
            height_offset=height_offset,
            return_mesh=visualize_smpl,
            g1_smplx=g1_smplx,
        )
    else:
        raise ValueError(f"Unsupported file type: {file_path}")

    if pelvis_rotation != 0.0:
        extra_rot = R.from_rotvec(np.array([pelvis_rotation, 0, 0]), degrees=True)
        for frame in range(len(result["smplx_data_frames"])):
            for joint_name in result["smplx_data_frames"][frame].keys():
                if joint_name == "pelvis":
                    pos, quat = result["smplx_data_frames"][frame][joint_name]
                    existing_rot = R.from_quat(quat, scalar_first=True)
                    result["smplx_data_frames"][frame][joint_name] = (
                        pos,
                        (existing_rot * extra_rot).as_quat(scalar_first=True),
                    )

    if visualize_smpl:

        from vispy import app, scene
        from vispy.scene import visuals
        import vispy.util

        keys = result["smplx_data_frames"][0].keys()
        smplx_joints = np.array(
            [
                [result["smplx_data_frames"][f][k][0] for k in keys]
                for f in range(len(result["smplx_data_frames"]))
            ]
        )
        mesh_verts = result["mesh_verts"].copy()

        mesh_verts = mesh_verts[PAD_FRAMES:]
        smplx_joints = smplx_joints[PAD_FRAMES:]

        obj_name = [x for x in result["obj_data"].keys() if x != "table"][0]
        object_mesh = result["obj_data"][obj_name]["mesh"]

        rot_mats = R.from_quat(
            result["obj_data"][obj_name]["rot_quats"], scalar_first=True
        ).as_matrix()
        object_vertices = (
            np.matmul(object_mesh.vertices, rot_mats.transpose(0, 2, 1))
            + result["obj_data"][obj_name]["transl"][:, None]
        )

        canvas = scene.SceneCanvas(keys="interactive", bgcolor="white", size=(800, 600), show=True)
        view = canvas.central_widget.add_view()
        view.camera = "turntable"  # 'arcball' also works

        paused = False

        @canvas.events.key_press.connect
        def on_key_press(event):
            nonlocal paused
            if event.key == vispy.util.keys.SPACE:
                paused = not paused

        vertex_colors = 0.5 * (object_mesh.vertex_normals + 1)
        vertex_colors = np.pad(
            vertex_colors, ((0, 0), (0, 1)), mode="constant", constant_values=1.0
        )

        obj_mesh_vis = visuals.Mesh(
            vertices=object_mesh.vertices,
            faces=object_mesh.faces,
            vertex_colors=vertex_colors,
            shading=None,
        )
        obj_mesh_vis.transform = scene.transforms.MatrixTransform()
        view.add(obj_mesh_vis)

        sbj_mesh_vis = visuals.Mesh(
            vertices=result["mesh_verts"][0],
            faces=result["mesh_faces"],
            color=(0.5, 0.7, 0.5, 1.0),
            shading="smooth",
        )
        sbj_mesh_vis.transform = scene.transforms.MatrixTransform()
        view.add(sbj_mesh_vis)

        points_vis = visuals.Markers()
        points_vis.set_data(smplx_joints[0], face_color="red", size=4, edge_width=0)
        points_vis.set_gl_state(depth_test=False)
        view.add(points_vis)

        view.camera.set_range()  # auto-fit the initial geometry
        F = mesh_verts.shape[0]
        frame = 0

        def update(event):

            nonlocal frame
            f = frame % F

            obj_mesh_vis.set_data(
                vertices=object_vertices[f], faces=object_mesh.faces, vertex_colors=vertex_colors
            )
            sbj_mesh_vis.set_data(vertices=mesh_verts[f], faces=result["mesh_faces"])
            points_vis.set_data(smplx_joints[f], face_color="red", size=4, edge_width=0)

            if not paused:
                frame += 1
            canvas.update()

        _timer = app.Timer(interval=1.0 / 30, connect=update, start=True)  # ~30 FPS  # noqa: F841
        app.run()

    obj_name = [x for x in result["obj_data"].keys() if x != "table"][0]

    smplx_data_frames = result["smplx_data_frames"]
    transl = result["transl"]

    height_scale = result["height_scale"]

    # Initialize retargeting system
    retarget = GMR(
        src_human="smplx",
        tgt_robot=robot,
        verbose=False,
        use_velocity_limit=True,
    )

    if no_viewer:
        robot_motion_viewer = None
    else:
        ensure_listener_started()
        # Initialize robot motion viewer
        robot_motion_viewer = ModifiedRobotMotionViewer(
            robot_type=robot,
            motion_fps=target_fps,
            transparent_robot=transparent_robot,
            record_video=False,
            video_path=None,
            obj_meshes=[result["obj_data"][k]["mesh"] for k in result["obj_data"].keys()],
        )

    obj_tran_quat_array = [
        np.concatenate(
            [result["obj_data"][k]["transl"], result["obj_data"][k]["rot_quats"]], axis=-1
        )
        for k in result["obj_data"].keys()
    ]
    obj_tran_quat_array = np.concatenate(obj_tran_quat_array, axis=-1)

    qpos_list = []

    if robot_motion_viewer is not None:
        robot_motion_viewer.start_recording()

    paused = robot_motion_viewer is not None
    init_config = {
        "left_hip_pitch_joint": -0.02513099494689817,
        "left_hip_roll_joint": 0.01899441634277485,
        "left_hip_yaw_joint": -0.0022360307624332693,
        "left_knee_joint": 0.11007054150034166,
        "left_ankle_pitch_joint": -0.1221834339385838,
        "left_ankle_roll_joint": -0.011626410915664456,
        "right_hip_pitch_joint": -0.030570498570421812,
        "right_hip_roll_joint": -0.006224588598396952,
        "right_hip_yaw_joint": 0.001794682938890269,
        "right_knee_joint": 0.114228855772871,
        "right_ankle_pitch_joint": -0.12131391886319735,
        "right_ankle_roll_joint": 0.01381159522898908,
        "waist_yaw_joint": 0.012472488516680173,
        "waist_roll_joint": 0.00837222514718666,
        "waist_pitch_joint": -0.1503562559954008,
        "left_shoulder_pitch_joint": 0.08441059680998222,
        "left_shoulder_roll_joint": 1.5839950244664192,
        "left_shoulder_yaw_joint": -0.017186157456786547,
        "left_elbow_joint": 1.5488449453974993,
        "left_wrist_roll_joint": -0.09783657909787696,
        "left_wrist_pitch_joint": 0.11174492130168037,
        "left_wrist_yaw_joint": -0.16397002611636324,
        "right_shoulder_pitch_joint": 0.23667226628271232,
        "right_shoulder_roll_joint": -1.5811611629436342,
        "right_shoulder_yaw_joint": -0.13223036620527512,
        "right_elbow_joint": 1.5319065338082463,
        "right_wrist_roll_joint": 0.07233085465420917,
        "right_wrist_pitch_joint": 0.09702733597666197,
        "right_wrist_yaw_joint": 0.14902005719460554,
    }

    for i in range(retarget.model.nu):
        name = retarget.model.actuator(i).name
        if name in init_config:
            retarget.configuration.data.qpos[i + 7] = init_config[name]

    qpos = retarget.configuration.data.qpos.copy()
    root_rot = R.from_quat(qpos[3:7]).as_rotvec()

    pbar = tqdm(total=len(smplx_data_frames))
    i = 0
    while i < len(smplx_data_frames):
        if not paused:
            pbar.n = i
            pbar.refresh()

        # Retarget
        frame = np.clip(i, 0, len(smplx_data_frames) - 1)
        if not paused:

            frames = smplx_data_frames[frame]

            # add "palm" joints with the position of the wrist and the rotation of the middle1 joint:
            # x axis needs to be from wrist to middle1 joint:, z axis from pinky to index knuckle
            weights = np.array([1.0, 1.0, 1.0, 1.0])
            weights /= np.sum(weights)

            for hand in ["left", "right"]:

                # get the local z axis from the line of the fingertips - it's a big clunky robot hand
                # and we want it to eg align perpendicular to the shaft of a hammer when the
                # robot is picking it up. This is hopefully a good hint at the object it's aligning with.
                z_axis = frames[f"{hand}_index3"][0] - frames[f"{hand}_ring3"][0]
                if hand == "right":
                    z_axis = -z_axis
                z_axis = z_axis / np.linalg.norm(z_axis)

                finger_names = [
                    f"{hand}_index1",
                    f"{hand}_middle1",
                    f"{hand}_ring1",
                    f"{hand}_pinky1",
                ]
                knuckles_mean = sum(
                    [w * frames[finger_name][0] for w, finger_name in zip(weights, finger_names)]
                )
                x_axis = knuckles_mean - frames[f"{hand}_wrist"][0]
                x_axis = x_axis - np.dot(x_axis, z_axis) * z_axis
                x_axis = x_axis / np.linalg.norm(x_axis)

                y_axis = np.cross(z_axis, x_axis)

                if hand == "left":
                    rotmat = np.column_stack([x_axis, y_axis, z_axis])
                else:
                    # not 100% sure why you have to negate the x and z axes to get the
                    # expected results here...
                    rotmat = np.column_stack([-x_axis, y_axis, -z_axis])

                all_hand_joint_names = [
                    f"{hand}_index1",
                    f"{hand}_index2",
                    # f"{hand}_index3",
                    f"{hand}_middle1",
                    f"{hand}_middle2",
                    # f"{hand}_middle3",
                    f"{hand}_pinky1",
                    f"{hand}_pinky2",
                    # f"{hand}_pinky3",
                    f"{hand}_ring1",
                    f"{hand}_ring2",
                    # f"{hand}_ring3",
                    f"{hand}_thumb1",
                    f"{hand}_thumb2",
                    # f"{hand}_thumb3",
                ]
                all_joint_positions = np.array(
                    [frames[joint_name][0] for joint_name in all_hand_joint_names]
                )
                all_joint_positions_mean = np.mean(all_joint_positions, axis=0)

                rotation = R.from_matrix(rotmat).as_quat(scalar_first=True)
                frames[f"{hand}_wrist"] = (all_joint_positions_mean - x_axis * 0.1, rotation)

            retarget.configuration.data.qpos[:3] = transl[frame]
            qpos = retarget.retarget(frames)

        # Visualize
        obj_frame = np.clip(i - PAD_FRAMES, 0, len(smplx_data_frames) - 1)
        if robot_motion_viewer is not None:
            robot_motion_viewer.step(
                root_pos=qpos[:3],
                root_rot=qpos[3:7],
                dof_pos=qpos[7:],
                human_motion_data=(
                    smplx_data_frames[frame]
                    if not hasattr(retarget, "scaled_human_data")
                    else retarget.scaled_human_data
                ),
                human_pos_offset=np.array([0.0, 0.0, 0.0]),
                show_human_body_name=False,
                obj_pose=obj_tran_quat_array[obj_frame],
                follow_camera=False,
                all_joint_pos=smplx_data_frames[frame],
            )

        if space_pressed_since_last_call():
            paused = not paused

        if not paused:
            # skip the first frame as it's padding:
            qpos_list.append(qpos)
            i += 1

    if robot_motion_viewer is not None:
        robot_motion_viewer.close()

    if output_dir is None:
        return

    os.makedirs(output_dir, exist_ok=True)

    # Create output directories
    output_dir_objects = os.path.join(output_dir, "objects")
    output_dir_robot = os.path.join(output_dir, "robot")
    output_dir_mesh = os.path.join(output_dir, "object_usd")
    output_dir_common = os.path.join(output_dir, "common")
    output_dir_meta = os.path.join(output_dir, "meta")

    os.makedirs(output_dir_objects, exist_ok=True)
    os.makedirs(output_dir_robot, exist_ok=True)
    os.makedirs(output_dir_mesh, exist_ok=True)
    os.makedirs(output_dir_common, exist_ok=True)
    os.makedirs(output_dir_meta, exist_ok=True)

    object_output_file = os.path.join(output_dir_objects, f"{seq_name}.pkl")
    robot_output_file = os.path.join(output_dir_robot, f"{seq_name}.pkl")
    meta_output_file = os.path.join(output_dir_meta, f"{seq_name}.pkl")

    output_qpos_list = qpos_list[PAD_FRAMES:]  # Skip padding frames

    if zero_out_wrist:
        wrist_indices = [
            i for i in range(retarget.model.nu) if "wrist" in retarget.model.actuator(i).name
        ]
        for qpos in output_qpos_list:
            for idx in wrist_indices:
                qpos[7 + idx] = 0.0

    object_name = next(k for k in result["obj_data"].keys() if k != "table")
    if downstairs_initial_height_correction:
        _apply_downstairs_initial_height_correction(
            output_qpos_list=output_qpos_list,
            retarget=retarget,
            object_mesh=result["obj_data"][object_name]["mesh"],
            object_transl=result["obj_data"][object_name]["transl"],
            object_rot_quats=result["obj_data"][object_name]["rot_quats"],
            foot_body_names=_parse_csv(downstairs_foot_body_names),
            radius=downstairs_height_radius,
            clearance=downstairs_height_clearance,
        )

    print(
        f"  Skipping {PAD_FRAMES} padding frames for output (saving {len(output_qpos_list)} frames)"
    )

    root_pos = np.array([qpos[:3] for qpos in output_qpos_list])
    root_rot = np.array([qpos[3:7][[1, 2, 3, 0]] for qpos in output_qpos_list])  # wxyz to xyzw
    dof_pos = np.array([qpos[7:] for qpos in output_qpos_list])

    motion_data = {
        "fps": target_fps,
        "root_pos": root_pos,
        "root_rot": root_rot,
        "dof_pos": dof_pos,
        "local_body_pos": None,
        "link_body_list": None,
    }

    convert_gmr_data_to_motion_lib(motion_data, robot_output_file, retarget.model, seq_name)

    # Prepare object data in motion library format
    # Shape: (T, N, D) where N is number of objects (1 in this case)

    # Object data is NOT padded (only SMPLX body data is), so no trimming needed
    object_transl = result["obj_data"][object_name]["transl"]
    object_rot_quats = result["obj_data"][object_name]["rot_quats"]
    object_data = {
        seq_name: {
            "root_pos": object_transl[:, np.newaxis, :].astype(np.float32),  # (T, 1, 3)
            "root_quat": object_rot_quats[:, np.newaxis, :].astype(np.float32),  # (T, 1, 4)
            "object_name": obj_name,
            "fps": float(target_fps),
        }
    }

    # Add object scale if available (GeniHOI data)
    obj_scale = result.get("obj_scale")
    if obj_scale is not None:
        object_data[seq_name]["scale"] = obj_scale

    if "obj_contact_points" in result["obj_data"][object_name]:
        contact_points = result["obj_data"][object_name]["obj_contact_points"]
        if isinstance(contact_points, dict):
            for hand in ("left_hand", "right_hand"):
                if hand in contact_points and isinstance(contact_points[hand], dict):
                    object_data[seq_name][f"contact_points_{hand}"] = {
                        k: v.astype(np.float32) if hasattr(v, "astype") else v
                        for k, v in contact_points[hand].items()
                    }

    joblib.dump(object_data, object_output_file, compress=True)
    print(f"Saved object motion to {object_output_file}")

    if "table" in result["obj_data"]:
        table_name = "table"
        table_transl = result["obj_data"][table_name]["transl"]
        table_rot_quats = result["obj_data"][table_name]["rot_quats"]
        # Metadata (fixed values, saved separately per sequence)
        meta = {
            "table_pos": table_transl[table_transl.shape[0] // 2]
            .astype(np.float32)
            .tolist(),  # [x, y, z]
            "table_quat": table_rot_quats[table_rot_quats.shape[0] // 2]
            .astype(np.float32)
            .tolist(),  # [w, x, y, z]
            "object_name": obj_name,
            "scene_scale": float(height_scale),
            "smpl_height": 1.66,
        }

        joblib.dump(meta, meta_output_file, compress=True)
        copy_mesh_file(mesh_dir, table_name, table_name, output_dir_common)
        print(f"Saved metadata to {meta_output_file}")

    elif "scene_table" in result:
        # GeniHOI data: write table geometry info from scene_data to meta
        meta = {
            "object_name": obj_name,
        }
        meta.update(result["scene_table"])

        joblib.dump(meta, meta_output_file, compress=True)
        print(f"Saved GeniHOI metadata to {meta_output_file}")

    else:
        meta = {
            "object_name": obj_name,
            "scene_scale": float(height_scale),
            "smpl_height": 1.66,
        }
        joblib.dump(meta, meta_output_file, compress=True)
        print(f"Saved minimal metadata (no table) to {meta_output_file}")

    if file_path.endswith(".npz"):
        copy_mesh_file(mesh_dir, obj_name, seq_name, output_dir_mesh)
    else:

        # convert object mesh to USD (with texture uniquification to avoid collisions)
        object_mesh_path = Path(result["obj_data"][object_name]["file_path"])
        output_usd_path = str(Path(output_dir_mesh) / f"{seq_name}.usd")

        with tempfile.TemporaryDirectory() as tmp_dir:
            patched_obj = prepare_unique_mesh(object_mesh_path, seq_name, Path(tmp_dir))
            cmd = [
                sys.executable,
                os.path.join(os.path.dirname(__file__), "convert_mesh.py"),
                str(patched_obj),
                str(output_usd_path),
                "--headless",
                "--mass",
                "1.0",
                "--scale",
                str(mesh_scale),
                "--collision-approximation",
                "meshSimplification",
            ]
            subprocess.run(cmd, check=True)

        # Convert collision approximation to SDF mesh
        convert_to_sdf(Path(output_usd_path))
        # Fix texture paths to be relative (portable across machines)
        fixup_texture_paths(Path(output_usd_path), seq_name)
        # Verify + repair physics schemas. IsaacLab's MeshConverter can raise
        # `RuntimeError: Accessed invalid null prim` mid-conversion for some
        # OBJs whose intermediate USD's defaultPrim doesn't compose cleanly
        # with the /<basename>/geometry wrapper. The error is swallowed (Kit
        # eats the non-zero exit code), so subprocess.run(check=True) returns
        # success but the USD is missing RigidBodyAPI / MassAPI / CollisionAPI.
        # This trips up IsaacLab downstream with "no contact sensors / no
        # rigid bodies". Detect and apply the missing schemas here.
        ensure_physics_schemas(Path(output_usd_path), mass=1.0)


def run_main(
    data_dir,
    all,
    seq_name,
    name_filter,
    robot,
    smplx_model_path,
    output_dir,
    transparent_robot,
    no_viewer,
    mesh_dir,
    scene_scale=1.0,
    target_fps=30,
    mesh_scale=1.0,
    zero_out_wrist=False,
    height_offset=0.0,
    visualize_smpl=False,
    pelvis_rotation=0.0,
    g1_smplx=None,
    file=None,
    num_job_chunks=1,
    job_chunk_idx=0,
    downstairs_initial_height_correction=False,
    downstairs_height_radius=0.08,
    downstairs_height_clearance=0.015,
    downstairs_foot_body_names="left_toe_link,right_toe_link,left_ankle_roll_link,right_ankle_roll_link",
) -> None:

    if seq_name is None and not all and not file:
        raise ValueError("Either --seq_name, --all, or --file must be specified")

    if file:
        file_path = Path(file)
        if file_path.is_dir():
            file_path = file_path / "hoi_data" / "hoi_data.pkl"
        if not file_path.exists():
            raise ValueError(f"File not found: {file_path}")
        retarget_single_sequence(
            str(file_path),
            robot,
            smplx_model_path,
            output_dir,
            transparent_robot,
            no_viewer,
            mesh_dir,
            scene_scale=scene_scale,
            target_fps=target_fps,
            mesh_scale=mesh_scale,
            zero_out_wrist=zero_out_wrist,
            height_offset=height_offset,
            visualize_smpl=visualize_smpl,
            pelvis_rotation=pelvis_rotation,
            g1_smplx=g1_smplx,
            downstairs_initial_height_correction=downstairs_initial_height_correction,
            downstairs_height_radius=downstairs_height_radius,
            downstairs_height_clearance=downstairs_height_clearance,
            downstairs_foot_body_names=downstairs_foot_body_names,
        )
        return

    if all:

        files = glob.glob(os.path.join(data_dir, "s*/*.npz"))
        if not files:
            # Search for GeniHOI pkl files with multiple patterns:
            # 1. data_dir/generation/4dhoi_recon_valid/<Dataset>/<Category>/<Seq>/hoi_data/hoi_data.pkl
            # 2. data_dir/<Dataset>/<Category>/<Seq>/hoi_data/hoi_data.pkl (if data_dir already points to 4dhoi_recon_valid)
            for pattern in [
                os.path.join(data_dir, "generation/4dhoi_recon_valid/*/*/hoi_data/hoi_data.pkl"),
                os.path.join(data_dir, "generation/4dhoi_recon_valid/*/*/*/hoi_data/hoi_data.pkl"),
                os.path.join(data_dir, "*/*/hoi_data/hoi_data.pkl"),
                os.path.join(data_dir, "*/*/*/hoi_data/hoi_data.pkl"),
            ]:
                files = glob.glob(pattern)
                if files:
                    break
        if name_filter is not None:
            files = [file for file in files if name_filter in file]

        # Chunk-aware parallelism: when fanned out across N workers,
        # each chunk processes only files[i::N] of the sorted list.
        files = sorted(files)[job_chunk_idx::num_job_chunks]

        for file_path in files:
            print(file_path)
        for i, file_path in enumerate(files):
            print(f"Processing {i+1} of {len(files)}: {file_path}")
            retarget_single_sequence(
                file_path,
                robot,
                smplx_model_path,
                output_dir,
                transparent_robot,
                no_viewer,
                mesh_dir,
                scene_scale=scene_scale,
                target_fps=target_fps,
                mesh_scale=mesh_scale,
                zero_out_wrist=zero_out_wrist,
                height_offset=height_offset,
                visualize_smpl=visualize_smpl,
                pelvis_rotation=pelvis_rotation,
                g1_smplx=g1_smplx,
                downstairs_initial_height_correction=downstairs_initial_height_correction,
                downstairs_height_radius=downstairs_height_radius,
                downstairs_height_clearance=downstairs_height_clearance,
                downstairs_foot_body_names=downstairs_foot_body_names,
            )

    else:
        seq_file = os.path.join(data_dir, seq_name + ".npz")
        if not os.path.exists(seq_file):
            # Search for the sequence in any dataset subdirectory
            candidates = (
                glob.glob(
                    os.path.join(
                        data_dir,
                        "generation/4dhoi_recon_valid",
                        "*",
                        seq_name,
                        "hoi_data",
                        "hoi_data.pkl",
                    )
                )
                + glob.glob(
                    os.path.join(
                        data_dir,
                        "generation/4dhoi_recon_valid",
                        "*",
                        "*",
                        seq_name,
                        "hoi_data",
                        "hoi_data.pkl",
                    )
                )
                + glob.glob(os.path.join(data_dir, "*", seq_name, "hoi_data", "hoi_data.pkl"))
                + glob.glob(os.path.join(data_dir, "*", "*", seq_name, "hoi_data", "hoi_data.pkl"))
            )
            if candidates:
                seq_file = candidates[0]
            else:
                # Fallback to original ComAsset path
                seq_file = os.path.join(
                    data_dir,
                    "generation",
                    "4dhoi_recon_valid",
                    "ComAsset",
                    seq_name,
                    "hoi_data",
                    "hoi_data.pkl",
                )
        if not os.path.exists(seq_file):
            raise ValueError(f"Couldn't find sequence {seq_file} in {data_dir}")
        retarget_single_sequence(
            seq_file,
            robot,
            smplx_model_path,
            output_dir,
            transparent_robot,
            no_viewer,
            mesh_dir,
            scene_scale=scene_scale,
            target_fps=target_fps,
            mesh_scale=mesh_scale,
            zero_out_wrist=zero_out_wrist,
            height_offset=height_offset,
            visualize_smpl=visualize_smpl,
            pelvis_rotation=pelvis_rotation,
            g1_smplx=g1_smplx,
            downstairs_initial_height_correction=downstairs_initial_height_correction,
            downstairs_height_radius=downstairs_height_radius,
            downstairs_height_clearance=downstairs_height_clearance,
            downstairs_foot_body_names=downstairs_foot_body_names,
        )


def main():
    parser = argparse.ArgumentParser(
        description="Convert GRAB SMPLX motion to robot motion using GMR retargeting.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
    python scripts/motion/grab_retarget_raw.py \\
        --seq_name s1_mug_drink_4 \\
        --robot unitree_g1 \\
        --output_dir data/motion_lib_grab/
        """,
    )

    parser.add_argument(
        "--seq_name",
        type=str,
        default=None,
        required=False,
        help="Name of the sequence to convert (e.g., s1_mug_drink_4)",
    )
    parser.add_argument(
        "--name_filter",
        type=str,
        default=None,
        required=False,
        help="Filter sequences by name (e.g., s1_*)",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Convert all sequences in the dataset",
    )

    parser.add_argument(
        "--file",
        type=str,
        default=None,
        help="Direct path to a single HOI folder or hoi_data.pkl file. "
        "Bypasses --data_dir search logic.",
    )

    parser.add_argument(
        "--data_dir",
        type=str,
        default="data/grab_raw/",
        help="Path to GRAB dataset directory",
    )

    parser.add_argument(
        "--robot",
        choices=[
            "unitree_g1",
            "unitree_g1_with_hands",
        ],
        default="unitree_g1",
        help="Target robot type for retargeting",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory for robot motion",
    )

    parser.add_argument(
        "--smplx_model_path",
        type=str,
        default="imports/GEM-SMPL/inputs/checkpoints/body_models",
        help="Path to SMPLX body models",
    )

    parser.add_argument(
        "--transparent_robot",
        action="store_true",
        help="Make the robot transparent",
    )

    parser.add_argument(
        "--no_viewer",
        action="store_true",
        help="Do not show the viewer",
    )

    parser.add_argument(
        "--mesh_dir",
        type=str,
        default="data/PHC_Lab/mesh/grab",
        help="Directory containing pre-converted object mesh USD files",
    )

    parser.add_argument(
        "--fps",
        type=int,
        default=24,
        help="FPS of the source motion data (default: 24 for GeniHOI, use 30 for GRAB)",
    )

    parser.add_argument(
        "--mesh_scale",
        type=float,
        default=1.0,
        help="Scale factor applied to the object mesh USD (default: 1.0, use 0.6 for terrain data)",
    )

    parser.add_argument(
        "--zero_out_wrist",
        action="store_true",
        help="Zero out wrist joint positions after retargeting (useful for terrain/locomotion data)",
    )

    parser.add_argument(
        "--height_offset",
        type=float,
        default=0.0,
        help="Vertical offset applied to human and object positions (e.g., -0.1365). Table is not affected.",
    )

    parser.add_argument(
        "--downstairs_initial_height_correction",
        "--downstairs-initial-height-correction",
        action="store_true",
        help=(
            "For downstairs terrain motions, raise the robot trajectory if first-frame "
            "foot bodies are below the terrain mesh height under those feet."
        ),
    )

    parser.add_argument(
        "--downstairs_height_radius",
        "--downstairs-height-radius",
        type=float,
        default=0.08,
        help="Fallback XY radius for terrain height lookup when no mesh triangle contains a foot.",
    )

    parser.add_argument(
        "--downstairs_height_clearance",
        "--downstairs-height-clearance",
        type=float,
        default=0.015,
        help="Extra clearance above terrain for downstairs initial height correction.",
    )

    parser.add_argument(
        "--downstairs_foot_body_names",
        "--downstairs-foot-body-names",
        type=str,
        default="left_toe_link,right_toe_link,left_ankle_roll_link,right_ankle_roll_link",
        help=(
            "Comma-separated MuJoCo body names whose first-frame body positions are "
            "used as foot points for downstairs correction."
        ),
    )

    parser.add_argument(
        "--visualize_smpl",
        action="store_true",
        help="Visualize the smpl motion.",
    )

    parser.add_argument(
        "--scene_scale",
        type=float,
        default=1.0,
        help="Scale factor applied to the scene (default: 1.0).",
    )

    parser.add_argument(
        "--pelvis_rotation",
        type=float,
        default=0.0,
        help="Extra rotation (degrees) applied to the pelvis around the X axis.",
    )

    parser.add_argument(
        "--no_g1_proportions",
        action="store_true",
        default=False,
        help="Disable G1 bone proportion adjustment. By default, SMPLX joints "
        "are scaled to match G1 robot proportions for better retargeting.",
    )

    parser.add_argument("--num_job_chunks", type=int, default=1)
    parser.add_argument("--job_chunk_idx", type=int, default=0)

    args = parser.parse_args()

    # Create G1-proportioned SMPLX model unless disabled.
    # Uses pre-baked T-pose + scale from the G1 SMPL-X body profile; a one-time bake
    # utility (out-of-repo) regenerates them if the underlying Jason betas
    # change, which is rare.
    g1_smplx = None
    if not args.no_g1_proportions:
        g1_smplx = G1ProportionSMPLX.from_g1_smplx_dir(args.smplx_model_path)

    run_main(
        args.data_dir,
        args.all,
        args.seq_name,
        args.name_filter,
        args.robot,
        args.smplx_model_path,
        args.output_dir,
        args.transparent_robot,
        args.no_viewer,
        args.mesh_dir,
        scene_scale=args.scene_scale,
        target_fps=args.fps,
        mesh_scale=args.mesh_scale,
        zero_out_wrist=args.zero_out_wrist,
        height_offset=args.height_offset,
        visualize_smpl=args.visualize_smpl,
        pelvis_rotation=args.pelvis_rotation,
        g1_smplx=g1_smplx,
        file=args.file,
        num_job_chunks=args.num_job_chunks,
        job_chunk_idx=args.job_chunk_idx,
        downstairs_initial_height_correction=args.downstairs_initial_height_correction,
        downstairs_height_radius=args.downstairs_height_radius,
        downstairs_height_clearance=args.downstairs_height_clearance,
        downstairs_foot_body_names=args.downstairs_foot_body_names,
    )


if __name__ == "__main__":
    main()
