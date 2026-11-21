from __future__ import annotations

import copy
import dataclasses
import glob
import json
import os
import re

from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg, ViewerCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import (
    CameraCfg,
    ContactSensorCfg,
    FrameTransformerCfg,
    TiledCamera,
    TiledCameraCfg,
)
import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.sim.utils import clone
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
import joblib
import pxr

from gear_sonic.envs.manager_env.mdp import terrain
from gear_sonic.envs.manager_env.robots import g1
from gear_sonic.rp2.direct_directory_roster import (
    normalize_direct_spec,
    resolve_direct_roster_for_env_cfg,
)
from gear_sonic.rp2.discrete_contact import (
    CONTACT_BODY_GROUPS,
    G1_CONTACT_GROUP_LINKS,
)
from gear_sonic.rp2.joint_roster import resolve_rank_local_roster
from gear_sonic.trl.utils import common


class SelectedEnvironmentTiledCamera(TiledCamera):
    """A tiled camera whose camera rows correspond to selected global env IDs.

    InteractiveScene forwards global environment IDs to every sensor on reset.
    A sparse camera view has fewer rows, so translate those IDs to the view's
    local row numbers before delegating to the standard TiledCamera reset.
    """

    def _initialize_impl(self):
        super()._initialize_impl()
        self._global_to_local_env_id = {}
        for local_id, prim in enumerate(self._view.prims):
            match = re.search(r"/env_(\d+)/", prim.GetPath().pathString)
            if match is None:
                raise RuntimeError(
                    "selected tiled-camera prim has no environment ID: "
                    f"{prim.GetPath().pathString}"
                )
            global_id = int(match.group(1))
            if global_id in self._global_to_local_env_id:
                raise RuntimeError("selected tiled-camera environment IDs are duplicated")
            self._global_to_local_env_id[global_id] = local_id

    def reset(self, env_ids=None):
        if env_ids is None:
            return super().reset(None)
        if isinstance(env_ids, slice):
            if env_ids == slice(None):
                return super().reset(None)
            raise ValueError("selected tiled camera does not support partial slices")
        ids = env_ids.tolist() if hasattr(env_ids, "tolist") else list(env_ids)
        local_ids = [
            self._global_to_local_env_id[int(env_id)]
            for env_id in ids
            if int(env_id) in self._global_to_local_env_id
        ]
        return super().reset(local_ids)


def _static_context_is_configured(static_context_usd_path, joint_roster_view):
    """Return whether this scene branch will spawn static context."""
    return bool(static_context_usd_path) or joint_roster_view is not None


ROBOT_RUNTIME_INFO = {
    "g1": {"robot_has_hands": False, "hand_type": None},
    "g1_no_wrist": {"robot_has_hands": False, "hand_type": None},
    "g1_dex": {"robot_has_hands": True, "hand_type": "dex"},
    "g1_model_12": {"robot_has_hands": False, "hand_type": None},
    "g1_model_12_dex": {"robot_has_hands": True, "hand_type": "dex"},
    "g1_model_12_dex_thor": {"robot_has_hands": True, "hand_type": "dex"},
    "g1_model_12_dex_thor_dev": {"robot_has_hands": True, "hand_type": "dex"},
    "g1_model_12_dex_delayed": {"robot_has_hands": True, "hand_type": "dex"},
    "g1_model_12_dex_waist": {"robot_has_hands": True, "hand_type": "dex"},
    "g1_43dof": {"robot_has_hands": True, "hand_type": "dex"},
    "g1_43dof_waist_collision": {"robot_has_hands": True, "hand_type": "dex"},
    "g1_43dof_yam_core_collision": {"robot_has_hands": True, "hand_type": "dex"},
    "g1_43dof_larger_hip_no_head_elbow": {
        "robot_has_hands": True,
        "hand_type": "dex",
    },
    "g1_33dof_yam": {"robot_has_hands": True, "hand_type": "yam"},
    "h2": {"robot_has_hands": False, "hand_type": None},
}

_DEX_RIGHT_FRAME_LINKS = [
    "right_hand_thumb_2_link",
    "right_hand_index_1_link",
    "right_hand_middle_1_link",
]
_DEX_LEFT_FINGER_LINKS = [
    "left_hand_thumb_2_link",
    "left_hand_index_1_link",
    "left_hand_middle_1_link",
    "left_hand_palm_link",
]
_DEX_RIGHT_FINGER_LINKS = [
    "right_hand_thumb_2_link",
    "right_hand_index_1_link",
    "right_hand_middle_1_link",
    "right_hand_palm_link",
]
_DEX_RIGHT_BROAD_CONTACT_LINKS = [
    "right_wrist_roll_link",
    "right_wrist_pitch_link",
    "right_wrist_yaw_link",
    "right_hand_palm_link",
    "right_hand_index_0_link",
    "right_hand_index_1_link",
    "right_hand_middle_0_link",
    "right_hand_middle_1_link",
    "right_hand_thumb_0_link",
    "right_hand_thumb_1_link",
    "right_hand_thumb_2_link",
]
_YAM_RIGHT_FRAME_LINKS = [
    "right_yam_YAM_LinearRailGripper_Part_7_1",
    "right_yam_Assembly_finger_1_V210_2",
    "right_yam_Assembly_finger_2_V210_2",
]
_YAM_RIGHT_FINGER_LINKS = [
    "right_yam_Assembly_finger_1_V210_2",
    "right_yam_Assembly_finger_2_V210_2",
]
_YAM_LEFT_FINGER_LINKS = [
    "left_yam_Assembly_finger_1_V210_2",
    "left_yam_Assembly_finger_2_V210_2",
]
_YAM_RIGHT_BROAD_CONTACT_LINKS = [
    "right_wrist_roll_link",
    "right_wrist_pitch_link",
    "right_wrist_yaw_link",
    # "right_yam_padding_dm_gripper_0",  # not present in new gripper USDA
    # "right_yam_padding_dm_gripper_1",  # not present in new gripper USDA
    "right_yam_sensor_mount_1",
    "right_yam_Assembly_motor_1",
    "right_yam_YAM_LinearRailGripper_Part_7_1",
    "right_yam_Assembly_finger_1_Part_1_1",
    "right_yam_Assembly_finger_1_yam_finray_single_piece_adapter_1",
    "right_yam_Assembly_finger_1_V210_2",
    "right_yam_Assembly_finger_2_Part_1_1",
    "right_yam_Assembly_finger_2_yam_finray_single_piece_adapter_1",
    "right_yam_Assembly_finger_2_V210_2",
    "right_yam_a_06_00001_REV3_1",
]


def _default_hand_sensor_links(hand_type: str | None) -> dict[str, list[str]]:
    if hand_type == "yam":
        return {
            "object_to_hand_frame_links": list(_YAM_RIGHT_FRAME_LINKS),
            "contact_sensor_finger_links": list(_YAM_RIGHT_FINGER_LINKS),
            "left_contact_sensor_finger_links": list(_YAM_LEFT_FINGER_LINKS),
            "table_to_hand_contact_bodies": list(_YAM_RIGHT_BROAD_CONTACT_LINKS),
            "object_to_robot_contact_bodies": list(_YAM_RIGHT_BROAD_CONTACT_LINKS),
            "table_to_robot_contact_bodies": list(_YAM_RIGHT_BROAD_CONTACT_LINKS),
        }
    if hand_type == "dex":
        return {
            "object_to_hand_frame_links": list(_DEX_RIGHT_FRAME_LINKS),
            "contact_sensor_finger_links": list(_DEX_RIGHT_FINGER_LINKS),
            "left_contact_sensor_finger_links": list(_DEX_LEFT_FINGER_LINKS),
            "table_to_hand_contact_bodies": list(_DEX_RIGHT_BROAD_CONTACT_LINKS),
            "object_to_robot_contact_bodies": list(_DEX_RIGHT_BROAD_CONTACT_LINKS),
            "table_to_robot_contact_bodies": list(_DEX_RIGHT_BROAD_CONTACT_LINKS),
        }
    return {
        "object_to_hand_frame_links": [],
        "contact_sensor_finger_links": [],
        "left_contact_sensor_finger_links": [],
        "table_to_hand_contact_bodies": [],
        "object_to_robot_contact_bodies": [],
        "table_to_robot_contact_bodies": [],
    }


def _robot_body_paths(link_names: list[str]) -> list[str]:
    return [f"{{ENV_REGEX_NS}}/Robot/{link_name}" for link_name in link_names]


def _frame_name_from_link(link_name: str) -> str:
    return re.sub(r"[^0-9a-zA-Z]+", "_", link_name).strip("_").lower()


def _load_opencv_params_from_usd(usd_path: str) -> dict:
    """Parse a USD file to extract OpenCV lens distortion parameters.

    Args:
        usd_path: Path to the USD file containing camera with OpenCV distortion.

    Returns:
        Dictionary with keys: fx, fy, cx, cy, k1-k6, p1, p2, s1-s4

    Raises:
        FileNotFoundError: If USD file doesn't exist
        ValueError: If required intrinsics (fx, fy, cx, cy) are missing
    """
    with open(usd_path) as f:
        content = f.read()

    params = {}

    # Parse the USD text format for omni:lensdistortion:opencvPinhole attributes
    patterns = {
        "fx": r"omni:lensdistortion:opencvPinhole:fx\s*=\s*([-\d.]+)",
        "fy": r"omni:lensdistortion:opencvPinhole:fy\s*=\s*([-\d.]+)",
        "cx": r"omni:lensdistortion:opencvPinhole:cx\s*=\s*([-\d.]+)",
        "cy": r"omni:lensdistortion:opencvPinhole:cy\s*=\s*([-\d.]+)",
        "k1": r"omni:lensdistortion:opencvPinhole:k1\s*=\s*([-\d.]+)",
        "k2": r"omni:lensdistortion:opencvPinhole:k2\s*=\s*([-\d.]+)",
        "k3": r"omni:lensdistortion:opencvPinhole:k3\s*=\s*([-\d.]+)",
        "k4": r"omni:lensdistortion:opencvPinhole:k4\s*=\s*([-\d.]+)",
        "k5": r"omni:lensdistortion:opencvPinhole:k5\s*=\s*([-\d.]+)",
        "k6": r"omni:lensdistortion:opencvPinhole:k6\s*=\s*([-\d.]+)",
        "p1": r"omni:lensdistortion:opencvPinhole:p1\s*=\s*([-\d.]+)",
        "p2": r"omni:lensdistortion:opencvPinhole:p2\s*=\s*([-\d.]+)",
        "s1": r"omni:lensdistortion:opencvPinhole:s1\s*=\s*([-\d.]+)",
        "s2": r"omni:lensdistortion:opencvPinhole:s2\s*=\s*([-\d.]+)",
        "s3": r"omni:lensdistortion:opencvPinhole:s3\s*=\s*([-\d.]+)",
        "s4": r"omni:lensdistortion:opencvPinhole:s4\s*=\s*([-\d.]+)",
    }

    for key, pattern in patterns.items():
        match = re.search(pattern, content)
        if match:
            params[key] = float(match.group(1))

    # Assert required intrinsics exist
    required_intrinsics = ["fx", "fy", "cx", "cy"]
    missing = [k for k in required_intrinsics if k not in params]
    if missing:
        raise ValueError(f"USD file {usd_path} missing required OpenCV intrinsics: {missing}")

    # Assert at least k1 distortion exists (otherwise why use OpenCV mode?)
    if "k1" not in params:
        raise ValueError(
            f"USD file {usd_path} missing distortion coefficient k1 - use pinhole camera instead"
        )

    print(f"Loaded OpenCV params from {usd_path}: {params}")  # noqa: T201
    return params


@clone
def spawn_opencv_camera(
    prim_path: str,
    cfg,
    translation: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None,
    **kwargs,  # noqa: ARG001
) -> pxr.Usd.Prim:
    """Create a camera with OpenCV lens distortion applied at spawn time.

    This spawner creates a standard camera and then applies the OmniLensDistortionOpenCvPinholeAPI
    schema with the distortion parameters from the config. This happens BEFORE environment cloning,
    so all cloned environments will have the correct distortion properties.
    """
    import pxr  # noqa: F811 — re-import to ensure availability in this scope
    import isaacsim.core.utils.prims as prim_utils

    # Create the camera prim
    if not prim_utils.is_prim_path_valid(prim_path):
        prim_utils.create_prim(
            prim_path, "Camera", translation=translation, orientation=orientation
        )
    else:
        raise ValueError(f"A prim already exists at path: '{prim_path}'.")

    prim = prim_utils.get_prim_at_path(prim_path)

    # Set clipping range (the only standard camera param that matters for OpenCV mode)
    if hasattr(cfg, "clipping_range") and cfg.clipping_range is not None:
        prim.GetAttribute("clippingRange").Set(cfg.clipping_range)

    # Apply OmniLensDistortionOpenCvPinholeAPI schema via Sdf layer
    # Note: This only works with headless=false (use xvfb-run for headless rendering)
    stage = prim.GetStage()
    layer = stage.GetRootLayer()
    prim_spec = layer.GetPrimAtPath(prim_path)
    if prim_spec is None:
        prim_spec = pxr.Sdf.CreatePrimInLayer(layer, prim_path)  # noqa: F823

    api_schemas = prim_spec.GetInfo("apiSchemas")
    if api_schemas is None:
        api_schemas = pxr.Sdf.TokenListOp()
    prepend_items = list(api_schemas.prependedItems) if api_schemas.prependedItems else []
    if "OmniLensDistortionOpenCvPinholeAPI" not in prepend_items:
        prepend_items.append("OmniLensDistortionOpenCvPinholeAPI")
        api_schemas.prependedItems = prepend_items
        prim_spec.SetInfo("apiSchemas", api_schemas)

    # Set the distortion model
    prim.CreateAttribute("omni:lensdistortion:model", pxr.Sdf.ValueTypeNames.Token).Set(
        "opencvPinhole"
    )

    # Set OpenCV distortion parameters from config
    opencv_attrs = {
        "omni:lensdistortion:opencvPinhole:fx": ("opencv_fx", pxr.Sdf.ValueTypeNames.Float),
        "omni:lensdistortion:opencvPinhole:fy": ("opencv_fy", pxr.Sdf.ValueTypeNames.Float),
        "omni:lensdistortion:opencvPinhole:cx": ("opencv_cx", pxr.Sdf.ValueTypeNames.Float),
        "omni:lensdistortion:opencvPinhole:cy": ("opencv_cy", pxr.Sdf.ValueTypeNames.Float),
        "omni:lensdistortion:opencvPinhole:k1": ("opencv_k1", pxr.Sdf.ValueTypeNames.Float),
        "omni:lensdistortion:opencvPinhole:k2": ("opencv_k2", pxr.Sdf.ValueTypeNames.Float),
        "omni:lensdistortion:opencvPinhole:k3": ("opencv_k3", pxr.Sdf.ValueTypeNames.Float),
        "omni:lensdistortion:opencvPinhole:k4": ("opencv_k4", pxr.Sdf.ValueTypeNames.Float),
        "omni:lensdistortion:opencvPinhole:k5": ("opencv_k5", pxr.Sdf.ValueTypeNames.Float),
        "omni:lensdistortion:opencvPinhole:k6": ("opencv_k6", pxr.Sdf.ValueTypeNames.Float),
        "omni:lensdistortion:opencvPinhole:p1": ("opencv_p1", pxr.Sdf.ValueTypeNames.Float),
        "omni:lensdistortion:opencvPinhole:p2": ("opencv_p2", pxr.Sdf.ValueTypeNames.Float),
        "omni:lensdistortion:opencvPinhole:s1": ("opencv_s1", pxr.Sdf.ValueTypeNames.Float),
        "omni:lensdistortion:opencvPinhole:s2": ("opencv_s2", pxr.Sdf.ValueTypeNames.Float),
        "omni:lensdistortion:opencvPinhole:s3": ("opencv_s3", pxr.Sdf.ValueTypeNames.Float),
        "omni:lensdistortion:opencvPinhole:s4": ("opencv_s4", pxr.Sdf.ValueTypeNames.Float),
    }

    for attr_name, (cfg_name, attr_type) in opencv_attrs.items():
        if hasattr(cfg, cfg_name):
            value = getattr(cfg, cfg_name)
            if value is not None:
                prim.CreateAttribute(attr_name, attr_type).Set(value)

    # Set image size
    if hasattr(cfg, "opencv_image_size") and cfg.opencv_image_size is not None:
        import pxr

        prim.CreateAttribute(
            "omni:lensdistortion:opencvPinhole:imageSize", pxr.Sdf.ValueTypeNames.Int2
        ).Set(pxr.Gf.Vec2i(cfg.opencv_image_size[0], cfg.opencv_image_size[1]))
        print(  # noqa: T201
            f"[DEBUG] spawn_opencv_camera: Set imageSize to {cfg.opencv_image_size}"
        )  # noqa: T201

    # Debug: print all the values that were set
    print(f"[DEBUG] spawn_opencv_camera at {prim_path}:")  # noqa: T201
    print(  # noqa: T201
        f"[DEBUG]   fx={getattr(cfg, 'opencv_fx', None)}, fy={getattr(cfg, 'opencv_fy', None)}"
    )  # noqa: T201
    print(  # noqa: T201
        f"[DEBUG]   cx={getattr(cfg, 'opencv_cx', None)}, cy={getattr(cfg, 'opencv_cy', None)}"
    )  # noqa: T201
    print(f"[DEBUG]   image_size={getattr(cfg, 'opencv_image_size', None)}")  # noqa: T201

    return prim


@configclass
class OpenCVCameraCfg:
    """Camera config with OpenCV lens distortion parameters."""

    func = spawn_opencv_camera
    copy_from_source: bool = True  # Required by IsaacLab spawner

    clipping_range: tuple[float, float] = (0.01, 20.0)

    # These must exist or Camera.__init__ crashes, but values don't matter -
    # OpenCV distortion API (fx/fy/cx/cy) overrides the projection at render time
    horizontal_aperture: float = 1.0
    vertical_aperture: float | None = None

    # OpenCV intrinsics (required)
    opencv_fx: float = None
    opencv_fy: float = None
    opencv_cx: float = None
    opencv_cy: float = None
    opencv_image_size: tuple[int, int] = None

    # OpenCV distortion coefficients (required - at least k1 should be non-zero)
    opencv_k1: float = None
    opencv_k2: float = 0.0
    opencv_k3: float = 0.0
    opencv_k4: float = 0.0
    opencv_k5: float = 0.0
    opencv_k6: float = 0.0
    opencv_p1: float = 0.0
    opencv_p2: float = 0.0
    opencv_s1: float = 0.0
    opencv_s2: float = 0.0
    opencv_s3: float = 0.0
    opencv_s4: float = 0.0


def _resolve_object_usd_paths(usd_path):
    def expand_regex(path_pattern):
        if os.path.exists(path_pattern):
            return [os.path.abspath(path_pattern)]
        directory = os.path.dirname(path_pattern) or "."
        if not os.path.isdir(directory):
            return []
        base_pattern = os.path.basename(path_pattern)
        try:
            regex = re.compile(base_pattern)
        except re.error as exc:
            raise ValueError(f"Invalid object_usd_path regex: {base_pattern}") from exc
        matched = [
            os.path.abspath(os.path.join(directory, name))
            for name in os.listdir(directory)
            if regex.fullmatch(name)
        ]
        matched.sort(key=lambda p: os.path.basename(p))
        return matched

    if isinstance(usd_path, list | tuple):
        raw_paths = list(usd_path)
    else:
        raw_paths = [usd_path]

    expanded_paths = []
    used_pattern = False
    for path in raw_paths:
        if isinstance(path, str):
            pattern_matches = expand_regex(path)
            if len(pattern_matches) == 1:
                expanded_paths.append(os.path.abspath(pattern_matches[0]))
            elif len(pattern_matches) > 1:
                used_pattern = True
                expanded_paths.extend(pattern_matches)
            else:
                raise ValueError(f"object_usd_path regex did not match any files: {path}")
        else:
            expanded_paths.append(path)

    if not expanded_paths:
        expanded_paths.append(os.path.abspath(raw_paths[0]))
    if used_pattern:
        expanded_paths.sort(key=lambda p: os.path.basename(p))
    return expanded_paths


def _read_object_usd_list(path: str) -> list[str]:
    with open(path, encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def _parse_full_scene_slot_dynamic_indices(value) -> set[int] | None:
    if value is None or value == "":
        return None
    if isinstance(value, str):
        if value.lower() in {"none", "null", "false"}:
            return None
        return {int(item) for item in re.split(r"[,\s]+", value.strip()) if item}
    if isinstance(value, int):
        return {int(value)}
    if isinstance(value, list | tuple | set):
        return {int(item) for item in value}
    return None


def _slot_flag_from_indices(idx: int, indices: set[int] | None, default: bool) -> bool:
    if indices is None:
        return bool(default)
    return int(idx) in indices


def _resolve_full_scene_usd_list_dir(usd_list_dir: str) -> tuple[list[str], dict[str, list[str]]]:
    """Resolve full_scene_object_usds/<motion_key>.txt files.

    Returns a unique USD path list for scene spawning and a per-motion stem map
    used later to place each motion's object trajectories into the right assets.
    """
    list_files = sorted(
        os.path.join(usd_list_dir, name)
        for name in os.listdir(usd_list_dir)
        if name.endswith(".txt")
    )
    resolved_paths: list[str] = []
    seen: set[str] = set()
    stems_by_motion: dict[str, list[str]] = {}
    for list_file in list_files:
        motion_key = os.path.splitext(os.path.basename(list_file))[0]
        paths = _read_object_usd_list(list_file)
        stems_by_motion[motion_key] = [
            os.path.splitext(os.path.basename(path))[0] for path in paths
        ]
        for path in paths:
            abs_path = os.path.abspath(path)
            if abs_path not in seen:
                resolved_paths.append(abs_path)
                seen.add(abs_path)
    return resolved_paths, stems_by_motion


def _ordered_full_scene_motion_keys(
    usd_list_dir: str,
    motion_file_path,
    filter_motion_keys,
    max_unique_motions,
    interleave_by_object: bool,
    motion_lib_cfg: dict,
) -> list[str]:
    list_keys = {
        os.path.splitext(name)[0]
        for name in os.listdir(usd_list_dir)
        if name.endswith(".txt")
    }
    if motion_file_path and os.path.isdir(str(motion_file_path)):
        all_robot_stems = {
            os.path.splitext(f)[0]
            for f in os.listdir(str(motion_file_path))
            if f.endswith(".pkl")
        }
        if filter_motion_keys:
            motion_keys = [k for k in filter_motion_keys if k in all_robot_stems and k in list_keys]
        else:
            motion_keys = sorted(k for k in all_robot_stems if k in list_keys)
    else:
        motion_keys = sorted(list_keys)

    shard_rank = motion_lib_cfg.get("motion_shard_rank", 0)
    shard_ws = motion_lib_cfg.get("motion_shard_world_size", 1)
    if shard_ws > 1:
        total = len(motion_keys)
        start = (shard_rank * total) // shard_ws
        end = ((shard_rank + 1) * total) // shard_ws
        motion_keys = motion_keys[start:end]
        print(  # noqa: T201
            f"[Full-Scene Slots] Rank {shard_rank}/{shard_ws}: "
            f"motions [{start}:{end}] = {len(motion_keys)}/{total}"
        )

    if max_unique_motions and len(motion_keys) > max_unique_motions:
        if interleave_by_object:
            from gear_sonic.utils.motion_lib.motion_lib_base import interleave_keys_by_object

            motion_keys = interleave_keys_by_object(motion_keys)
        motion_keys = motion_keys[:max_unique_motions]
        print(f"[Full-Scene Slots] Capped to {max_unique_motions} motions")  # noqa: T201
    return motion_keys


def _resolve_full_scene_usd_slot_lists(
    usd_list_dir: str,
    motion_file_path,
    filter_motion_keys,
    max_unique_motions,
    interleave_by_object: bool,
    motion_lib_cfg: dict,
    max_slots: int,
) -> tuple[list[list[str]], dict[str, list[str]]]:
    """Return slot-wise USD lists ordered to match motion assignment.

    MultiUsdFileCfg(random_choice=False) picks ``usd_path[env_id % len(usd_path)]``.
    When use_paired_motions also assigns ``motion_id = env_id % num_motions``,
    each object slot can therefore select the USD belonging to that env's motion.
    Missing per-motion slots are padded with an arbitrary USD; reset code keeps
    those padded slots inactive and far from the playable scene.
    """
    motion_keys = _ordered_full_scene_motion_keys(
        usd_list_dir,
        motion_file_path,
        filter_motion_keys,
        max_unique_motions,
        interleave_by_object,
        motion_lib_cfg,
    )
    paths_by_motion: dict[str, list[str]] = {}
    stems_by_motion: dict[str, list[str]] = {}
    fallback_path = None
    for motion_key in motion_keys:
        list_path = os.path.join(usd_list_dir, f"{motion_key}.txt")
        paths = [os.path.abspath(path) for path in _read_object_usd_list(list_path)]
        paths_by_motion[motion_key] = paths
        stems_by_motion[motion_key] = [
            os.path.splitext(os.path.basename(path))[0] for path in paths
        ]
        if fallback_path is None and paths:
            fallback_path = paths[0]
    if fallback_path is None:
        raise ValueError(f"No object USDs found in full-scene list dir: {usd_list_dir}")

    observed_max = max((len(paths) for paths in paths_by_motion.values()), default=1)
    slot_count = max(1, min(int(max_slots or observed_max), observed_max))
    slot_paths: list[list[str]] = []
    for slot_idx in range(slot_count):
        slot_paths.append(
            [
                paths_by_motion[motion_key][slot_idx]
                if slot_idx < len(paths_by_motion[motion_key])
                else fallback_path
                for motion_key in motion_keys
            ]
        )
    return slot_paths, stems_by_motion


def _resolve_paired_motion_usd_paths(
    usd_dir: str,
    motion_file_path,
    filter_motion_keys,
    max_unique_motions,
    interleave_by_object: bool,
    motion_lib_cfg: dict,
) -> list[str]:
    """Resolve per-motion USDs in the same order used by paired motion loading."""
    if not usd_dir:
        return []
    if not os.path.isdir(usd_dir):
        return _resolve_object_usd_paths(usd_dir)
    if motion_file_path and os.path.isdir(str(motion_file_path)):
        if filter_motion_keys:
            all_robot_stems = {
                os.path.splitext(f)[0]
                for f in os.listdir(str(motion_file_path))
                if f.endswith(".pkl")
            }
            motion_stems = [k for k in filter_motion_keys if k in all_robot_stems]
        else:
            motion_stems = sorted(
                os.path.splitext(f)[0]
                for f in os.listdir(str(motion_file_path))
                if f.endswith(".pkl")
            )

        shard_rank = motion_lib_cfg.get("motion_shard_rank", 0)
        shard_ws = motion_lib_cfg.get("motion_shard_world_size", 1)
        if shard_ws > 1:
            total = len(motion_stems)
            start = (shard_rank * total) // shard_ws
            end = ((shard_rank + 1) * total) // shard_ws
            motion_stems = motion_stems[start:end]

        if max_unique_motions and len(motion_stems) > max_unique_motions:
            if interleave_by_object:
                from gear_sonic.utils.motion_lib.motion_lib_base import interleave_keys_by_object

                motion_stems = interleave_keys_by_object(motion_stems)
            else:
                motion_stems = sorted(motion_stems)
            motion_stems = motion_stems[:max_unique_motions]

        resolved_paths = []
        for stem in motion_stems:
            candidates = [
                os.path.join(usd_dir, stem + ".usd"),
                os.path.join(usd_dir, stem + ".usda"),
            ]
            usd_file = next((p for p in candidates if os.path.isfile(p)), None)
            if usd_file is not None:
                resolved_paths.append(os.path.abspath(usd_file))
            else:
                print(f"[WARNING] No paired USD for motion '{stem}' in {usd_dir}")  # noqa: T201
        return resolved_paths

    return sorted(
        os.path.abspath(os.path.join(usd_dir, f))
        for f in os.listdir(usd_dir)
        if f.endswith((".usd", ".usda"))
    )


def _load_object_role_metadata(path: str | None) -> dict | None:
    if not path:
        return None
    path = os.path.abspath(path)
    if not os.path.isfile(path):
        print(f"[ObjectDynamics] Role metadata missing: {path}")  # noqa: T201
        return None
    with open(path) as f:
        data = json.load(f)
    data.setdefault("default", {})
    data.setdefault("rules", [])
    return data


def _load_object_physics_overrides(path: str | None) -> dict | None:
    if not path:
        return None
    path = os.path.abspath(path)
    if not os.path.isfile(path):
        print(f"[ObjectPhysics] Override metadata missing: {path}")  # noqa: T201
        return None
    with open(path) as f:
        data = json.load(f)
    data.setdefault("default", {})
    data.setdefault("objects", {})
    data.setdefault("rules", [])
    print(f"[ObjectPhysics] Loaded overrides from {path}")  # noqa: T201
    return data


def _object_role_for_stem(stem: str, roles: dict | None) -> dict:
    if roles is None:
        return {}
    role = dict(roles.get("default", {}))
    exact = roles.get("objects", {}).get(stem)
    if exact:
        role.update(exact)
    for rule in roles.get("rules", []):
        contains = rule.get("contains")
        pattern = rule.get("match")
        matched = bool(contains and contains in stem)
        if pattern:
            matched = matched or re.search(pattern, stem) is not None
        if not matched:
            continue
        for key, value in rule.items():
            if key not in {"contains", "match"}:
                role[key] = value
    return role


def _object_physics_override_for_stem(
    stem: str, role: dict, overrides: dict | None
) -> dict:
    if overrides is None:
        return {}
    physics = dict(overrides.get("default", {}))
    exact = overrides.get("objects", {}).get(stem)
    if exact:
        physics.update(exact)
    role_name = role.get("role")
    for rule in overrides.get("rules", []):
        predicates = []
        contains = rule.get("contains")
        if contains:
            predicates.append(contains in stem)
        pattern = rule.get("match")
        if pattern:
            predicates.append(re.search(pattern, stem) is not None)
        rule_role = rule.get("role")
        if rule_role:
            if isinstance(rule_role, list):
                predicates.append(role_name in rule_role)
            else:
                predicates.append(role_name == rule_role)
        matched = bool(predicates) and all(predicates)
        if not matched:
            continue
        for key, value in rule.items():
            if key not in {"contains", "match", "role"}:
                physics[key] = value
    return physics


@configclass
class MySceneCfg(InteractiveSceneCfg):
    """Configuration for the terrain scene with a legged robot."""

    def __init__(self, config, commands=None, **kwargs):  # noqa: ARG002
        super().__init__()

        _mcfg = (commands or {}).get("motion", {}).get("motion_lib_cfg", {}) or {}
        motion_file_path = _mcfg.get("motion_file", None)
        filter_motion_keys = _mcfg.get("filter_motion_keys", None)
        max_unique_motions = _mcfg.get("max_unique_motions", None)
        interleave_by_object = _mcfg.get("interleave_by_object", False)

        self.num_envs = config.get("num_envs", 4096)
        use_asset_compatible_motions = bool(
            (commands or {})
            .get("motion", {})
            .get("use_asset_compatible_motions", False)
        )
        joint_roster_view = None
        # Direct-directory roster (docs/DIRECT_DIRECTORY_ROSTER.md): the pack directory
        # itself is the input; the rank's slot view is derived in memory here.
        direct_spec = normalize_direct_spec(_mcfg.get("joint_roster_direct_dirs"))
        if direct_spec and not use_asset_compatible_motions:
            raise ValueError(
                "motion_lib_cfg.joint_roster_direct_dirs is set but "
                "commands.motion.use_asset_compatible_motions is false: the direct-directory "
                "roster is consumed only by the asset-compatible route (enable it or unset "
                "JOINT_ROSTER_DIRECT_DIRS)"
            )
        if use_asset_compatible_motions:
            manifest_path = _mcfg.get("joint_source_manifest")
            if direct_spec and manifest_path:
                raise ValueError(
                    "ambiguous joint roster input: both motion_lib_cfg.joint_roster_direct_dirs "
                    f"({direct_spec!r}) and motion_lib_cfg.joint_source_manifest "
                    f"({manifest_path!r}) are set; use exactly one"
                )
            if not manifest_path and not direct_spec:
                raise ValueError(
                    "use_asset_compatible_motions requires "
                    "motion_lib_cfg.joint_source_manifest or "
                    "motion_lib_cfg.joint_roster_direct_dirs"
                )
            rank = int(
                os.environ.get(
                    "RANK",
                    _mcfg.get("motion_shard_rank", 0),
                )
            )
            world_size = int(
                os.environ.get(
                    "WORLD_SIZE",
                    _mcfg.get("motion_shard_world_size", 1),
                )
            )
            plan_cache_mode = str(
                _mcfg.get("joint_roster_scene_plan_cache_mode", "disabled") or "disabled"
            ).strip().lower()
            if direct_spec:
                # Native in-memory resolution only: the rank scene-plan cache is never
                # consulted or written for a directory input (no file identity exists; a
                # persisted rank plan would be a stale-able roster index). The helper reports
                # the requested cache mode in its log lines.
                joint_roster_view, direct_log_lines = resolve_direct_roster_for_env_cfg(
                    _mcfg,
                    num_envs=self.num_envs,
                    rank=rank,
                    world_size=world_size,
                )
                for line in direct_log_lines:
                    print(line)  # noqa: T201
            elif plan_cache_mode != "disabled":
                # Optional run-local plan cache (default disabled): a compatible
                # hit reuses the resolved Python rank plan; a miss runs the same
                # resolver and publishes it. This never persists a PhysX scene.
                from gear_sonic.rp2.rank_scene_plan_cache import (  # noqa: PLC0415
                    resolve_rank_scene_plan_cached,
                )

                joint_roster_view, plan_cache_status = resolve_rank_scene_plan_cached(
                    resolve_rank_local_roster,
                    manifest_path,
                    num_envs=self.num_envs,
                    rank=rank,
                    world_size=world_size,
                    runtime_target_fps_hz=float(_mcfg.get("target_fps", 50)),
                    cache_mode=plan_cache_mode,
                    cache_dir=_mcfg.get("joint_roster_scene_plan_cache_dir"),
                )
                print(  # noqa: T201
                    "[JointRoster] startup_phase=rank_scene_plan_cache "
                    f"cache_mode={plan_cache_status.get('mode')} rank={rank} "
                    f"world_size={world_size} num_envs={self.num_envs} "
                    f"elapsed_s={float(plan_cache_status.get('elapsed_seconds', 0.0)):.3f} "
                    f"entry={plan_cache_status.get('entry')}"
                    + (
                        f" write_error={plan_cache_status['write_error']}"
                        if "write_error" in plan_cache_status
                        else ""
                    )
                )
            else:
                joint_roster_view = resolve_rank_local_roster(
                    manifest_path,
                    num_envs=self.num_envs,
                    rank=rank,
                    world_size=world_size,
                    runtime_target_fps_hz=float(
                        _mcfg.get("target_fps", 50)
                    ),
                )
            motion_file_path = joint_roster_view["motion_dir"]
            filter_motion_keys = joint_roster_view["motion_aliases"]
            max_unique_motions = None
            self._tmp_joint_roster_view = joint_roster_view

        self.env_spacing = config.get("env_spacing", 2.5)

        # Allow config to override replicate_physics (default True)
        # Set to False for per-environment randomization (e.g., table size)
        self.replicate_physics = config.get("replicate_physics", True)

        self.eval_camera = None
        if config.get("render_results", False):
            render_env_ids = config.get("render_env_ids", None)
            if render_env_ids is None:
                eval_camera_prim_path = "/World/envs/env_.*/eval_camera"
            else:
                render_env_ids = [int(env_id) for env_id in render_env_ids]
                if not render_env_ids:
                    raise ValueError("render_env_ids must not be empty")
                alternatives = "|".join(str(env_id) for env_id in render_env_ids)
                eval_camera_prim_path = (
                    f"/World/envs/env_({alternatives})/eval_camera"
                )
            self.eval_camera = TiledCameraCfg(
                prim_path=eval_camera_prim_path,
                offset=TiledCameraCfg.OffsetCfg(
                    pos=(0, 0, 0), rot=(1, 0, 0, 0), convention="world"
                ),
                data_types=["rgb"],
                spawn=sim_utils.PinholeCameraCfg(
                    focal_length=5.0,
                    focus_distance=50.0,
                    horizontal_aperture=5,
                    clipping_range=(0.1, 20.0),
                ),
                width=config.get("render_width", 1920),
                height=config.get("render_height", 1080),
            )
            if render_env_ids is not None:
                self.eval_camera.class_type = SelectedEnvironmentTiledCamera

        # Single global overview camera (not per-env) for replay videos
        self.overview_camera = None
        self.overview_camera_side = None
        if config.get("overview_camera", False):
            _cam_w = config.get("render_width", 1920)
            _cam_h = config.get("render_height", 1080)
            _cam_cfg = dict(
                data_types=["rgb"],
                spawn=sim_utils.PinholeCameraCfg(
                    focal_length=5.0,
                    focus_distance=100.0,
                    horizontal_aperture=10.0,
                    clipping_range=(0.1, 500.0),
                ),
                width=_cam_w,
                height=_cam_h,
            )
            self.overview_camera = CameraCfg(
                prim_path="/World/OverviewCamera",
                offset=CameraCfg.OffsetCfg(pos=(0, 0, 50), rot=(1, 0, 0, 0), convention="world"),
                **_cam_cfg,
            )
            self.overview_camera_side = CameraCfg(
                prim_path="/World/OverviewCameraSide",
                offset=CameraCfg.OffsetCfg(pos=(0, 0, 50), rot=(1, 0, 0, 0), convention="world"),
                **_cam_cfg,
            )

        # ground terrain
        terrain_type = config.get("terrain_type", "plane")
        if terrain_type == "plane":
            self.terrain = TerrainImporterCfg(
                prim_path="/World/ground",
                terrain_type="plane",
                collision_group=-1,
                physics_material=sim_utils.RigidBodyMaterialCfg(
                    friction_combine_mode="multiply",
                    restitution_combine_mode="multiply",
                    static_friction=1.0,
                    dynamic_friction=1.0,
                ),
                visual_material=sim_utils.MdlFileCfg(
                    mdl_path="{NVIDIA_NUCLEUS_DIR}/Materials/Base/Architecture/Shingles_01.mdl",
                    project_uvw=True,
                ),
            )
        elif terrain_type == "trimesh":
            self.terrain = TerrainImporterCfg(
                prim_path="/World/ground",
                terrain_type="generator",
                terrain_generator=terrain.ROUGH_TERRAINS_CFG,
                max_init_terrain_level=10,
                collision_group=1,
                physics_material=sim_utils.RigidBodyMaterialCfg(
                    friction_combine_mode="multiply",
                    restitution_combine_mode="multiply",
                    static_friction=1.0,
                    dynamic_friction=1.0,
                ),
                visual_material=sim_utils.MdlFileCfg(
                    mdl_path="{NVIDIA_NUCLEUS_DIR}/Materials/Base/Architecture/Shingles_01.mdl",
                    project_uvw=True,
                ),
                debug_vis=False,
            )
        else:
            raise ValueError(f"Unknown terrain type: {terrain_type}")

        # robots
        self.robot: ArticulationCfg = dataclasses.MISSING

        # lights
        if not config.get("render_ego_random", False):
            self.light = AssetBaseCfg(
                prim_path="/World/light",
                spawn=sim_utils.DistantLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
            )

            self.sky_light = AssetBaseCfg(
                prim_path="/World/skyLight",
                spawn=sim_utils.DomeLightCfg(color=(0.13, 0.13, 0.13), intensity=1000.0),
            )
        else:
            self.sky_light = AssetBaseCfg(
                prim_path="/World/skyLight",
                spawn=sim_utils.DomeLightCfg(color=(0.13, 0.13, 0.13), intensity=1000.0),
            )
            # from isaac_playground.env_rand.domelight import RandomDomeLightCfg
            # self.sky_light = RandomDomeLightCfg(
            #     prim_path="/World/skyLight",
            #     texture_file_folder="../rl_data/HDRIs",
            #     dynamic_randomize_texture=True,
            #     dynamic_randomize_texture_interval=1.0
            # )

        # [D perf migration] The 53-body robot net-force sensor is kept by default
        # (legacy configs stay identical). Its only yaml consumer in the D family
        # (undesired_contacts_no_ankle_hand) is null and no code references it,
        # while history_length=3 forces a full PhysX refresh on every physics
        # substep; `enable_robot_net_contact_sensor: false` removes it.
        # Evidence: runs/d_perf_migration_20261011/consumer_map/sensors_consumer_map.md
        if config.get("enable_robot_net_contact_sensor", True):
            self.contact_forces = ContactSensorCfg(
                prim_path="{ENV_REGEX_NS}/Robot/.*",
                history_length=3,
                track_air_time=True,
                force_threshold=10.0,
                debug_vis=False,
            )

        # Feet-to-terrain contact sensor: isolates contact forces between feet and
        # the terrain Object so rewards can penalize horizontal forces without
        # contamination from gravity or internal joint forces.
        # Pattern: prim_path = Object (sensed body), filter = feet (contact partners).
        # force_matrix_w shape: [num_envs, 1, 2, 3] — force on Object from each foot.
        if config.get("add_object", False) and config.get("enable_feet_terrain_contact", True):
            self.feet_terrain_contact_sensor = ContactSensorCfg(
                prim_path="{ENV_REGEX_NS}/Object",
                filter_prim_paths_expr=_robot_body_paths(
                    ["left_ankle_roll_link", "right_ankle_roll_link"]
                ),
                history_length=2,
                track_air_time=False,
            )

        # Robot-specific metadata is used for optional hand sensors and link defaults.
        robot_type = config.get("robot", {}).get("type", "g1")
        robot_info = ROBOT_RUNTIME_INFO.get(
            robot_type, {"robot_has_hands": False, "hand_type": None}
        )
        robot_has_hands = config.get("robot_has_hands", robot_info["robot_has_hands"])
        hand_type = config.get("hand_type", robot_info["hand_type"])
        hand_sensor_defaults = _default_hand_sensor_links(hand_type)

        motion_meta_info_path = config.get("motion_meta_info_path", None)
        motion_meta_info = None

        usd_path = config.get("object_usd_path", "")
        if motion_meta_info_path is None and isinstance(usd_path, str) and os.path.isfile(usd_path):
            usd_root, usd_ext = os.path.splitext(usd_path)
            if usd_ext.lower() in {".usd", ".usda"}:
                motion_meta_info_path = (usd_root + ".pkl").replace("object_usd", "meta")
        if motion_meta_info_path is not None and os.path.isfile(motion_meta_info_path):
            motion_meta_info = joblib.load(motion_meta_info_path)

        # Scene scale: config override > meta file > default 1.0
        scene_scale = config.get("scene_scale", 1.0)
        if motion_meta_info is not None:
            scene_scale = motion_meta_info.get("scene_scale", 1.0)

        # Ensure scene_scale is never None
        if scene_scale is None:
            scene_scale = 1.0

        # Object with rigid body (optional, can be configured via config)
        # Supports: single file, directory, list, or regex patterns for object_usd_path
        #
        # multi_object_per_env=True:  All objects spawned in every env (one active at a time)
        # multi_object_per_env=False: One object per env (MultiUsdFileCfg cycles through)
        if config.get("add_object", False):
            usd_path = config.get("object_usd_path", f"{os.getcwd()}/data/wheelchair.usd")
            object_is_dynamic = config.get("object_is_dynamic", False)
            object_dynamics_from_roles = config.get("object_dynamics_from_roles", False)
            object_roles_path = config.get("object_roles_path", None)
            if object_roles_path is None:
                object_roles_path = (commands or {}).get("motion", {}).get(
                    "object_roles_path", None
                )
            object_roles = (
                _load_object_role_metadata(object_roles_path)
                if object_dynamics_from_roles
                else None
            )
            object_physics_overrides_path = config.get("object_physics_overrides_path", None)
            object_physics_overrides = _load_object_physics_overrides(
                object_physics_overrides_path
            )
            object_dynamics_by_stem: dict[str, bool] = {}
            object_physics_overrides_by_stem: dict[str, dict] = {}
            default_object_mass = config.get("object_mass", None)
            default_object_density = config.get("object_density", None)

            def _object_spawn_metadata(stem: str) -> tuple[dict, bool, dict]:
                if object_roles is not None:
                    role = _object_role_for_stem(stem, object_roles)
                    dynamic = bool(role.get("dynamic", object_is_dynamic))
                else:
                    role = {}
                    dynamic = bool(object_is_dynamic)
                physics = _object_physics_override_for_stem(
                    stem, role, object_physics_overrides
                )
                object_dynamics_by_stem[stem] = dynamic
                if physics:
                    object_physics_overrides_by_stem[stem] = dict(physics)
                return role, dynamic, physics

            def _object_rigid_props(dynamic: bool, physics: dict):
                rigid_kwargs = {
                    "kinematic_enabled": not dynamic,
                    "max_depenetration_velocity": physics.get(
                        "max_depenetration_velocity", 1.0
                    ),
                }
                for key in (
                    "linear_damping",
                    "angular_damping",
                    "max_linear_velocity",
                    "max_angular_velocity",
                    "max_contact_impulse",
                    "enable_gyroscopic_forces",
                    "retain_accelerations",
                    "solver_position_iteration_count",
                    "solver_velocity_iteration_count",
                    "sleep_threshold",
                    "stabilization_threshold",
                ):
                    value = physics.get(key)
                    if value is not None:
                        rigid_kwargs[key] = value
                return sim_utils.RigidBodyPropertiesCfg(**rigid_kwargs)

            def _object_mass_props(physics: dict):
                if physics.get("mass") is not None:
                    return sim_utils.MassPropertiesCfg(mass=physics["mass"])
                if physics.get("density") is not None:
                    return sim_utils.MassPropertiesCfg(density=physics["density"])
                if default_object_mass is not None:
                    return sim_utils.MassPropertiesCfg(mass=default_object_mass)
                if default_object_density is not None:
                    return sim_utils.MassPropertiesCfg(density=default_object_density)
                return None

            def _object_collision_props(physics: dict):
                collision_kwargs = {
                    "collision_enabled": physics.get(
                        "collision_enabled", object_collision_enabled
                    )
                }
                for key in (
                    "contact_offset",
                    "rest_offset",
                    "torsional_patch_radius",
                    "min_torsional_patch_radius",
                ):
                    value = physics.get(key, config.get(f"object_{key}", None))
                    if value is not None:
                        collision_kwargs[key] = value
                return sim_utils.CollisionPropertiesCfg(**collision_kwargs)

            object_collision_enabled = config.get("object_collision_enabled", True)
            multi_object_per_env = config.get("multi_object_per_env", False)
            object_color = config.get("object_color", None)  # e.g. [0.6, 0.4, 0.2] for wood-brown
            object_sensor_prim_path = "{ENV_REGEX_NS}/Object"
            multi_object_sensor_prim_paths = []

            # Check if terrain-aware mode is enabled
            terrain_motion_dir = config.get("terrain_motion_dir", None)
            flat_usd_cfg = config.get("flat_usd_path", None)
            terrain_usd_cfg = config.get("terrain_usd_path", None)
            _terrain_aware = terrain_motion_dir or (flat_usd_cfg and terrain_usd_cfg)

            if _terrain_aware:
                # ── TERRAIN-AWARE MODE ──
                # Per-rank slicing with terrain/flat interleaving.
                # Build a per-rank USD list that places terrain USDs at positions
                # matching commands.py's _env_is_terrain mask (env_id % (R+1) == R).
                _cfg_rank = _mcfg.get("motion_shard_rank", None)
                if _cfg_rank is not None:
                    rank = int(_cfg_rank)
                else:
                    rank = int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0")))
                num_envs = config.get("num_envs", 1024)
                flat_motion_dir = config.get("flat_motion_dir", None)

                # --- Discover terrain keys + USDs ---
                if terrain_motion_dir:
                    terrain_motion_dir = os.path.abspath(terrain_motion_dir)
                    terrain_usd_dir = os.path.join(terrain_motion_dir, "object_usd")
                    terrain_robot_dir = os.path.join(terrain_motion_dir, "robot")

                    # Scan USDs: {stem: absolute_path}
                    usd_by_stem = {}
                    for f in glob.glob(
                        os.path.join(terrain_usd_dir, "**", "*.usd"), recursive=True
                    ):
                        stem = os.path.splitext(os.path.basename(f))[0]
                        if stem == "flat_placeholder":
                            continue
                        usd_by_stem[stem] = os.path.abspath(f)

                    # Scan robot PKLs and enforce 1:1 matching with USDs by stem
                    all_terrain_keys = []
                    terrain_usd_paths = []
                    missing_usds = []
                    for f in sorted(
                        glob.glob(os.path.join(terrain_robot_dir, "**", "*.pkl"), recursive=True)
                    ):
                        if f.endswith("metadata.pkl"):
                            continue
                        stem = os.path.splitext(os.path.basename(f))[0]
                        if stem in usd_by_stem:
                            all_terrain_keys.append(stem)
                            terrain_usd_paths.append(usd_by_stem[stem])
                        else:
                            missing_usds.append(stem)
                    if missing_usds:
                        total = len(all_terrain_keys) + len(missing_usds)
                        raise FileNotFoundError(
                            f"[TerrainAutoDiscover] {len(missing_usds)}/{total} robot motions "
                            f"have no matching USD in {terrain_usd_dir}. "
                            f"First missing: {missing_usds[:5]}. Every robot PKL must have a "
                            f"USD with the same stem (required for use_paired_motions rendering)."
                        )

                    # Reorder terrain keys/USDs to match filter_motion_keys when set
                    # (e.g. eval render mode sorts motions by failure priority). Without this,
                    # env k would track motion[k] but render USD[k] from the unsorted scan.
                    if filter_motion_keys and all_terrain_keys:
                        key_to_usd = dict(zip(all_terrain_keys, terrain_usd_paths))
                        reordered = [k for k in filter_motion_keys if k in key_to_usd]
                        if reordered:
                            all_terrain_keys = reordered
                            terrain_usd_paths = [key_to_usd[k] for k in reordered]
                            print(  # noqa: T201
                                f"[TerrainAutoDiscover] Reordered {len(reordered)} terrain "
                                f"USDs/keys to match filter_motion_keys"
                            )

                    print(  # noqa: T201
                        f"[TerrainAutoDiscover] Found {len(all_terrain_keys)} terrain motions "
                        f"from {terrain_motion_dir}"
                    )
                else:
                    # Legacy: read terrain from txt list files
                    terrain_usd_paths = _resolve_object_usd_paths([terrain_usd_cfg])
                    terrain_keys_path = config.get("terrain_motion_keys_path", None)
                    if terrain_keys_path:
                        with open(terrain_keys_path) as _f:
                            all_terrain_keys = [l.strip() for l in _f if l.strip()]
                    else:
                        all_terrain_keys = []

                # --- Discover flat keys ---
                if flat_motion_dir:
                    flat_motion_dir = os.path.abspath(flat_motion_dir)
                    all_flat_keys = sorted(
                        [
                            os.path.splitext(os.path.basename(f))[0]
                            for f in glob.glob(
                                os.path.join(flat_motion_dir, "**", "*.pkl"), recursive=True
                            )
                            if not f.endswith("metadata.pkl")
                        ]
                    )
                    print(
                        f"[TerrainAutoDiscover] Found {len(all_flat_keys)} flat motions from {flat_motion_dir}"
                    )  # noqa: T201
                else:
                    # Legacy: read flat from txt list file
                    flat_keys_path = config.get("flat_motion_keys_path", None)
                    if flat_keys_path:
                        with open(flat_keys_path) as _f:
                            all_flat_keys = [l.strip() for l in _f if l.strip()]
                    else:
                        all_flat_keys = []

                # --- Flat placeholder USD ---
                # Look at dataset root first (new layout where object_usd/ holds only
                # per-motion USDs), then fall back to object_usd/ for older datasets.
                if flat_usd_cfg:
                    flat_template = os.path.abspath(flat_usd_cfg)
                elif terrain_motion_dir:
                    _fp_root = os.path.join(terrain_motion_dir, "flat_placeholder.usd")
                    _fp_legacy = os.path.join(
                        terrain_motion_dir, "object_usd", "flat_placeholder.usd"
                    )
                    if os.path.isfile(_fp_root):
                        flat_template = os.path.abspath(_fp_root)
                    elif os.path.isfile(_fp_legacy):
                        flat_template = os.path.abspath(_fp_legacy)
                    else:
                        raise FileNotFoundError(
                            f"No flat_placeholder.usd found at {_fp_root} or {_fp_legacy}. "
                            f"Set flat_usd_path explicitly."
                        )
                else:
                    flat_template = _resolve_object_usd_paths([flat_usd_cfg])[0]

                # --- Common terrain-aware slicing logic ---
                R = config.get("flat_to_terrain_ratio", 3)
                block_size = R + 1

                n_terrain_envs = num_envs // block_size
                n_flat_envs = num_envs - n_terrain_envs

                def _slice_wrap(lst, start, count):
                    if not lst:
                        return []
                    return [lst[(start + i) % len(lst)] for i in range(count)]

                # --- USD slicing (terrain only; flat uses placeholder) ---
                terrain_usd_slice = _slice_wrap(
                    terrain_usd_paths, rank * n_terrain_envs, n_terrain_envs
                )

                usd_paths = []
                ti = 0
                for env_id in range(num_envs):
                    if env_id % block_size == R and ti < len(terrain_usd_slice):
                        usd_paths.append(terrain_usd_slice[ti])
                        ti += 1
                    else:
                        usd_paths.append(flat_template)

                # --- Motion keys slicing ---
                if all_terrain_keys:
                    terrain_key_slice = _slice_wrap(
                        all_terrain_keys, rank * n_terrain_envs, n_terrain_envs
                    )

                    flat_per_rank = max(
                        n_flat_envs,
                        int(n_flat_envs * len(all_flat_keys) / max(len(all_terrain_keys), 1)),
                    )
                    flat_per_rank = min(flat_per_rank, len(all_flat_keys))
                    flat_key_slice = _slice_wrap(all_flat_keys, rank * flat_per_rank, flat_per_rank)

                    # Motion keys file: terrain first (index 0..N-1), then flat (index N..)
                    per_rank_keys = terrain_key_slice + flat_key_slice
                    keys_file = f"/tmp/rank_{rank}_motion_keys.txt"
                    with open(keys_file, "w") as _kf:
                        _kf.write("\n".join(per_rank_keys) + "\n")
                    config["per_rank_motion_keys_file"] = keys_file
                    config["n_terrain_motions"] = n_terrain_envs
                    if flat_motion_dir:
                        config["flat_motion_dir"] = flat_motion_dir

                    print(  # noqa: T201
                        f"[PerRankUSD] rank={rank}, num_envs={num_envs}, "
                        f"flat_envs={n_flat_envs}, terrain_envs={n_terrain_envs}, "
                        f"ratio={R}:1, unique_usd={len(set(usd_paths))}, "
                        f"terrain_motions={len(terrain_key_slice)}, "
                        f"flat_motions={len(flat_key_slice)}"
                    )
                    print(f"[PerRankUSD] terrain_keys[0:3]={terrain_key_slice[:3]}")  # noqa: T201
                    print(
                        f"[PerRankUSD] terrain_usd[0:3]={[os.path.basename(p) for p in terrain_usd_slice[:3]]}"
                    )  # noqa: T201
                    print(f"[PerRankUSD] flat_keys[0:3]={flat_key_slice[:3]}")  # noqa: T201

                    # Stash expected motion-key order for the commands.py
                    # use_paired_motions check. motion_lib loads motions in
                    # per_rank_keys order (terrain + flat, deduped), so
                    # verifying motion_lib.curr_motion_keys == these stems
                    # catches any ordering regression in terrain-aware loading.
                    self._tmp_usd_stems = list(
                        dict.fromkeys(list(terrain_key_slice) + list(flat_key_slice))
                    )

                object_mass = config.get("object_mass", None)
                object_density = config.get("object_density", None)
                if object_mass is not None:
                    mass_props = sim_utils.MassPropertiesCfg(mass=object_mass)
                elif object_density is not None:
                    mass_props = sim_utils.MassPropertiesCfg(density=object_density)
                else:
                    mass_props = None
                object_opacity = config.get("object_opacity", 1.0)
                if object_color is not None:
                    visual_material = sim_utils.PreviewSurfaceCfg(
                        diffuse_color=tuple(object_color[:3]),
                        opacity=object_opacity,
                        metallic=object_color[3] if len(object_color) > 3 else 0.0,
                        roughness=object_color[4] if len(object_color) > 4 else 0.5,
                    )
                elif object_opacity < 1.0:
                    visual_material = sim_utils.PreviewSurfaceCfg(opacity=object_opacity)
                else:
                    visual_material = None

                self.replicate_physics = False
                self.object = RigidObjectCfg(
                    prim_path="{ENV_REGEX_NS}/Object",
                    spawn=sim_utils.MultiUsdFileCfg(
                        usd_path=usd_paths,
                        random_choice=False,
                        activate_contact_sensors=True,
                        rigid_props=sim_utils.RigidBodyPropertiesCfg(
                            kinematic_enabled=not object_is_dynamic,
                            max_depenetration_velocity=1.0,
                        ),
                        mass_props=mass_props,
                        collision_props=_object_collision_props({}),
                        visual_material=visual_material,
                        scale=(scene_scale, scene_scale, scene_scale),
                    ),
                    init_state=RigidObjectCfg.InitialStateCfg(
                        pos=tuple(config.get("object_position", [2.0, 0.0, 0.0]))
                    ),
                )
            else:
                # --- Step 1: Resolve usd_path to a list of absolute paths ---
                object_usd_stems_by_motion = {}
                full_scene_slot_paths_by_index = None
                if isinstance(usd_path, list):
                    resolved_paths = _resolve_object_usd_paths(usd_path)
                elif isinstance(usd_path, str) and usd_path.endswith(".txt") and os.path.isfile(usd_path):
                    # Text file with one USD path per line (from build_object_usd_order.py)
                    resolved_paths = _read_object_usd_list(usd_path)
                    print(f"[Object USD] Loaded {len(resolved_paths)} paths from {usd_path}")
                elif os.path.isdir(usd_path):
                    full_scene_list_files = sorted(
                        f for f in os.listdir(usd_path) if f.endswith(".txt")
                    )
                    if multi_object_per_env and full_scene_list_files:
                        if config.get("full_scene_slots_per_env", False):
                            full_scene_slot_paths_by_index, object_usd_stems_by_motion = (
                                _resolve_full_scene_usd_slot_lists(
                                    usd_path,
                                    motion_file_path,
                                    filter_motion_keys,
                                    max_unique_motions,
                                    interleave_by_object,
                                    _mcfg,
                                    _mcfg.get("max_num_objects", 1),
                                )
                            )
                            resolved_paths = [
                                paths[0] for paths in full_scene_slot_paths_by_index if paths
                            ]
                            print(  # noqa: T201
                                f"[Object USD] Loaded {len(full_scene_slot_paths_by_index)} "
                                f"per-env object slots from {len(object_usd_stems_by_motion)} "
                                f"per-motion lists in {usd_path}"
                            )
                        else:
                            resolved_paths, object_usd_stems_by_motion = (
                                _resolve_full_scene_usd_list_dir(usd_path)
                            )
                            print(  # noqa: T201
                                f"[Object USD] Loaded {len(resolved_paths)} unique USDs "
                                f"from {len(object_usd_stems_by_motion)} per-motion lists in {usd_path}"
                            )
                    elif motion_file_path and os.path.isdir(str(motion_file_path)):
                        # Derive USD list from motion files — guarantees 1:1 pairing
                        if filter_motion_keys:
                            # filter_motion_keys is ordered by eval metrics; preserve that order
                            # so USD index matches motion index (required for use_paired_motions)
                            all_robot_stems = {
                                os.path.splitext(f)[0]
                                for f in os.listdir(str(motion_file_path))
                                if f.endswith(".pkl")
                            }
                            motion_stems = [k for k in filter_motion_keys if k in all_robot_stems]
                        else:
                            motion_stems = sorted(
                                os.path.splitext(f)[0]
                                for f in os.listdir(str(motion_file_path))
                                if f.endswith(".pkl")
                            )
                        # Rank-based sharding: only load this GPU's slice of motions/USDs
                        _shard_rank = _mcfg.get("motion_shard_rank", 0)
                        _shard_ws = _mcfg.get("motion_shard_world_size", 1)
                        if _shard_ws > 1:
                            _total = len(motion_stems)
                            _start = (_shard_rank * _total) // _shard_ws
                            _end = ((_shard_rank + 1) * _total) // _shard_ws
                            motion_stems = motion_stems[_start:_end]
                            print(
                                f"[USD Shard] Rank {_shard_rank}/{_shard_ws}: "
                                f"USDs [{_start}:{_end}] = {len(motion_stems)}/{_total}"
                            )
                        # Cap to max_unique_motions (must match motion lib's deterministic cap)
                        if max_unique_motions and len(motion_stems) > max_unique_motions:
                            if interleave_by_object:
                                from gear_sonic.utils.motion_lib.motion_lib_base import interleave_keys_by_object
                                motion_stems = interleave_keys_by_object(motion_stems)
                            else:
                                motion_stems = sorted(motion_stems)
                            motion_stems = motion_stems[:max_unique_motions]
                            print(f"[USD] Capped to {max_unique_motions} motions (max_unique_motions)")
                        resolved_paths = []
                        for stem in motion_stems:
                            usd_file = os.path.abspath(os.path.join(usd_path, stem + ".usd"))
                            if os.path.isfile(usd_file):
                                resolved_paths.append(usd_file)
                                # One-per-env paired assets still need a per-motion
                                # lookup. Without it, command metadata falls back to
                                # the global USD list and every env resolves the
                                # first stem's role/BPS metadata.
                                object_usd_stems_by_motion[stem] = [
                                    os.path.splitext(os.path.basename(usd_file))[0]
                                ]
                            else:
                                print(f"[WARNING] No USD for motion '{stem}', skipping")
                        print(
                            f"[Object USD] Derived {len(resolved_paths)} USDs from "
                            f"{len(motion_stems)} motions (motion-driven pairing)"
                        )
                    else:
                        resolved_paths = sorted(
                            os.path.abspath(os.path.join(usd_path, f))
                            for f in os.listdir(usd_path)
                            if f.endswith(".usd")
                        )
                else:
                    resolved_paths = [os.path.abspath(usd_path)]

                # Populate role/physics metadata for one-per-env assets as well as
                # multi-object assets. MultiUsdFileCfg must use common rigid-body
                # properties, but the command can apply per-env mass overrides at
                # reset when this map is available.
                if not multi_object_per_env:
                    for path in resolved_paths:
                        _object_spawn_metadata(
                            os.path.splitext(os.path.basename(path))[0]
                        )

                # --- Step 2: Spawn based on multi_object_per_env flag ---
                if multi_object_per_env:
                    if full_scene_slot_paths_by_index is not None:
                        for stems in object_usd_stems_by_motion.values():
                            for stem in stems:
                                _object_spawn_metadata(stem)
                        slot_dynamic_default = bool(
                            config.get("full_scene_slot_objects_dynamic", object_is_dynamic)
                        )
                        slot_dynamic_indices = _parse_full_scene_slot_dynamic_indices(
                            config.get("full_scene_slot_dynamic_indices", None)
                        )
                        slot_contact_sensor_default = bool(
                            config.get("full_scene_slot_activate_contact_sensors", True)
                        )
                        slot_contact_sensor_indices = _parse_full_scene_slot_dynamic_indices(
                            config.get("full_scene_slot_contact_sensor_indices", None)
                        )
                        slot_collision_default = bool(
                            config.get("full_scene_slot_collision_enabled", object_collision_enabled)
                        )
                        slot_collision_indices = _parse_full_scene_slot_dynamic_indices(
                            config.get("full_scene_slot_collision_indices", None)
                        )
                        object_opacity = config.get("object_opacity", 1.0)
                        visual_material = (
                            sim_utils.PreviewSurfaceCfg(opacity=object_opacity)
                            if object_opacity < 1.0
                            else None
                        )
                        z_spacing = 10.0
                        for idx, slot_paths in enumerate(full_scene_slot_paths_by_index):
                            slot_dynamic = _slot_flag_from_indices(
                                idx, slot_dynamic_indices, slot_dynamic_default
                            )
                            slot_contact_sensors = _slot_flag_from_indices(
                                idx, slot_contact_sensor_indices, slot_contact_sensor_default
                            )
                            slot_collision_enabled = _slot_flag_from_indices(
                                idx, slot_collision_indices, slot_collision_default
                            )
                            obj_name_safe = str(idx)
                            obj_prim_path = f"{{ENV_REGEX_NS}}/Object_{obj_name_safe}"
                            if idx == 0:
                                object_sensor_prim_path = obj_prim_path
                            multi_object_sensor_prim_paths.append((idx, obj_name_safe, obj_prim_path))
                            init_z = -100.0 - idx * z_spacing
                            setattr(
                                self,
                                f"object_{obj_name_safe}",
                                RigidObjectCfg(
                                    prim_path=obj_prim_path,
                                    spawn=sim_utils.MultiUsdFileCfg(
                                        usd_path=slot_paths,
                                        random_choice=False,
                                        activate_contact_sensors=slot_contact_sensors,
                                        scale=(scene_scale, scene_scale, scene_scale),
                                        rigid_props=_object_rigid_props(slot_dynamic, {}),
                                        mass_props=_object_mass_props({}),
                                        collision_props=_object_collision_props(
                                            {"collision_enabled": slot_collision_enabled}
                                        ),
                                        visual_material=visual_material,
                                    ),
                                    init_state=RigidObjectCfg.InitialStateCfg(
                                        pos=(1000.0, 0.0, init_z)
                                    ),
                                ),
                            )
                        print(  # noqa: T201
                            "[Full-Scene Slots] Spawned "
                            f"{len(full_scene_slot_paths_by_index)} object slots for "
                            f"{len(object_usd_stems_by_motion)} paired motion(s)"
                        )
                        if slot_contact_sensor_indices is not None:
                            print(  # noqa: T201
                                "[Full-Scene Slots] Slot contact sensors enabled for "
                                f"indices={sorted(slot_contact_sensor_indices)}"
                            )
                        if slot_collision_indices is not None:
                            print(  # noqa: T201
                                "[Full-Scene Slots] Slot collisions enabled for "
                                f"indices={sorted(slot_collision_indices)}"
                            )
                        if object_roles is not None:
                            dynamic_count = sum(object_dynamics_by_stem.values())
                            print(  # noqa: T201
                                f"[ObjectDynamics] Role metadata contains "
                                f"{dynamic_count}/{len(object_dynamics_by_stem)} dynamic objects; "
                                "slot rigid props use "
                                f"full_scene_slot_dynamic_indices={sorted(slot_dynamic_indices)}"
                                if slot_dynamic_indices is not None
                                else (
                                    f"[ObjectDynamics] Role metadata contains "
                                    f"{dynamic_count}/{len(object_dynamics_by_stem)} dynamic objects; "
                                    f"slot rigid props use full_scene_slot_objects_dynamic={slot_dynamic_default}"
                                )
                            )
                    else:
                        # MULTI-OBJECT MODE: One RigidObjectCfg per USD, all present in every env.
                        # commands.py detects this by checking for object_* entries in scene.rigid_objects.
                        #
                        # CRITICAL: Initial positions must be spread apart to avoid collision pairs!
                        # If all objects spawn at (0,0,0), PhysX creates O(N²) pairs at scene creation.
                        # Spread in Z (vertical) since envs only vary in X,Y.
                        z_spacing = 10.0
                        for idx, path in enumerate(resolved_paths):
                            obj_name = os.path.splitext(os.path.basename(path))[0]
                            _, obj_dynamic, obj_physics = _object_spawn_metadata(obj_name)
                            obj_name_safe = obj_name.replace("-", "_")
                            obj_prim_path = f"{{ENV_REGEX_NS}}/Object_{obj_name_safe}"
                            if idx == 0:
                                object_sensor_prim_path = obj_prim_path
                            multi_object_sensor_prim_paths.append((idx, obj_name_safe, obj_prim_path))
                            init_z = -100.0 - idx * z_spacing
                            setattr(
                                self,
                                f"object_{obj_name_safe}",
                                RigidObjectCfg(
                                    prim_path=obj_prim_path,
                                    spawn=sim_utils.UsdFileCfg(
                                        usd_path=path,
                                        activate_contact_sensors=True,
                                        scale=(scene_scale, scene_scale, scene_scale),
                                        rigid_props=_object_rigid_props(obj_dynamic, obj_physics),
                                        mass_props=_object_mass_props(obj_physics),
                                        collision_props=_object_collision_props(obj_physics),
                                    ),
                                    init_state=RigidObjectCfg.InitialStateCfg(
                                        pos=(1000.0, 0.0, init_z)
                                    ),
                                ),
                            )
                        print(  # noqa: T201
                            f"[Multi-Object Mode] Spawned {len(resolved_paths)} objects at spread Z positions"
                        )
                        if object_roles is not None:
                            dynamic_count = sum(object_dynamics_by_stem.values())
                            print(  # noqa: T201
                                f"[ObjectDynamics] Role-aware dynamics enabled: "
                                f"{dynamic_count}/{len(object_dynamics_by_stem)} objects dynamic"
                            )
                        if object_physics_overrides is not None:
                            print(  # noqa: T201
                                "[ObjectPhysics] Applied overrides to "
                                f"{len(object_physics_overrides_by_stem)}/{len(resolved_paths)} objects"
                            )
                elif len(resolved_paths) == 1:
                    # SINGLE OBJECT MODE
                    object_mass = config.get("object_mass", None)
                    object_density = config.get("object_density", None)
                    if object_mass is not None:
                        mass_props = sim_utils.MassPropertiesCfg(mass=object_mass)
                    elif object_density is not None:
                        mass_props = sim_utils.MassPropertiesCfg(density=object_density)
                    else:
                        mass_props = None
                    object_opacity = config.get("object_opacity", 1.0)
                    if object_color is not None:
                        visual_material = sim_utils.PreviewSurfaceCfg(
                            diffuse_color=tuple(object_color[:3]),
                            opacity=object_opacity,
                            metallic=object_color[3] if len(object_color) > 3 else 0.0,
                            roughness=object_color[4] if len(object_color) > 4 else 0.5,
                        )
                    elif object_opacity < 1.0:
                        visual_material = sim_utils.PreviewSurfaceCfg(opacity=object_opacity)
                    else:
                        visual_material = None
                    self.object = RigidObjectCfg(
                        prim_path="{ENV_REGEX_NS}/Object",
                        spawn=sim_utils.UsdFileCfg(
                            usd_path=resolved_paths[0],
                            activate_contact_sensors=True,
                            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                                kinematic_enabled=not object_is_dynamic,
                                max_depenetration_velocity=1.0,
                            ),
                            mass_props=mass_props,
                            collision_props=_object_collision_props({}),
                            visual_material=visual_material,
                            scale=(scene_scale, scene_scale, scene_scale),
                        ),
                        init_state=RigidObjectCfg.InitialStateCfg(
                            pos=tuple(config.get("object_position", [2.0, 0.0, 0.0]))
                        ),
                    )
                else:
                    # ONE-PER-ENV MODE: Different object per env via MultiUsdFileCfg
                    self.replicate_physics = False
                    object_mass = config.get("object_mass", None)
                    object_density = config.get("object_density", None)
                    if object_mass is not None:
                        mass_props = sim_utils.MassPropertiesCfg(mass=object_mass)
                    elif object_density is not None:
                        mass_props = sim_utils.MassPropertiesCfg(density=object_density)
                    else:
                        mass_props = None
                    object_opacity = config.get("object_opacity", 1.0)
                    visual_material = (
                        sim_utils.PreviewSurfaceCfg(opacity=object_opacity)
                        if object_opacity < 1.0
                        else None
                    )
                    self.object = RigidObjectCfg(
                        prim_path="{ENV_REGEX_NS}/Object",
                        spawn=sim_utils.MultiUsdFileCfg(
                            usd_path=resolved_paths,
                            random_choice=False,
                            activate_contact_sensors=True,
                            scale=(scene_scale, scene_scale, scene_scale),
                            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                                kinematic_enabled=not object_is_dynamic,
                                max_depenetration_velocity=1.0,
                            ),
                            mass_props=mass_props,
                            collision_props=_object_collision_props({}),
                            visual_material=visual_material,
                        ),
                        init_state=RigidObjectCfg.InitialStateCfg(
                            pos=tuple(config.get("object_position", [2.0, 0.0, 0.0]))
                        ),
                    )

                # Stash USD stems temporarily — moved to ModularTrackingEnvCfg immediately
                # after scene creation (cannot remain here or IsaacLab treats it as asset)
                if full_scene_slot_paths_by_index is not None:
                    self._tmp_usd_stems = [
                        str(idx) for idx in range(len(full_scene_slot_paths_by_index))
                    ]
                    self._tmp_object_usd_slot_count = len(full_scene_slot_paths_by_index)
                else:
                    self._tmp_usd_stems = [
                        os.path.splitext(os.path.basename(p))[0] for p in resolved_paths
                    ]
                    self._tmp_object_usd_slot_count = 0
                self._tmp_object_usd_stems_by_motion = object_usd_stems_by_motion
                self._tmp_object_dynamics_by_stem = dict(object_dynamics_by_stem)
                self._tmp_object_physics_overrides_by_stem = dict(
                    object_physics_overrides_by_stem
                )

            static_context_usd_path = config.get("static_context_usd_path", "")
            if static_context_usd_path:
                static_context_paths = _resolve_paired_motion_usd_paths(
                    static_context_usd_path,
                    motion_file_path,
                    filter_motion_keys,
                    max_unique_motions,
                    interleave_by_object,
                    _mcfg,
                )
                if not static_context_paths:
                    raise ValueError(
                        f"static_context_usd_path produced no USDs: {static_context_usd_path}"
                    )
                if len(static_context_paths) > 1:
                    self.replicate_physics = False

                static_context_collision_enabled = bool(
                    config.get("static_context_collision_enabled", True)
                )
                static_context_contact_sensors = bool(
                    config.get("static_context_activate_contact_sensors", False)
                )
                static_context_kinematic = bool(
                    config.get("static_context_kinematic", True)
                )
                self.static_context = AssetBaseCfg(
                    prim_path="{ENV_REGEX_NS}/StaticContext",
                    spawn=sim_utils.MultiUsdFileCfg(
                        usd_path=static_context_paths,
                        random_choice=False,
                        activate_contact_sensors=static_context_contact_sensors,
                        scale=tuple(
                            config.get(
                                "static_context_scale",
                                (scene_scale, scene_scale, scene_scale),
                            )
                        ),
                        rigid_props=sim_utils.RigidBodyPropertiesCfg(
                            kinematic_enabled=static_context_kinematic,
                            max_depenetration_velocity=1.0,
                        ),
                        collision_props=sim_utils.CollisionPropertiesCfg(
                            collision_enabled=static_context_collision_enabled,
                        ),
                    ),
                    init_state=AssetBaseCfg.InitialStateCfg(
                        pos=tuple(config.get("static_context_position", [0.0, 0.0, 0.0]))
                    ),
                )
                print(  # noqa: T201
                    "[Static Context] Spawned "
                    f"{len(static_context_paths)} paired context USDs from {static_context_usd_path}; "
                    f"collision={static_context_collision_enabled}, "
                    f"contact_sensors={static_context_contact_sensors}, "
                    f"kinematic={static_context_kinematic}"
                )

            articulated_context_usd_path = config.get("articulated_context_usd_path", "")
            if articulated_context_usd_path:
                articulated_context_paths = _resolve_paired_motion_usd_paths(
                    articulated_context_usd_path,
                    motion_file_path,
                    filter_motion_keys,
                    max_unique_motions,
                    interleave_by_object,
                    _mcfg,
                )
                if not articulated_context_paths:
                    raise ValueError(
                        "articulated_context_usd_path produced no USDs: "
                        f"{articulated_context_usd_path}"
                    )
                if len(articulated_context_paths) > 1:
                    self.replicate_physics = False

                articulated_context_mode = str(
                    config.get("articulated_context_mode", "articulation")
                )
                articulated_context_collision_enabled = bool(
                    config.get("articulated_context_collision_enabled", True)
                )
                articulated_context_contact_sensors = bool(
                    config.get("articulated_context_activate_contact_sensors", False)
                )
                articulated_context_kinematic = bool(
                    config.get("articulated_context_kinematic", False)
                )
                articulated_context_spawn = sim_utils.MultiUsdFileCfg(
                    usd_path=articulated_context_paths,
                    random_choice=False,
                    activate_contact_sensors=articulated_context_contact_sensors,
                    scale=tuple(
                        config.get(
                            "articulated_context_scale",
                            (scene_scale, scene_scale, scene_scale),
                        )
                    ),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(
                        kinematic_enabled=articulated_context_kinematic,
                        max_depenetration_velocity=float(
                            config.get(
                                "articulated_context_max_depenetration_velocity", 1.0
                            )
                        ),
                    ),
                    collision_props=sim_utils.CollisionPropertiesCfg(
                        collision_enabled=articulated_context_collision_enabled,
                    ),
                )
                articulated_context_pos = tuple(
                    config.get("articulated_context_position", [0.0, 0.0, 0.0])
                )
                if articulated_context_mode == "articulation":
                    articulated_context_spawn.articulation_props = (
                        sim_utils.ArticulationRootPropertiesCfg(
                            enabled_self_collisions=bool(
                                config.get(
                                    "articulated_context_enabled_self_collisions",
                                    False,
                                )
                            ),
                            solver_position_iteration_count=int(
                                config.get(
                                    "articulated_context_solver_position_iterations",
                                    8,
                                )
                            ),
                            solver_velocity_iteration_count=int(
                                config.get(
                                    "articulated_context_solver_velocity_iterations",
                                    4,
                                )
                            ),
                        )
                    )
                    self.articulated_context = ArticulationCfg(
                        prim_path="{ENV_REGEX_NS}/ArticulatedContext",
                        spawn=articulated_context_spawn,
                        articulation_root_prim_path=config.get(
                            "articulated_context_root_prim_path", None
                        ),
                        init_state=ArticulationCfg.InitialStateCfg(
                            pos=articulated_context_pos,
                            joint_pos=config.get(
                                "articulated_context_initial_joint_pos", {".*": 0.0}
                            ),
                            joint_vel=config.get(
                                "articulated_context_initial_joint_vel", {".*": 0.0}
                            ),
                        ),
                        soft_joint_pos_limit_factor=float(
                            config.get(
                                "articulated_context_soft_joint_pos_limit_factor", 1.0
                            )
                        ),
                        actuators={},
                    )
                elif articulated_context_mode == "asset":
                    self.articulated_context = AssetBaseCfg(
                        prim_path="{ENV_REGEX_NS}/ArticulatedContext",
                        spawn=articulated_context_spawn,
                        init_state=AssetBaseCfg.InitialStateCfg(
                            pos=articulated_context_pos
                        ),
                    )
                else:
                    raise ValueError(
                        "articulated_context_mode must be 'articulation' or 'asset', "
                        f"got {articulated_context_mode!r}"
                    )
                self._tmp_articulated_context_usd_paths = list(articulated_context_paths)
                print(  # noqa: T201
                    "[Articulated Context] Spawned "
                    f"{len(articulated_context_paths)} paired USDs from "
                    f"{articulated_context_usd_path}; "
                    f"mode={articulated_context_mode}, "
                    f"collision={articulated_context_collision_enabled}, "
                    f"contact_sensors={articulated_context_contact_sensors}, "
                    f"kinematic={articulated_context_kinematic}"
                )

            if robot_has_hands:
                hand_frame_links = list(
                    config.get(
                        "object_to_hand_frame_links",
                        hand_sensor_defaults["object_to_hand_frame_links"],
                    )
                )
                if not hand_frame_links:
                    raise ValueError(
                        f"Robot type {robot_type} enables hand sensors but has no "
                        "object_to_hand_frame_links configured."
                    )
                # Frame transformer for hand-object tracking
                self.object_to_hand_frame_transformer = FrameTransformerCfg(
                    prim_path=object_sensor_prim_path,  # Source: active/primary object
                    target_frames=[
                        FrameTransformerCfg.FrameCfg(
                            name=_frame_name_from_link(link_name),
                            prim_path=f"{{ENV_REGEX_NS}}/Robot/{link_name}",
                        )
                        for link_name in hand_frame_links
                    ],
                )

                # Object-to-hand contact sensor for grasp detection (only for robots with hands)
                # force_matrix_w gives shape [num_envs, 1, N, 3] - force on Object from each finger link
                # Configurable via contact_sensor_finger_links (list of link names without Robot/ prefix)
                # Default: tips + palm (4 links). For fuller grasp detection, use all finger segments:
                #   [right_hand_palm_link, right_hand_thumb_0_link, right_hand_thumb_1_link,
                #    right_hand_thumb_2_link, right_hand_index_0_link, right_hand_index_1_link,
                #    right_hand_middle_0_link, right_hand_middle_1_link]
                custom_finger_links = list(
                    config.get(
                        "contact_sensor_finger_links",
                        hand_sensor_defaults["contact_sensor_finger_links"],
                    )
                )
                finger_tip_bodies = _robot_body_paths(custom_finger_links)
                self.object_to_hand_contact_sensor = ContactSensorCfg(
                    prim_path=object_sensor_prim_path,
                    filter_prim_paths_expr=finger_tip_bodies,
                    history_length=2,
                    track_air_time=False,
                )

                left_finger_tip_links = list(
                    config.get(
                        "left_contact_sensor_finger_links",
                        hand_sensor_defaults["left_contact_sensor_finger_links"],
                    )
                )
                if left_finger_tip_links:
                    self.object_to_left_hand_contact_sensor = ContactSensorCfg(
                        prim_path=object_sensor_prim_path,
                        filter_prim_paths_expr=_robot_body_paths(left_finger_tip_links),
                        history_length=2,
                        track_air_time=False,
                    )

                if (
                    multi_object_per_env
                    and config.get("enable_multi_object_contact_sensors", False)
                    and multi_object_sensor_prim_paths
                ):
                    contact_sensor_map = {}
                    support_contact_links = list(
                        config.get("support_contact_sensor_body_links", [])
                    )
                    support_contact_bodies = _robot_body_paths(support_contact_links)
                    enable_support_contact = (
                        config.get("enable_multi_object_support_contact_sensors", False)
                        and len(support_contact_bodies) > 0
                    )
                    for obj_idx, obj_name_safe, obj_prim_path in multi_object_sensor_prim_paths:
                        right_sensor_name = f"rp2_contact_right_{obj_idx:03d}_{obj_name_safe}"
                        setattr(
                            self,
                            right_sensor_name,
                            ContactSensorCfg(
                                prim_path=obj_prim_path,
                                filter_prim_paths_expr=finger_tip_bodies,
                                history_length=2,
                                track_air_time=False,
                            ),
                        )
                        entry = {
                            "object_index": int(obj_idx),
                            "object_stem": obj_name_safe,
                            "prim_path": obj_prim_path,
                            "right_hand_sensor": right_sensor_name,
                            "left_hand_sensor": None,
                            "support_body_sensor": None,
                        }
                        if left_finger_tip_links:
                            left_sensor_name = f"rp2_contact_left_{obj_idx:03d}_{obj_name_safe}"
                            setattr(
                                self,
                                left_sensor_name,
                                ContactSensorCfg(
                                    prim_path=obj_prim_path,
                                    filter_prim_paths_expr=_robot_body_paths(left_finger_tip_links),
                                    history_length=2,
                                    track_air_time=False,
                                ),
                            )
                            entry["left_hand_sensor"] = left_sensor_name
                        if enable_support_contact:
                            support_sensor_name = f"rp2_contact_support_{obj_idx:03d}_{obj_name_safe}"
                            setattr(
                                self,
                                support_sensor_name,
                                ContactSensorCfg(
                                    prim_path=obj_prim_path,
                                    filter_prim_paths_expr=support_contact_bodies,
                                    history_length=2,
                                    track_air_time=False,
                                ),
                            )
                            entry["support_body_sensor"] = support_sensor_name
                        contact_sensor_map[obj_name_safe] = entry
                    self._tmp_multi_object_contact_sensor_map = contact_sensor_map
                    print(  # noqa: T201
                        "[Multi-Object Contact] Added "
                        f"{len(contact_sensor_map)} explicit object contact sensor pairs"
                        + (" plus support-body sensors" if enable_support_contact else "")
                    )

        # Opt-in active articulated object. This is intentionally separate from
        # the legacy rigid-object branch above so existing experiments retain
        # exactly their original scene composition and sensor paths.
        active_articulation_usd_path = config.get("active_articulation_usd_path", "")
        if joint_roster_view is not None and active_articulation_usd_path:
            raise ValueError(
                "Joint-roster mode resolves active articulation USDs from the "
                "manifest; active_articulation_usd_path must be empty."
            )
        if active_articulation_usd_path or joint_roster_view is not None:
            if config.get("add_object", False):
                raise ValueError(
                    "active_articulation_usd_path requires manager_env.config.add_object=false; "
                    "otherwise both the legacy rigid object and articulation would be spawned."
                )

            if joint_roster_view is not None:
                active_articulation_paths = [
                    str(slot["active_articulation_usd"])
                    for slot in joint_roster_view["slots"]
                ]
            else:
                active_articulation_paths = _resolve_paired_motion_usd_paths(
                    active_articulation_usd_path,
                    motion_file_path,
                    filter_motion_keys,
                    max_unique_motions,
                    interleave_by_object,
                    _mcfg,
                )
            if not active_articulation_paths:
                raise ValueError(
                    "active_articulation_usd_path produced no USDs: "
                    f"{active_articulation_usd_path}"
                )
            if len(active_articulation_paths) > 1:
                self.replicate_physics = False

            active_articulation_spawn = sim_utils.MultiUsdFileCfg(
                usd_path=active_articulation_paths,
                random_choice=False,
                activate_contact_sensors=bool(
                    config.get("active_articulation_activate_contact_sensors", True)
                ),
                scale=tuple(
                    config.get(
                        "active_articulation_scale",
                        (scene_scale, scene_scale, scene_scale),
                    )
                ),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    kinematic_enabled=False,
                    max_depenetration_velocity=float(
                        config.get("active_articulation_max_depenetration_velocity", 1.0)
                    ),
                ),
                collision_props=sim_utils.CollisionPropertiesCfg(
                    collision_enabled=bool(
                        config.get("active_articulation_collision_enabled", True)
                    ),
                ),
            )
            if joint_roster_view is not None:
                # PhysX tensor views require one articulation topology per
                # rank. Rigid manipulands remain floating; revolute and
                # prismatic mechanisms keep their authored support fixed.
                fix_root_link = (
                    joint_roster_view["rank_entity_kind"] != "rigid"
                )
            else:
                fix_root_link = config.get(
                    "active_articulation_fix_root_link", True
                )
            if fix_root_link is not None:
                fix_root_link = bool(fix_root_link)
            active_articulation_spawn.articulation_props = (
                sim_utils.ArticulationRootPropertiesCfg(
                    fix_root_link=fix_root_link,
                    enabled_self_collisions=bool(
                        config.get("active_articulation_enabled_self_collisions", False)
                    ),
                    solver_position_iteration_count=int(
                        config.get("active_articulation_solver_position_iterations", 8)
                    ),
                    solver_velocity_iteration_count=int(
                        config.get("active_articulation_solver_velocity_iterations", 4)
                    ),
                )
            )
            passive_joint_static_friction = float(
                config.get("active_articulation_passive_joint_static_friction", 0.0)
            )
            passive_joint_dynamic_friction = float(
                config.get("active_articulation_passive_joint_dynamic_friction", 0.0)
            )
            passive_joint_viscous_friction = float(
                config.get("active_articulation_passive_joint_viscous_friction", 0.0)
            )
            passive_joint_damping = float(
                config.get("active_articulation_passive_joint_damping", 0.0)
            )
            passive_joint_values = {
                "static_friction": passive_joint_static_friction,
                "dynamic_friction": passive_joint_dynamic_friction,
                "viscous_friction": passive_joint_viscous_friction,
                "damping": passive_joint_damping,
            }
            invalid_passive_values = {
                name: value for name, value in passive_joint_values.items() if value < 0.0
            }
            if invalid_passive_values:
                raise ValueError(
                    "Active-articulation passive joint parameters must be non-negative: "
                    f"{invalid_passive_values}"
                )
            if joint_roster_view is not None and any(
                value > 0.0 for value in passive_joint_values.values()
            ):
                raise ValueError(
                    "Joint-roster mode resolves passive mechanism resistance "
                    "per asset signature; global active-articulation passive "
                    "joint parameters must remain zero."
                )
            active_articulation_actuators = {}
            if any(value > 0.0 for value in passive_joint_values.values()):
                # A zero-stiffness implicit actuator authors only physical joint
                # resistance. It does not track a position target or replay the
                # reference mechanism trajectory.
                active_articulation_actuators["passive_mechanism"] = (
                    ImplicitActuatorCfg(
                        joint_names_expr=[".*"],
                        stiffness=0.0,
                        damping=passive_joint_damping,
                        friction=passive_joint_static_friction,
                        dynamic_friction=passive_joint_dynamic_friction,
                        viscous_friction=passive_joint_viscous_friction,
                    )
                )

            self.active_articulation = ArticulationCfg(
                prim_path="{ENV_REGEX_NS}/ActiveArticulation",
                spawn=active_articulation_spawn,
                articulation_root_prim_path=config.get(
                    "active_articulation_root_prim_path", None
                ),
                init_state=ArticulationCfg.InitialStateCfg(
                    pos=tuple(config.get("active_articulation_position", [0.0, 0.0, 0.0])),
                    rot=tuple(config.get("active_articulation_rotation", [1.0, 0.0, 0.0, 0.0])),
                    joint_pos=config.get(
                        "active_articulation_initial_joint_pos", {".*": 0.0}
                    ),
                    joint_vel=config.get(
                        "active_articulation_initial_joint_vel", {".*": 0.0}
                    ),
                ),
                soft_joint_pos_limit_factor=float(
                    config.get("active_articulation_soft_joint_pos_limit_factor", 1.0)
                ),
                actuators=active_articulation_actuators,
            )
            self._tmp_active_articulation_usd_paths = list(active_articulation_paths)
            if joint_roster_view is not None:
                self._tmp_active_articulation_usd_stems = [
                    str(slot["representative_alias"])
                    for slot in joint_roster_view["slots"]
                ]
                # Optional per-slot physics overrides for the joint-roster
                # path. The same manager_env.config.object_physics_overrides_path
                # JSON (default/objects/rules) is matched on each slot's
                # object_key (falling back to its representative alias). The
                # values are applied to the live PhysX view after sim start by
                # the motion command (see _apply_joint_roster_physics_overrides);
                # without an overrides path every slot resolves to {} and the
                # scene is untouched.
                roster_overrides = _load_object_physics_overrides(
                    config.get("object_physics_overrides_path", None)
                )
                self._tmp_joint_roster_physics_overrides = [
                    _object_physics_override_for_stem(
                        str(slot.get("object_key") or slot["representative_alias"]),
                        {},
                        roster_overrides,
                    )
                    for slot in joint_roster_view["slots"]
                ]
            else:
                self._tmp_active_articulation_usd_stems = [
                    os.path.splitext(os.path.basename(path))[0]
                    for path in active_articulation_paths
                ]

            # Inactive scene context remains one passive, collision-only asset.
            # It is never exposed through object observations or rewards.
            static_context_usd_path = config.get("static_context_usd_path", "")
            if joint_roster_view is not None and static_context_usd_path:
                raise ValueError(
                    "Joint-roster mode resolves static context USDs from the "
                    "manifest; static_context_usd_path must be empty."
                )
            static_context_is_configured = _static_context_is_configured(
                static_context_usd_path,
                joint_roster_view,
            )
            if static_context_is_configured:
                if joint_roster_view is not None:
                    static_context_paths = [
                        str(slot["static_context_usd"])
                        for slot in joint_roster_view["slots"]
                    ]
                else:
                    static_context_paths = _resolve_paired_motion_usd_paths(
                        static_context_usd_path,
                        motion_file_path,
                        filter_motion_keys,
                        max_unique_motions,
                        interleave_by_object,
                        _mcfg,
                    )
                if not static_context_paths:
                    raise ValueError(
                        f"static_context_usd_path produced no USDs: {static_context_usd_path}"
                    )
                if len(static_context_paths) > 1:
                    self.replicate_physics = False
                static_context_collision_enabled = bool(
                    config.get("static_context_collision_enabled", True)
                )
                static_context_contact_sensors = bool(
                    config.get("static_context_activate_contact_sensors", False)
                )
                static_context_kinematic = bool(
                    config.get("static_context_kinematic", True)
                )
                self.static_context = AssetBaseCfg(
                    prim_path="{ENV_REGEX_NS}/StaticContext",
                    spawn=sim_utils.MultiUsdFileCfg(
                        usd_path=static_context_paths,
                        random_choice=False,
                        activate_contact_sensors=static_context_contact_sensors,
                        scale=tuple(
                            config.get(
                                "static_context_scale",
                                (scene_scale, scene_scale, scene_scale),
                            )
                        ),
                        rigid_props=sim_utils.RigidBodyPropertiesCfg(
                            kinematic_enabled=static_context_kinematic,
                            max_depenetration_velocity=1.0,
                        ),
                        collision_props=sim_utils.CollisionPropertiesCfg(
                            collision_enabled=static_context_collision_enabled,
                        ),
                    ),
                    init_state=AssetBaseCfg.InitialStateCfg(
                        pos=tuple(config.get("static_context_position", [0.0, 0.0, 0.0]))
                    ),
                )
                if (
                    static_context_contact_sensors
                    and robot_has_hands
                    and bool(
                        config.get(
                            "static_context_enable_legacy_hand_contact_sensor",
                            True,
                        )
                    )
                ):
                    static_context_hand_bodies = _robot_body_paths(
                        list(
                            config.get(
                                "static_context_to_hand_contact_bodies",
                                hand_sensor_defaults["table_to_hand_contact_bodies"],
                            )
                        )
                    )
                    self.static_context_to_hand_contact_sensor = ContactSensorCfg(
                        prim_path="{ENV_REGEX_NS}/StaticContext/.*",
                        filter_prim_paths_expr=static_context_hand_bodies,
                        history_length=2,
                        track_air_time=False,
                    )

            if robot_has_hands:
                tracked_prim_path = config.get(
                    "active_articulation_tracked_prim_path",
                    "{ENV_REGEX_NS}/ActiveArticulation/panel",
                )
                contact_prim_path = (
                    config.get("active_articulation_contact_prim_path")
                    or tracked_prim_path
                )
                hand_frame_links = list(
                    config.get(
                        "object_to_hand_frame_links",
                        hand_sensor_defaults["object_to_hand_frame_links"],
                    )
                )
                if not hand_frame_links:
                    raise ValueError(
                        f"Robot type {robot_type} enables hand sensors but has no "
                        "object_to_hand_frame_links configured."
                    )
                self.object_to_hand_frame_transformer = FrameTransformerCfg(
                    prim_path=tracked_prim_path,
                    target_frames=[
                        FrameTransformerCfg.FrameCfg(
                            name=_frame_name_from_link(link_name),
                            prim_path=f"{{ENV_REGEX_NS}}/Robot/{link_name}",
                        )
                        for link_name in hand_frame_links
                    ],
                )

                right_finger_links = list(
                    config.get(
                        "contact_sensor_finger_links",
                        hand_sensor_defaults["contact_sensor_finger_links"],
                    )
                )
                self.object_to_hand_contact_sensor = ContactSensorCfg(
                    prim_path=contact_prim_path,
                    filter_prim_paths_expr=_robot_body_paths(right_finger_links),
                    history_length=2,
                    track_air_time=False,
                )
                left_finger_links = list(
                    config.get(
                        "left_contact_sensor_finger_links",
                        hand_sensor_defaults["left_contact_sensor_finger_links"],
                    )
                )
                if left_finger_links:
                    self.object_to_left_hand_contact_sensor = ContactSensorCfg(
                        prim_path=contact_prim_path,
                        filter_prim_paths_expr=_robot_body_paths(left_finger_links),
                        history_length=2,
                        track_air_time=False,
                    )
                if bool(
                    config.get(
                        "active_articulation_enable_broad_hand_contact_sensor", False
                    )
                ):
                    right_broad_links = list(
                        hand_sensor_defaults["object_to_robot_contact_bodies"]
                    )
                    left_broad_links = [
                        name.replace("right_", "left_", 1)
                        for name in right_broad_links
                        if name.startswith("right_")
                    ]
                    broad_links = list(
                        config.get(
                            "active_articulation_broad_hand_contact_bodies",
                            right_broad_links + left_broad_links,
                        )
                    )
                    self.active_articulation_to_broad_hand_contact_sensor = (
                        ContactSensorCfg(
                            prim_path=contact_prim_path,
                            filter_prim_paths_expr=_robot_body_paths(broad_links),
                            history_length=2,
                            track_air_time=False,
                        )
                    )
                if bool(
                    config.get(
                        "active_articulation_enable_support_body_contact_sensor",
                        False,
                    )
                ):
                    support_body_links = list(
                        config.get(
                            "active_articulation_support_body_contact_bodies",
                            [],
                        )
                    )
                    if not support_body_links:
                        raise ValueError(
                            "active articulation support contact sensing requires "
                            "active_articulation_support_body_contact_bodies"
                        )
                    self.active_articulation_to_support_body_contact_sensor = (
                        ContactSensorCfg(
                            prim_path=contact_prim_path,
                            filter_prim_paths_expr=_robot_body_paths(
                                support_body_links
                            ),
                            history_length=2,
                            track_air_time=False,
                        )
                    )

            if bool(
                config.get("enable_canonical_full_body_contact_sensors", False)
            ):
                canonical_robot_links = list(
                    dict.fromkeys(
                        link
                        for group_name in CONTACT_BODY_GROUPS
                        for link in G1_CONTACT_GROUP_LINKS[group_name]
                    )
                )
                canonical_robot_paths = _robot_body_paths(
                    canonical_robot_links
                )
                for body_name in ("frame", "panel"):
                    setattr(
                        self,
                        (
                            "canonical_active_entity_contact_sensor__"
                            f"{body_name}"
                        ),
                        ContactSensorCfg(
                            prim_path=(
                                "{ENV_REGEX_NS}/ActiveArticulation/"
                                f"{body_name}"
                            ),
                            filter_prim_paths_expr=canonical_robot_paths,
                            history_length=2,
                            track_air_time=False,
                        ),
                    )
                # Collision-only static-context USDs have no rigid body to host
                # a sensor; such configs must opt out and report it explicitly.
                if static_context_is_configured and bool(
                    config.get("canonical_static_context_contact_sensor_enabled", True)
                ):
                    self.canonical_static_context_contact_sensor = ContactSensorCfg(
                        prim_path="{ENV_REGEX_NS}/StaticContext",
                        filter_prim_paths_expr=canonical_robot_paths,
                        history_length=2,
                        track_air_time=False,
                    )
                for link_name in canonical_robot_links:
                    setattr(
                        self,
                        f"canonical_ground_contact_sensor__{link_name}",
                        ContactSensorCfg(
                            prim_path=(
                                "{ENV_REGEX_NS}/Robot/" + link_name
                            ),
                            # The terrain root is an Xform; PhysX filters must
                            # name the collider beneath it to report forces.
                            filter_prim_paths_expr=[
                                config.get(
                                    "canonical_ground_contact_filter_prim_path",
                                    "/World/ground",
                                )
                            ],
                            history_length=2,
                            track_air_time=False,
                        ),
                    )

            print(  # noqa: T201
                "[Active Articulation] Spawned "
                f"{len(active_articulation_paths)} paired USDs; "
                f"tracked_body={config.get('active_articulation_tracked_body_name', 'panel')}; "
                f"contact_prim={config.get('active_articulation_contact_prim_path') or 'tracked'}; "
                f"static_context={static_context_is_configured}; "
                "canonical_full_body_contact="
                f"{bool(config.get('enable_canonical_full_body_contact_sensors', False))}; "
                "canonical_contact_topology="
                "entity_side_frame_panel_static_plus_per_link_ground"
            )

        if config.get("enable_depth_camera", False):
            import math

            from gear_sonic.sensors import DEPTH_NOISE_REGISTRY, DepthCameraCfg
            from isaaclab.sensors.ray_caster import MultiMeshRayCasterCfg
            from isaaclab.sensors.ray_caster.patterns import PinholeCameraPatternCfg

            depth_cfg = config.get("depth_camera", {})
            depth_attached_link = depth_cfg.get("attached_link", "torso_link")
            depth_pos_offset = tuple(
                depth_cfg.get("pos_offset", [0.0487988662332928, 0.01, 0.4378029937970051])
            )
            depth_rot_offset = tuple(
                depth_cfg.get(
                    "rot_offset",
                    [0.9135367613482678, 0.004363309284746571, 0.4067366430758002, 0.0],
                )
            )
            depth_resolution = depth_cfg.get("resolution", [36, 64])
            depth_clipping = depth_cfg.get("clipping_range", [0.1, 2.5])
            depth_fovx = depth_cfg.get("fov_x", 89.51)
            depth_fovy = depth_cfg.get("fov_y", 58.29)
            depth_history_length = depth_cfg.get("history_length", 37)
            depth_fixed_pitch_roll = depth_cfg.get("fixed_pitch_roll", False)
            depth_crop_region = tuple(depth_cfg.get("crop_region", [18, 0, 16, 16]))
            depth_resize_shape = depth_cfg.get("resize_shape", None)
            if depth_resize_shape is not None:
                depth_resize_shape = tuple(depth_resize_shape)

            RTC = MultiMeshRayCasterCfg.RaycastTargetCfg
            object_mesh_prim = depth_cfg.get(
                "object_mesh_prim", "/World/envs/env_.*/Object/Geometry/Object/terrain"
            )
            mesh_targets = [
                RTC(prim_expr="/World/ground", is_shared=True, track_mesh_transforms=False),
            ]
            if object_mesh_prim:
                mesh_targets.append(RTC(prim_expr=object_mesh_prim, track_mesh_transforms=True))
            g1_links = [
                "left_shoulder_pitch_link",
                "left_shoulder_roll_link",
                "left_shoulder_yaw_link",
                "left_elbow_link",
                "left_wrist_roll_link",
                "left_wrist_pitch_link",
                "left_wrist_yaw_link",
                "right_shoulder_pitch_link",
                "right_shoulder_roll_link",
                "right_shoulder_yaw_link",
                "right_elbow_link",
                "right_wrist_roll_link",
                "right_wrist_pitch_link",
                "right_wrist_yaw_link",
                "left_hip_pitch_link",
                "left_hip_roll_link",
                "left_hip_yaw_link",
                "left_knee_link",
                "left_ankle_pitch_link",
                "left_ankle_roll_link",
                "right_hip_pitch_link",
                "right_hip_roll_link",
                "right_hip_yaw_link",
                "right_knee_link",
                "right_ankle_pitch_link",
                "right_ankle_roll_link",
            ]
            for link in g1_links:
                mesh_targets.append(
                    RTC(
                        prim_expr=f"/World/envs/env_.*/Robot/{link}/visuals",
                        track_mesh_transforms=True,
                    )
                )

            _default_noise = [
                "contour",
                "stereo",
                "artifact",
                "gaussian_blur",
                "random_gaussian",
                "latency",
            ]
            noise_pipeline_cfg = depth_cfg.get("noise_pipeline", _default_noise)
            noise_pipeline = []
            for entry in noise_pipeline_cfg:
                if isinstance(entry, str):
                    if entry not in DEPTH_NOISE_REGISTRY:
                        raise ValueError(
                            f"Unknown depth noise type '{entry}'. "
                            f"Available: {list(DEPTH_NOISE_REGISTRY.keys())}"
                        )
                    noise_pipeline.append(DEPTH_NOISE_REGISTRY[entry]())
                elif isinstance(entry, dict):
                    noise_type = entry.pop("type", None)
                    if noise_type is None or noise_type not in DEPTH_NOISE_REGISTRY:
                        raise ValueError(
                            f"Noise dict must have 'type' key from {list(DEPTH_NOISE_REGISTRY.keys())}"
                        )
                    noise_pipeline.append(DEPTH_NOISE_REGISTRY[noise_type](**entry))

            self.depth_camera = DepthCameraCfg(
                prim_path=f"{{ENV_REGEX_NS}}/Robot/{depth_attached_link}",
                mesh_prim_paths=mesh_targets,
                offset=DepthCameraCfg.OffsetCfg(
                    pos=depth_pos_offset, rot=depth_rot_offset, convention="world"
                ),
                data_types=["distance_to_image_plane"],
                depth_clipping_behavior="max",
                pattern_cfg=PinholeCameraPatternCfg(
                    focal_length=1.0,
                    horizontal_aperture=2 * math.tan(math.radians(depth_fovx) / 2),
                    vertical_aperture=2 * math.tan(math.radians(depth_fovy) / 2),
                    width=depth_resolution[1],
                    height=depth_resolution[0],
                ),
                max_distance=depth_clipping[1],
                history_length=depth_history_length,
                fixed_pitch_roll=depth_fixed_pitch_roll,
                crop_region=depth_crop_region,
                resize_shape=depth_resize_shape,
                noise_pipeline=noise_pipeline,
                update_period=0.02,
                debug_vis=False,
            )


        if config.get("add_table", False):
            # Table initial position and orientation from config or meta
            # (per-env update in commands.py for multi-motion)
            if "table_position" in config:
                table_pos = config["table_position"]
            elif motion_meta_info is not None and "table_pos" in motion_meta_info:
                table_pos = motion_meta_info["table_pos"]
            else:
                table_pos = [0.0, 0.0, 0.8]

            # Apply table_offset if configured (additive offset to table position)
            table_offset = config.get("table_offset", None)
            if table_offset is not None:
                table_pos[0] += table_offset[0]
                table_pos[1] += table_offset[1]
                table_pos[2] += table_offset[2]

            # Table quaternion (w, x, y, z) - GRAB data uses 90° X-rotation
            if "table_quat" in config:
                table_quat = tuple(config["table_quat"])
            elif motion_meta_info is not None and "table_quat" in motion_meta_info:
                table_quat = tuple(motion_meta_info["table_quat"])
            else:
                # Default for GRAB table USD: 90° rotation around X-axis
                table_quat = (0.7071068, -0.7071068, 0.0, 0.0)

            # Use table USD when: scene_scale is explicitly set OR table_usd_path is provided OR meta info exists
            use_table_usd = config.get("table_usd_path") is not None

            if not use_table_usd:
                # Use simple cuboid table for GeniHOI or default scenarios
                # Table size can be configured via config or meta file
                # For GeniHOI: table_size = [width, depth, thickness] from meta
                # Default: 1.0 x 0.6 x 0.04 (reasonable for kitchen/desk scenarios)

                if "table_size" in config:
                    table_size = config["table_size"]
                    table_width = table_size[0]
                    table_depth = table_size[1]
                    table_thickness = table_size[2]
                elif motion_meta_info is not None and "table_size" in motion_meta_info:
                    table_size = motion_meta_info["table_size"]
                    table_width = table_size[0]
                    table_depth = table_size[1]
                    table_thickness = table_size[2]
                else:
                    # Default cuboid size - large enough for most GeniHOI scenarios
                    table_width = 1.0
                    table_depth = 0.6
                    table_thickness = 0.04

                # For cuboid tables, use identity quaternion (no rotation needed)
                table_quat_cuboid = (1.0, 0.0, 0.0, 0.0)

                self.table = RigidObjectCfg(
                    prim_path="{ENV_REGEX_NS}/Table",
                    spawn=sim_utils.CuboidCfg(
                        activate_contact_sensors=True,
                        size=(table_width, table_depth, table_thickness),
                        rigid_props=sim_utils.RigidBodyPropertiesCfg(
                            kinematic_enabled=True,
                            max_depenetration_velocity=1.0,
                        ),
                        mass_props=sim_utils.MassPropertiesCfg(density=500.0),
                        collision_props=sim_utils.CollisionPropertiesCfg(
                            collision_enabled=True,
                        ),
                        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.5, 0.35, 0.2)),
                    ),
                    init_state=RigidObjectCfg.InitialStateCfg(
                        pos=tuple(table_pos), rot=table_quat_cuboid
                    ),
                )
            else:
                # Use table USD with scene_scale and proper rotation
                self.table = RigidObjectCfg(
                    prim_path="{ENV_REGEX_NS}/Table",
                    spawn=sim_utils.UsdFileCfg(
                        activate_contact_sensors=True,
                        usd_path=config.get(
                            "table_usd_path", "data/motion_lib_grab/common/table.usda"
                        ),
                        rigid_props=sim_utils.RigidBodyPropertiesCfg(
                            kinematic_enabled=True,
                            max_depenetration_velocity=1.0,
                        ),
                        collision_props=sim_utils.CollisionPropertiesCfg(
                            collision_enabled=True,
                        ),
                        scale=(scene_scale, scene_scale, scene_scale),
                    ),
                    init_state=RigidObjectCfg.InitialStateCfg(pos=tuple(table_pos), rot=table_quat),
                )

            # Table-to-hand contact sensor for walk_stand_grasp rewards (only for robots with hands)
            if robot_has_hands:
                # Use Table as prim_path and explicitly list hand bodies in filter_prim_paths_expr.
                table_contact_bodies = _robot_body_paths(
                    list(
                        config.get(
                            "table_to_hand_contact_bodies",
                            hand_sensor_defaults["table_to_hand_contact_bodies"],
                        )
                    )
                )
                self.table_to_hand_contact_sensor = ContactSensorCfg(
                    prim_path="{ENV_REGEX_NS}/Table",
                    filter_prim_paths_expr=table_contact_bodies,
                    history_length=2,
                    track_air_time=False,
                )

            # Object-to-table contact sensor for detecting object-table contact forces
            # Sensor is attached to Object, filters for contact with Table
            # force_matrix_w gives shape [num_envs, 1, 1, 3] - force on Object from Table
            if config.get("add_object", False):
                self.object_to_table_contact_sensor = ContactSensorCfg(
                    prim_path=object_sensor_prim_path,
                    filter_prim_paths_expr=["{ENV_REGEX_NS}/Table"],
                    history_length=2,
                    track_air_time=False,
                )

            # object_to_robot and table_to_robot contact sensors are only used
            # by the robot_table_contact_before_object termination, which is
            # redundant when hand_table_contact_termination is active.
            # Disabled by default to improve throughput (~10% on YAM).
            # To re-enable, set enable_robot_table_contact_sensors: true.
            if config.get("enable_robot_table_contact_sensors", False):
                object_to_robot_contact_bodies = _robot_body_paths(
                    list(
                        config.get(
                            "object_to_robot_contact_bodies",
                            hand_sensor_defaults["object_to_robot_contact_bodies"],
                        )
                    )
                )
                table_to_robot_contact_bodies = _robot_body_paths(
                    list(
                        config.get(
                            "table_to_robot_contact_bodies",
                            hand_sensor_defaults["table_to_robot_contact_bodies"],
                        )
                    )
                )
                if config.get("add_object", False):
                    self.object_to_robot_contact_sensor = ContactSensorCfg(
                        prim_path=object_sensor_prim_path,
                        filter_prim_paths_expr=object_to_robot_contact_bodies,
                        history_length=2,
                        track_air_time=False,
                    )

                if config.get("add_table", False):
                    self.table_to_robot_contact_sensor = ContactSensorCfg(
                        prim_path="{ENV_REGEX_NS}/Table",
                        filter_prim_paths_expr=table_to_robot_contact_bodies,
                        history_length=2,
                        track_air_time=False,
                    )

        # # TODO: Do this better.  # noqa: TD002, TD003
        # self.ego_camera = TiledCameraCfg(
        #     prim_path="{ENV_REGEX_NS}/Robot/head_link/HeadCamera",
        #     # parent=SceneEntityCfg(
        #     #     name="robot",
        #     #     body_names=["head_link"],
        #     # ),
        #     # offset=TiledCameraCfg.OffsetCfg(pos=camera_pos_offset, rot=camera_rot_offset, convention="world"),
        #     data_types=['rgb'],
        #     spawn=sim_utils.PinholeCameraCfg(focal_length=5.0, focus_distance=50.0, horizontal_aperture=5, clipping_range=(0.1, 20.0)),  # noqa: E501
        #     width=384,
        #     height=384,
        #     debug_vis=True,
        # )

        # TODO: Do this better.  # noqa: TD002, TD003
        # Copied from gear_sonic/config/simulator/isaacsim.yaml
        # enable_cameras flag creates the ego camera for vision-based policies
        if config.get("enable_cameras", False) or config.get("render_ego", False):
            # Get camera config from nested cameras dict
            cameras_cfg = config.get("cameras", {})

            # Choose camera type: fisheye, pinhole, or opencv (with lens distortion)
            if config.get("render_ego_opencv_usd", False):
                # Load OpenCV distortion parameters from USD file
                opencv_usd_path = cameras_cfg.get(
                    "opencv_usd_path", "runs/oak_camera.usda"  # Default path
                )
                print(  # noqa: T201
                    f"USING OPENCV CAMERA - loading distortion from: {opencv_usd_path}"
                )  # noqa: T201

                # Parse the USD file to extract distortion parameters
                opencv_params = _load_opencv_params_from_usd(opencv_usd_path)

                # Camera resolution [H, W] - required, no fallback
                camera_resolution = cameras_cfg["camera_resolution"]

                # Use custom OpenCV spawner that applies distortion at spawn time (before cloning)
                print(  # noqa: T201
                    f"[DEBUG] Creating OpenCVCameraCfg with resolution={camera_resolution} -> image_size={(camera_resolution[1], camera_resolution[0])}"  # noqa: E501
                )
                print(  # noqa: T201
                    f"[DEBUG] fx={opencv_params['fx']}, fy={opencv_params['fy']}, cx={opencv_params['cx']}, cy={opencv_params['cy']}"  # noqa: E501
                )
                print(f"[DEBUG] k1={opencv_params['k1']}")  # noqa: T201
                camera_spawn_cfg = OpenCVCameraCfg(
                    clipping_range=(0.01, 20.0),
                    opencv_fx=opencv_params["fx"],
                    opencv_fy=opencv_params["fy"],
                    opencv_cx=opencv_params["cx"],
                    opencv_cy=opencv_params["cy"],
                    opencv_k1=opencv_params["k1"],  # Required - validated by parser
                    opencv_k2=opencv_params.get("k2", 0.0),
                    opencv_k3=opencv_params.get("k3", 0.0),
                    opencv_k4=opencv_params.get("k4", 0.0),
                    opencv_k5=opencv_params.get("k5", 0.0),
                    opencv_k6=opencv_params.get("k6", 0.0),
                    opencv_p1=opencv_params.get("p1", 0.0),
                    opencv_p2=opencv_params.get("p2", 0.0),
                    opencv_s1=opencv_params.get("s1", 0.0),
                    opencv_s2=opencv_params.get("s2", 0.0),
                    opencv_s3=opencv_params.get("s3", 0.0),
                    opencv_s4=opencv_params.get("s4", 0.0),
                    opencv_image_size=(camera_resolution[1], camera_resolution[0]),  # (W, H)
                )
            elif config.get("render_ego_fisheye", False):
                print("USING FISHEYE CAMERA" * 1000)  # noqa: T201
                camera_spawn_cfg = sim_utils.FisheyeCameraCfg(
                    projection_type=config.get("render_ego_fisheye_projection", "fisheyeSpherical"),
                    fisheye_max_fov=config.get("render_ego_fisheye_fov", 180.0),
                    focus_distance=0.5,
                    clipping_range=(0.1, 20.0),
                )
            elif config.get("render_ego_fisheye_polynomial", False):
                camera_spawn_cfg = sim_utils.FisheyeCameraCfg(
                    projection_type="fisheyePolynomial",
                    fisheye_max_fov=360.0,
                    focus_distance=0.5,
                    clipping_range=(0.1, 20.0),
                )
            else:
                # Convert clipping_range to tuple if it's a list or string (from YAML config)
                clipping_range = cameras_cfg.get("camera_clipping_range", (0.1, 20.0))
                if isinstance(clipping_range, str):
                    # Parse string like "(0.01,20.0)" to tuple
                    clipping_range = tuple(map(float, clipping_range.strip("()").split(",")))
                elif isinstance(clipping_range, list):
                    clipping_range = tuple(clipping_range)
                camera_spawn_cfg = sim_utils.PinholeCameraCfg(
                    focal_length=cameras_cfg.get("camera_focal_length", 1.88),
                    focus_distance=cameras_cfg.get("camera_focus_distance", 0.5),
                    horizontal_aperture=cameras_cfg.get("camera_horizontal_aperture", 2.6035),
                    vertical_aperture=cameras_cfg.get("camera_vertical_aperture", 1.4621),
                    clipping_range=clipping_range,
                )

            # Camera attachment link (e.g., "d435_link" for RealSense D435)
            # If not specified, camera is attached to robot root
            camera_attached_link = cameras_cfg.get("camera_attached_link", "")
            if camera_attached_link:
                camera_prim_path = f"{{ENV_REGEX_NS}}/Robot/{camera_attached_link}/ego_camera"
            else:
                camera_prim_path = "{ENV_REGEX_NS}/Robot/ego_camera"

            # Camera position and rotation offsets (relative to attached link)
            camera_pos_offset = tuple(cameras_cfg.get("camera_pos_offset", [0.0, 0.0, 0.0]))
            camera_rot_offset = tuple(
                cameras_cfg.get("camera_rot_offset", [1.0, 0.0, 0.0, 0.0])
            )  # wxyz quaternion

            # Camera data types (e.g., ["rgb"], ["rgb", "depth"])
            camera_data_types = cameras_cfg.get("camera_data_types", ["rgb"])

            # Camera resolution [H, W]
            camera_resolution = cameras_cfg.get("camera_resolution", [108, 192])

            self.ego_camera = TiledCameraCfg(
                prim_path=camera_prim_path,
                offset=TiledCameraCfg.OffsetCfg(
                    pos=camera_pos_offset, rot=camera_rot_offset, convention="world"
                ),
                data_types=camera_data_types,
                spawn=camera_spawn_cfg,
                height=camera_resolution[0],
                width=camera_resolution[1],
                debug_vis=True,
                update_period=0.0,
                update_latest_camera_pose=True,
            )

        # [D perf migration] Opt-in contact-sensor cadence controls ported from the
        # audited HiPHI "sensorperf" bundle (OLD modular_tracking_env_cfg.py
        # post-pass). Keys absent => no behavior change.
        #   contact_sensor_history_length: int   applied to every ContactSensorCfg
        #   contact_sensor_update_period: float  applied to every ContactSensorCfg
        #   contact_sensor_parity_shadow: bool   duplicate every ContactSensorCfg with
        #       its ORIGINAL cadence under 'shadow__<name>' so the runtime probe
        #       (gear_sonic/utils/contact_sensor_parity.py) can compare the force
        #       tensors consumed under the overridden cadence against the legacy
        #       cadence on the same physics state. The 'shadow__' PREFIX keeps the
        #       twins out of the canonical '<family>__<link>' name matching.
        # IsaacLab 0.54.2 (sensor_base.py:184-192): history_length > 0 forces a
        # full recompute on every scene.update() (4x per policy step at sim_dt
        # 0.005 / decimation 4); history_length == 0 makes the sensor lazy (one
        # recompute on the first post-step data access). update_period is not
        # changed unless requested: with update_period 0.0 the lazy sensor is
        # always outdated after a physics step, which avoids the float32
        # timestamp-drift caveat of update_period=0.02 documented in
        # runs/d_perf_migration_20261011/consumer_map/sensors_consumer_map.md.
        contact_sensor_update_period = config.get("contact_sensor_update_period", None)
        contact_sensor_history_length = config.get("contact_sensor_history_length", None)
        contact_sensor_parity_shadow = bool(config.get("contact_sensor_parity_shadow", False))
        if (
            contact_sensor_parity_shadow
            or contact_sensor_update_period is not None
            or contact_sensor_history_length is not None
        ):
            sensor_items = [
                (name, value)
                for name, value in vars(self).items()
                if isinstance(value, ContactSensorCfg)
            ]
            if contact_sensor_parity_shadow:
                for name, sensor_cfg_value in sensor_items:
                    setattr(self, f"shadow__{name}", copy.deepcopy(sensor_cfg_value))
            for _name, sensor_cfg_value in sensor_items:
                if contact_sensor_update_period is not None:
                    sensor_cfg_value.update_period = float(contact_sensor_update_period)
                if contact_sensor_history_length is not None:
                    sensor_cfg_value.history_length = int(contact_sensor_history_length)
            print(  # noqa: T201
                "[Contact Sensor Perf] "
                f"history_length={contact_sensor_history_length} "
                f"update_period={contact_sensor_update_period} "
                f"applied_to={len(sensor_items)} ContactSensorCfg entries; "
                f"parity_shadow={contact_sensor_parity_shadow}; "
                f"robot_net_sensor={bool(config.get('enable_robot_net_contact_sensor', True))}"
            )


@configclass
class ModularTrackingEnvCfg(ManagerBasedRLEnvCfg):
    """Modular configuration for the tracking environment that uses Hydra composition."""

    def __init__(  # noqa: D417
        self,
        config,
        actions,
        observations,
        rewards,
        terminations,
        commands,
        events,
        curriculum,
        recorders,
        **kwargs,  # noqa: ARG002
    ):
        """Initialize the configuration with Hydra config.

        Args:
            hydra_cfg: Hydra configuration containing component specifications
        """
        super().__init__()
        self._setup_from_hydra(
            config,
            actions,
            observations,
            rewards,
            terminations,
            commands,
            events,
            curriculum,
            recorders,
        )
        self.override_settings()

    def _setup_from_hydra(
        self,
        config,
        actions,
        observations,
        rewards,
        terminations,
        commands,
        events,
        curriculum,
        recorders,
    ):
        """Setup configuration from Hydra config."""
        self.config = config
        # Scene settings
        self.scene = MySceneCfg(
            config=config,
            commands=commands,
        )
        # Store USD stems for motion-USD pairing verification (outside scene cfg to avoid
        # IsaacLab treating it as an asset config)
        self.object_usd_stems = getattr(self.scene, "_tmp_usd_stems", None)
        if hasattr(self.scene, "_tmp_usd_stems"):
            del self.scene._tmp_usd_stems
        self.object_usd_stems_by_motion = getattr(
            self.scene, "_tmp_object_usd_stems_by_motion", {}
        )
        if hasattr(self.scene, "_tmp_object_usd_stems_by_motion"):
            del self.scene._tmp_object_usd_stems_by_motion
        self.object_usd_slot_count = getattr(self.scene, "_tmp_object_usd_slot_count", 0)
        if hasattr(self.scene, "_tmp_object_usd_slot_count"):
            del self.scene._tmp_object_usd_slot_count
        self.object_dynamics_by_stem = getattr(self.scene, "_tmp_object_dynamics_by_stem", {})
        if hasattr(self.scene, "_tmp_object_dynamics_by_stem"):
            del self.scene._tmp_object_dynamics_by_stem
        self.object_physics_overrides_by_stem = getattr(
            self.scene, "_tmp_object_physics_overrides_by_stem", {}
        )
        if hasattr(self.scene, "_tmp_object_physics_overrides_by_stem"):
            del self.scene._tmp_object_physics_overrides_by_stem
        self.articulated_context_usd_paths = getattr(
            self.scene, "_tmp_articulated_context_usd_paths", []
        )
        if hasattr(self.scene, "_tmp_articulated_context_usd_paths"):
            del self.scene._tmp_articulated_context_usd_paths
        self.active_articulation_usd_paths = getattr(
            self.scene, "_tmp_active_articulation_usd_paths", []
        )
        if hasattr(self.scene, "_tmp_active_articulation_usd_paths"):
            del self.scene._tmp_active_articulation_usd_paths
        self.active_articulation_usd_stems = getattr(
            self.scene, "_tmp_active_articulation_usd_stems", None
        )
        if hasattr(self.scene, "_tmp_active_articulation_usd_stems"):
            del self.scene._tmp_active_articulation_usd_stems
        self.joint_roster_view = getattr(
            self.scene, "_tmp_joint_roster_view", None
        )
        if hasattr(self.scene, "_tmp_joint_roster_view"):
            del self.scene._tmp_joint_roster_view
        self.joint_roster_physics_overrides = getattr(
            self.scene, "_tmp_joint_roster_physics_overrides", None
        )
        if hasattr(self.scene, "_tmp_joint_roster_physics_overrides"):
            del self.scene._tmp_joint_roster_physics_overrides
        self.multi_object_contact_sensor_map = getattr(
            self.scene, "_tmp_multi_object_contact_sensor_map", {}
        )
        if hasattr(self.scene, "_tmp_multi_object_contact_sensor_map"):
            del self.scene._tmp_multi_object_contact_sensor_map
        # Instantiate components using Hydra
        self.actions = common.custom_instantiate(actions, _recursive=True)
        self.observations = common.custom_instantiate(observations, _recursive=True)
        self._retarget_legacy_object_scene_entities()
        self.rewards = common.custom_instantiate(rewards, _recursive=True)
        self.terminations = common.custom_instantiate(terminations, _recursive=True)
        self.commands = common.custom_instantiate(commands, _recursive=True)
        self.events = common.custom_instantiate(events, _recursive=True)
        # Debug: Print instantiated events
        print(f"[DEBUG] Instantiated events: {self.events}")  # noqa: T201
        if hasattr(self.events, "__dict__"):
            for event_name, event_cfg in self.events.__dict__.items():
                if event_cfg is not None and not event_name.startswith("_"):
                    print(  # noqa: T201
                        f"[DEBUG]   Event '{event_name}': {type(event_cfg).__name__}"
                    )  # noqa: T201
        self.curriculum = common.custom_instantiate(curriculum, _recursive=True)
        self.recorders = common.custom_instantiate(recorders, _recursive=True)

    def _retarget_legacy_object_scene_entities(self):
        """Point legacy single-object observation cfgs at an existing multi-object scene entry."""
        if self.config.get("active_articulation_usd_path", "") or self.joint_roster_view is not None:
            first_object_name = "active_articulation"
        else:
            if not self.config.get("multi_object_per_env", False):
                return
            if not self.object_usd_stems:
                return
            if getattr(self, "object_usd_slot_count", 0):
                first_object_name = "object_0"
            else:
                first_object_name = f"object_{self.object_usd_stems[0].replace('-', '_')}"
        updated_terms = []
        for group_name, group_cfg in vars(self.observations).items():
            if group_name.startswith("_") or group_cfg is None:
                continue
            for term_name, term_cfg in vars(group_cfg).items():
                params = getattr(term_cfg, "params", None)
                if not isinstance(params, dict):
                    continue
                asset_cfg = params.get("asset_cfg")
                if getattr(asset_cfg, "name", None) == "object":
                    asset_cfg.name = first_object_name
                    updated_terms.append(f"{group_name}/{term_name}")

        if updated_terms:
            print(  # noqa: T201
                f"[Object Scene] Retargeted legacy SceneEntityCfg(name='object') "
                f"terms to {first_object_name}: {updated_terms}"
            )

    def override_settings(self):
        config = self.config
        # General settings
        self.decimation = config.get("decimation", 4)
        self.episode_length_s = config.get("episode_length_s", 10.0)

        # Simulation settings
        self.sim.dt = config.get("sim_dt", 0.005)
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material
        self.sim.physx.gpu_max_rigid_contact_count = config.get(
            "physx_gpu_max_rigid_contact_count", self.sim.physx.gpu_max_rigid_contact_count
        )
        self.sim.physx.gpu_max_rigid_patch_count = config.get(
            "physx_gpu_max_rigid_patch_count", 10 * 2**16
        )

        # Increase collision stack size for scenes with complex collision meshes (e.g. staircases)
        gpu_collision_stack_size_exp = config.get("gpu_collision_stack_size_exp", 26)
        self.sim.physx.gpu_collision_stack_size = 2**gpu_collision_stack_size_exp

        # Increase PhysX GPU memory only for multi-object scenes
        # These prevent "totalAggregatePairsCapacity" errors when many objects are spawned
        # Check if object_usd_path is a directory (multi-object mode)
        object_usd_path = config.get("object_usd_path", "")
        if config.get("add_object", False) and (
            isinstance(object_usd_path, list) or os.path.isdir(object_usd_path)
        ):
            # With proper Z-spacing of initial positions, collision pairs should be minimal
            # These are moderate values that should work for 1000+ envs
            self.sim.physx.gpu_found_lost_pairs_capacity = 2**24  # ~16M
            self.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 2**24
            self.sim.physx.gpu_total_aggregate_pairs_capacity = 2**21  # ~2M

        # Viewer settings
        viewer_config = config.get("viewer", {})
        self.viewer = ViewerCfg(
            eye=viewer_config.get("eye", [4.5, 0.0, 4.0]),
            lookat=viewer_config.get("lookat", [0.0, 0.0, 0.0]),
        )

        robot_mapping = {
            "g1": {
                "robot_cfg": g1.G1_CYLINDER_CFG,
                "action_scale": g1.G1_ACTION_SCALE,
                "isaaclab_to_mujoco_mapping": g1.G1_ISAACLAB_TO_MUJOCO_MAPPING,
                "robot_has_hands": False,
                "hand_type": None,
            },
            "g1_no_wrist": {
                "robot_cfg": g1.G1_CYLINDER_CFG,
                "action_scale": g1.G1_ACTION_SCALE_NO_WRIST,
                "isaaclab_to_mujoco_mapping": g1.G1_ISAACLAB_TO_MUJOCO_MAPPING,
                "robot_has_hands": False,
                "hand_type": None,
            },
            "g1_render": {
                "robot_cfg": g1.G1_TEXTURED_CFG,
                "action_scale": g1.G1_ACTION_SCALE,
                "isaaclab_to_mujoco_mapping": g1.G1_ISAACLAB_TO_MUJOCO_MAPPING,
                "robot_has_hands": False,
                "hand_type": None,
            },
            "g1_no_wrist_render": {
                "robot_cfg": g1.G1_TEXTURED_CFG,
                "action_scale": g1.G1_ACTION_SCALE_NO_WRIST,
                "isaaclab_to_mujoco_mapping": g1.G1_ISAACLAB_TO_MUJOCO_MAPPING,
                "robot_has_hands": False,
                "hand_type": None,
            },
            "g1_dex": {
                "robot_cfg": g1.G1_CYLINDER_DEX_CFG,
                "action_scale": g1.G1_ACTION_SCALE,
                "isaaclab_to_mujoco_mapping": g1.G1_ISAACLAB_TO_MUJOCO_MAPPING,
                "robot_has_hands": True,
                "hand_type": "dex",
            },
            "g1_model_12": {
                "robot_cfg": g1.G1_CYLINDER_MODEL_12_CFG,
                "action_scale": g1.G1_MODEL_12_ACTION_SCALE,
                "isaaclab_to_mujoco_mapping": g1.G1_ISAACLAB_TO_MUJOCO_MAPPING,
                "robot_has_hands": False,
                "hand_type": None,
            },
            "g1_model_12_dex": {
                "robot_cfg": g1.G1_CYLINDER_MODEL_12_DEX_CFG,
                "action_scale": g1.G1_MODEL_12_ACTION_SCALE,
                "isaaclab_to_mujoco_mapping": g1.G1_ISAACLAB_TO_MUJOCO_MAPPING,
                "robot_has_hands": True,
                "hand_type": "dex",
            },
            "g1_model_12_dex_usd": {
                "robot_cfg": g1.G1_CYLINDER_MODEL_12_DEX_USD_CFG,
                "action_scale": g1.G1_MODEL_12_ACTION_SCALE,
                "isaaclab_to_mujoco_mapping": g1.G1_ISAACLAB_TO_MUJOCO_MAPPING,
                "robot_has_hands": True,
                "hand_type": "dex",
            },
            "g1_model_12_dex_thor": {
                "robot_cfg": g1.G1_CYLINDER_MODEL_12_DEX_THOR_CFG,
                "action_scale": g1.G1_MODEL_12_ACTION_SCALE,
                "isaaclab_to_mujoco_mapping": g1.G1_ISAACLAB_TO_MUJOCO_MAPPING,
                "robot_has_hands": True,
                "hand_type": "dex",
            },
            "g1_model_12_dex_delayed": {
                "robot_cfg": g1.G1_CYLINDER_MODEL_12_DEX_DELAYED_CFG,
                "action_scale": g1.G1_MODEL_12_ACTION_SCALE,
                "isaaclab_to_mujoco_mapping": g1.G1_ISAACLAB_TO_MUJOCO_MAPPING,
                "robot_has_hands": True,
                "hand_type": "dex",
            },
            "g1_model_12_dex_waist": {
                "robot_cfg": g1.G1_CYLINDER_MODEL_12_DEX_WAIST_CFG,
                "action_scale": g1.G1_MODEL_12_DEX_WAIST_ACTION_SCALE,
                "isaaclab_to_mujoco_mapping": g1.G1_ISAACLAB_TO_MUJOCO_MAPPING,
                "robot_has_hands": True,
                "hand_type": "dex",
            },
            "g1_43dof": {
                "robot_cfg": g1.G1_43DOF_CFG,
                "action_scale": g1.G1_43DOF_ACTION_SCALE,
                "isaaclab_to_mujoco_mapping": g1.G1_ISAACLAB_TO_MUJOCO_MAPPING,
                "robot_has_hands": True,
                "hand_type": "dex",
            },
            "g1_43dof_waist_collision": {
                "robot_cfg": g1.G1_43DOF_WAIST_COLLISION_CFG,
                "action_scale": g1.G1_43DOF_ACTION_SCALE,
                "isaaclab_to_mujoco_mapping": g1.G1_ISAACLAB_TO_MUJOCO_MAPPING,
                "robot_has_hands": True,
                "hand_type": "dex",
            },
            "g1_43dof_yam_core_collision": {
                "robot_cfg": g1.G1_43DOF_YAM_CORE_COLLISION_CFG,
                "action_scale": g1.G1_43DOF_ACTION_SCALE,
                "isaaclab_to_mujoco_mapping": g1.G1_ISAACLAB_TO_MUJOCO_MAPPING,
                "robot_has_hands": True,
                "hand_type": "dex",
            },
            "g1_43dof_larger_hip_no_head_elbow": {
                "robot_cfg": g1.G1_43DOF_LARGER_HIP_NO_HEAD_ELBOW_CFG,
                "action_scale": g1.G1_43DOF_ACTION_SCALE,
                "isaaclab_to_mujoco_mapping": g1.G1_ISAACLAB_TO_MUJOCO_MAPPING,
                "robot_has_hands": True,
                "hand_type": "dex",
            },
            "g1_33dof_yam": {
                "robot_cfg": g1.G1_33DOF_YAM_CFG,
                "action_scale": g1.G1_33DOF_YAM_ACTION_SCALE,
                "isaaclab_to_mujoco_mapping": g1.G1_ISAACLAB_TO_MUJOCO_MAPPING,
                "robot_has_hands": True,
                "hand_type": "yam",
            },
        }

        robot_type = config["robot"].get("type", "g1")
        self.scene.robot = robot_mapping[robot_type]["robot_cfg"].replace(
            prim_path="{ENV_REGEX_NS}/Robot"
        )
        self.actions.joint_pos.scale = robot_mapping[config["robot"].get("type", "g1")][
            "action_scale"
        ]
        self.isaaclab_to_mujoco_mapping = robot_mapping[config["robot"].get("type", "g1")][
            "isaaclab_to_mujoco_mapping"
        ]

        # curriculum? WARNING HARDCODED
        import importlib

        if (
            hasattr(self.curriculum, "force_push_curriculum")
            and self.curriculum.force_push_curriculum
        ):
            module = importlib.import_module("gear_sonic.envs.manager_env.mdp")
            self.curriculum.force_push_curriculum.params["modify_fn"] = getattr(
                module, "step_curriculum"
            )

        if (
            hasattr(self.curriculum, "force_push_linear_curriculum")
            and self.curriculum.force_push_linear_curriculum
        ):
            module = importlib.import_module("gear_sonic.envs.manager_env.mdp")
            self.curriculum.force_push_linear_curriculum.params["modify_fn"] = getattr(
                module, "linear_curriculum"
            )
