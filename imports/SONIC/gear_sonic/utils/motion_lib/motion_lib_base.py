import enum
import gc
import glob
import json
import os
import os.path as osp
import random
import re
import resource
import time
from collections.abc import Mapping
from pathlib import Path

import easydict
import joblib
from loguru import logger
import numpy as np
from rich import progress
from scipy.spatial import transform
import torch
import torch.multiprocessing as mp

from gear_sonic.isaac_utils import rotations
from gear_sonic.rp2.discrete_contact import (
    CONTACT_BODY_GROUPS,
    CONTACT_ENTITY_CLASSES,
)
from gear_sonic.rp2.distributed_sampler import (
    ADAPTIVE_COUNT_STATE_KEYS,
    MOTION_IDENTITY_METADATA_KEY,
    PROVENANCE_KEY,
    RESTORE_NEUTRAL,
    RESTORE_REBASE,
    RESTORE_RETAIN,
    BinKey,
    BinStatistics,
    CanonicalAggregate,
    canonical_metadata_equal,
    deserialize_sampler_checkpoint,
    merge_object_records_by_identity,
    migrate_sampler_checkpoint_roster_transition,
    normalize_object_records,
    plan_adaptive_restore,
    rebase_counts,
    rebase_sampler_checkpoint_counts,
    serialize_canonical_checkpoint,
)
from gear_sonic.rp2.joint_payload_contract import (
    validate_joint_training_payload_pair,
)
from gear_sonic.rp2.joint_source_roster import normalize_joint_source_roster
from gear_sonic.rp2.source_relation import (
    DEFAULT_SOURCE_RELATION_MAPPING,
    select_full_scene_source_object_indices,
)
from gear_sonic.trl.utils import common
from gear_sonic.utils.motion_lib import load_jobs, motion_tensor_cache, skeleton
from gear_sonic.utils.motion_lib.contact_label_provenance import native_contact_label_authority


def interleave_keys_by_object(keys):
    """Reorder motion keys so objects are interleaved: Oa_m0, Ob_m0, ..., Oa_m1, Ob_m1, ...

    When used with max_unique_motions, the first N keys will cover as many distinct
    object classes as possible rather than exhausting one class before moving to the next.

    Object class is extracted as everything before the last underscore-delimited instance
    number in the part preceding '_jason_' (e.g. 'apple_5_jason_...' → class 'apple').
    """
    from collections import defaultdict

    def obj_class(k):
        return "_".join(k.split("_jason_")[0].split("_")[:-1])

    by_class = defaultdict(list)
    for k in sorted(keys):
        by_class[obj_class(k)].append(k)

    classes = sorted(by_class.keys())
    max_len = max(len(v) for v in by_class.values())
    interleaved = []
    for i in range(max_len):
        for cls in classes:
            if i < len(by_class[cls]):
                interleaved.append(by_class[cls][i])
    return interleaved


class FixHeightMode(enum.Enum):
    no_fix = 0
    full_fix = 1
    ankle_fix = 2


class MotionlibMode(enum.Enum):
    file = 1
    directory = 2


def to_torch(tensor):
    if torch.is_tensor(tensor):
        return tensor
    else:
        return torch.from_numpy(tensor)


def _joint_source_identity_present(motion_lib_cfg):
    """True when the motion library runs on canonical joint-source identities.

    Either identity input selects the same scientific semantics (paired 50 Hz
    reference clock, required reference_contact_state, unknown ``-1`` hand-action
    padding, robot/object pair validation): a roster manifest FILE
    (``joint_source_manifest``) or the in-memory roster of a direct-directory
    view (``joint_source_roster``, label never opened).
    """
    return bool(motion_lib_cfg.get("joint_source_manifest")) or bool(
        motion_lib_cfg.get("joint_source_roster")
    )


def _hand_action_frames(
    motion_file_data, key, *, num_frames, target_fps, paired_clock, time_map_resample=False
):
    """Discrete hand-action labels of one motion on the target clock.

    Present labels are resampled. By default this keeps the legacy
    nearest-neighbor index map (round(linspace(0, src-1, num_frames))), which the
    released task-general tracking data was prepared and evaluated with.
    ``time_map_resample=True`` (motion_lib cfg ``hand_action_time_map_resample``)
    uses the shared reference time map instead. A paired-clock motion without
    labels gets the UNKNOWN padding (``-1``, never a measured negative);
    otherwise ``None`` (legacy motions carry no hand-action channel).
    """
    if key in motion_file_data:
        if time_map_resample:
            return (
                resample_discrete_frames(
                    motion_file_data[key],
                    source_fps=motion_file_data["fps"],
                    target_fps=target_fps,
                    num_frames=num_frames,
                )
                .cpu()
                .numpy()
            )
        raw_action = motion_file_data[key]
        src_len = len(raw_action)
        if src_len != num_frames:
            indices = np.round(np.linspace(0, src_len - 1, num_frames)).astype(int)
            raw_action = raw_action[indices]
        return raw_action
    if paired_clock:
        return np.full((num_frames,), -1.0, dtype=np.float32)
    return None


def _missing_reference_contact_state(
    num_frames,
    *,
    motion_key,
    motion_lib_cfg,
):
    """Return the legacy neutral fallback or reject incomplete joint-roster data."""
    if _joint_source_identity_present(motion_lib_cfg):
        raise ValueError(
            "Canonical joint-multisource motion "
            f"{motion_key!r} is missing required reference_contact_state"
        )
    return torch.zeros(
        num_frames,
        len(CONTACT_ENTITY_CLASSES),
        len(CONTACT_BODY_GROUPS),
        dtype=torch.int8,
    )


def _canonicalize_retarget_initialization_dof(value, *, width, key):
    """Keep the shared body-DOF prefix and reject incomplete reset states."""
    if value.ndim != 2 or value.shape[1] < width:
        raise ValueError(
            f"Expected {key} to contain at least {width} canonical body "
            f"joints, got {tuple(value.shape)}"
        )
    return value[:, :width]


def is_navigation_motion(motion_key):
    return (
        motion_key.startswith("2025")
        or motion_key.startswith("walking_2025")
        or motion_key.startswith("running_2025")
        or motion_key.startswith("slow_walk_2025")
    )


def interpolate_translation_data(
    data,
    source_fps,
    target_fps,
    num_frames,
    max_num_objects=1,
    pad_value=0.0,
):
    """Interpolate translation-like data (e.g., root_pos, contact_points) to target FPS.

    Args:
        data: Tensor of shape (T, N, D) where T=frames, N=num_objects, D=dims (e.g., 3 for pos)
        source_fps: Original frame rate
        target_fps: Target frame rate
        num_frames: Target number of frames
        max_num_objects: Maximum number of objects to pad to
        pad_value: Value to use for padding

    Returns:
        Interpolated tensor of shape (num_frames, max_num_objects, D)
    """
    from gear_sonic.trl.utils import math

    data = torch.tensor(data).float() if not torch.is_tensor(data) else data.float()
    N_objects = data.shape[1]
    D = data.shape[2]

    # Interpolate to target FPS if needed
    if source_fps != target_fps:
        # Reshape to (T, N_objects*D) for batch interpolation
        data_flat = data.reshape(data.shape[0], -1)
        data_interp = math.interpolate_pose(
            data_flat,
            source_fps=source_fps,
            target_fps=target_fps,
            device=data.device,
            interpolation_type="linear",
        )
        data = data_interp.reshape(-1, N_objects, D)

    # Trim or pad to match num_frames
    if data.shape[0] > num_frames:
        data = data[:num_frames]
    elif data.shape[0] < num_frames:
        padding = data[-1:].repeat(num_frames - data.shape[0], 1, 1)
        data = torch.cat([data, padding], dim=0)

    # Pad or trim to max_num_objects
    if data.shape[1] < max_num_objects:
        padding = torch.full(
            (data.shape[0], max_num_objects - data.shape[1], D),
            pad_value,
            dtype=data.dtype,
            device=data.device,
        )
        data = torch.cat([data, padding], dim=1)
    else:
        data = data[:, :max_num_objects]

    return data


def interpolate_quaternion_data(
    data,
    source_fps,
    target_fps,
    num_frames,
    max_num_objects=1,
):
    """Interpolate quaternion data (e.g., root_quat) to target FPS using slerp.

    Args:
        data: Tensor of shape (T, N, 4) where T=frames, N=num_objects
        source_fps: Original frame rate
        target_fps: Target frame rate
        num_frames: Target number of frames
        max_num_objects: Maximum number of objects to pad to

    Returns:
        Interpolated tensor of shape (num_frames, max_num_objects, 4)
    """
    from gear_sonic.trl.utils import math

    data = torch.tensor(data).float() if not torch.is_tensor(data) else data.float()
    N_objects = data.shape[1]

    # Interpolate to target FPS if needed
    if source_fps != target_fps:
        # Reshape to (T, N_objects*4) for batch interpolation
        data_flat = data.reshape(data.shape[0], -1)
        data_interp = math.interpolate_pose(
            data_flat,
            source_fps=source_fps,
            target_fps=target_fps,
            device=data.device,
            interpolation_type="slerp",
            rot_type="quat",
        )
        data = data_interp.reshape(-1, N_objects, 4)

    # Trim or pad to match num_frames
    if data.shape[0] > num_frames:
        data = data[:num_frames]
    elif data.shape[0] < num_frames:
        padding = data[-1:].repeat(num_frames - data.shape[0], 1, 1)
        data = torch.cat([data, padding], dim=0)

    # Pad or trim to max_num_objects
    if data.shape[1] < max_num_objects:
        padding = torch.zeros(
            data.shape[0],
            max_num_objects - data.shape[1],
            4,
            dtype=data.dtype,
            device=data.device,
        )
        padding[:, :, 0] = 1.0  # w=1 for identity quaternion
        data = torch.cat([data, padding], dim=1)
    else:
        data = data[:, :max_num_objects]

    return data


def resample_discrete_frames(
    data,
    source_fps,
    target_fps,
    num_frames,
):
    """Resample arbitrary per-frame labels with the canonical time map."""
    value = torch.as_tensor(data)
    if value.ndim < 1:
        raise ValueError(
            f"Expected per-frame labels with rank >= 1, got {tuple(value.shape)}"
        )
    if value.shape[0] == 0:
        raise ValueError("Cannot resample an empty discrete label sequence")
    source_fps = float(source_fps)
    target_fps = float(target_fps)
    if (
        not np.isfinite(source_fps)
        or source_fps <= 0.0
        or not np.isfinite(target_fps)
        or target_fps <= 0.0
        or int(num_frames) <= 0
    ):
        raise ValueError(
            "Discrete resampling requires positive finite source/target FPS "
            f"and frame count, got {source_fps}/{target_fps}/{num_frames}"
        )
    target_times = (
        torch.arange(
            int(num_frames),
            device=value.device,
            dtype=torch.float64,
        )
        / target_fps
    )
    # Use the same explicit half-up rule as the preprocessing contract.
    # torch.round uses bankers' rounding at exact half-frame boundaries.
    frame_idx = torch.floor(target_times * source_fps + 0.5).long()
    frame_idx = torch.clamp(frame_idx, min=0, max=value.shape[0] - 1)
    return value[frame_idx]


def interpolate_object_mask(
    data,
    source_fps,
    target_fps,
    num_frames,
    max_num_objects=1,
    pad_value=0.0,
):
    """Resample a per-frame, per-object mask to motion-lib frame count.

    Masks are labels rather than geometry, so nearest-neighbor resampling avoids
    blending inactive and active phases.
    """
    data = torch.tensor(data).float() if not torch.is_tensor(data) else data.float()
    if data.ndim == 1:
        data = data[:, None]
    elif data.ndim == 3 and data.shape[-1] == 1:
        data = data.squeeze(-1)
    if data.ndim != 2:
        raise ValueError(f"Expected object mask shape (T, N), got {tuple(data.shape)}")

    if data.shape[0] == 0:
        data = torch.full(
            (num_frames, data.shape[1] if data.shape[1] > 0 else max_num_objects),
            pad_value,
            dtype=torch.float32,
            device=data.device,
        )
    elif data.shape[0] != num_frames or source_fps != target_fps:
        data = resample_discrete_frames(
            data,
            source_fps,
            target_fps,
            num_frames,
        )

    if data.shape[0] > num_frames:
        data = data[:num_frames]
    elif data.shape[0] < num_frames:
        padding = data[-1:].repeat(num_frames - data.shape[0], 1)
        data = torch.cat([data, padding], dim=0)

    if data.shape[1] < max_num_objects:
        padding = torch.full(
            (data.shape[0], max_num_objects - data.shape[1]),
            pad_value,
            dtype=data.dtype,
            device=data.device,
        )
        data = torch.cat([data, padding], dim=1)
    else:
        data = data[:, :max_num_objects]

    return data


def interpolate_contact_center(
    contact_points_dict,
    source_fps,
    target_fps,
    num_frames,
):
    """Compute contact center and in_contact label from raw contact points.

    Directly scales source frame indices to target frame space, avoiding
    dense array interpolation that would blend real positions with zeros.

    Args:
        contact_points_dict: Dict mapping frame_idx -> (N_points, 3) array
        source_fps: Original frame rate
        target_fps: Target frame rate
        num_frames: Target number of frames

    Returns:
        Tuple of:
            contact_center: Tensor of shape (num_frames, 3)
            in_contact: Tensor of shape (num_frames,) with binary labels
    """
    if not contact_points_dict:
        return torch.zeros(num_frames, 3), torch.zeros(num_frames)

    fps_ratio = target_fps / source_fps
    contact_center = torch.zeros(num_frames, 3)
    in_contact = torch.zeros(num_frames)

    for src_idx, points in contact_points_dict.items():
        if not (hasattr(points, "shape") and len(points) > 0):
            continue
        # Scale source frame range to target frame space
        t_start = max(0, int(src_idx * fps_ratio))
        t_end = min(num_frames, int((src_idx + 1) * fps_ratio) + 1)
        center = torch.from_numpy(points.mean(axis=0).astype(np.float32))
        contact_center[t_start:t_end] = center
        in_contact[t_start:t_end] = 1.0

    return contact_center, in_contact


class MotionLibBase:
    def __init__(self, motion_lib_cfg, num_envs, device):
        self.m_cfg = motion_lib_cfg
        self.motion_fps_scale = self.m_cfg.get("motion_fps_scale", 1.0)
        self._sim_fps = 1 / self.m_cfg.get("step_dt", 1 / 50)
        self.target_fps = self.m_cfg.get("target_fps", 50)
        retarget_dof_count = self.m_cfg.get(
            "retarget_initialization_dof_count"
        )
        self.retarget_initialization_dof_count = (
            None if retarget_dof_count is None else int(retarget_dof_count)
        )
        if (
            self.retarget_initialization_dof_count is not None
            and self.retarget_initialization_dof_count <= 0
        ):
            raise ValueError(
                "retarget_initialization_dof_count must be positive"
            )
        # Paired-clock / reference-contact semantics follow the identity input
        # (manifest file or in-memory roster), identically for both.
        self.require_paired_reference_clock = _joint_source_identity_present(
            self.m_cfg
        )
        if self.require_paired_reference_clock and abs(
            float(self.target_fps) - 50.0
        ) > 1.0e-9:
            raise ValueError(
                "Joint multi-source runtime must use the canonical 50 Hz "
                f"reference clock, got {self.target_fps}"
            )
        self.adaptive_sampling_cfg = self.m_cfg.get("adaptive_sampling") or {}
        self.allow_roster_transition = bool(
            self.adaptive_sampling_cfg.get(
                "allow_roster_transition",
                self.adaptive_sampling_cfg.get(
                    "allow_additions_only_roster_transition",
                    False,
                ),
            )
        )
        # Optional plain list of motion identities ("<source>::<motion_key>") whose
        # keyed sampler statistics must reset on a roster transition even though
        # their bin metadata is unchanged. No receipt/sidecar is involved.
        self.roster_replacement_motion_identities = tuple(
            str(value)
            for value in (
                self.adaptive_sampling_cfg.get(
                    "roster_replacement_motion_identities"
                )
                or ()
            )
        )
        self.allow_additions_only_roster_transition = (
            self.allow_roster_transition
        )
        self.all_motions_loaded = False

        self.debug = motion_lib_cfg.get("debug", False)
        self.use_parallel_fk = motion_lib_cfg.get("use_parallel_fk", False)
        self.num_envs = num_envs
        self._device = device
        self.mesh_parsers = None
        self.has_action = False
        skeleton_file = Path(self.m_cfg.asset.assetRoot) / self.m_cfg.asset.assetFileName
        self.skeleton_tree = skeleton.SkeletonTree.from_mjcf(skeleton_file)
        logger.info(f"Loaded skeleton from {skeleton_file}")
        logger.info(f"Loading motion data from {self.m_cfg.motion_file}...")
        self.load_data(self.m_cfg.motion_file)
        self.init_joint_source_tracking()
        self.use_adaptive_sampling = self.adaptive_sampling_cfg.get("enable", False)
        self._check_effective_adaptive_tracking()
        if self.use_adaptive_sampling:
            self.init_adaptive_sampling()
        self.setup_constants(
            fix_height=motion_lib_cfg.get("fix_height", FixHeightMode.no_fix),
            multi_thread=self.m_cfg.get("multi_thread", True),
        )

        self.vid_smpl_pose = None
        self.vid_smpl_joints = None
        self.smpl_data = None
        smpl_motion_file = motion_lib_cfg.get("smpl_motion_file", None)
        self.smpl_data_keys = set()
        if smpl_motion_file is not None:
            if smpl_motion_file in ("dummy", "zeros"):
                # Generate dummy zero SMPL data so SMPL observation terms work
                # without needing to null them out in the config.
                self.smpl_data = [None] * len(self._motion_data_keys)
            elif osp.exists(smpl_motion_file):
                if osp.isfile(smpl_motion_file):
                    self.smpl_data = joblib.load(smpl_motion_file)
                    self.smpl_data_keys = set(self.smpl_data.keys())
                    self.smpl_data = [
                        (self.smpl_data[k] if k in self.smpl_data else None)  # noqa: SIM401
                        for k in self._motion_data_keys
                    ]
                else:
                    self.smpl_data = []
                    smpl_pkl_files = set(
                        glob.glob(osp.join(smpl_motion_file, "**", "*.pkl"), recursive=True)
                    )
                    for k in self._motion_data_keys:
                        seq = os.path.basename(k)
                        smpl_path = osp.join(smpl_motion_file, seq + ".pkl")
                        if self.debug or smpl_path in smpl_pkl_files:
                            self.smpl_data.append({"seq": seq, "path": smpl_path})
                            self.smpl_data_keys.add(seq)
                        else:
                            self.smpl_data.append(None)
            else:
                self.smpl_data = [None] * len(self._motion_data_keys)

        self.smpl_y_up = motion_lib_cfg.get("smpl_y_up", False)

        # SOMA skeleton data loading (parallel to SMPL)
        self.soma_data = None
        soma_motion_file = motion_lib_cfg.get("soma_motion_file", None)
        self.soma_data_keys = set()
        self.soma_y_up = motion_lib_cfg.get("soma_y_up", True)  # BVH is Y-up by default
        self.num_soma_joints = motion_lib_cfg.get("num_soma_joints", 26)
        if soma_motion_file is not None:
            if soma_motion_file in ("dummy", "zeros"):
                self.soma_data = [None] * len(self._motion_data_keys)
            elif osp.exists(soma_motion_file):
                if osp.isfile(soma_motion_file):
                    self.soma_data = joblib.load(soma_motion_file)
                    self.soma_data_keys = set(self.soma_data.keys())
                    self.soma_data = [
                        (self.soma_data[k] if k in self.soma_data else None)  # noqa: SIM401
                        for k in self._motion_data_keys
                    ]
                else:
                    # Directory mode: per-motion PKL files (may be nested in subdirs)
                    soma_index = {
                        osp.splitext(osp.basename(f))[0]: f
                        for f in glob.glob(
                            osp.join(soma_motion_file, "**", "*.pkl"), recursive=True
                        )
                    }
                    self.soma_data = []
                    for k in self._motion_data_keys:
                        seq = os.path.basename(k)
                        soma_path = soma_index.get(seq)
                        if soma_path is not None or self.debug:
                            self.soma_data.append(
                                {
                                    "seq": seq,
                                    "path": soma_path or osp.join(soma_motion_file, seq + ".pkl"),
                                }
                            )
                            self.soma_data_keys.add(seq)
                        else:
                            self.soma_data.append(None)
            else:
                self.soma_data = [None] * len(self._motion_data_keys)

        # Object data loading (similar to SMPL data)
        self.object_data = None
        object_motion_file = motion_lib_cfg.get("object_motion_file", None)
        self.object_data_keys = set()
        self.max_num_objects = motion_lib_cfg.get("max_num_objects", 1)
        if object_motion_file is not None:
            if osp.isfile(object_motion_file):
                self.object_data = joblib.load(object_motion_file)
                self.object_data_keys = set(self.object_data.keys())
                self.object_data = [
                    (self.object_data[k] if k in self.object_data else None)  # noqa: SIM401
                    for k in self._motion_data_keys
                ]
            else:
                self.object_data = []
                # TODO: osp.exists() can be very expensive, consider using a set of all object pkl files
                # like in the smpl data loading above.
                for k in self._motion_data_keys:
                    seq = os.path.basename(k)
                    object_path = osp.join(object_motion_file, seq + ".pkl")
                    if self.debug or osp.exists(object_path):
                        self.object_data.append({"seq": seq, "path": object_path})
                        self.object_data_keys.add(seq)
                    else:
                        self.object_data.append(None)
        self._init_bps_directory(motion_lib_cfg)
        # randomize the upper body poses condition
        self.randomize_upper_body_poses = self.m_cfg.get("cat_upper_body_poses", False)
        self.cat_upper_body_poses_prob = self.m_cfg.get("cat_upper_body_poses_prob", 0.0)
        # The default prefixes for the upper body augmentation -- generated by the kinematic planner.
        self.upper_body_augment_prefixes = self.m_cfg.get(
            "upper_body_augment_prefixes",
            ["2025", "walking_2025", "running_2025", "slow_walk_2025"],
        )
        # Wrist joint noise augmentation config
        self.randomize_wrist_poses = self.m_cfg.get("randomize_wrist_poses", False)
        self.randomize_wrist_prob = self.m_cfg.get("randomize_wrist_prob", 0.3)
        self.randomize_wrist_std = self.m_cfg.get("randomize_wrist_std", 0.1)
        # MuJoCo DOF indices for wrist joints (L/R roll/pitch/yaw)
        self.wrist_mujoco_dof_indices = [19, 20, 21, 26, 27, 28]

    def load_data(self, motion_file):
        if osp.isfile(motion_file):
            self.mode = MotionlibMode.file
            self._motion_data_load = joblib.load(motion_file)
        else:
            assert osp.isdir(
                motion_file
            ), f"Expected motion_file to be a directory, got: {motion_file}"
            self.mode = MotionlibMode.directory
            selected = None if self.debug else self._rank_selected_motion_paths(motion_file)
            if selected is not None:
                # The caller already named this rank's motions (filter_motion_keys = the rank-local
                # aliases of the roster view): build their paths directly and read only the
                # directory's own metadata.pkl, instead of a recursive glob of the whole corpus and
                # a per-entry subdirectory walk for metadata files.
                self._motion_data_load = selected
                direct_metadata = osp.join(motion_file, "metadata.pkl")
                if osp.exists(direct_metadata):
                    for k, v in joblib.load(direct_metadata).items():
                        if k in self._motion_data_load:
                            self._motion_data_load[k].update(v)
                print(  # noqa: T201
                    f"Loaded {len(self._motion_data_load)} motion files (rank-selected paths)"
                )
            elif self.debug:
                self._motion_data_load = {}
            else:
                # sorted() is critical: motion_lib_base.py:1187's `sample_idxes`
                # picks motions by index [0..N-1]. Without sorted(), filesystem
                # glob order is non-deterministic, so different runs / different
                # repos load motions in different order — env i ends up training
                # on completely different objects. Was the root cause of GRAIL bp3
                # iter-1 reward divergence (0.54 vs gr00t 0.18 with same seed).
                self._motion_data_load = {
                    osp.splitext(osp.basename(f))[0]: {"path": f}
                    for f in sorted(
                        glob.glob(osp.join(motion_file, "**", "*.pkl"), recursive=True)
                    )
                    if not f.endswith("metadata.pkl")
                }

            if selected is None:
                metadata_files = []
                # Check for metadata.pkl directly in motion_file directory
                direct_metadata = osp.join(motion_file, "metadata.pkl")
                if osp.exists(direct_metadata):
                    metadata_files.append(direct_metadata)
                # Also check subdirectories for metadata.pkl
                all_sub_dirs = os.listdir(motion_file)
                if self.debug:
                    all_sub_dirs = all_sub_dirs[:1]
                for sub_dir in all_sub_dirs:
                    sub_dir_path = osp.join(motion_file, sub_dir)
                    if osp.isdir(sub_dir_path):
                        sub_meta = osp.join(sub_dir_path, "metadata.pkl")
                        if osp.exists(sub_meta):
                            metadata_files.append(sub_meta)

                for metadata_file in metadata_files:
                    metadata = joblib.load(metadata_file)
                    if self.debug:
                        metadata = {
                            k: v
                            for k, v in list(metadata.items())[:5]
                            if osp.exists(f"{sub_dir_path}/{k}.pkl")
                        }
                    for k, v in metadata.items():
                        if self.debug:
                            self._motion_data_load[k] = {"path": f"{sub_dir_path}/{k}.pkl"}
                        if (
                            k in self._motion_data_load
                        ):  # metadata file can have more motion sequences than in the directory. Only load the necessary ones.  # noqa: E501
                            self._motion_data_load[k].update(v)

                print(f"Loaded {len(self._motion_data_load)} motion files")  # noqa: T201

        data_list = self._motion_data_load

        filter_motion_keys = self.m_cfg.get("filter_motion_keys", None)
        if filter_motion_keys is not None:
            if isinstance(filter_motion_keys, str):  # noqa: SIM108
                patterns = [filter_motion_keys]
            else:
                patterns = list(filter_motion_keys)

            if all(pattern in data_list for pattern in patterns):
                matched_keys = [pattern for pattern in patterns if pattern in data_list]
            else:
                compiled = []
                for pattern in patterns:
                    try:
                        compiled.append(re.compile(pattern))
                    except re.error as exc:
                        raise ValueError(f"Invalid filter_motion_keys regex: {pattern}") from exc

                matched_keys = [
                    k for k in data_list if any(regex.fullmatch(k) for regex in compiled)
                ]
                matched_keys.sort()
            data_list = {k: data_list[k] for k in matched_keys}

        remove_motion_keys = self.m_cfg.get("remove_motion_keys", None)
        if remove_motion_keys is not None:
            # Remove any motion whose key starts with any of the remove_motion_keys prefixes
            keys_to_remove = [
                k for k in data_list if any(k.startswith(prefix) for prefix in remove_motion_keys)
            ]
            for k in keys_to_remove:
                del data_list[k]

        # Rank-based motion sharding: each GPU loads only its slice of motions.
        # Applied BEFORE max_unique_motions so each shard sees its correct partition
        # rather than a random sample of the globally-capped set.
        motion_shard_rank = self.m_cfg.get("motion_shard_rank", 0)
        motion_shard_world_size = self.m_cfg.get("motion_shard_world_size", 1)
        if motion_shard_world_size > 1:
            all_keys = sorted(data_list.keys())
            total = len(all_keys)
            start = (motion_shard_rank * total) // motion_shard_world_size
            end = ((motion_shard_rank + 1) * total) // motion_shard_world_size
            shard_keys = all_keys[start:end]
            data_list = {k: data_list[k] for k in shard_keys}
            print(  # noqa: T201
                f"[Motion Shard] Rank {motion_shard_rank}/{motion_shard_world_size}: "
                f"motions [{start}:{end}] = {len(shard_keys)}/{total}"
            )

        max_unique_motions = self.m_cfg.get("max_unique_motions", None)
        if max_unique_motions is not None and len(data_list) > max_unique_motions:
            keys = (
                interleave_keys_by_object(data_list.keys())
                if self.m_cfg.get("interleave_by_object", False)
                else sorted(data_list.keys())
            )
            selected = keys[:max_unique_motions]
            data_list = {k: data_list[k] for k in selected}
            print(  # noqa: T201
                f"Limited to {max_unique_motions} motions (from {len(keys)})"
            )  # noqa: RUF100, T201

        self._motion_data_list = np.array(list(data_list.values()))
        self._motion_data_keys = np.array(list(data_list.keys()))

        # # HACK: Force specific motion only
        # _FORCE_MOTION_KEY = "canned_food_31_jason_rigged_001_indoor2-v4_rand00063_000065"
        # if _FORCE_MOTION_KEY in self._motion_data_keys:
        #     idx = list(self._motion_data_keys).index(_FORCE_MOTION_KEY)
        #     self._motion_data_list = np.array([self._motion_data_list[idx]])
        #     self._motion_data_keys = np.array([_FORCE_MOTION_KEY])

        self._num_unique_motions = len(self._motion_data_list)
        logger.info(f"Loaded {self._num_unique_motions} motions")

    def _rank_selected_motion_paths(self, motion_file):
        """``{alias: {"path": <motion_file>/<alias>.pkl}}`` for an explicit alias list, else None.

        Applies when ``filter_motion_keys`` is a list of exact names and every one of them has
        its file directly under ``motion_file`` (the unified-cohort layout); otherwise the
        existing discovery runs unchanged (regex filters, nested layouts, missing files).
        """
        keys = self.m_cfg.get("filter_motion_keys", None)
        if isinstance(keys, str) or not isinstance(keys, (list, tuple)) or not keys:
            return None
        paths = {}
        for key in keys:
            key = str(key)
            path = osp.join(motion_file, f"{key}.pkl")
            if not osp.isfile(path):
                return None
            paths[key] = {"path": path}
        return paths

    def _should_augment_upper_body(self, motion_key):
        """Check if motion key matches any prefix for upper body augmentation"""  # noqa: D415
        return any(motion_key.startswith(prefix) for prefix in self.upper_body_augment_prefixes)

    def setup_constants(self, fix_height=FixHeightMode.full_fix, multi_thread=True):
        self.fix_height = fix_height
        self.multi_thread = multi_thread

        #### Termination history
        self._curr_motion_ids = None
        self._termination_history = torch.zeros(self._num_unique_motions).to(self._device)
        self._success_rate = torch.zeros(self._num_unique_motions).to(self._device)
        self._sampling_history = torch.zeros(self._num_unique_motions).to(self._device)
        self._sampling_prob = (
            torch.ones(self._num_unique_motions).to(self._device) / self._num_unique_motions
        )  # For use in sampling batches

    def update_soft_sampling_weight(self, failed_keys):
        # sampling weight based on evaluation, only "mostly" trained on "failed" sequences. Auto PMCP.
        if len(failed_keys) > 0:
            all_keys = self._motion_data_keys.tolist()
            indexes = [all_keys.index(k) for k in failed_keys]
            self._termination_history[indexes] += 1
            self.update_sampling_prob(self._termination_history)

            print(  # noqa: T201
                "############################################################ Auto PMCP ############################################################"  # noqa: E501
            )
            print(  # noqa: T201
                f"Training mostly on {len(self._sampling_prob.cpu().nonzero())} seqs "
            )  # noqa: RUF100, T201
            print(  # noqa: T201
                self._motion_data_keys[self._sampling_prob.cpu().nonzero()].flatten()
            )  # noqa: RUF100, T201
            print(  # noqa: T201
                "###############################################################################################################################"
            )
        else:
            all_keys = self._motion_data_keys.tolist()
            self._sampling_prob = (
                torch.ones(self._num_unique_motions).to(self._device) / self._num_unique_motions
            )  # For use in sampling batches

    def update_sampling_prob(self, termination_history):
        if (
            len(termination_history) == len(self._termination_history)
            and termination_history.sum() > 0
        ):
            self._sampling_prob[:] = termination_history / termination_history.sum()
            if self._sampling_prob[self._curr_motion_ids].sum() == 0:
                self._sampling_prob[self._curr_motion_ids] += 1e-6
                self._sampling_prob[:] = self._sampling_prob[:] / self._sampling_prob[:].sum()
            self._sampling_batch_prob = (
                self._sampling_prob[self._curr_motion_ids]
                / self._sampling_prob[self._curr_motion_ids].sum()
            )
            self._termination_history = termination_history
            return True
        else:
            return False

    def get_motion_actions(self, motion_ids, motion_times):
        motion_len = self._motion_lengths[motion_ids]
        num_frames = self._motion_num_frames[motion_ids]
        dt = self._motion_dt[motion_ids]
        # import ipdb; ipdb.set_trace()
        frame_idx0, frame_idx1, blend = self._calc_frame_blend(
            motion_times, motion_len, num_frames, dt
        )
        f0l = frame_idx0 + self.length_starts[motion_ids]
        f1l = frame_idx1 + self.length_starts[motion_ids]  # noqa: F841

        action = self._motion_actions[f0l]
        return action

    def get_time_step_total(self, motion_ids):
        return self._motion_num_frames[motion_ids]

    @property
    def body_indexes(self):
        return self.m_cfg.get("body_indexes_data", None)

    # ------------------------------------------------------------------
    # FP16 reference storage + tracked/full dedup
    # (_compress_reference_channels/_upcast_reference_rows/
    # _tracked_body_rows). Only these six derived channels are stored fp16; world
    # positions, dof_pos/hand_dof_pos, indices, int8 contact state and bool masks
    # stay exact. Every consumer receives fp32 rows through the single upcast
    # boundary; the 14-body tracked copies are gathered from *_full instead of
    # being materialized as second full-length tensors. Opt-in here
    # (motion_lib cfg ``fp16_reference_tensors``, default False) until measured.
    # ------------------------------------------------------------------
    _FP16_REFERENCE_CHANNELS = (
        "body_quat_w_full",
        "body_lin_vel_w_full",
        "body_ang_vel_w_full",
        "root_linv_vel_w",
        "root_ang_vel_w",
        "dof_vel",
    )

    def _fp16_reference_enabled(self) -> bool:
        value = self.m_cfg.get("fp16_reference_tensors", False)
        if isinstance(value, str):  # Hydra/CLI strings: "false"/"0" must switch it OFF
            return value.strip().lower() in ("1", "true", "yes", "on")
        return bool(value)

    def _compress_reference_channels(self):
        """Store the eligible reference channels fp16 (idempotent; no-op when disabled)."""
        if not self._fp16_reference_enabled():
            return
        for name in self._FP16_REFERENCE_CHANNELS:
            value = self.__dict__.get(name, None)
            if torch.is_tensor(value) and value.dtype == torch.float32:
                setattr(self, name, value.to(torch.float16))

    def _dedup_tracked_reference_tensors(self):
        """Drop materialized tracked-body duplicates once the *_full channels exist."""
        for name in ("body_pos_w", "body_quat_w", "body_lin_vel_w", "body_ang_vel_w"):
            full = self.__dict__.get(f"{name}_full", None)
            if torch.is_tensor(full) and name in self.__dict__:
                delattr(self, name)
        self.__dict__.pop("_tracked_body_subset_idx", None)

    def _post_restore_reference_state(self):
        """Cache hit or native load: enforce the configured storage precision and dedup."""
        self._dedup_tracked_reference_tensors()
        if not self._fp16_reference_enabled():
            # Type correctness: an FP32 run never consumes fp16 channels (a mislabeled bundle
            # is rejected here and the caller falls back to the native loader).
            bad = [
                name for name in self._FP16_REFERENCE_CHANNELS
                if torch.is_tensor(self.__dict__.get(name)) and self.__dict__[name].dtype == torch.float16
            ]
            if bad:
                raise ValueError(f"fp16 reference channels restored while fp16_reference_tensors is off: {bad}")
        self._compress_reference_channels()

    @staticmethod
    def _upcast_reference_rows(rows):
        """Single fp16 -> fp32 consumption boundary for stored reference channels."""
        if torch.is_tensor(rows) and rows.dtype == torch.float16:
            return rows.float()
        return rows

    def _tracked_body_rows(self, rows):
        """Gather the tracked-body subset (``body_indexes``) from ``*_full`` rows.

        Exact: ``full[rows][:, idx] == full[:, idx][rows]``. When no
        ``mujoco_to_isaaclab_body`` reordering is configured the sliced set never
        existed and rows pass through unchanged.
        """
        idx = self.__dict__.get("_tracked_body_subset_idx", False)
        if idx is False:
            if (
                "mujoco_to_isaaclab_body" in self.m_cfg.keys()  # noqa: SIM118
                and getattr(self, "body_indexes", None) is not None
            ):
                idx = torch.as_tensor(
                    list(self.body_indexes), dtype=torch.long, device=rows.device
                )
            else:
                idx = None
            self._tracked_body_subset_idx = idx
        if idx is None:
            return self._upcast_reference_rows(rows)
        if idx.device != rows.device:
            idx = idx.to(rows.device)
            self._tracked_body_subset_idx = idx
        return self._upcast_reference_rows(rows.index_select(-2, idx))

    def get_dof_pos(self, motion_ids, motion_steps):
        length_starts = self.length_starts[motion_ids]

        return self._upcast_reference_rows(self.dof_pos[motion_steps + length_starts])

    def get_dof_vel(self, motion_ids, motion_steps):
        length_starts = self.length_starts[motion_ids]
        return self._upcast_reference_rows(self.dof_vel[motion_steps + length_starts])

    def get_hand_dof_pos(self, motion_ids, motion_steps):
        """Get hand DOF positions if available (for 43-DOF motion on 43-DOF robot)."""
        if not hasattr(self, "hand_dof_pos") or self.hand_dof_pos is None:
            return None
        length_starts = self.length_starts[motion_ids]
        return self._upcast_reference_rows(self.hand_dof_pos[motion_steps + length_starts])

    def get_body_pos_w(self, motion_ids, motion_steps):
        length_starts = self.length_starts[motion_ids]

        return self._tracked_body_rows(self.body_pos_w_full[motion_steps + length_starts])

    def get_body_quat_w(self, motion_ids, motion_steps):
        length_starts = self.length_starts[motion_ids]
        return self._tracked_body_rows(self.body_quat_w_full[motion_steps + length_starts])

    def get_body_lin_vel_w(self, motion_ids, motion_steps):
        length_starts = self.length_starts[motion_ids]
        return self._tracked_body_rows(self.body_lin_vel_w_full[motion_steps + length_starts])

    def get_body_ang_vel_w(self, motion_ids, motion_steps):
        length_starts = self.length_starts[motion_ids]
        return self._tracked_body_rows(self.body_ang_vel_w_full[motion_steps + length_starts])

    # Full body data getters (all bodies, not sliced by body_indexes)
    def get_body_pos_w_full(self, motion_ids, motion_steps):
        """Get full body positions (all bodies, IsaacLab order)."""
        length_starts = self.length_starts[motion_ids]
        return self.body_pos_w_full[motion_steps + length_starts]

    def get_body_quat_w_full(self, motion_ids, motion_steps):
        """Get full body quaternions (all bodies, IsaacLab order, wxyz)."""
        length_starts = self.length_starts[motion_ids]
        return self._upcast_reference_rows(self.body_quat_w_full[motion_steps + length_starts])

    def get_body_lin_vel_w_full(self, motion_ids, motion_steps):
        """Get full body linear velocities (all bodies, IsaacLab order)."""
        length_starts = self.length_starts[motion_ids]
        return self._upcast_reference_rows(self.body_lin_vel_w_full[motion_steps + length_starts])

    def get_body_ang_vel_w_full(self, motion_ids, motion_steps):
        """Get full body angular velocities (all bodies, IsaacLab order)."""
        length_starts = self.length_starts[motion_ids]
        return self._upcast_reference_rows(self.body_ang_vel_w_full[motion_steps + length_starts])

    def get_root_pos_w(self, motion_ids, motion_steps):
        length_starts = self.length_starts[motion_ids]
        return self.body_pos_w_full[motion_steps + length_starts, 0, :]

    def get_root_quat_w(self, motion_ids, motion_steps):
        length_starts = self.length_starts[motion_ids]
        return self._upcast_reference_rows(self.body_quat_w_full[motion_steps + length_starts, 0, :])

    def get_root_lin_vel_w(self, motion_ids, motion_steps):
        length_starts = self.length_starts[motion_ids]
        return self._upcast_reference_rows(self.body_lin_vel_w_full[motion_steps + length_starts, 0, :])

    def get_root_ang_vel_w(self, motion_ids, motion_steps):
        length_starts = self.length_starts[motion_ids]
        return self._upcast_reference_rows(self.body_ang_vel_w_full[motion_steps + length_starts, 0, :])

    def get_smpl_pose(self, motion_ids, motion_steps):
        length_starts = self.length_starts[motion_ids]
        return self._motion_smpl_poses[motion_steps + length_starts]

    def get_smpl_joints(self, motion_ids, motion_steps):
        length_starts = self.length_starts[motion_ids]
        return self._motion_smpl_joints[motion_steps + length_starts]

    def get_smpl_transl(self, motion_ids, motion_steps):
        length_starts = self.length_starts[motion_ids]
        return self._motion_smpl_transl[motion_steps + length_starts]

    @staticmethod
    def _resample_continuous_tensor(data, fps_source, fps_target):
        """Resample a continuous tensor using the shared reference time map.

        Uses the same arange(0, duration, 1/fps_target) formula as
        torch_humanoid_batch.interploate_pose so every source remains aligned
        with the robot reference.
        """
        n_src = data.shape[0]
        duration = (n_src - 1) / fps_source
        tgt_times = torch.arange(0, duration, 1 / fps_target, dtype=torch.float32)
        n_tgt = len(tgt_times)
        if n_tgt <= 1:
            return data[:1]
        # Compute blend weights (same logic as _compute_frame_blend)
        phase = tgt_times / duration
        idx0 = (phase * (n_src - 1)).floor().long()
        idx1 = torch.minimum(idx0 + 1, torch.tensor(n_src - 1))
        blend = (phase * (n_src - 1) - idx0).float()
        # Reshape blend for broadcasting with arbitrary trailing dims
        for _ in range(data.dim() - 1):
            blend = blend.unsqueeze(-1)
        return data[idx0] * (1 - blend) + data[idx1] * blend

    def get_soma_joints(self, motion_ids, motion_steps):
        length_starts = self.length_starts[motion_ids]
        return self._motion_soma_joints[motion_steps + length_starts]

    def get_soma_root_quat(self, motion_ids, motion_steps):
        length_starts = self.length_starts[motion_ids]
        return self._motion_soma_root_quat[motion_steps + length_starts]

    def get_soma_transl(self, motion_ids, motion_steps):
        length_starts = self.length_starts[motion_ids]
        return self._motion_soma_transl[motion_steps + length_starts]

    def get_object_root_pos(self, motion_ids, motion_steps):
        length_starts = self.length_starts[motion_ids]
        return self._motion_object_root_pos[motion_steps + length_starts]

    def _init_bps_directory(self, motion_lib_cfg):
        """BPS (Basis Point Set) object shape encoding: per-object <stem>.npy files under bps_dir.

        Nothing is listed or read here. The directory's file NAMES are listed once when a native
        load assembles the batch (``_bps_available_stems``) and only the vectors of the motions this
        rank loads are read then (``_bps_vector``); a reference tensor cache hit restores
        ``_motion_object_bps`` without touching the directory. The shared basis file is not consumed
        by the observation / sampling path and is not read or re-validated.
        """
        self._bps_lookup: dict = {}  # object_stem -> (d,) np.ndarray, filled per selected motion
        self._bps_dir = None
        self._bps_stems = None  # frozenset of stems once listed
        self._bps_dim: int = 0
        bps_dir = motion_lib_cfg.get("bps_dir", None)
        if bps_dir is not None and osp.isdir(bps_dir):
            self._bps_dir = bps_dir
            self._bps_dim = int(motion_lib_cfg.get("bps_expected_dim", 10))
        elif bps_dir is not None:
            logger.warning(f"[BPS] bps_dir not found: {bps_dir} — BPS obs will be zeros")
        self._motion_object_bps = None

    def _bps_available_stems(self):
        """The BPS stems present in bps_dir (listed once per process, names only)."""
        if self._bps_stems is None:
            self._bps_stems = frozenset(
                osp.splitext(f)[0]
                for f in os.listdir(self._bps_dir)
                if f.endswith(".npy") and not f.startswith("_")
            ) if self._bps_dir is not None else frozenset()
            logger.info(
                f"[BPS] {len(self._bps_stems)} object vectors available in {self._bps_dir}/ "
                "(read per selected motion)"
            )
        return self._bps_stems

    def _bps_vector(self, stem):
        """The (d,) vector of one BPS stem, read from bps_dir on first use and kept for this process."""
        vec = self._bps_lookup.get(stem)
        if vec is None:
            vec = np.load(osp.join(self._bps_dir, f"{stem}.npy"))
            self._bps_lookup[stem] = vec
            self._bps_dim = int(vec.shape[0])
        return vec

    def get_object_bps(self, motion_ids: "torch.Tensor") -> "torch.Tensor":
        """Get BPS (Basis Point Set) shape encoding for objects at the given motion indices.

        BPS is static per object (no frame dependence). Returns zeros if not loaded.
        """
        if self._motion_object_bps is None:
            value = torch.zeros(
                len(motion_ids), max(self._bps_dim, 1), device=self._device
            )
        else:
            value = self._motion_object_bps[motion_ids]
        if bool(self.m_cfg.get("bps_require_nonzero_runtime", False)):
            expected_dim = int(self.m_cfg.get("bps_expected_dim", 10))
            expected_shape = (len(motion_ids), expected_dim)
            if tuple(value.shape) != expected_shape:
                raise RuntimeError(
                    "[BPS] Joint-training runtime shape is "
                    f"{tuple(value.shape)}, expected {expected_shape}"
                )
            minimum_norm = float(
                self.m_cfg.get("bps_minimum_vector_norm", 1.0e-8)
            )
            norms = torch.linalg.vector_norm(value.float(), dim=-1)
            invalid = (~torch.isfinite(norms)) | (norms <= minimum_norm)
            if invalid.any():
                bad = torch.nonzero(invalid, as_tuple=False).flatten()
                bad_motion_ids = motion_ids[bad].detach().cpu().tolist()
                raise RuntimeError(
                    "[BPS] Joint-training runtime produced zero/non-finite "
                    f"shape observations for motion ids {bad_motion_ids[:8]}"
                )
        return value

    def get_object_root_quat(self, motion_ids, motion_steps):
        length_starts = self.length_starts[motion_ids]
        return self._motion_object_root_quat[motion_steps + length_starts]

    def get_object_lin_vel(self, motion_ids, motion_steps):
        """Get object linear velocity from motion library."""
        length_starts = self.length_starts[motion_ids]
        return self._motion_object_lin_vel[motion_steps + length_starts]

    def get_object_ang_vel(self, motion_ids, motion_steps):
        """Get object angular velocity from motion library."""
        length_starts = self.length_starts[motion_ids]
        return self._motion_object_ang_vel[motion_steps + length_starts]

    def get_active_articulation_root_pos(self, motion_ids, motion_steps):
        length_starts = self.length_starts[motion_ids]
        return self._motion_active_articulation_root_pos[motion_steps + length_starts]

    def get_active_articulation_root_quat(self, motion_ids, motion_steps):
        length_starts = self.length_starts[motion_ids]
        return self._motion_active_articulation_root_quat[motion_steps + length_starts]

    def get_active_articulation_root_lin_vel(self, motion_ids, motion_steps):
        length_starts = self.length_starts[motion_ids]
        return self._motion_active_articulation_root_lin_vel[motion_steps + length_starts]

    def get_active_articulation_root_ang_vel(self, motion_ids, motion_steps):
        length_starts = self.length_starts[motion_ids]
        return self._motion_active_articulation_root_ang_vel[motion_steps + length_starts]

    def get_active_articulation_joint_pos(self, motion_ids, motion_steps):
        length_starts = self.length_starts[motion_ids]
        return self._motion_active_articulation_joint_pos[motion_steps + length_starts]

    def get_active_articulation_joint_vel(self, motion_ids, motion_steps):
        length_starts = self.length_starts[motion_ids]
        return self._motion_active_articulation_joint_vel[motion_steps + length_starts]

    def get_active_articulation_joint_valid_mask(self, motion_ids, motion_steps):
        length_starts = self.length_starts[motion_ids]
        return self._motion_active_articulation_joint_valid_mask[
            motion_steps + length_starts
        ]

    def get_active_articulation_root_dynamic_mask(self, motion_ids, motion_steps):
        length_starts = self.length_starts[motion_ids]
        return self._motion_active_articulation_root_dynamic_mask[
            motion_steps + length_starts
        ]

    def _label_indices(self, motion_ids, motion_steps):
        """Flat indices into concatenated per-motion label tensors.

        Steps past a clip's end would otherwise silently read the next
        motion's labels (or run out of bounds on the last motion), so clamp
        to each motion's own final frame.
        """
        max_steps = self._motion_num_frames[motion_ids] - 1
        steps = torch.minimum(motion_steps, max_steps).clamp_min(0)
        return steps + self.length_starts[motion_ids]

    def get_object_contact_center(self, motion_ids, motion_steps, hand="right_hand"):
        """Get the per-hand object-local contact target from the motion library.

        Legacy packs store the mean of exact contact points. New RP2 packs store
        the nearest validated source-hand surface point in the same field.

        Args:
            motion_ids: (N,) tensor of motion indices
            motion_steps: (N,) tensor of frame indices within each motion
            hand: Which hand's contact center to return ("left_hand" or "right_hand")

        Returns:
            Tensor of shape (N, 3) with contact center positions in object-local frame,
            or None if not available.
        """
        attr = f"_motion_object_contact_center_{'left' if hand == 'left_hand' else 'right'}"
        if not hasattr(self, attr):
            return None
        return getattr(self, attr)[self._label_indices(motion_ids, motion_steps)]

    def get_object_in_contact(self, motion_ids, motion_steps, hand="right_hand"):
        """Get binary in_contact label for the given hand.

        Args:
            motion_ids: (N,) tensor of motion indices
            motion_steps: (N,) tensor of frame indices within each motion
            hand: Which hand ("left_hand" or "right_hand")

        Returns:
            Tensor of shape (N,) with 1.0 if in contact, 0.0 otherwise,
            or None if not available.
        """
        attr = f"_motion_object_in_contact_{'left' if hand == 'left_hand' else 'right'}"
        if not hasattr(self, attr):
            return None
        return getattr(self, attr)[self._label_indices(motion_ids, motion_steps)]

    def _get_object_mask(self, attr, motion_ids, motion_steps):
        if not hasattr(self, attr):
            return None
        return getattr(self, attr)[self._label_indices(motion_ids, motion_steps)]

    def get_object_interaction_mask(self, motion_ids, motion_steps):
        """Get per-object interaction labels for the current reference frame."""
        return self._get_object_mask("_motion_object_interaction_mask", motion_ids, motion_steps)

    def get_object_motion_mask(self, motion_ids, motion_steps):
        """Get per-object motion-derived activity labels for the current reference frame."""
        return self._get_object_mask("_motion_object_motion_mask", motion_ids, motion_steps)

    def get_object_contact_mask(self, motion_ids, motion_steps, hand="right_hand"):
        """Get optional per-object contact labels for a hand at the current frame."""
        side = "left" if hand == "left_hand" else "right"
        return self._get_object_mask(f"_motion_object_contact_mask_{side}", motion_ids, motion_steps)

    def get_object_relation_vector(self, motion_ids, motion_steps, hand="right_hand"):
        """Get object-local SOMA hand-object relation vectors for a hand."""
        side = "left" if hand == "left_hand" else "right"
        attr = f"_motion_object_relation_vector_{side}"
        if not hasattr(self, attr):
            return None
        return getattr(self, attr)[self._label_indices(motion_ids, motion_steps)]

    def get_object_relation_mask(self, motion_ids, motion_steps, hand="right_hand"):
        """Get optional per-object labels for dense hand-object relation edges."""
        side = "left" if hand == "left_hand" else "right"
        return self._get_object_mask(f"_motion_object_relation_mask_{side}", motion_ids, motion_steps)

    def get_object_wrist_relation_vector(self, motion_ids, motion_steps, hand="right_hand"):
        """Get object-local SOMA wrist-object relation vectors for a hand."""
        side = "left" if hand == "left_hand" else "right"
        attr = f"_motion_object_wrist_relation_vector_{side}"
        if not hasattr(self, attr):
            return None
        return getattr(self, attr)[self._label_indices(motion_ids, motion_steps)]

    def get_object_wrist_relation_mask(self, motion_ids, motion_steps, hand="right_hand"):
        """Get validity labels for stable SOMA wrist-object relation vectors."""
        side = "left" if hand == "left_hand" else "right"
        return self._get_object_mask(
            f"_motion_object_wrist_relation_mask_{side}", motion_ids, motion_steps
        )

    def get_object_wrist_surface_distance(self, motion_ids, motion_steps, hand="right_hand"):
        """Get reference SOMA wrist-to-object-surface distances."""
        side = "left" if hand == "left_hand" else "right"
        return self._get_object_mask(
            f"_motion_object_wrist_surface_distance_{side}", motion_ids, motion_steps
        )

    def get_rp2_source_relation_vector(self, motion_ids, motion_steps):
        """Get dense source-SOMA link-to-object vectors in the human heading frame."""
        return self._get_object_mask(
            "_motion_rp2_source_relation_vector", motion_ids, motion_steps
        )

    def get_rp2_source_relation_mask(self, motion_ids, motion_steps):
        """Get validity masks for source-SOMA relation vectors."""
        return self._get_object_mask(
            "_motion_rp2_source_relation_mask", motion_ids, motion_steps
        )

    def get_rp2_source_surface_relation_vector(self, motion_ids, motion_steps):
        """Get source-link-to-nearest-object-surface vectors."""
        return self._get_object_mask(
            "_motion_rp2_source_surface_relation_vector", motion_ids, motion_steps
        )

    def get_rp2_source_surface_relation_mask(self, motion_ids, motion_steps):
        """Get validity masks for source surface-relation vectors."""
        return self._get_object_mask(
            "_motion_rp2_source_surface_relation_mask", motion_ids, motion_steps
        )

    def get_rp2_source_surface_point_local(self, motion_ids, motion_steps):
        """Get temporally stable nearest-surface anchors in object-local space."""
        return self._get_object_mask(
            "_motion_rp2_source_surface_point_local", motion_ids, motion_steps
        )

    def get_rp2_source_surface_signed_distance(self, motion_ids, motion_steps):
        """Get source signed distances to the object mesh surface."""
        return self._get_object_mask(
            "_motion_rp2_source_surface_signed_distance", motion_ids, motion_steps
        )

    def get_rp2_source_surface_penetration_mask(self, motion_ids, motion_steps):
        """Get source-link penetration labels used for surface projection."""
        return self._get_object_mask(
            "_motion_rp2_source_surface_penetration_mask", motion_ids, motion_steps
        )

    def get_rp2_source_contact_probability(self, motion_ids, motion_steps):
        """Get smooth left/right source-contact probabilities."""
        return self._get_object_mask(
            "_motion_rp2_source_contact_probability", motion_ids, motion_steps
        )

    def get_rp2_source_contact_mask(self, motion_ids, motion_steps):
        """Get unsmoothed left/right source-contact labels for diagnostics."""
        return self._get_object_mask(
            "_motion_rp2_source_contact_mask", motion_ids, motion_steps
        )

    def get_reference_contact_state(self, motion_ids, motion_steps):
        """Get canonical ternary contact labels [N, entity_class, body_group]."""
        return self._get_object_mask(
            "_motion_reference_contact_state", motion_ids, motion_steps
        )

    # -- label presence / provenance (None when the loaded state predates these channels) --------
    def get_reference_contact_state_valid(self, motion_ids, motion_steps):
        """1.0 where the clip carried a canonical reference_contact_state, 0.0 where it was filled."""
        return self._get_object_mask("_motion_reference_contact_state_valid", motion_ids, motion_steps)

    def get_object_in_contact_valid(self, motion_ids, motion_steps, hand="right_hand"):
        """1.0 where the clip carried the legacy per-hand in_contact label, 0.0 where it was zero-padded."""
        side = "left" if hand == "left_hand" else "right"
        return self._get_object_mask(f"_motion_object_in_contact_{side}_valid", motion_ids, motion_steps)

    def get_object_contact_mask(self, motion_ids, motion_steps, hand="right_hand"):
        """Legacy per-hand per-object contact mask [N, objects] (None when absent)."""
        side = "left" if hand == "left_hand" else "right"
        return self._get_object_mask(f"_motion_object_contact_mask_{side}", motion_ids, motion_steps)

    def get_object_contact_mask_valid(self, motion_ids, motion_steps, hand="right_hand"):
        """1.0 where the clip carried the legacy per-hand object contact mask, 0.0 where it was zero-padded."""
        side = "left" if hand == "left_hand" else "right"
        return self._get_object_mask(f"_motion_object_contact_mask_{side}_valid", motion_ids, motion_steps)

    def get_contact_label_native_authoritative(self, motion_ids, motion_steps):
        """1.0 where the clip's native binary hand label is authoritative (valid 0 = known no contact)."""
        return self._get_object_mask("_motion_contact_label_native_authoritative", motion_ids, motion_steps)

    def get_rp2_full_scene_source_relation_vector(self, motion_ids, motion_steps):
        """Get full-scene source-SOMA link-to-object vectors [N, O, L, 3]."""
        return self._get_object_mask(
            "_motion_rp2_full_scene_source_relation_vector", motion_ids, motion_steps
        )

    def get_rp2_full_scene_source_relation_mask(self, motion_ids, motion_steps):
        """Get full-scene source relation validity masks [N, O, L]."""
        return self._get_object_mask(
            "_motion_rp2_full_scene_source_relation_mask", motion_ids, motion_steps
        )

    def get_rp2_full_scene_source_surface_relation_vector(
        self, motion_ids, motion_steps
    ):
        """Get full-scene source-link-to-surface vectors [N, O, L, 3]."""
        return self._get_object_mask(
            "_motion_rp2_full_scene_source_surface_relation_vector",
            motion_ids,
            motion_steps,
        )

    def get_rp2_full_scene_source_surface_relation_mask(
        self, motion_ids, motion_steps
    ):
        """Get full-scene surface-relation validity masks [N, O, L]."""
        return self._get_object_mask(
            "_motion_rp2_full_scene_source_surface_relation_mask",
            motion_ids,
            motion_steps,
        )

    def get_rp2_full_scene_source_surface_point_local(
        self, motion_ids, motion_steps
    ):
        """Get full-scene object-local surface anchors [N, O, L, 3]."""
        return self._get_object_mask(
            "_motion_rp2_full_scene_source_surface_point_local",
            motion_ids,
            motion_steps,
        )

    def get_rp2_full_scene_source_surface_signed_distance(
        self, motion_ids, motion_steps
    ):
        """Get full-scene source signed surface distances [N, O, L]."""
        return self._get_object_mask(
            "_motion_rp2_full_scene_source_surface_signed_distance",
            motion_ids,
            motion_steps,
        )

    def get_rp2_full_scene_source_surface_penetration_mask(
        self, motion_ids, motion_steps
    ):
        """Get full-scene source penetration labels [N, O, L]."""
        return self._get_object_mask(
            "_motion_rp2_full_scene_source_surface_penetration_mask",
            motion_ids,
            motion_steps,
        )

    def get_rp2_full_scene_source_contact_probability(self, motion_ids, motion_steps):
        """Get full-scene smooth left/right source-contact probabilities [N, O, 2]."""
        return self._get_object_mask(
            "_motion_rp2_full_scene_source_contact_probability", motion_ids, motion_steps
        )

    def get_rp2_full_scene_source_contact_mask(self, motion_ids, motion_steps):
        """Get full-scene unsmoothed left/right source-contact labels [N, O, 2]."""
        return self._get_object_mask(
            "_motion_rp2_full_scene_source_contact_mask", motion_ids, motion_steps
        )

    def get_rp2_source_root_pos(self, motion_ids, motion_steps):
        """Get scaled source-SOMA pelvis positions for source-only reset."""
        return self._get_object_mask("_motion_rp2_source_root_pos", motion_ids, motion_steps)

    def get_rp2_source_root_heading_xy(self, motion_ids, motion_steps):
        """Get source-SOMA yaw headings as normalized XY vectors."""
        return self._get_object_mask(
            "_motion_rp2_source_root_heading_xy", motion_ids, motion_steps
        )

    def get_rp2_retarget_initialization(self, motion_ids, motion_steps):
        """Get optional G1 state used only for reset initialization."""
        required = (
            "_motion_rp2_retarget_init_root_pos",
            "_motion_rp2_retarget_init_root_quat_wxyz",
            "_motion_rp2_retarget_init_root_lin_vel",
            "_motion_rp2_retarget_init_root_ang_vel",
            "_motion_rp2_retarget_init_dof",
            "_motion_rp2_retarget_init_dof_vel",
        )
        if not all(hasattr(self, name) for name in required):
            return None
        indices = self._label_indices(motion_ids, motion_steps)
        return tuple(getattr(self, name)[indices] for name in required)

    def get_rp2_retarget_initialization_valid(self, motion_ids, motion_steps):
        """Return whether each reset frame carries authoritative retarget state."""
        if not hasattr(self, "_motion_rp2_retarget_init_valid"):
            return None
        indices = self._label_indices(motion_ids, motion_steps)
        return self._motion_rp2_retarget_init_valid[indices]

    def get_hand_action(self, motion_ids, motion_steps, hand="right_hand"):
        """Get discrete hand action (open/closed) for the given hand.

        Args:
            motion_ids: (N,) tensor of motion indices
            motion_steps: (N,) tensor of frame indices within each motion
            hand: Which hand ("left_hand" or "right_hand")

        Returns:
            Tensor of shape (N,) with -1.0 = open, +1.0 = closed,
            or None if not available.
        """
        attr = f"_motion_hand_action_{'left' if hand == 'left_hand' else 'right'}"
        if not hasattr(self, attr):
            return None
        return getattr(self, attr)[self._label_indices(motion_ids, motion_steps)]

    def get_feet_l(self, motion_ids, motion_steps):
        length_starts = self.length_starts[motion_ids]
        return self.feet_l[motion_steps + length_starts]

    def get_feet_r(self, motion_ids, motion_steps):
        length_starts = self.length_starts[motion_ids]
        return self.feet_r[motion_steps + length_starts]

    def get_motion_state(self, motion_ids, motion_times, offset=None):
        motion_len = self._motion_lengths[motion_ids]
        num_frames = self._motion_num_frames[motion_ids]
        dt = self._motion_dt[motion_ids]

        frame_idx0, frame_idx1, blend = self._calc_frame_blend(
            motion_times, motion_len, num_frames, dt
        )
        f0l = frame_idx0 + self.length_starts[motion_ids]
        f1l = frame_idx1 + self.length_starts[motion_ids]

        if "dof_pos" in self.__dict__:
            local_rot0 = self._upcast_reference_rows(self.dof_pos[f0l])
            local_rot1 = self._upcast_reference_rows(self.dof_pos[f1l])
        else:
            local_rot0 = self.body_pos_b[f0l]
            local_rot1 = self.body_pos_b[f1l]

        body_lin_vel_w0 = self._tracked_body_rows(self.body_lin_vel_w_full[f0l])
        body_lin_vel_w1 = self._tracked_body_rows(self.body_lin_vel_w_full[f1l])

        body_ang_vel0 = self._tracked_body_rows(self.body_ang_vel_w_full[f0l])
        body_ang_vel1 = self._tracked_body_rows(self.body_ang_vel_w_full[f1l])

        body_pos_w0 = self._tracked_body_rows(self.body_pos_w_full[f0l])
        body_pos_w1 = self._tracked_body_rows(self.body_pos_w_full[f1l])

        dof_vel0 = self._upcast_reference_rows(self.dof_vel[f0l])
        dof_vel1 = self._upcast_reference_rows(self.dof_vel[f1l])

        vals = [
            local_rot0,
            local_rot1,
            body_lin_vel_w0,
            body_lin_vel_w1,
            body_ang_vel0,
            body_ang_vel1,
            body_pos_w0,
            body_pos_w1,
            dof_vel0,
            dof_vel1,
        ]
        for v in vals:
            assert v.dtype != torch.float64

        blend = blend.unsqueeze(-1)

        blend_exp = blend.unsqueeze(-1)

        if offset is None:
            body_pos_w = (
                1.0 - blend_exp
            ) * body_pos_w0 + blend_exp * body_pos_w1  # ZL: apply offset
        else:
            body_pos_w = (
                (1.0 - blend_exp) * body_pos_w0 + blend_exp * body_pos_w1 + offset[..., None, :]
            )  # ZL: apply offset

        body_lin_vel_w = (1.0 - blend_exp) * body_lin_vel_w0 + blend_exp * body_lin_vel_w1
        body_ang_vel_w = (1.0 - blend_exp) * body_ang_vel0 + blend_exp * body_ang_vel1

        if "dof_pos" in self.__dict__:  # Robot Joints
            dof_vel = (1.0 - blend) * dof_vel0 + blend * dof_vel1
            dof_pos = (1.0 - blend) * local_rot0 + blend * local_rot1
        else:
            dof_vel = (1.0 - blend_exp) * dof_vel0 + blend_exp * dof_vel1
            local_rot = rotations.slerp(local_rot0, local_rot1, torch.unsqueeze(blend, axis=-1))
            dof_pos = self._local_rotation_to_dof_smpl(local_rot)

        body_quat_w0 = self._tracked_body_rows(self.body_quat_w_full[f0l])
        body_quat_w1 = self._tracked_body_rows(self.body_quat_w_full[f1l])
        body_quat_w = rotations.slerp(body_quat_w0, body_quat_w1, blend_exp)
        return_dict = {}

        if "gts_t" in self.__dict__:
            body_pos_w_t0 = self.body_pos_t_w[f0l]
            body_pos_w_t1 = self.body_pos_t_w[f1l]

            body_quat_t0 = self.body_quat_t_w[f0l]
            body_quat_t1 = self.body_quat_t_w[f1l]

            body_lin_vel_w_t0 = self.body_lin_vel_t_w[f0l]
            body_lin_vel_w_t1 = self.body_lin_vel_t_w[f1l]

            body_ang_vel_t0 = self.body_ang_vel_t_w[f0l]
            body_ang_vel_t1 = self.body_ang_vel_t_w[f1l]
            if offset is None:
                body_pos_t_w = (1.0 - blend_exp) * body_pos_w_t0 + blend_exp * body_pos_w_t1
            else:
                body_pos_t_w = (
                    (1.0 - blend_exp) * body_pos_w_t0
                    + blend_exp * body_pos_w_t1
                    + offset[..., None, :]
                )
            body_quat_t_w = rotations.slerp(body_quat_t0, body_quat_t1, blend_exp)
            body_lin_vel_t_w = (1.0 - blend_exp) * body_lin_vel_w_t0 + blend_exp * body_lin_vel_w_t1
            body_ang_vel_t_w = (1.0 - blend_exp) * body_ang_vel_t0 + blend_exp * body_ang_vel_t1
        else:
            body_pos_t_w = body_pos_w
            body_quat_t_w = body_quat_w
            body_lin_vel_t_w = body_lin_vel_w
            body_ang_vel_t_w = body_ang_vel_w

        if self.smpl_data is not None:
            smpl_pose0 = self._motion_smpl_poses[f0l]
            smpl_pose1 = self._motion_smpl_poses[f1l]
            smpl_pose = (1.0 - blend) * smpl_pose0 + blend * smpl_pose1
            return_dict.update({"smpl_pose": smpl_pose.clone()})

            if hasattr(self, "_motion_smpl_joints"):
                smpl_joints0 = self._motion_smpl_joints[f0l]
                smpl_joints1 = self._motion_smpl_joints[f1l]
                smpl_joints = (1.0 - blend_exp) * smpl_joints0 + blend_exp * smpl_joints1
                return_dict.update({"smpl_joints": smpl_joints.clone()})

            if hasattr(self, "_motion_smpl_transl"):
                smpl_transl0 = self._motion_smpl_transl[f0l]
                smpl_transl1 = self._motion_smpl_transl[f1l]
                smpl_transl = (1.0 - blend_exp) * smpl_transl0 + blend_exp * smpl_transl1
                return_dict.update({"smpl_transl": smpl_transl.clone()})

        if self.soma_data is not None:
            if hasattr(self, "_motion_soma_joints"):
                soma_joints0 = self._motion_soma_joints[f0l]
                soma_joints1 = self._motion_soma_joints[f1l]
                soma_joints = (1.0 - blend_exp) * soma_joints0 + blend_exp * soma_joints1
                return_dict.update({"soma_joints": soma_joints.clone()})

            if hasattr(self, "_motion_soma_root_quat"):
                # For quaternions, use slerp (approximate with linear blend + normalize)
                soma_rq0 = self._motion_soma_root_quat[f0l]
                soma_rq1 = self._motion_soma_root_quat[f1l]
                soma_root_quat = (1.0 - blend_exp) * soma_rq0 + blend_exp * soma_rq1
                soma_root_quat = soma_root_quat / (soma_root_quat.norm(dim=-1, keepdim=True) + 1e-8)
                return_dict.update({"soma_root_quat": soma_root_quat.clone()})

            if hasattr(self, "_motion_soma_transl"):
                soma_transl0 = self._motion_soma_transl[f0l]
                soma_transl1 = self._motion_soma_transl[f1l]
                soma_transl = (1.0 - blend_exp) * soma_transl0 + blend_exp * soma_transl1
                return_dict.update({"soma_transl": soma_transl.clone()})

        if self.object_data is not None:
            if hasattr(self, "_motion_object_root_pos"):
                object_root_pos0 = self._motion_object_root_pos[f0l]
                object_root_pos1 = self._motion_object_root_pos[f1l]
                object_root_pos = (
                    1.0 - blend_exp
                ) * object_root_pos0 + blend_exp * object_root_pos1
                return_dict.update({"object_root_pos": object_root_pos.clone()})

            if hasattr(self, "_motion_object_root_quat"):
                object_root_quat0 = self._motion_object_root_quat[f0l]
                object_root_quat1 = self._motion_object_root_quat[f1l]
                # Use slerp for quaternion interpolation
                object_root_quat = rotations.slerp(object_root_quat0, object_root_quat1, blend_exp)
                return_dict.update({"object_root_quat": object_root_quat.clone()})

        return_dict.update(
            {
                "root_pos": body_pos_w[..., 0, :].clone(),
                "root_rot": body_quat_w[..., 0, :].clone(),
                "dof_pos": dof_pos.clone(),
                "root_vel": body_lin_vel_w[..., 0, :].clone(),
                "root_ang_vel": body_ang_vel_w[..., 0, :].clone(),
                "dof_vel": dof_vel.clone(),
                "motion_aa": self._motion_aa[f0l].clone(),
                "motion_bodies": self._motion_bodies[motion_ids].clone(),
                "body_pos_w": body_pos_w.clone(),
                "body_quat_w": body_quat_w.clone(),
                "body_lin_vel_w": body_lin_vel_w.clone(),
                "body_ang_vel_w": body_ang_vel_w.clone(),
                "body_pos_w_t": body_pos_t_w.clone(),
                "body_quat_t": body_quat_t_w.clone(),
                "body_lin_vel_w_t": body_lin_vel_t_w.clone(),
                "body_ang_vel_t": body_ang_vel_t_w.clone(),
            }
        )
        if "feet_l" in self.__dict__:
            blend_int = blend.round().int()
            feet_l = torch.where(blend_int == 0, self.feet_l[f0l], self.feet_l[f1l])
            feet_r = torch.where(blend_int == 0, self.feet_r[f0l], self.feet_r[f1l])
            return_dict.update(
                {
                    "feet_l": feet_l.clone().bool(),
                    "feet_r": feet_r.clone().bool(),
                }
            )
        return return_dict

    def load_all_motions(self):
        self.all_motions_loaded = True
        self.load_motions(random_sample=False, num_motions_to_load=self._num_unique_motions)

    def load_motions_terrain_aware(self, n_terrain, n_flat_to_load):
        """Load motions with terrain keys locked at front, flat randomly sampled after.

        n_terrain is the number of terrain *envs* (may exceed unique terrain motions).
        Assumes _motion_data_keys are ordered: terrain keys first, then flat keys
        (set by per_rank_motion_keys_file). When unique terrain motions < n_terrain,
        terrain indices are oversampled with repeats.
        """
        n_unique_terrain = min(n_terrain, self._num_unique_motions)
        n_flat_available = max(self._num_unique_motions - n_unique_terrain, 0)
        n_flat_to_load = min(n_flat_to_load, n_flat_available)

        # Terrain: indices [0, n_unique_terrain) — oversample with repeats if unique < n_terrain
        if n_unique_terrain >= n_terrain:
            terrain_idxes = torch.arange(n_terrain)
        else:
            # Repeat terrain indices to fill n_terrain slots
            repeats = n_terrain // n_unique_terrain
            remainder = n_terrain % n_unique_terrain
            terrain_idxes = torch.cat([
                torch.arange(n_unique_terrain).repeat(repeats),
                torch.arange(remainder),
            ])

        # Flat: random sample from [n_terrain_unique, _num_unique_motions)
        if n_flat_available == 0:
            flat_idxes = torch.tensor([], dtype=torch.long)
        elif n_flat_to_load >= n_flat_available:
            flat_idxes = torch.arange(n_unique_terrain, self._num_unique_motions)
        else:
            flat_perm = torch.randperm(n_flat_available)[:n_flat_to_load]
            flat_idxes = flat_perm + n_unique_terrain

        sample_idxes = torch.cat([terrain_idxes, flat_idxes])
        self.all_motions_loaded = len(sample_idxes) >= self._num_unique_motions
        self.load_motions(
            random_sample=False,
            num_motions_to_load=len(sample_idxes),
            _override_sample_idxes=sample_idxes,
        )
        # Store params so resample_motion() can re-invoke with same terrain, new flat
        self._terrain_aware_n_terrain = n_terrain
        self._terrain_aware_n_flat = n_flat_to_load
        print(  # noqa: T201
            f"[TerrainAwareLoad] {n_terrain} terrain (unique={n_unique_terrain}) + {len(flat_idxes)} flat "
            f"(total={len(sample_idxes)}, unique_motions={self._num_unique_motions}, flat_available={n_flat_available})"
        )

    def load_motions_for_training(self, max_num_seqs=None):
        if self.all_motions_loaded:
            print("All motions already loaded!!! No need to resample.")  # noqa: T201
            return False

        if self.m_cfg.get("override_num_motions_to_load", None) is not None:
            max_num_seqs = self.m_cfg.override_num_motions_to_load

        # Optional run-local reference tensor cache (default disabled). A
        # compatible hit restores the assembled reference tensors; a miss runs
        # the unchanged native loader and publishes the bundle for the next
        # ordinary same-layout initialization (e.g. auto-resume). Sampler,
        # RNG, adaptive and checkpoint state are never cached.
        if self._motion_tensor_cache_mode() != "disabled" and self._load_motions_for_training_cached(
            max_num_seqs
        ):
            return True

        # Option to load unique motions (no duplicates) - useful for replay/evaluation
        load_unique = self.m_cfg.get("load_unique_motions", False)

        if (
            max_num_seqs is None
        ):  # if not specified, load all motions, can OOM if the dataset is too large.
            max_num_seqs = self._num_unique_motions
            self.all_motions_loaded = True
            self.load_motions(random_sample=False, num_motions_to_load=self._num_unique_motions)
        elif (
            max_num_seqs >= self._num_unique_motions
        ):  # if specified but more than the number of unique motions, load all motions as well.
            self.all_motions_loaded = True
            self.load_motions(random_sample=False, num_motions_to_load=self._num_unique_motions)
        else:  # if there are more motions than specified, then randomly sample the requested number of motions.
            self.all_motions_loaded = False
            # Use random_sample=False when load_unique=True to avoid duplicates
            self.load_motions(random_sample=not load_unique, num_motions_to_load=max_num_seqs)
            if load_unique:
                print(  # noqa: T201
                    f"[MotionLib] Loaded {max_num_seqs} unique motions (no duplicates)"
                )  # noqa: RUF100, T201
        return True

    def _motion_tensor_cache_mode(self):
        cache_cfg = self.m_cfg.get("motion_tensor_cache") or {}
        if not isinstance(cache_cfg, Mapping):
            raise TypeError("motion_lib_cfg.motion_tensor_cache must be a mapping")
        return motion_tensor_cache.normalize_cache_mode(cache_cfg.get("mode", "disabled"))

    def _load_motions_for_training_cached(self, max_num_seqs):
        """Load all rank-local motions through the optional reference tensor cache.

        Returns True when the reference tensors are initialized (cache hit or
        cold native load with an attempted publish). Returns False to let the
        unchanged native branches run when caching does not apply.
        """
        cache_cfg = self.m_cfg.get("motion_tensor_cache") or {}
        mode = motion_tensor_cache.normalize_cache_mode(cache_cfg.get("mode", "disabled"))
        cache_dir = cache_cfg.get("cache_dir")
        rank = int(os.environ.get("RANK", self.m_cfg.get("motion_shard_rank", 0) or 0))
        world_size = int(
            os.environ.get("WORLD_SIZE", self.m_cfg.get("motion_shard_world_size", 1) or 1)
        )
        loads_all_rank_motions = max_num_seqs is None or int(max_num_seqs) >= int(
            self._num_unique_motions
        )
        problems = motion_tensor_cache.nondeterministic_load_options(
            self.m_cfg, loads_all_rank_motions=loads_all_rank_motions
        )
        if not cache_dir:
            problems.append("motion_tensor_cache.cache_dir is not set")
        if problems:
            logger.warning(
                "startup_phase=motion_tensor_cache cache_mode=not_applicable "
                f"rank={rank} reasons={problems}; using the native loader"
            )
            return False

        started = time.perf_counter()
        cache = None
        loaded = None
        identity = None
        try:
            identity = motion_tensor_cache.build_motion_tensor_cache_identity(
                self, rank=rank, world_size=world_size, num_envs=int(self.num_envs)
            )
            payload_mode = motion_tensor_cache.effective_payload_mode(self)
            cache = motion_tensor_cache.MotionTensorCache(
                cache_dir=cache_dir, identity=identity, mode=mode, payload_mode=payload_mode
            )
            try:
                loaded = cache.load()
            except motion_tensor_cache.MotionTensorCacheMiss:
                loaded = None  # read_only miss: try compatible entries, then the native loader
            if loaded is None and bool(cache_cfg.get("reuse_compatible_entry", True)):
                # The bundle is per unique motion/frame, not per env slot: a bundle of
                # the same rank/world whose payload identity is compatible (layout
                # fields may differ; FP32 source may serve an FP16-opted run; the
                # skeleton identity is exactly what the builder recorded - resolved
                # path/size/mtime, no content hashing; everything else strict) is
                # validated through the normal read path and reused, then the
                # configured precision/dedup is applied after restore.
                loaded = motion_tensor_cache.find_reusable_entry(cache_dir, identity, payload_mode=payload_mode)
            if loaded is None and mode == "read_only":
                # read_only only controls cache WRITES; it is never permission to train.
                logger.warning(
                    "startup_phase=motion_tensor_cache cache_mode=miss_native_fallback "
                    f"rank={rank} cache_key={identity.get('cache_key')} cache_dir={cache_dir}; "
                    "no compatible entry, using the native loader (nothing will be written)"
                )
        except (
            motion_tensor_cache.MotionTensorCacheMiss,
            OSError,
            ValueError,
            TypeError,
            RuntimeError,
            json.JSONDecodeError,
        ) as exc:
            logger.warning(
                "startup_phase=motion_tensor_cache cache_mode=unavailable "
                f"rank={rank} error={exc!r}; using the native loader"
            )
            loaded = None
        if loaded is not None:
            tensors, state_metadata, status = loaded
            try:
                motion_tensor_cache.restore_motionlib_reference_state(
                    self, tensors, state_metadata, device=self._device
                )
                # Enforce the configured storage precision / dedup on restored state
                # (identity already excludes mismatched fp16 settings; this also
                # drops tracked duplicates stored by older bundles).
                self._post_restore_reference_state()
            except (ValueError, TypeError, RuntimeError) as exc:
                logger.warning(
                    "startup_phase=motion_tensor_cache cache_mode=restore_failed "
                    f"rank={rank} error={exc!r}; using the native loader"
                )
                loaded = None
        if loaded is not None:
            # Same caller-side state the native branch derives for a full load.
            self.num_joints = len(self.skeleton_tree.node_names)
            self._curr_motion_ids = torch.arange(
                self._num_motions, dtype=torch.long, device=self._device
            )
            self._sampling_batch_prob = (
                self._sampling_prob[self._curr_motion_ids]
                / self._sampling_prob[self._curr_motion_ids].sum()
            )
            self.res_non_nav_dataset = {}
            self.all_motions_loaded = True
            if self.use_adaptive_sampling:
                self._finish_deferred_adaptive_init()
                self.update_adaptive_sampling_motion_frames()
            self._motion_tensor_cache_status = dict(status)
            logger.info(
                "startup_phase=motion_tensor_cache "
                f"cache_mode={status['mode']} rank={rank} world_size={world_size} "
                f"motions={self._num_motions} frames={self.body_pos_w_full.shape[0]} "
                f"elapsed_s={time.perf_counter() - started:.3f} entry={status['entry']}"
            )
            return True

        # Cold miss: the unchanged native loader, then an optional publish.
        attributes_before_load = frozenset(vars(self))
        self.all_motions_loaded = True
        self.load_motions(random_sample=False, num_motions_to_load=self._num_unique_motions)
        if cache is None:
            self._motion_tensor_cache_status = {"mode": "unavailable"}
            return True
        if mode != "read_write":
            # read_only: native state is initialized, nothing is published.
            self._motion_tensor_cache_status = {"mode": "miss_native_fallback", "entry": str(cache.entry)}
            logger.info(
                "startup_phase=motion_tensor_cache cache_mode=miss_native_fallback "
                f"rank={rank} world_size={world_size} motions={self._num_motions} "
                f"frames={self.body_pos_w_full.shape[0]} elapsed_s={time.perf_counter() - started:.3f}"
            )
            return True
        try:
            tensors, state_metadata = motion_tensor_cache.capture_motionlib_reference_state(
                self,
                attributes_before_load=attributes_before_load
                | getattr(self, "_deferred_adaptive_attribute_names", frozenset()),
            )
        except Exception as exc:  # noqa: BLE001 - optional cache IO only; native state is initialized
            status = {"mode": "write_failed", "error": repr(exc), "entry": str(cache.entry)}
            logger.warning(
                "startup_phase=motion_tensor_cache cache_mode=write_failed "
                f"rank={rank} error={exc!r}; native initialization kept"
            )
            self._motion_tensor_cache_status = dict(status)
            return True
        # The publication (20 GB class torch.save + fsync + atomic rename) leaves the startup critical path: it runs
        # on a background thread over the CPU copies the capture just made; training starts on the native state now.
        native_elapsed = time.perf_counter() - started
        frames = int(self.body_pos_w_full.shape[0])
        motions = int(self._num_motions)

        def _on_publish_done(status: dict, rank=rank, world_size=world_size, frames=frames, motions=motions) -> None:
            self._motion_tensor_cache_status = dict(status)
            logger.info(
                "startup_phase=motion_tensor_cache "
                f"cache_mode={status['mode']} rank={rank} world_size={world_size} "
                f"motions={motions} frames={frames} bytes={status.get('tensor_bytes_total')} "
                f"publish_s={status.get('publish_seconds', 0.0):.3f} entry={status.get('entry')}"
                + (f" error={status.get('error')}" if status.get("mode") == "write_failed" else "")
            )

        self._motion_tensor_cache_publish = motion_tensor_cache.BackgroundPublish(
            cache, tensors, state_metadata, on_done=_on_publish_done
        )
        del tensors
        self._motion_tensor_cache_status = dict(self._motion_tensor_cache_publish.status)
        logger.info(
            "startup_phase=motion_tensor_cache cache_mode=computed_publish_pending "
            f"rank={rank} world_size={world_size} motions={motions} frames={frames} "
            f"elapsed_s={native_elapsed:.3f} entry={cache.entry} (publication continues in the background)"
        )
        return True

    def load_motions_for_evaluation(self, start_idx=0):
        # disable this check to avoid upper body poses randomization in evaluation
        # if self.all_motions_loaded:
        #     print("All motions already loaded!!! No need to resample.")
        #     return

        if (
            self._num_unique_motions > self.num_envs
        ):  # if number of motions is more than number of envs, then we should only partially load the motions.
            self.all_motions_loaded = False
            self.load_motions(
                random_sample=False,
                num_motions_to_load=self.num_envs,
                start_idx=start_idx,
                is_evaluation=True,
            )
        else:
            self.all_motions_loaded = True
            self.load_motions(
                random_sample=False,
                num_motions_to_load=self._num_unique_motions,
                start_idx=start_idx,
                is_evaluation=True,
            )

    def load_motions(
        self,
        random_sample=True,
        start_idx=0,
        max_len=-1,
        target_heading=None,
        num_motions_to_load=None,
        is_evaluation=False,
        _override_sample_idxes=None,
    ):

        if "gts" in self.__dict__:
            for _name in (
                "body_pos_w", "body_quat_w", "body_pos_b", "root_linv_vel_w", "root_ang_vel_w",
                "body_ang_vel_w", "body_lin_vel_w", "dof_vels", "dof_pos",
                "body_pos_w_full", "body_quat_w_full", "body_lin_vel_w_full", "body_ang_vel_w_full",
                "_tracked_body_subset_idx",
            ):
                self.__dict__.pop(_name, None)
            if "gts_t" in self.__dict__:
                del (
                    self.body_pos_t_w,
                    self.body_quat_t_w,
                    self.body_lin_vel_t_w,
                    self.body_ang_vel_t_w,
                )

        motions = []
        _motion_lengths = []
        _motion_fps = []
        _motion_source_fps = []
        _motion_dt = []
        _motion_num_frames = []
        _motion_bodies = []
        _motion_aa = []
        has_action = False  # noqa: F841
        _motion_actions = []
        _motion_smpl_poses = []
        _motion_smpl_joints = []
        _motion_smpl_transl = []
        _motion_soma_joints = []
        _motion_soma_root_quat = []
        _motion_soma_transl = []
        _motion_object_root_pos = []
        _motion_object_root_quat = []
        _motion_active_articulation_root_pos = []
        _motion_active_articulation_root_quat = []
        _motion_active_articulation_root_lin_vel = []
        _motion_active_articulation_root_ang_vel = []
        _motion_active_articulation_joint_pos = []
        _motion_active_articulation_joint_vel = []
        _motion_active_articulation_joint_valid_mask = []
        _motion_active_articulation_root_dynamic_mask = []
        _motion_object_contact_center_left = []
        _motion_object_contact_center_right = []
        _motion_object_in_contact_left = []
        _motion_object_in_contact_right = []
        _motion_object_name_candidates = []
        _motion_object_interaction_mask = []
        _motion_object_motion_mask = []
        _motion_object_contact_mask_left = []
        _motion_object_contact_mask_right = []
        # Per-clip label PRESENCE channels (1 = the source clip carried the label, 0 = loader padding) and
        # per-clip provenance (1 = the clip's native binary hand label is authoritative, e.g. OmniContact
        # wrist contact_info). Plain per-frame tensors, captured by the optional tensor cache like every
        # other reference channel; absent (old bundle) -> readers treat presence as unknown. No gate.
        _motion_object_in_contact_left_valid = []
        _motion_object_in_contact_right_valid = []
        _motion_object_contact_mask_left_valid = []
        _motion_object_contact_mask_right_valid = []
        _motion_reference_contact_state_valid = []
        _motion_contact_label_native_authoritative = []
        has_object_contact_center_left = False
        has_object_contact_center_right = False
        has_object_in_contact_left = False
        has_object_in_contact_right = False
        has_object_contact_mask_left = False
        has_object_contact_mask_right = False
        _motion_object_relation_vector_left = []
        _motion_object_relation_vector_right = []
        _motion_object_relation_mask_left = []
        _motion_object_relation_mask_right = []
        has_object_relation_vector_left = False
        has_object_relation_vector_right = False
        has_object_relation_mask_left = False
        has_object_relation_mask_right = False
        _motion_object_wrist_relation_vector_left = []
        _motion_object_wrist_relation_vector_right = []
        _motion_object_wrist_relation_mask_left = []
        _motion_object_wrist_relation_mask_right = []
        _motion_object_wrist_surface_distance_left = []
        _motion_object_wrist_surface_distance_right = []
        has_object_wrist_relation_vector_left = False
        has_object_wrist_relation_vector_right = False
        has_object_wrist_relation_mask_left = False
        has_object_wrist_relation_mask_right = False
        has_object_wrist_surface_distance_left = False
        has_object_wrist_surface_distance_right = False
        source_relation_num_links = int(
            self.m_cfg.get("rp2_source_relation_num_links", 14)
        )
        _motion_rp2_source_relation_vector = []
        _motion_rp2_source_relation_mask = []
        _motion_rp2_source_surface_relation_vector = []
        _motion_rp2_source_surface_relation_mask = []
        _motion_rp2_source_surface_point_local = []
        _motion_rp2_source_surface_signed_distance = []
        _motion_rp2_source_surface_penetration_mask = []
        _motion_rp2_source_contact_probability = []
        _motion_rp2_source_contact_mask = []
        _motion_reference_contact_state = []
        _motion_rp2_full_scene_source_relation_vector = []
        _motion_rp2_full_scene_source_relation_mask = []
        _motion_rp2_full_scene_source_surface_relation_vector = []
        _motion_rp2_full_scene_source_surface_relation_mask = []
        _motion_rp2_full_scene_source_surface_point_local = []
        _motion_rp2_full_scene_source_surface_signed_distance = []
        _motion_rp2_full_scene_source_surface_penetration_mask = []
        _motion_rp2_full_scene_source_contact_probability = []
        _motion_rp2_full_scene_source_contact_mask = []
        _motion_rp2_source_root_pos = []
        _motion_rp2_source_root_heading_xy = []
        _motion_rp2_retarget_init_root_pos = []
        _motion_rp2_retarget_init_root_quat_wxyz = []
        _motion_rp2_retarget_init_root_lin_vel = []
        _motion_rp2_retarget_init_root_ang_vel = []
        _motion_rp2_retarget_init_dof = []
        _motion_rp2_retarget_init_dof_vel = []
        _motion_rp2_retarget_init_valid = []
        has_rp2_source_relation_vector = False
        has_rp2_source_relation_mask = False
        has_rp2_source_surface_relation_vector = False
        has_rp2_source_surface_relation_mask = False
        has_rp2_source_surface_point_local = False
        has_rp2_source_surface_signed_distance = False
        has_rp2_source_surface_penetration_mask = False
        has_rp2_source_contact_probability = False
        has_rp2_source_contact_mask = False
        has_reference_contact_state = False
        has_rp2_full_scene_source_relation_vector = False
        has_rp2_full_scene_source_relation_mask = False
        has_rp2_full_scene_source_surface_relation_vector = False
        has_rp2_full_scene_source_surface_relation_mask = False
        has_rp2_full_scene_source_surface_point_local = False
        has_rp2_full_scene_source_surface_signed_distance = False
        has_rp2_full_scene_source_surface_penetration_mask = False
        has_rp2_full_scene_source_contact_probability = False
        has_rp2_full_scene_source_contact_mask = False
        has_rp2_source_root_pos = False
        has_rp2_source_root_heading_xy = False
        has_rp2_retarget_initialization = False
        _motion_hand_action_left = []
        _motion_hand_action_right = []

        total_len = 0.0
        self.num_joints = len(self.skeleton_tree.node_names)
        if num_motions_to_load is None:  # noqa: SIM108
            num_motion_to_load = self.num_envs
        else:
            num_motion_to_load = num_motions_to_load

        if self.use_adaptive_sampling:
            self._prepare_adaptive_motion_selection(
                random_sample=random_sample,
                explicit_selection=_override_sample_idxes is not None,
            )

        if _override_sample_idxes is not None:
            sample_idxes = _override_sample_idxes.to(self._device)
            num_motion_to_load = len(sample_idxes)
        elif random_sample:
            sample_idxes = torch.multinomial(
                self._sampling_prob, num_samples=num_motion_to_load, replacement=True
            ).to(self._device)
        else:  # start_idx only used for non-random sampling.
            sample_idxes = torch.clamp(
                torch.arange(num_motion_to_load) + start_idx, max=self._num_unique_motions - 1
            ).to(self._device)

        # sample_idxes = torch.tensor([self._motion_data_keys.tolist().index("0-KIT_8_WalkInClockwiseCircle04_poses")]).to(self._device)  # noqa: E501
        self._curr_motion_ids = sample_idxes
        self.curr_motion_keys = (
            [self._motion_data_keys[sample_idxes.cpu()]]
            if sample_idxes.numel() == 1
            else self._motion_data_keys[sample_idxes.cpu()].tolist()
        )
        self._sampling_batch_prob = (
            self._sampling_prob[self._curr_motion_ids]
            / self._sampling_prob[self._curr_motion_ids].sum()
        )

        logger.info(f"Loading {num_motion_to_load} motions...")
        logger.info(f"Sampling motion: {sample_idxes[:10]}, ....")
        logger.info(f"Current motion keys: {self.curr_motion_keys[:10]}, ....")

        motion_data_list = self._motion_data_list[sample_idxes.cpu().numpy()]
        if self.smpl_data is not None:
            smpl_data_list = [self.smpl_data[idx] for idx in sample_idxes.cpu().numpy()]
        else:
            smpl_data_list = None
        if self.object_data is not None:
            object_data_list = [self.object_data[idx] for idx in sample_idxes.cpu().numpy()]
        else:
            object_data_list = None
        if self.soma_data is not None:
            soma_data_list = [self.soma_data[idx] for idx in sample_idxes.cpu().numpy()]
        else:
            soma_data_list = None
        torch.set_num_threads(1)

        # Increase file descriptor limit to prevent "too many open files" error
        try:
            soft_limit, hard_limit = resource.getrlimit(resource.RLIMIT_NOFILE)
            target_limit = 1048576

            # Try to set both soft and hard limits
            if soft_limit < target_limit:
                try:
                    # First try to increase hard limit (requires root)
                    resource.setrlimit(resource.RLIMIT_NOFILE, (target_limit, target_limit))
                    logger.info(
                        f"Increased file descriptor limits from {soft_limit}/{hard_limit} to {target_limit}/{target_limit}"  # noqa: E501
                    )
                except PermissionError:
                    # Fallback to increasing only soft limit up to hard limit
                    new_soft = min(target_limit, hard_limit)
                    resource.setrlimit(resource.RLIMIT_NOFILE, (new_soft, hard_limit))
                    logger.info(
                        f"Increased soft file descriptor limit from {soft_limit} to {new_soft} (hard limit: {hard_limit})"  # noqa: E501
                    )
        except Exception as e:  # noqa: BLE001
            logger.warning(f"Could not increase file descriptor limit: {e}")

        num_jobs = min(min(mp.cpu_count(), 32), len(motion_data_list))  # noqa: PLW3301

        if num_jobs <= 8 or not self.multi_thread or len(motion_data_list) <= 128:
            num_jobs = 1

        # Fork-free parallel load: when process workers are off (MOTION_LIB_MULTI_THREAD=false in every
        # formal leg because forking after CUDA init is unsafe), the same chunks run on threads of this process
        # (gear_sonic/utils/motion_lib/load_jobs.py). Serial path: MOTION_LIB_LOAD_THREADS=1 or a single motion.
        use_threads = False
        if num_jobs == 1:
            thread_jobs = load_jobs.loader_thread_count(len(motion_data_list))
            if thread_jobs > 1:
                num_jobs = thread_jobs
                use_threads = True

        # Single-GPU teardown fix: the Manager server is a
        # forked child of the Isaac Sim process; with one job it was created but never used and never
        # shut down, so after the trainer's os._exit() it outlived the parent holding the step's stdout
        # pipe (a smoke run hung 7.9 min in teardown). Create it only when workers exist and shut
        # it down once results are gathered. Loading semantics/order are unchanged.
        manager = mp.Manager() if (num_jobs > 1 and not use_threads) else None
        queue = manager.Queue() if manager is not None else (load_jobs.new_queue() if use_threads else None)

        logger.info(f"Loading motions with {num_jobs} jobs..." + (" (threads, fork-free)" if use_threads else ""))
        self.res_non_nav_dataset = {}
        res_acc = {}  # using dictionary ensures order of the results.
        workers = []

        # if self.randomize_upper_body_poses:
        # self.cat_upper_body_poses_prob = 1.0
        if self.randomize_upper_body_poses and not is_evaluation:

            # get indices that are in navigation dataset
            nav_indices = [
                i
                for i in range(len(motion_data_list))
                if self._should_augment_upper_body(self.curr_motion_keys[i])
            ]
            other_indices = [
                i
                for i in range(len(motion_data_list))
                if not self._should_augment_upper_body(self.curr_motion_keys[i])
            ]

            nav_motion_data_list = [motion_data_list[i] for i in nav_indices]
            other_motion_data_list = [motion_data_list[i] for i in other_indices]

            if self.smpl_data is not None:
                nav_smpl_data_list = [smpl_data_list[i] for i in nav_indices]
                other_smpl_data_list = [smpl_data_list[i] for i in other_indices]
            else:
                nav_smpl_data_list = None
                other_smpl_data_list = None

            if self.object_data is not None:
                nav_object_data_list = [object_data_list[i] for i in nav_indices]
                other_object_data_list = [object_data_list[i] for i in other_indices]
            else:
                nav_object_data_list = None
                other_object_data_list = None

            if soma_data_list is not None:
                nav_soma_data_list = [soma_data_list[i] for i in nav_indices]
                other_soma_data_list = [soma_data_list[i] for i in other_indices]
            else:
                nav_soma_data_list = None
                other_soma_data_list = None

            # load non-navigation dataset first
            if len(other_motion_data_list) > 0:
                jobs = other_motion_data_list
                chunk = np.ceil(len(jobs) / num_jobs).astype(int)
                ids = np.array(other_indices)  # Use original indices, not sequential

                jobs = [
                    (
                        ids[i : i + chunk],
                        jobs[i : i + chunk],
                        (
                            None
                            if other_smpl_data_list is None
                            else other_smpl_data_list[i : i + chunk]
                        ),
                        (
                            None
                            if other_object_data_list is None
                            else other_object_data_list[i : i + chunk]
                        ),
                        (
                            None
                            if other_soma_data_list is None
                            else other_soma_data_list[i : i + chunk]
                        ),
                        self.fix_height,
                        target_heading,
                        max_len,
                        is_evaluation,
                    )
                    for i in range(0, len(jobs), chunk)
                ]

                res_acc.update(self._run_load_jobs(jobs, queue=queue, use_threads=use_threads,
                                                   label="Gathering results for non-navigation dataset..."))

                self.res_non_nav_dataset = res_acc.copy()

            # load navigation dataset
            if len(nav_motion_data_list) > 0:

                jobs = nav_motion_data_list
                chunk = np.ceil(len(jobs) / num_jobs).astype(int)
                ids = np.array(nav_indices)  # Use original indices, not sequential

                jobs = [
                    (
                        ids[i : i + chunk],
                        jobs[i : i + chunk],
                        nav_smpl_data_list[
                            i : i + chunk
                        ],  # navigation dataset would never have smpl data. This is always empty.
                        (
                            None
                            if nav_object_data_list is None
                            else nav_object_data_list[i : i + chunk]
                        ),
                        (None if nav_soma_data_list is None else nav_soma_data_list[i : i + chunk]),
                        self.fix_height,
                        target_heading,
                        max_len,
                        is_evaluation,
                    )
                    for i in range(0, len(jobs), chunk)
                ]
                res_acc.update(self._run_load_jobs(jobs, queue=queue, use_threads=use_threads,
                                                   label="Gathering results for navigation dataset..."))

        else:
            jobs = motion_data_list
            chunk = np.ceil(len(jobs) / num_jobs).astype(int)
            ids = np.arange(len(jobs))

            jobs = [
                (
                    ids[i : i + chunk],
                    jobs[i : i + chunk],
                    None if smpl_data_list is None else smpl_data_list[i : i + chunk],
                    None if object_data_list is None else object_data_list[i : i + chunk],
                    None if soma_data_list is None else soma_data_list[i : i + chunk],
                    self.fix_height,
                    target_heading,
                    max_len,
                    is_evaluation,
                )
                for i in range(0, len(jobs), chunk)
            ]

            res_acc.update(self._run_load_jobs(jobs, queue=queue, use_threads=use_threads, label="Gathering results..."))

            nav_indices = []
            other_indices = list(range(len(motions)))

        if manager is not None:
            try:
                manager.shutdown()
            except Exception as exc:  # noqa: BLE001 - results are already gathered; teardown only
                logger.warning(f"motion loader manager shutdown failed: {exc!r}")
            manager = None

        for f in progress.track(range(len(res_acc)), description="Processing motions..."):
            motion_file_data, curr_motion = res_acc[f]
            motion_fps = int(curr_motion.fps * self.motion_fps_scale)
            curr_dt = 1.0 / motion_fps
            num_frames = curr_motion.global_rotation.shape[0]

            curr_len = 1.0 / motion_fps * (num_frames - 1)

            if "beta" in motion_file_data:
                _motion_aa.append(motion_file_data["pose_aa"].reshape(-1, self.num_joints * 3))
                _motion_bodies.append(curr_motion.gender_beta)
            else:
                _motion_aa.append(np.zeros((num_frames, self.num_joints * 3)))
                _motion_bodies.append(torch.zeros(17))

            _motion_fps.append(motion_fps)
            _motion_source_fps.append(float(motion_file_data["fps"]))
            _motion_dt.append(curr_dt)
            _motion_num_frames.append(num_frames)
            motions.append(curr_motion)
            _motion_lengths.append(curr_len)
            if self.has_action:
                _motion_actions.append(curr_motion.action)
            if self.smpl_data is not None:
                _motion_smpl_poses.append(curr_motion["smpl_pose"])
                if "smpl_joints" in curr_motion:
                    _motion_smpl_joints.append(curr_motion["smpl_joints"])
                if "smpl_transl" in curr_motion:
                    _motion_smpl_transl.append(curr_motion["smpl_transl"])
            if self.soma_data is not None:
                if "soma_joints" in curr_motion:
                    _motion_soma_joints.append(curr_motion["soma_joints"])
                if "soma_root_quat" in curr_motion:
                    _motion_soma_root_quat.append(curr_motion["soma_root_quat"])
                if "soma_transl" in curr_motion:
                    _motion_soma_transl.append(curr_motion["soma_transl"])
            if self.object_data is not None:
                def _as_object_mask(value):
                    if value.ndim == 1:
                        value = value.unsqueeze(-1)
                    elif value.ndim == 3 and value.shape[-1] == 1:
                        value = value.squeeze(-1)
                    if value.ndim != 2:
                        raise ValueError(
                            f"Expected object mask [frames, objects], got {tuple(value.shape)}"
                        )
                    return value

                def _pad_object_axis(value, *, expected_tail, pad_value=0.0):
                    if tuple(value.shape[2:]) != tuple(expected_tail):
                        raise ValueError(
                            "Expected full-scene source tensor [frames, objects, "
                            f"{expected_tail}], got {tuple(value.shape)}"
                        )
                    selected_indices = select_full_scene_source_object_indices(
                        value.shape[1],
                        self.max_num_objects,
                        target_object_index=curr_motion.get(
                            "rp2_box29_target_object_index"
                        ),
                        declared_source_object_count=curr_motion.get(
                            "rp2_box29_source_full_scene_object_count"
                        ),
                    )
                    if len(selected_indices) != value.shape[1]:
                        value = value[:, list(selected_indices)]
                    if value.shape[1] == self.max_num_objects:
                        return value
                    padded_shape = (
                        value.shape[0],
                        self.max_num_objects,
                        *value.shape[2:],
                    )
                    padded = torch.full(
                        padded_shape,
                        float(pad_value),
                        dtype=value.dtype,
                        device=value.device,
                    )
                    padded[:, : value.shape[1]] = value
                    return padded

                if "object_root_pos" in curr_motion:
                    _motion_object_root_pos.append(curr_motion["object_root_pos"])
                if "object_root_quat" in curr_motion:
                    _motion_object_root_quat.append(curr_motion["object_root_quat"])
                active_reference_lists = {
                    "active_articulation_root_pos": _motion_active_articulation_root_pos,
                    "active_articulation_root_quat": _motion_active_articulation_root_quat,
                    "active_articulation_root_lin_vel": _motion_active_articulation_root_lin_vel,
                    "active_articulation_root_ang_vel": _motion_active_articulation_root_ang_vel,
                    "active_articulation_joint_pos": _motion_active_articulation_joint_pos,
                    "active_articulation_joint_vel": _motion_active_articulation_joint_vel,
                }
                for key, destination in active_reference_lists.items():
                    if key not in curr_motion:
                        continue
                    value = curr_motion[key]
                    if value.ndim == 1:
                        value = value.unsqueeze(-1)
                    elif value.ndim == 3 and value.shape[1] == 1:
                        value = value[:, 0]
                    destination.append(value)
                if "active_articulation_joint_pos" in curr_motion:
                    joint_valid_mask = curr_motion.get(
                        "active_articulation_joint_valid_mask",
                        torch.ones_like(curr_motion["active_articulation_joint_pos"]),
                    )
                    if joint_valid_mask.ndim == 1:
                        joint_valid_mask = joint_valid_mask.unsqueeze(-1)
                    elif joint_valid_mask.ndim == 3 and joint_valid_mask.shape[1] == 1:
                        joint_valid_mask = joint_valid_mask[:, 0]
                    _motion_active_articulation_joint_valid_mask.append(
                        joint_valid_mask.float()
                    )
                    root_dynamic_mask = curr_motion.get(
                        "active_articulation_root_dynamic_mask",
                        torch.ones(num_frames, 1),
                    )
                    if root_dynamic_mask.ndim == 1:
                        root_dynamic_mask = root_dynamic_mask.unsqueeze(-1)
                    elif root_dynamic_mask.ndim == 3 and root_dynamic_mask.shape[1] == 1:
                        root_dynamic_mask = root_dynamic_mask[:, 0]
                    _motion_active_articulation_root_dynamic_mask.append(
                        root_dynamic_mask.float()
                    )
                if "object_contact_center_left" in curr_motion:
                    _motion_object_contact_center_left.append(
                        curr_motion["object_contact_center_left"]
                    )
                    has_object_contact_center_left = True
                else:
                    _motion_object_contact_center_left.append(torch.zeros(num_frames, 3))
                if "object_in_contact_left" in curr_motion:
                    _motion_object_in_contact_left.append(curr_motion["object_in_contact_left"])
                    has_object_in_contact_left = True
                    _motion_object_in_contact_left_valid.append(torch.ones(num_frames))
                else:
                    _motion_object_in_contact_left.append(torch.zeros(num_frames))
                    _motion_object_in_contact_left_valid.append(torch.zeros(num_frames))
                if "object_contact_center_right" in curr_motion:
                    _motion_object_contact_center_right.append(
                        curr_motion["object_contact_center_right"]
                    )
                    has_object_contact_center_right = True
                else:
                    _motion_object_contact_center_right.append(torch.zeros(num_frames, 3))
                if "object_in_contact_right" in curr_motion:
                    _motion_object_in_contact_right.append(curr_motion["object_in_contact_right"])
                    has_object_in_contact_right = True
                    _motion_object_in_contact_right_valid.append(torch.ones(num_frames))
                else:
                    _motion_object_in_contact_right.append(torch.zeros(num_frames))
                    _motion_object_in_contact_right_valid.append(torch.zeros(num_frames))
                _motion_object_name_candidates.append(
                    list(curr_motion.get("object_name_candidates", []))
                )
                if "object_interaction_mask" in curr_motion:
                    _motion_object_interaction_mask.append(
                        _as_object_mask(curr_motion["object_interaction_mask"])
                    )
                else:
                    _motion_object_interaction_mask.append(
                        torch.zeros(num_frames, self.max_num_objects)
                    )
                if "object_motion_mask" in curr_motion:
                    _motion_object_motion_mask.append(
                        _as_object_mask(curr_motion["object_motion_mask"])
                    )
                else:
                    _motion_object_motion_mask.append(torch.zeros(num_frames, self.max_num_objects))
                if "object_contact_mask_left" in curr_motion:
                    _motion_object_contact_mask_left.append(
                        _as_object_mask(curr_motion["object_contact_mask_left"])
                    )
                    has_object_contact_mask_left = True
                    _motion_object_contact_mask_left_valid.append(torch.ones(num_frames))
                else:
                    _motion_object_contact_mask_left.append(
                        torch.zeros(num_frames, self.max_num_objects)
                    )
                    _motion_object_contact_mask_left_valid.append(torch.zeros(num_frames))
                if "object_contact_mask_right" in curr_motion:
                    _motion_object_contact_mask_right.append(
                        _as_object_mask(curr_motion["object_contact_mask_right"])
                    )
                    has_object_contact_mask_right = True
                    _motion_object_contact_mask_right_valid.append(torch.ones(num_frames))
                else:
                    _motion_object_contact_mask_right.append(
                        torch.zeros(num_frames, self.max_num_objects)
                    )
                    _motion_object_contact_mask_right_valid.append(torch.zeros(num_frames))
                # Provenance: is this clip's native binary hand label authoritative (a valid 0 = known no
                # contact even where the canonical conversion says neutral)? Read an explicit per-clip pack
                # field when the producer wrote one; otherwise the configured motion-key prefixes
                # (default: OmniContact packs, keys starting with "omni"). Documented assumption, no scan.
                _native_flag, _native_reason = native_contact_label_authority(
                    str(self.curr_motion_keys[f]),
                    curr_motion,
                    curr_motion.get("_object_record_provenance"),
                    key_prefixes=self.m_cfg.get("legacy_contact_native_authoritative_key_prefixes", None),
                )
                _motion_contact_label_native_authoritative.append(
                    torch.full((num_frames,), 1.0 if bool(_native_flag) else 0.0)
                )
                if "object_relation_vector_left" in curr_motion:
                    _motion_object_relation_vector_left.append(
                        curr_motion["object_relation_vector_left"]
                    )
                    has_object_relation_vector_left = True
                else:
                    _motion_object_relation_vector_left.append(
                        torch.zeros(num_frames, self.max_num_objects, 3)
                    )
                if "object_relation_vector_right" in curr_motion:
                    _motion_object_relation_vector_right.append(
                        curr_motion["object_relation_vector_right"]
                    )
                    has_object_relation_vector_right = True
                else:
                    _motion_object_relation_vector_right.append(
                        torch.zeros(num_frames, self.max_num_objects, 3)
                    )
                if "object_relation_mask_left" in curr_motion:
                    _motion_object_relation_mask_left.append(
                        _as_object_mask(curr_motion["object_relation_mask_left"])
                    )
                    has_object_relation_mask_left = True
                else:
                    _motion_object_relation_mask_left.append(
                        torch.zeros(num_frames, self.max_num_objects)
                    )
                if "object_relation_mask_right" in curr_motion:
                    _motion_object_relation_mask_right.append(
                        _as_object_mask(curr_motion["object_relation_mask_right"])
                    )
                    has_object_relation_mask_right = True
                else:
                    _motion_object_relation_mask_right.append(
                        torch.zeros(num_frames, self.max_num_objects)
                    )
                for side in ("left", "right"):
                    vector_key = f"object_wrist_relation_vector_{side}"
                    mask_key = f"object_wrist_relation_mask_{side}"
                    distance_key = f"object_wrist_surface_distance_{side}"
                    vector_values = (
                        _motion_object_wrist_relation_vector_left
                        if side == "left"
                        else _motion_object_wrist_relation_vector_right
                    )
                    mask_values = (
                        _motion_object_wrist_relation_mask_left
                        if side == "left"
                        else _motion_object_wrist_relation_mask_right
                    )
                    distance_values = (
                        _motion_object_wrist_surface_distance_left
                        if side == "left"
                        else _motion_object_wrist_surface_distance_right
                    )
                    if vector_key in curr_motion:
                        vector_values.append(curr_motion[vector_key])
                        if side == "left":
                            has_object_wrist_relation_vector_left = True
                        else:
                            has_object_wrist_relation_vector_right = True
                    else:
                        vector_values.append(
                            torch.zeros(num_frames, self.max_num_objects, 3)
                        )
                    if mask_key in curr_motion:
                        mask_values.append(_as_object_mask(curr_motion[mask_key]))
                        if side == "left":
                            has_object_wrist_relation_mask_left = True
                        else:
                            has_object_wrist_relation_mask_right = True
                    else:
                        mask_values.append(torch.zeros(num_frames, self.max_num_objects))
                    if distance_key in curr_motion:
                        distance_values.append(_as_object_mask(curr_motion[distance_key]))
                        if side == "left":
                            has_object_wrist_surface_distance_left = True
                        else:
                            has_object_wrist_surface_distance_right = True
                    else:
                        distance_values.append(torch.zeros(num_frames, self.max_num_objects))
                if "rp2_source_relation_vector" in curr_motion:
                    source_relation = curr_motion["rp2_source_relation_vector"]
                    if source_relation.shape[1:] != (source_relation_num_links, 3):
                        raise ValueError(
                            "Expected rp2_source_relation_vector "
                            f"[frames, {source_relation_num_links}, 3], got "
                            f"{tuple(source_relation.shape)}"
                        )
                    expected_robot_names = [
                        robot_name for robot_name, _ in DEFAULT_SOURCE_RELATION_MAPPING
                    ]
                    expected_soma_names = [
                        soma_name for _, soma_name in DEFAULT_SOURCE_RELATION_MAPPING
                    ]
                    if (
                        curr_motion.get("rp2_source_relation_robot_body_names")
                        != expected_robot_names
                    ):
                        raise ValueError(
                            "rp2_source_relation_vector G1 body order does not match "
                            "DEFAULT_SOURCE_RELATION_MAPPING"
                        )
                    if (
                        curr_motion.get("rp2_source_relation_soma_joint_names")
                        != expected_soma_names
                    ):
                        raise ValueError(
                            "rp2_source_relation_vector SOMA joint order does not match "
                            "DEFAULT_SOURCE_RELATION_MAPPING"
                        )
                    _motion_rp2_source_relation_vector.append(source_relation)
                    has_rp2_source_relation_vector = True
                else:
                    _motion_rp2_source_relation_vector.append(
                        torch.zeros(num_frames, source_relation_num_links, 3)
                    )
                if "rp2_source_relation_mask" in curr_motion:
                    source_mask = curr_motion["rp2_source_relation_mask"]
                    if source_mask.shape[1:] != (source_relation_num_links,):
                        raise ValueError(
                            "Expected rp2_source_relation_mask "
                            f"[frames, {source_relation_num_links}], got {tuple(source_mask.shape)}"
                        )
                    _motion_rp2_source_relation_mask.append(source_mask)
                    has_rp2_source_relation_mask = True
                else:
                    _motion_rp2_source_relation_mask.append(
                        torch.zeros(num_frames, source_relation_num_links)
                    )
                for key, destination, expected_tail in (
                    (
                        "rp2_source_surface_relation_vector",
                        _motion_rp2_source_surface_relation_vector,
                        (source_relation_num_links, 3),
                    ),
                    (
                        "rp2_source_surface_relation_mask",
                        _motion_rp2_source_surface_relation_mask,
                        (source_relation_num_links,),
                    ),
                    (
                        "rp2_source_surface_point_local",
                        _motion_rp2_source_surface_point_local,
                        (source_relation_num_links, 3),
                    ),
                    (
                        "rp2_source_surface_signed_distance",
                        _motion_rp2_source_surface_signed_distance,
                        (source_relation_num_links,),
                    ),
                    (
                        "rp2_source_surface_penetration_mask",
                        _motion_rp2_source_surface_penetration_mask,
                        (source_relation_num_links,),
                    ),
                ):
                    value = curr_motion.get(key)
                    if value is not None:
                        if value.shape[1:] != expected_tail:
                            raise ValueError(
                                f"Expected {key} [frames, {', '.join(map(str, expected_tail))}], "
                                f"got {tuple(value.shape)}"
                            )
                        destination.append(value)
                    else:
                        destination.append(torch.zeros(num_frames, *expected_tail))
                if "rp2_source_surface_relation_vector" in curr_motion:
                    has_rp2_source_surface_relation_vector = True
                if "rp2_source_surface_relation_mask" in curr_motion:
                    has_rp2_source_surface_relation_mask = True
                if "rp2_source_surface_point_local" in curr_motion:
                    has_rp2_source_surface_point_local = True
                if "rp2_source_surface_signed_distance" in curr_motion:
                    has_rp2_source_surface_signed_distance = True
                if "rp2_source_surface_penetration_mask" in curr_motion:
                    has_rp2_source_surface_penetration_mask = True
                bilateral_source_channels = (
                    (
                        "rp2_source_contact_probability",
                        _motion_rp2_source_contact_probability,
                        "has_rp2_source_contact_probability",
                    ),
                    (
                        "rp2_source_contact_mask",
                        _motion_rp2_source_contact_mask,
                        "has_rp2_source_contact_mask",
                    ),
                    (
                        "rp2_source_root_heading_xy",
                        _motion_rp2_source_root_heading_xy,
                        "has_rp2_source_root_heading_xy",
                    ),
                )
                for key, destination, _ in bilateral_source_channels:
                    value = curr_motion.get(key)
                    if value is not None:
                        if value.shape[1:] != (2,):
                            raise ValueError(f"Expected {key} [frames, 2], got {tuple(value.shape)}")
                        destination.append(value)
                    else:
                        destination.append(torch.zeros(num_frames, 2))
                if "rp2_source_contact_probability" in curr_motion:
                    has_rp2_source_contact_probability = True
                if "rp2_source_contact_mask" in curr_motion:
                    has_rp2_source_contact_mask = True
                if "rp2_source_root_heading_xy" in curr_motion:
                    has_rp2_source_root_heading_xy = True
                reference_contact_state = curr_motion.get("reference_contact_state")
                if reference_contact_state is not None:
                    expected_tail = (
                        len(CONTACT_ENTITY_CLASSES),
                        len(CONTACT_BODY_GROUPS),
                    )
                    if reference_contact_state.shape[1:] != expected_tail:
                        raise ValueError(
                            "Expected reference_contact_state "
                            f"[frames, {expected_tail[0]}, {expected_tail[1]}], got "
                            f"{tuple(reference_contact_state.shape)}"
                        )
                    if curr_motion.get("reference_contact_entity_classes") != list(
                        CONTACT_ENTITY_CLASSES
                    ):
                        raise ValueError(
                            "reference_contact_state entity order does not match "
                            "CONTACT_ENTITY_CLASSES"
                        )
                    if curr_motion.get("reference_contact_body_groups") != list(
                        CONTACT_BODY_GROUPS
                    ):
                        raise ValueError(
                            "reference_contact_state body order does not match "
                            "CONTACT_BODY_GROUPS"
                        )
                    rounded_contact = reference_contact_state.round()
                    if not torch.equal(
                        reference_contact_state.to(dtype=torch.float32),
                        rounded_contact.to(dtype=torch.float32),
                    ):
                        raise ValueError(
                            "reference_contact_state must contain discrete integer labels"
                        )
                    unique_contact = set(
                        int(value)
                        for value in torch.unique(rounded_contact).cpu().tolist()
                    )
                    if not unique_contact.issubset({-1, 0, 1}):
                        raise ValueError(
                            "reference_contact_state contains labels outside "
                            f"{{-1, 0, +1}}: {sorted(unique_contact)}"
                        )
                    _motion_reference_contact_state.append(
                        rounded_contact.to(dtype=torch.int8)
                    )
                    has_reference_contact_state = True
                    _motion_reference_contact_state_valid.append(torch.ones(num_frames))
                else:
                    _motion_reference_contact_state.append(
                        _missing_reference_contact_state(
                            num_frames,
                            motion_key=self.curr_motion_keys[f],
                            motion_lib_cfg=self.m_cfg,
                        )
                    )
                    _motion_reference_contact_state_valid.append(torch.zeros(num_frames))
                if "rp2_full_scene_source_relation_vector" in curr_motion:
                    source_relation = curr_motion["rp2_full_scene_source_relation_vector"]
                    if source_relation.shape[2:] != (source_relation_num_links, 3):
                        raise ValueError(
                            "Expected rp2_full_scene_source_relation_vector "
                            f"[frames, objects, {source_relation_num_links}, 3], got "
                            f"{tuple(source_relation.shape)}"
                        )
                    expected_robot_names = [
                        robot_name for robot_name, _ in DEFAULT_SOURCE_RELATION_MAPPING
                    ]
                    expected_soma_names = [
                        soma_name for _, soma_name in DEFAULT_SOURCE_RELATION_MAPPING
                    ]
                    if (
                        curr_motion.get("rp2_full_scene_source_relation_robot_body_names")
                        != expected_robot_names
                    ):
                        raise ValueError(
                            "rp2_full_scene_source_relation_vector G1 body order does not "
                            "match DEFAULT_SOURCE_RELATION_MAPPING"
                        )
                    if (
                        curr_motion.get("rp2_full_scene_source_relation_soma_joint_names")
                        != expected_soma_names
                    ):
                        raise ValueError(
                            "rp2_full_scene_source_relation_vector SOMA joint order does not "
                            "match DEFAULT_SOURCE_RELATION_MAPPING"
                        )
                    _motion_rp2_full_scene_source_relation_vector.append(
                        _pad_object_axis(source_relation, expected_tail=(source_relation_num_links, 3))
                    )
                    has_rp2_full_scene_source_relation_vector = True
                else:
                    _motion_rp2_full_scene_source_relation_vector.append(
                        torch.zeros(num_frames, self.max_num_objects, source_relation_num_links, 3)
                    )
                if "rp2_full_scene_source_relation_mask" in curr_motion:
                    source_mask = curr_motion["rp2_full_scene_source_relation_mask"]
                    if source_mask.shape[2:] != (source_relation_num_links,):
                        raise ValueError(
                            "Expected rp2_full_scene_source_relation_mask "
                            f"[frames, objects, {source_relation_num_links}], got "
                            f"{tuple(source_mask.shape)}"
                        )
                    _motion_rp2_full_scene_source_relation_mask.append(
                        _pad_object_axis(source_mask, expected_tail=(source_relation_num_links,))
                    )
                    has_rp2_full_scene_source_relation_mask = True
                else:
                    _motion_rp2_full_scene_source_relation_mask.append(
                        torch.zeros(num_frames, self.max_num_objects, source_relation_num_links)
                    )
                for key, destination, expected_tail in (
                    (
                        "rp2_full_scene_source_surface_relation_vector",
                        _motion_rp2_full_scene_source_surface_relation_vector,
                        (source_relation_num_links, 3),
                    ),
                    (
                        "rp2_full_scene_source_surface_relation_mask",
                        _motion_rp2_full_scene_source_surface_relation_mask,
                        (source_relation_num_links,),
                    ),
                    (
                        "rp2_full_scene_source_surface_point_local",
                        _motion_rp2_full_scene_source_surface_point_local,
                        (source_relation_num_links, 3),
                    ),
                    (
                        "rp2_full_scene_source_surface_signed_distance",
                        _motion_rp2_full_scene_source_surface_signed_distance,
                        (source_relation_num_links,),
                    ),
                    (
                        "rp2_full_scene_source_surface_penetration_mask",
                        _motion_rp2_full_scene_source_surface_penetration_mask,
                        (source_relation_num_links,),
                    ),
                ):
                    value = curr_motion.get(key)
                    if value is not None:
                        if value.shape[2:] != expected_tail:
                            raise ValueError(
                                f"Expected {key} [frames, objects, "
                                f"{', '.join(map(str, expected_tail))}], got "
                                f"{tuple(value.shape)}"
                            )
                        destination.append(
                            _pad_object_axis(value, expected_tail=expected_tail)
                        )
                    else:
                        destination.append(
                            torch.zeros(
                                num_frames,
                                self.max_num_objects,
                                *expected_tail,
                            )
                        )
                if "rp2_full_scene_source_surface_relation_vector" in curr_motion:
                    has_rp2_full_scene_source_surface_relation_vector = True
                if "rp2_full_scene_source_surface_relation_mask" in curr_motion:
                    has_rp2_full_scene_source_surface_relation_mask = True
                if "rp2_full_scene_source_surface_point_local" in curr_motion:
                    has_rp2_full_scene_source_surface_point_local = True
                if "rp2_full_scene_source_surface_signed_distance" in curr_motion:
                    has_rp2_full_scene_source_surface_signed_distance = True
                if "rp2_full_scene_source_surface_penetration_mask" in curr_motion:
                    has_rp2_full_scene_source_surface_penetration_mask = True
                for key, destination, expected_tail in (
                    (
                        "rp2_full_scene_source_contact_probability",
                        _motion_rp2_full_scene_source_contact_probability,
                        (2,),
                    ),
                    (
                        "rp2_full_scene_source_contact_mask",
                        _motion_rp2_full_scene_source_contact_mask,
                        (2,),
                    ),
                ):
                    value = curr_motion.get(key)
                    if value is not None:
                        if value.shape[2:] != expected_tail:
                            raise ValueError(f"Expected {key} [frames, objects, 2], got {tuple(value.shape)}")
                        destination.append(_pad_object_axis(value, expected_tail=expected_tail))
                    else:
                        destination.append(torch.zeros(num_frames, self.max_num_objects, *expected_tail))
                if "rp2_full_scene_source_contact_probability" in curr_motion:
                    has_rp2_full_scene_source_contact_probability = True
                if "rp2_full_scene_source_contact_mask" in curr_motion:
                    has_rp2_full_scene_source_contact_mask = True
                if "rp2_source_root_pos" in curr_motion:
                    source_root_pos = curr_motion["rp2_source_root_pos"]
                    if source_root_pos.shape[1:] != (3,):
                        raise ValueError(
                            f"Expected rp2_source_root_pos [frames, 3], got {tuple(source_root_pos.shape)}"
                        )
                    _motion_rp2_source_root_pos.append(source_root_pos)
                    has_rp2_source_root_pos = True
                else:
                    _motion_rp2_source_root_pos.append(torch.zeros(num_frames, 3))
                retarget_dof_width = (
                    self.retarget_initialization_dof_count
                    or motion_file_data["dof"].shape[-1]
                )
                retarget_init_specs = (
                    ("rp2_retarget_init_root_pos", _motion_rp2_retarget_init_root_pos, 3),
                    (
                        "rp2_retarget_init_root_quat_wxyz",
                        _motion_rp2_retarget_init_root_quat_wxyz,
                        4,
                    ),
                    (
                        "rp2_retarget_init_root_lin_vel",
                        _motion_rp2_retarget_init_root_lin_vel,
                        3,
                    ),
                    (
                        "rp2_retarget_init_root_ang_vel",
                        _motion_rp2_retarget_init_root_ang_vel,
                        3,
                    ),
                    (
                        "rp2_retarget_init_dof",
                        _motion_rp2_retarget_init_dof,
                        retarget_dof_width,
                    ),
                    (
                        "rp2_retarget_init_dof_vel",
                        _motion_rp2_retarget_init_dof_vel,
                        retarget_dof_width,
                    ),
                )
                has_all_retarget_init = all(key in curr_motion for key, _, _ in retarget_init_specs)
                has_any_retarget_init = any(key in curr_motion for key, _, _ in retarget_init_specs)
                if has_any_retarget_init and not has_all_retarget_init:
                    missing = [key for key, _, _ in retarget_init_specs if key not in curr_motion]
                    raise ValueError(f"Incomplete RP2 retarget initialization channels: {missing}")
                for key, destination, width in retarget_init_specs:
                    value = curr_motion.get(key)
                    if value is None:
                        destination.append(torch.zeros(num_frames, width))
                    else:
                        if key in {
                            "rp2_retarget_init_dof",
                            "rp2_retarget_init_dof_vel",
                        }:
                            # Every promoted reference orders the canonical
                            # 29 body joints first; source-specific hand DOFs
                            # remain in separate hand-action channels.
                            value = _canonicalize_retarget_initialization_dof(
                                value,
                                width=width,
                                key=key,
                            )
                        if value.shape != (num_frames, width):
                            raise ValueError(
                                f"Expected {key} [{num_frames}, {width}], got {tuple(value.shape)}"
                            )
                        destination.append(value)
                _motion_rp2_retarget_init_valid.append(
                    torch.full(
                        (num_frames,),
                        has_all_retarget_init,
                        dtype=torch.bool,
                    )
                )
                has_rp2_retarget_initialization |= has_all_retarget_init
                for hand_key, hand_frames in (
                    ("hand_action_left", _motion_hand_action_left),
                    ("hand_action_right", _motion_hand_action_right),
                ):
                    frames = _hand_action_frames(
                        motion_file_data,
                        hand_key,
                        num_frames=num_frames,
                        target_fps=self.target_fps,
                        paired_clock=self.require_paired_reference_clock,
                        time_map_resample=bool(
                            self.m_cfg.get("hand_action_time_map_resample", False)
                        ),
                    )
                    if frames is not None:
                        hand_frames.append(frames)
            del curr_motion

        self._motion_lengths = torch.tensor(
            _motion_lengths, device=self._device, dtype=torch.float32
        )
        self._motion_fps = torch.tensor(_motion_fps, device=self._device, dtype=torch.float32)
        # The payload's own source fps (exact float, e.g. 29.97): the value the bin metadata and the
        # deferred adaptive init use; a reference tensor, so a cache hit restores it.
        self._motion_source_fps = torch.tensor(_motion_source_fps, device=self._device, dtype=torch.float64)
        self._motion_bodies = torch.stack(_motion_bodies).to(self._device).type(torch.float32)
        self._motion_aa = torch.tensor(
            np.concatenate(_motion_aa), device=self._device, dtype=torch.float32
        )

        if self.smpl_data is not None:
            self._motion_smpl_poses = torch.cat(_motion_smpl_poses, dim=0).float().to(self._device)
            if len(_motion_smpl_joints) > 0:
                self._motion_smpl_joints = (
                    torch.cat(_motion_smpl_joints, dim=0).float().to(self._device)
                )
            if len(_motion_smpl_transl) > 0:
                self._motion_smpl_transl = (
                    torch.cat(_motion_smpl_transl, dim=0).float().to(self._device)
                )
        if self.soma_data is not None:
            if len(_motion_soma_joints) > 0:
                self._motion_soma_joints = (
                    torch.cat(_motion_soma_joints, dim=0).float().to(self._device)
                )
            if len(_motion_soma_root_quat) > 0:
                self._motion_soma_root_quat = (
                    torch.cat(_motion_soma_root_quat, dim=0).float().to(self._device)
                )
            if len(_motion_soma_transl) > 0:
                self._motion_soma_transl = (
                    torch.cat(_motion_soma_transl, dim=0).float().to(self._device)
                )
        if self.object_data is not None:
            if len(_motion_object_root_pos) > 0:
                self._motion_object_root_pos = (
                    torch.cat(_motion_object_root_pos, dim=0).float().to(self._device)
                )
            if len(_motion_object_root_quat) > 0:
                self._motion_object_root_quat = (
                    torch.cat(_motion_object_root_quat, dim=0).float().to(self._device)
                )
            active_reference_lists = {
                "_motion_active_articulation_root_pos": _motion_active_articulation_root_pos,
                "_motion_active_articulation_root_quat": _motion_active_articulation_root_quat,
                "_motion_active_articulation_root_lin_vel": (
                    _motion_active_articulation_root_lin_vel
                ),
                "_motion_active_articulation_root_ang_vel": (
                    _motion_active_articulation_root_ang_vel
                ),
                "_motion_active_articulation_joint_pos": _motion_active_articulation_joint_pos,
                "_motion_active_articulation_joint_vel": _motion_active_articulation_joint_vel,
                "_motion_active_articulation_joint_valid_mask": (
                    _motion_active_articulation_joint_valid_mask
                ),
                "_motion_active_articulation_root_dynamic_mask": (
                    _motion_active_articulation_root_dynamic_mask
                ),
            }
            if any(active_reference_lists.values()):
                incomplete = {
                    name: len(values)
                    for name, values in active_reference_lists.items()
                    if len(values) != len(motions)
                }
                if incomplete:
                    raise ValueError(
                        "Active-articulation reference channels must be present for every "
                        f"loaded motion; counts={incomplete}, motions={len(motions)}"
                    )
                for name, values in active_reference_lists.items():
                    setattr(
                        self,
                        name,
                        torch.cat(values, dim=0).float().to(self._device),
                    )
            # Store per-hand contact centers and in_contact labels
            if has_object_contact_center_left:
                self._motion_object_contact_center_left = (
                    torch.cat(_motion_object_contact_center_left, dim=0).float().to(self._device)
                )
            forced_contact_tensors = self.effective_payload_mode()[
                "manifest_contact_tensors"
            ]
            if has_object_in_contact_left or forced_contact_tensors:
                self._motion_object_in_contact_left = (
                    torch.cat(_motion_object_in_contact_left, dim=0).float().to(self._device)
                )
            if has_object_contact_center_right:
                self._motion_object_contact_center_right = (
                    torch.cat(_motion_object_contact_center_right, dim=0).float().to(self._device)
                )
            if has_object_in_contact_right or forced_contact_tensors:
                self._motion_object_in_contact_right = (
                    torch.cat(_motion_object_in_contact_right, dim=0).float().to(self._device)
                )
            if len(_motion_object_interaction_mask) > 0:
                self._motion_object_interaction_mask = (
                    torch.cat(_motion_object_interaction_mask, dim=0).float().to(self._device)
                )
            if len(_motion_object_motion_mask) > 0:
                self._motion_object_motion_mask = (
                    torch.cat(_motion_object_motion_mask, dim=0).float().to(self._device)
                )
            if has_object_contact_mask_left:
                self._motion_object_contact_mask_left = (
                    torch.cat(_motion_object_contact_mask_left, dim=0).float().to(self._device)
                )
            if has_object_contact_mask_right:
                self._motion_object_contact_mask_right = (
                    torch.cat(_motion_object_contact_mask_right, dim=0).float().to(self._device)
                )
            # Label presence / provenance channels (always complete: one entry per loaded motion).
            for _name, _values in (
                ("_motion_object_in_contact_left_valid", _motion_object_in_contact_left_valid),
                ("_motion_object_in_contact_right_valid", _motion_object_in_contact_right_valid),
                ("_motion_object_contact_mask_left_valid", _motion_object_contact_mask_left_valid),
                ("_motion_object_contact_mask_right_valid", _motion_object_contact_mask_right_valid),
                ("_motion_reference_contact_state_valid", _motion_reference_contact_state_valid),
                ("_motion_contact_label_native_authoritative", _motion_contact_label_native_authoritative),
            ):
                if len(_values) == len(motions):
                    setattr(self, _name, torch.cat(_values, dim=0).float().to(self._device))
            if has_object_relation_vector_left:
                self._motion_object_relation_vector_left = (
                    torch.cat(_motion_object_relation_vector_left, dim=0).float().to(self._device)
                )
            if has_object_relation_vector_right:
                self._motion_object_relation_vector_right = (
                    torch.cat(_motion_object_relation_vector_right, dim=0).float().to(self._device)
                )
            if has_object_relation_mask_left:
                self._motion_object_relation_mask_left = (
                    torch.cat(_motion_object_relation_mask_left, dim=0).float().to(self._device)
                )
            if has_object_relation_mask_right:
                self._motion_object_relation_mask_right = (
                    torch.cat(_motion_object_relation_mask_right, dim=0).float().to(self._device)
                )
            wrist_channels = (
                (
                    has_object_wrist_relation_vector_left,
                    "_motion_object_wrist_relation_vector_left",
                    _motion_object_wrist_relation_vector_left,
                ),
                (
                    has_object_wrist_relation_vector_right,
                    "_motion_object_wrist_relation_vector_right",
                    _motion_object_wrist_relation_vector_right,
                ),
                (
                    has_object_wrist_relation_mask_left,
                    "_motion_object_wrist_relation_mask_left",
                    _motion_object_wrist_relation_mask_left,
                ),
                (
                    has_object_wrist_relation_mask_right,
                    "_motion_object_wrist_relation_mask_right",
                    _motion_object_wrist_relation_mask_right,
                ),
                (
                    has_object_wrist_surface_distance_left,
                    "_motion_object_wrist_surface_distance_left",
                    _motion_object_wrist_surface_distance_left,
                ),
                (
                    has_object_wrist_surface_distance_right,
                    "_motion_object_wrist_surface_distance_right",
                    _motion_object_wrist_surface_distance_right,
                ),
            )
            for present, name, values in wrist_channels:
                if present:
                    setattr(self, name, torch.cat(values, dim=0).float().to(self._device))
            source_relation_channels = (
                (
                    has_rp2_source_relation_vector,
                    "_motion_rp2_source_relation_vector",
                    _motion_rp2_source_relation_vector,
                ),
                (
                    has_rp2_source_relation_mask,
                    "_motion_rp2_source_relation_mask",
                    _motion_rp2_source_relation_mask,
                ),
                (
                    has_rp2_source_surface_relation_vector,
                    "_motion_rp2_source_surface_relation_vector",
                    _motion_rp2_source_surface_relation_vector,
                ),
                (
                    has_rp2_source_surface_relation_mask,
                    "_motion_rp2_source_surface_relation_mask",
                    _motion_rp2_source_surface_relation_mask,
                ),
                (
                    has_rp2_source_surface_point_local,
                    "_motion_rp2_source_surface_point_local",
                    _motion_rp2_source_surface_point_local,
                ),
                (
                    has_rp2_source_surface_signed_distance,
                    "_motion_rp2_source_surface_signed_distance",
                    _motion_rp2_source_surface_signed_distance,
                ),
                (
                    has_rp2_source_surface_penetration_mask,
                    "_motion_rp2_source_surface_penetration_mask",
                    _motion_rp2_source_surface_penetration_mask,
                ),
                (
                    has_rp2_source_contact_probability,
                    "_motion_rp2_source_contact_probability",
                    _motion_rp2_source_contact_probability,
                ),
                (
                    has_rp2_source_contact_mask,
                    "_motion_rp2_source_contact_mask",
                    _motion_rp2_source_contact_mask,
                ),
                (
                    has_rp2_full_scene_source_relation_vector,
                    "_motion_rp2_full_scene_source_relation_vector",
                    _motion_rp2_full_scene_source_relation_vector,
                ),
                (
                    has_rp2_full_scene_source_relation_mask,
                    "_motion_rp2_full_scene_source_relation_mask",
                    _motion_rp2_full_scene_source_relation_mask,
                ),
                (
                    has_rp2_full_scene_source_surface_relation_vector,
                    "_motion_rp2_full_scene_source_surface_relation_vector",
                    _motion_rp2_full_scene_source_surface_relation_vector,
                ),
                (
                    has_rp2_full_scene_source_surface_relation_mask,
                    "_motion_rp2_full_scene_source_surface_relation_mask",
                    _motion_rp2_full_scene_source_surface_relation_mask,
                ),
                (
                    has_rp2_full_scene_source_surface_point_local,
                    "_motion_rp2_full_scene_source_surface_point_local",
                    _motion_rp2_full_scene_source_surface_point_local,
                ),
                (
                    has_rp2_full_scene_source_surface_signed_distance,
                    "_motion_rp2_full_scene_source_surface_signed_distance",
                    _motion_rp2_full_scene_source_surface_signed_distance,
                ),
                (
                    has_rp2_full_scene_source_surface_penetration_mask,
                    "_motion_rp2_full_scene_source_surface_penetration_mask",
                    _motion_rp2_full_scene_source_surface_penetration_mask,
                ),
                (
                    has_rp2_full_scene_source_contact_probability,
                    "_motion_rp2_full_scene_source_contact_probability",
                    _motion_rp2_full_scene_source_contact_probability,
                ),
                (
                    has_rp2_full_scene_source_contact_mask,
                    "_motion_rp2_full_scene_source_contact_mask",
                    _motion_rp2_full_scene_source_contact_mask,
                ),
                (
                    has_rp2_source_root_pos,
                    "_motion_rp2_source_root_pos",
                    _motion_rp2_source_root_pos,
                ),
                (
                    has_rp2_source_root_heading_xy,
                    "_motion_rp2_source_root_heading_xy",
                    _motion_rp2_source_root_heading_xy,
                ),
            )
            for present, name, values in source_relation_channels:
                if present:
                    setattr(self, name, torch.cat(values, dim=0).float().to(self._device))
            if has_reference_contact_state:
                self._motion_reference_contact_state = torch.cat(
                    _motion_reference_contact_state, dim=0
                ).to(device=self._device, dtype=torch.int8)
            if has_rp2_retarget_initialization:
                retarget_init_channels = (
                    ("_motion_rp2_retarget_init_root_pos", _motion_rp2_retarget_init_root_pos),
                    (
                        "_motion_rp2_retarget_init_root_quat_wxyz",
                        _motion_rp2_retarget_init_root_quat_wxyz,
                    ),
                    (
                        "_motion_rp2_retarget_init_root_lin_vel",
                        _motion_rp2_retarget_init_root_lin_vel,
                    ),
                    (
                        "_motion_rp2_retarget_init_root_ang_vel",
                        _motion_rp2_retarget_init_root_ang_vel,
                    ),
                    ("_motion_rp2_retarget_init_dof", _motion_rp2_retarget_init_dof),
                    ("_motion_rp2_retarget_init_dof_vel", _motion_rp2_retarget_init_dof_vel),
                )
                for name, values in retarget_init_channels:
                    setattr(self, name, torch.cat(values, dim=0).float().to(self._device))
                self._motion_rp2_retarget_init_valid = torch.cat(
                    _motion_rp2_retarget_init_valid,
                    dim=0,
                ).to(device=self._device, dtype=torch.bool)
            if len(_motion_hand_action_left) > 0:
                self._motion_hand_action_left = (
                    torch.from_numpy(np.concatenate(_motion_hand_action_left, axis=0))
                    .float()
                    .to(self._device)
                )
            if len(_motion_hand_action_right) > 0:
                self._motion_hand_action_right = (
                    torch.from_numpy(np.concatenate(_motion_hand_action_right, axis=0))
                    .float()
                    .to(self._device)
                )
        self._motion_dt = torch.tensor(_motion_dt, device=self._device, dtype=torch.float32)

        # Compute object velocities from position/quaternion using finite differences
        if self.object_data is not None and hasattr(self, "_motion_object_root_pos"):
            self._compute_object_velocities(_motion_num_frames, _motion_dt)
        self._motion_num_frames = torch.tensor(_motion_num_frames, device=self._device)

        if self.has_action:
            self._motion_actions = torch.cat(_motion_actions, dim=0).float().to(self._device)
        self._num_motions = len(motions)

        self.body_pos_w = (
            torch.cat([m.global_translation for m in motions], dim=0).float().to(self._device)
        )
        self.body_quat_w = (
            torch.cat([m.global_rotation for m in motions], dim=0).float().to(self._device)
        )
        self.body_pos_b = (
            torch.cat([m.local_rotation for m in motions], dim=0).float().to(self._device)
        )
        self.root_linv_vel_w = (
            torch.cat([m.global_root_velocity for m in motions], dim=0).float().to(self._device)
        )
        self.root_ang_vel_w = (
            torch.cat([m.global_root_angular_velocity for m in motions], dim=0)
            .float()
            .to(self._device)
        )
        self.body_ang_vel_w = (
            torch.cat([m.global_angular_velocity for m in motions], dim=0).float().to(self._device)
        )
        self.body_lin_vel_w = (
            torch.cat([m.global_velocity for m in motions], dim=0).float().to(self._device)
        )
        self.dof_vel = torch.cat([m.dof_vels for m in motions], dim=0).float().to(self._device)
        self.feet_l = torch.cat([m.feet_l for m in motions], dim=0).float().to(self._device)
        self.feet_r = torch.cat([m.feet_r for m in motions], dim=0).float().to(self._device)

        # if "global_translation_extend" in motions[0].__dict__:
        #     self.body_pos_t_w = torch.cat([m.global_translation_extend for m in motions], dim=0).float().to(self._device)  # noqa: E501
        #     self.body_quat_t_w = torch.cat([m.global_rotation_extend for m in motions], dim=0).float().to(self._device)  # noqa: E501
        #     self.body_lin_vel_t_w = torch.cat([m.global_velocity_extend for m in motions], dim=0).float().to(self._device)  # noqa: E501
        #     self.body_ang_vel_t_w = torch.cat([m.global_angular_velocity_extend for m in motions], dim=0).float().to(self._device)  # noqa: E501
        #     self.feet_l = torch.cat([m.feet_l for m in motions], dim=0).float().to(self._device)
        #     self.feet_r = torch.cat([m.feet_r for m in motions], dim=0).float().to(self._device)

        if "dof_pos" in motions[0].__dict__:
            self.dof_pos = torch.cat([m.dof_pos for m in motions], dim=0).float().to(self._device)

        # Store hand DOF positions if available (for 43-DOF motion)
        if "hand_dof_pos" in motions[0].__dict__:
            self.hand_dof_pos = (
                torch.cat([m.hand_dof_pos for m in motions], dim=0).float().to(self._device)
            )
        else:
            self.hand_dof_pos = None

        lengths = self._motion_num_frames
        lengths_shifted = lengths.roll(1)
        lengths_shifted[0] = 0
        self.length_starts = lengths_shifted.cumsum(0)

        # Zero out initial root XY so all motions start at origin
        if self.m_cfg.get("zero_root_xy", False):
            print(  # noqa: T201
                f"[zero_root_xy] Zeroing initial root XY for {len(motions)} motions"
            )  # noqa: RUF100, T201
            for i in range(len(motions)):
                start = self.length_starts[i]
                end = start + self._motion_num_frames[i]
                init_xy = self.body_pos_w[start, 0, :2].clone()  # root body, XY
                print(  # noqa: T201
                    f"  Motion {i}: init_xy=[{init_xy[0]:.3f}, {init_xy[1]:.3f}], frames={self._motion_num_frames[i]}"  # noqa: E501
                )
                self.body_pos_w[start:end, :, :2] -= init_xy
                if (
                    hasattr(self, "_motion_object_root_pos")
                    and self._motion_object_root_pos is not None
                ):
                    self._motion_object_root_pos[start:end, :, :2] -= init_xy
                if hasattr(self, "_motion_active_articulation_root_pos"):
                    self._motion_active_articulation_root_pos[start:end, :2] -= init_xy

        self.motion_ids = torch.arange(len(motions), dtype=torch.long, device=self._device)

        # Build per-motion BPS tensor (static shape encoding, no frame dependence)
        self._motion_object_name_candidates = _motion_object_name_candidates
        if self._bps_dir is not None and self._bps_available_stems():
            import numpy as _np  # noqa: PLC0415

            from gear_sonic.utils.motion_lib.bps_resolution import (  # noqa: PLC0415
                resolve_object_bps_stem,
            )

            # Packs store per-object files named <motion_key>__<object_name>;
            # the bare-key lookup only matches legacy single-file packs. The
            # object-name resolution is opt-in so in-flight runs keep their
            # observation distribution across autoresume legs.
            resolve_by_object_name = bool(
                self.m_cfg.get("bps_object_name_resolution", False)
            )
            bps_rows = []
            unresolved_keys = []
            resolvable_only = 0
            for motion_idx, key in enumerate(self.curr_motion_keys):
                stem = os.path.basename(key)
                candidates = (
                    _motion_object_name_candidates[motion_idx]
                    if motion_idx < len(_motion_object_name_candidates)
                    else []
                )
                resolved = resolve_object_bps_stem(key, candidates, self._bps_stems)
                if resolved == stem or (resolve_by_object_name and resolved is not None):
                    vec = self._bps_vector(resolved)
                else:
                    if resolved is not None:
                        resolvable_only += 1
                    unresolved_keys.append(key)
                    vec = None  # zero row of the vector dim, filled once the dim is known
                bps_rows.append(vec)
            loaded = next((v for v in bps_rows if v is not None), None)
            row_dim = self._bps_dim if loaded is None else int(loaded.shape[0])   # np.stack rejects mixed shapes
            bps_rows = [
                _np.zeros(row_dim, dtype=_np.float32) if v is None else v for v in bps_rows
            ]
            matched = len(bps_rows) - len(unresolved_keys)
            logger.info(
                f"[BPS] Matched {matched}/{len(bps_rows)} motions to BPS rows "
                f"(object-name resolution {'on' if resolve_by_object_name else 'off'})"
            )
            if resolve_by_object_name and unresolved_keys:
                raise ValueError(
                    "[BPS] bps_object_name_resolution=true but "
                    f"{len(unresolved_keys)} motion(s) have no matching BPS file; "
                    f"first: {unresolved_keys[:5]}"
                )
            if not resolve_by_object_name and resolvable_only > 0:
                logger.warning(
                    f"[BPS] {resolvable_only} motion(s) only match per-object "
                    "<motion_key>__<object_name> BPS files and currently fall back "
                    "to ZERO shape encodings. Set motion_lib_cfg."
                    "bps_object_name_resolution=true to use them."
                )
            self._motion_object_bps = torch.tensor(
                _np.stack(bps_rows, axis=0), dtype=torch.float32, device=self._device
            )
        else:
            self._motion_object_bps = None

        motion_has_smpl = [
            self.curr_motion_keys[i] in self.smpl_data_keys for i in range(len(motions))
        ]
        self.motion_has_smpl = torch.tensor(motion_has_smpl, dtype=torch.bool, device=self._device)

        motion_has_soma = [
            self.curr_motion_keys[i] in self.soma_data_keys for i in range(len(motions))
        ]
        self.motion_has_soma = torch.tensor(motion_has_soma, dtype=torch.bool, device=self._device)

        motion_has_object = [
            self.curr_motion_keys[i] in self.object_data_keys for i in range(len(motions))
        ]
        self.motion_has_object = torch.tensor(
            motion_has_object, dtype=torch.bool, device=self._device
        )

        motion = motions[0]  # noqa: F841
        self.num_bodies = self.num_joints

        num_motions = self.num_motions()
        total_len = self.get_total_length()

        if self.use_adaptive_sampling:
            self._finish_deferred_adaptive_init()
            if getattr(self, "_adaptive_sequence_update_pending", False):
                # the pre-selection update skipped for a deferred explicit / sequential selection
                self._adaptive_sequence_update_pending = False
                self.update_adaptive_sampling_motion_sequences()
                self._sampling_batch_prob = (
                    self._sampling_prob[self._curr_motion_ids]
                    / self._sampling_prob[self._curr_motion_ids].sum()
                )
            self.update_adaptive_sampling_motion_frames()

        logger.info(
            f"Loaded {num_motions:d} motions with a total length of {total_len:.3f}s and {self.body_pos_w.shape[0]} frames."  # noqa: E501
        )

        del (
            motions,
            _motion_lengths,
            _motion_fps,
            _motion_dt,
            _motion_num_frames,
            _motion_bodies,
            _motion_aa,
            _motion_actions,
            _motion_smpl_poses,
            _motion_smpl_joints,
            _motion_smpl_transl,
            _motion_object_root_pos,
            _motion_object_root_quat,
        )
        gc.collect()
        torch.cuda.empty_cache()

        if "mujoco_to_isaaclab_body" in self.m_cfg.keys():  # noqa: SIM118
            self.dof_pos = self.dof_pos[:, self.m_cfg.mujoco_to_isaaclab_dof]
            self.dof_vel = self.dof_vel[:, self.m_cfg.mujoco_to_isaaclab_dof]

            # RP2 reset-only poses come from the same retargeted G1 motion
            # files as dof_pos, so they require the identical joint-order
            # conversion before being written into the IsaacLab articulation.
            if hasattr(self, "_motion_rp2_retarget_init_dof"):
                self._motion_rp2_retarget_init_dof = self._motion_rp2_retarget_init_dof[
                    :, self.m_cfg.mujoco_to_isaaclab_dof
                ]
                self._motion_rp2_retarget_init_dof_vel = (
                    self._motion_rp2_retarget_init_dof_vel[
                        :, self.m_cfg.mujoco_to_isaaclab_dof
                    ]
                )

            # Keep full body data (all bodies, IsaacLab order) before slicing
            self.num_bodies_full = len(self.m_cfg.mujoco_to_isaaclab_body)
            self.body_pos_w_full = self.body_pos_w[:, self.m_cfg.mujoco_to_isaaclab_body]
            self.body_quat_w_full = rotations.xyzw_to_wxyz(
                self.body_quat_w[:, self.m_cfg.mujoco_to_isaaclab_body]
            )
            self.body_lin_vel_w_full = self.body_lin_vel_w[:, self.m_cfg.mujoco_to_isaaclab_body]
            self.body_ang_vel_w_full = self.body_ang_vel_w[:, self.m_cfg.mujoco_to_isaaclab_body]

            # The tracked-body subset (body_indexes) is NOT materialized as duplicate
            # full-length tensors; consumers gather rows from *_full and slice via
            # _tracked_body_rows (exact same values).
            del self.body_pos_w, self.body_quat_w, self.body_lin_vel_w, self.body_ang_vel_w
            assert (
                self.m_cfg.get("anchor_body_idx_full", 0) == 0 and self.body_indexes[0] == 0
            ), "The anchor body has to be 0; otherwise will cause issues in the sliced body_indexes data's anchor."
        else:
            # No body reordering — full body data is the same as the original data
            self.body_pos_w_full = self.body_pos_w
            self.body_quat_w_full = self.body_quat_w
            self.body_lin_vel_w_full = self.body_lin_vel_w
            self.body_ang_vel_w_full = self.body_ang_vel_w
            self.num_bodies_full = self.body_pos_w.shape[2]
            # Drop the duplicate names so each channel is stored (and cached) once.
            del self.body_pos_w, self.body_quat_w, self.body_lin_vel_w, self.body_ang_vel_w

        # All load-time fp32 math (reorder, zero_root_xy, finite differences) is
        # done: store the eligible channels fp16 when configured. The tensor cache
        # captures the post-conversion state (bundles restore already compressed).
        self._compress_reference_channels()

    def foot_detect(self, positions, vel_thres, height_thresh):
        fid_l = self.m_cfg.get("left_foot_body_idx", [6])
        fid_r = self.m_cfg.get("right_foot_body_idx", [12])
        # fid_l, fid_r = [6], [12]
        velfactor = torch.tensor(
            [vel_thres] * len(fid_l), device=positions.device, dtype=positions.dtype
        )
        heightfactor = torch.tensor(
            [height_thresh] * len(fid_l), device=positions.device, dtype=positions.dtype
        )

        feet_l_xyz = (positions[1:, fid_l] - positions[:-1, fid_l]) ** 2
        feet_l_xyz = torch.cat([feet_l_xyz, feet_l_xyz[[-1]]], dim=0)
        feet_l_h = positions[:, fid_l, 2]
        feet_l = torch.logical_and(
            (feet_l_xyz.sum(dim=-1)) < velfactor, feet_l_h < heightfactor
        ).float()
        # feet_l = ((feet_l_x + feet_l_y + feet_l_z) < velfactor).float()

        feet_r_xyz = (positions[1:, fid_r] - positions[:-1, fid_r]) ** 2
        feet_r_xyz = torch.cat([feet_r_xyz, feet_r_xyz[[-1]]], dim=0)
        feet_r_h = positions[:, fid_r, 2]
        feet_r = torch.logical_and(
            (feet_r_xyz.sum(dim=-1)) < velfactor, feet_r_h < heightfactor
        ).float()
        # feet_r = (((feet_r_x + feet_r_y + feet_r_z) < velfactor)).float()
        return feet_l, feet_r

    def _compute_object_velocities(self, motion_num_frames, motion_dt):
        """Compute object linear and angular velocities from position and quaternion data.
        Uses finite differences: v = (p_{t+1} - p_t) / dt
        Handles motion boundaries properly (first frame uses forward difference).
        """  # noqa: D205
        total_frames = self._motion_object_root_pos.shape[0]
        num_objects = self._motion_object_root_pos.shape[1]

        # Initialize velocity tensors
        self._motion_object_lin_vel = torch.zeros_like(self._motion_object_root_pos)
        self._motion_object_ang_vel = torch.zeros(
            total_frames, num_objects, 3, device=self._device, dtype=torch.float32
        )

        # Compute length_starts for indexing (cumsum of frame counts)
        num_frames_tensor = torch.tensor(motion_num_frames, device=self._device)
        lengths_shifted = num_frames_tensor.roll(1)
        lengths_shifted[0] = 0
        length_starts = lengths_shifted.cumsum(0)

        # Compute velocities for each motion sequence separately
        for i, (start, num_frames, dt) in enumerate(  # noqa: B007
            zip(length_starts, motion_num_frames, motion_dt)  # noqa: B905
        ):
            start = start.item()  # noqa: PLW2901
            end = start + num_frames

            if num_frames < 2:
                continue  # Cannot compute velocity with less than 2 frames

            # Get position and quaternion for this motion
            pos = self._motion_object_root_pos[start:end]  # (T, N_obj, 3)
            quat = self._motion_object_root_quat[start:end]  # (T, N_obj, 4)

            # Compute linear velocity: v = (p_{t+1} - p_t) / dt
            lin_vel = (pos[1:] - pos[:-1]) / dt
            # First frame uses same velocity as second frame
            lin_vel = torch.cat([lin_vel[:1], lin_vel], dim=0)
            self._motion_object_lin_vel[start:end] = lin_vel

            # Compute angular velocity from quaternion difference using same method as robot body
            # ω = axis * angle / dt (same as _compute_angular_velocity in torch_humanoid_batch.py)
            q_curr = quat[:-1]  # (T-1, N_obj, 4)
            q_next = quat[1:]  # (T-1, N_obj, 4)

            # Compute quaternion difference: q_diff = q_next * q_curr^{-1}
            # Using quat_mul_norm and quat_inverse (w_last=False for xyzw format)
            diff_quat = rotations.quat_mul_norm(
                q_next, rotations.quat_inverse(q_curr, w_last=False), w_last=False
            )

            # Extract angle and axis from quaternion difference
            diff_angle, diff_axis = rotations.quat_angle_axis(diff_quat, w_last=False)

            # Angular velocity: ω = axis * angle / dt
            ang_vel = diff_axis * diff_angle.unsqueeze(-1) / dt
            # First frame uses same velocity as second frame
            ang_vel = torch.cat([ang_vel[:1], ang_vel], dim=0)
            self._motion_object_ang_vel[start:end] = ang_vel

        logger.info(f"Computed object velocities for {len(motion_num_frames)} motions")

    def fix_trans_height(self, pose_aa, trans, fix_height_mode):
        if fix_height_mode == FixHeightMode.no_fix:
            return trans, 0
        with torch.no_grad():

            mesh_obj = self.mesh_parsers.mesh_fk(pose_aa[None, :1], trans[None, :1])
            height_diff = np.asarray(mesh_obj.vertices)[..., 2].min()
            trans[..., 2] -= height_diff

            return trans, height_diff

    def effective_payload_mode(self):
        """The payload-shaping booleans this library ACTUALLY applies (cache hook; no gate).

        ``paired_reference_clock``: the constructor flag (identity input present:
        manifest file or in-memory roster) that selects the 50 Hz paired clock, the
        unknown ``-1`` hand-action padding, the required reference_contact_state
        and the robot/object pair validation. ``manifest_contact_tensors``: the
        left/right per-hand contact tensors are built even when no motion carries
        them (identity present, or a roster label ``joint_roster_manifest``). The
        shared tensor cache compares this mode between a bundle and the live
        library; a mismatch is an ordinary miss (native fill), never an error.
        """
        paired = bool(getattr(self, "require_paired_reference_clock", False))
        return {
            "paired_reference_clock": paired,
            "manifest_contact_tensors": bool(
                paired or self.m_cfg.get("joint_roster_manifest")
            ),
        }

    def _validate_paired_runtime_payload(self, curr_file, curr_object_data, curr_id):
        """Paired-clock checks of one robot/object payload pair (identity input: file or memory).

        The shared training ABI (``validate_joint_training_payload_pair`` with the
        roster's source id), the roster clock (``source_sample_fps_hz`` of the
        record, or the robot payload fps for memory records without meta) and the
        robot/object FPS agreement. Returns the object fps. Every error is the
        existing one; nothing is defaulted.
        """
        motion_key = self.curr_motion_keys[curr_id]
        expected_source = self.joint_source_motion_sources[curr_id]
        payload_contract = validate_joint_training_payload_pair(
            curr_file,
            curr_object_data,
            label=f"joint runtime payload {motion_key}",
            expected_source_id=expected_source,
        )
        expected_fps = self.joint_source_motion_sample_fps[curr_id]
        if not np.isclose(
            float(payload_contract["fps"]),
            expected_fps,
            rtol=0.0,
            atol=1.0e-9,
        ):
            raise ValueError(
                "Joint runtime payload clock differs from "
                f"the roster for {motion_key}: "
                f"{payload_contract['fps']} != "
                f"{expected_fps}"
            )
        if "fps" not in curr_object_data:
            raise KeyError(
                "Joint multi-source object reference "
                "is missing fps"
            )
        original_fps = float(curr_object_data["fps"])
        robot_fps = float(curr_file["fps"])
        if (
            not np.isfinite(original_fps)
            or original_fps <= 0.0
            or not np.isfinite(robot_fps)
            or robot_fps <= 0.0
        ):
            raise ValueError(
                "Joint multi-source paired reference has "
                f"invalid FPS {robot_fps} / {original_fps}"
            )
        if abs(original_fps - robot_fps) > 1.0e-9:
            raise ValueError(
                "Joint multi-source robot/object FPS "
                f"mismatch: {robot_fps} vs {original_fps}"
            )
        return original_fps

    def _run_load_jobs(self, jobs, *, queue, use_threads, label):
        """Run the load chunks ``jobs`` (chunk 0 inline as pid 0, the others as workers) and return the merged
        {motion index: (file, motion)} dictionary. Threads when fork workers are off (load_jobs.run_in_threads);
        otherwise the historical process workers: start, inline chunk 0, gather, join."""
        if use_threads:
            try:
                return load_jobs.run_in_threads(self.load_motion_with_skeleton, jobs, queue=queue)
            except RuntimeError as exc:
                # The threaded run is an accelerator: a failed chunk is logged and the SAME chunks are loaded serially
                # on this thread (the historical single-job path); a failure that persists there raises as before.
                logger.warning(f"threaded motion load failed ({exc}); loading the same chunks serially")
                res_serial = {}
                for job in jobs:
                    res_serial.update(self.load_motion_with_skeleton(*job, None, 0))
                return res_serial
        workers = []
        for i in range(1, len(jobs)):
            worker = mp.Process(target=self.load_motion_with_skeleton, args=(*jobs[i], queue, i))
            worker.start()
            workers.append(worker)
        res_acc = {}
        res_acc.update(self.load_motion_with_skeleton(*jobs[0], None, 0))
        for _ in progress.track(range(len(jobs) - 1), label):
            res_acc.update(queue.get())
        for worker in workers:
            worker.join()
            worker.close()
        return res_acc

    def load_motion_with_skeleton(
        self,
        ids,
        motion_data_list,
        smpl_data_list,
        object_data_list,
        soma_data_list,
        fix_height,
        target_heading,  # noqa: ARG002
        max_len,
        is_evaluation,
        queue,
        pid,
    ):
        # loading motion with the specified skeleton. Perfoming forward kinematics to get the joint positions
        res = {}

        if pid == 0:  # noqa: SIM108
            pbar = progress.track(range(len(ids)), description="Loading motions...")
        else:
            pbar = range(len(ids))

        for f in pbar:

            curr_id = ids[f]  # id for this datasample

            curr_file = motion_data_list[f]
            if "path" in curr_file:
                curr_file, *_ = joblib.load(
                    curr_file["path"]
                ).values()  # First value since it's a single item dictionary

            seq_len = curr_file["root_trans_offset"].shape[0]
            if max_len == -1 or seq_len < max_len:
                start, end = 0, seq_len
            else:
                start = random.randint(0, seq_len - max_len)
                end = start + max_len

            trans = to_torch(curr_file["root_trans_offset"]).clone()[start:end]
            pose_aa = to_torch(curr_file["pose_aa"][start:end]).clone()

            # import ipdb; ipdb.set_trace()
            if "action" in curr_file.keys():  # noqa: SIM118
                self.has_action = True

            if "fps" not in curr_file.keys():  # noqa: SIM118
                if self.require_paired_reference_clock:
                    raise KeyError(
                        "Joint multi-source robot reference is missing fps"
                    )
                curr_file["fps"] = 30.0
            if (
                not np.isfinite(float(curr_file["fps"]))
                or float(curr_file["fps"]) <= 0.0
            ):
                raise ValueError(
                    f"Robot reference has invalid fps {curr_file['fps']}"
                )
            dt = 1 / curr_file["fps"]  # noqa: F841

            B, J, N = pose_aa.shape
            freeze_frame_aug, freeze_idx = False, 0

            # self.m_cfg.freeze_frame_aug=True; is_evaluation=False; self.m_cfg.freeze_frame_prob=1
            # Debugging, force freeze frame augmentation

            if not is_evaluation and self.m_cfg.get("freeze_frame_aug", False):
                freeze_prob = self.m_cfg.get("freeze_frame_prob", 0.1)
                if np.random.random() < freeze_prob:  # noqa: NPY002
                    # Freeze the sequence at a random index
                    freeze_frame_aug = True
                    freeze_idx = np.random.randint(0, B)  # noqa: NPY002
                    # Repeat the frozen frame for all subsequent frames
                    pose_aa[freeze_idx:] = pose_aa[freeze_idx : freeze_idx + 1].clone()
                    trans[freeze_idx:] = trans[freeze_idx : freeze_idx + 1].clone()

            if not is_evaluation and self.m_cfg.get("randomize_heading", False):
                # ZL: this randomization is not combatiable with SMPL
                random_rot = np.zeros(3)
                random_rot[2] = np.pi * (2 * np.random.random() - 1.0)  # noqa: NPY002
                random_heading_rot = transform.Rotation.from_euler("xyz", random_rot)
                pose_aa = pose_aa.reshape(B, -1)
                pose_aa[:, :3] = torch.tensor(
                    (
                        random_heading_rot * transform.Rotation.from_rotvec(pose_aa[:, :3])
                    ).as_rotvec()
                )
                trans = torch.matmul(
                    trans, torch.from_numpy(random_heading_rot.as_matrix().T).float()
                )
                pose_aa = pose_aa.reshape(B, J, N)

            # self.cat_upper_body_poses_prob of the time, randomize the upper body poses and only for the motions are generated kinematically.  # noqa: E501
            randomize_upper_body_poses = (
                self.randomize_upper_body_poses
                and random.random() < self.cat_upper_body_poses_prob
                and (self._should_augment_upper_body(self.curr_motion_keys[curr_id]))
            )

            # only randomize the upper body poses if the non-navigation dataset is loaded
            if (
                randomize_upper_body_poses
                and self.res_non_nav_dataset is not None
                and len(self.res_non_nav_dataset) > 0
            ):
                # ZL: this randomization is not combatiable with SMPL, so only for kinematic generated data.
                # find the index for the upper body, skip the first index in pose_aa as it is the root.
                upper_body_indices = [
                    i for i in range(1, J) if i - 1 not in self.m_cfg.lower_joint_indices_mujoco
                ]
                # randomly select a motion from the non-navigation dataset
                selected_file, selected_motion = random.choice(
                    list(self.res_non_nav_dataset.values())
                )
                selected_pose_aa = to_torch(selected_file["pose_aa"])

                # Sample a matching slice from the selected motion
                # Use the same method as main code to determine sequence length
                selected_seq_len = selected_file["root_trans_offset"].shape[0]
                current_seq_len = pose_aa.shape[0]
                if selected_seq_len >= current_seq_len:
                    selected_start = random.randint(0, selected_seq_len - current_seq_len)
                    selected_end = selected_start + current_seq_len
                    selected_slice = selected_pose_aa[selected_start:selected_end]
                else:
                    # If selected motion is shorter, create a ping-pong (forward then backward) sequence
                    forward = selected_pose_aa
                    backward = selected_pose_aa.flip(dims=[0])  # reverse the sequence
                    # Concatenate forward and backward, excluding the last frame of forward to avoid duplication
                    extended = torch.cat([forward, backward[1:]], dim=0)

                    # If still not long enough, repeat the extended sequence
                    if extended.shape[0] < current_seq_len:
                        repeats = (current_seq_len + extended.shape[0] - 1) // extended.shape[
                            0
                        ]  # ceiling division
                        extended = extended.repeat(repeats, 1, 1)

                    selected_slice = extended[:current_seq_len]

                pose_aa[:, upper_body_indices] = selected_slice[:, upper_body_indices]

            # Wrist joint noise augmentation
            if (
                not is_evaluation
                and self.randomize_wrist_poses
                and random.random() < self.randomize_wrist_prob
            ):
                wrist_pose_aa_indices = [d + 1 for d in self.wrist_mujoco_dof_indices]
                noise = torch.randn(B, len(wrist_pose_aa_indices), N) * self.randomize_wrist_std
                pose_aa[:, wrist_pose_aa_indices] = pose_aa[:, wrist_pose_aa_indices] + noise

            if self.mesh_parsers is not None:
                trans, trans_fix = self.fix_trans_height(pose_aa, trans, fix_height_mode=fix_height)
                curr_motion = self.mesh_parsers.fk_batch(
                    pose_aa[None,],
                    trans[None,],
                    return_full=True,
                    fps=curr_file["fps"],
                    target_fps=self.target_fps,
                    interpolate_data=True,
                    use_parallel_fk=self.use_parallel_fk,
                )
                if self.smpl_data is not None:
                    curr_smpl_data = smpl_data_list[f]
                    if curr_smpl_data is not None:
                        if "path" in curr_smpl_data:
                            curr_smpl_data = joblib.load(curr_smpl_data["path"])

                        if curr_smpl_data["fps"] != self.target_fps:
                            smpl_pose = torch.tensor(curr_smpl_data["pose_aa"][start:end]).float()
                            smpl_pose[:, -6:] = 0.0
                            curr_motion["smpl_pose"] = self.mesh_parsers.interploate_pose(
                                None, smpl_pose[None,], curr_smpl_data["fps"], self.target_fps
                            )[1][0]
                        else:
                            smpl_pose = torch.tensor(curr_smpl_data["pose_aa"]).float()
                            smpl_pose[:, -6:] = 0.0
                            # new_seq_len = curr_motion['global_translation'].shape[1]
                            curr_motion["smpl_pose"] = smpl_pose
                        if "smpl_joints" in curr_smpl_data:
                            smpl_joints = torch.tensor(curr_smpl_data["smpl_joints"]).float()
                            curr_motion["smpl_joints"] = smpl_joints
                            if (
                                curr_motion["smpl_joints"].shape[0]
                                != curr_motion["global_translation"].shape[1]
                            ):
                                print(  # noqa: T201
                                    f"Length mismatch: smpl_joints={curr_motion['smpl_joints'].shape[0]}, "
                                    f"global_translation={curr_motion['global_translation'].shape[1]}"
                                )
                                print(smpl_data_list[f], motion_data_list[f])  # noqa: T201

                            assert (
                                curr_motion["smpl_joints"].shape[0]
                                == curr_motion["global_translation"].shape[1]
                            )
                        else:
                            num_frames = curr_motion["global_translation"].shape[1]
                            curr_motion["smpl_joints"] = torch.zeros(num_frames, 24, 3).to(
                                curr_motion["global_translation"]
                            )
                        if "transl" in curr_smpl_data:
                            transl = torch.tensor(curr_smpl_data["transl"]).float()
                            curr_motion["smpl_transl"] = transl
                            assert (
                                curr_motion["smpl_transl"].shape[0]
                                == curr_motion["global_translation"].shape[1]
                            )
                        else:
                            num_frames = curr_motion["global_translation"].shape[1]
                            curr_motion["smpl_transl"] = torch.zeros(num_frames, 3).to(
                                curr_motion["global_translation"]
                            )
                        assert (
                            curr_motion["smpl_pose"].shape[0]
                            == curr_motion["global_translation"].shape[1]
                        )

                        if freeze_frame_aug:
                            freeze_idx_new_fps = int(
                                freeze_idx * self.target_fps / curr_file["fps"]
                            )
                            curr_motion["smpl_pose"][freeze_idx_new_fps:] = curr_motion[
                                "smpl_pose"
                            ][freeze_idx_new_fps : freeze_idx_new_fps + 1].clone()
                            curr_motion["smpl_joints"][freeze_idx_new_fps:] = curr_motion[
                                "smpl_joints"
                            ][freeze_idx_new_fps : freeze_idx_new_fps + 1].clone()
                            curr_motion["smpl_transl"][freeze_idx_new_fps:] = curr_motion[
                                "smpl_transl"
                            ][freeze_idx_new_fps : freeze_idx_new_fps + 1].clone()
                    else:
                        curr_motion["smpl_pose"] = torch.zeros(
                            curr_motion["global_translation"].shape[1], 72
                        ).to(curr_motion["global_translation"])
                        curr_motion["smpl_joints"] = torch.zeros(
                            curr_motion["global_translation"].shape[1], 24, 3
                        ).to(curr_motion["global_translation"])
                        curr_motion["smpl_transl"] = torch.zeros(
                            curr_motion["global_translation"].shape[1], 3
                        ).to(curr_motion["global_translation"])
                    # print(curr_motion['smpl_pose'].shape, curr_motion['global_translation'].shape)

                # Load SOMA skeleton data if available
                if soma_data_list is not None:
                    curr_soma_data = soma_data_list[f]
                    if curr_soma_data is not None:
                        if "path" in curr_soma_data:
                            loaded = joblib.load(curr_soma_data["path"])
                            curr_soma_data, *_ = loaded.values()

                        num_frames = curr_motion["global_translation"].shape[1]
                        n_soma = self.num_soma_joints

                        # Resample SOMA data using the canonical interploate_pose formula
                        # to match robot frame count from fk_batch.
                        if (
                            self.require_paired_reference_clock
                            and "fps" not in curr_soma_data
                        ):
                            raise KeyError(
                                "Joint multi-source SOMA reference is missing fps"
                            )
                        soma_fps = float(
                            curr_soma_data.get("fps", self.target_fps)
                        )
                        if (
                            not np.isfinite(soma_fps)
                            or soma_fps <= 0.0
                        ):
                            raise ValueError(
                                f"SOMA reference has invalid fps {soma_fps}"
                            )
                        if (
                            self.require_paired_reference_clock
                            and abs(
                                soma_fps - float(curr_file["fps"])
                            )
                            > 1.0e-9
                        ):
                            raise ValueError(
                                "Joint multi-source robot/SOMA FPS mismatch: "
                                f"{curr_file['fps']} vs {soma_fps}"
                            )

                        if "soma_joints" in curr_soma_data:
                            soma_joints = torch.tensor(curr_soma_data["soma_joints"]).float()
                            soma_joints_orig_len = soma_joints.shape[0]
                            if soma_fps != self.target_fps:
                                soma_joints = self._resample_continuous_tensor(
                                    soma_joints, soma_fps, self.target_fps
                                )
                            curr_motion["soma_joints"] = soma_joints
                            assert soma_joints.shape[0] == num_frames, (
                                f"SOMA soma_joints length {soma_joints.shape[0]} != "
                                f"robot frames {num_frames} "
                                f"(soma_orig={soma_joints_orig_len} @ {soma_fps}fps, "
                                f"robot_orig={seq_len} @ {curr_file['fps']}fps, "
                                f"target_fps={self.target_fps})"
                            )
                        else:
                            curr_motion["soma_joints"] = torch.zeros(num_frames, n_soma, 3).to(
                                curr_motion["global_translation"]
                            )

                        if "soma_root_quat" in curr_soma_data:
                            soma_root_quat = torch.tensor(curr_soma_data["soma_root_quat"]).float()
                            if soma_fps != self.target_fps:
                                soma_root_quat = self._resample_continuous_tensor(
                                    soma_root_quat, soma_fps, self.target_fps
                                )
                                # Renormalize quaternions after linear interpolation
                                soma_root_quat = soma_root_quat / (
                                    soma_root_quat.norm(dim=-1, keepdim=True) + 1e-8
                                )
                            curr_motion["soma_root_quat"] = soma_root_quat
                            assert soma_root_quat.shape[0] == num_frames, (
                                f"SOMA soma_root_quat length {soma_root_quat.shape[0]} != "
                                f"robot frames {num_frames}"
                            )
                        else:
                            curr_motion["soma_root_quat"] = torch.zeros(num_frames, 4).to(
                                curr_motion["global_translation"]
                            )
                            curr_motion["soma_root_quat"][:, 0] = 1.0  # identity in wxyz

                        if "soma_transl" in curr_soma_data:
                            soma_transl = torch.tensor(curr_soma_data["soma_transl"]).float()
                            if soma_fps != self.target_fps:
                                soma_transl = self._resample_continuous_tensor(
                                    soma_transl, soma_fps, self.target_fps
                                )
                            curr_motion["soma_transl"] = soma_transl
                            assert soma_transl.shape[0] == num_frames, (
                                f"SOMA soma_transl length {soma_transl.shape[0]} != "
                                f"robot frames {num_frames}"
                            )
                        else:
                            curr_motion["soma_transl"] = torch.zeros(num_frames, 3).to(
                                curr_motion["global_translation"]
                            )

                        if freeze_frame_aug:
                            freeze_idx_new_fps = int(
                                freeze_idx * self.target_fps / curr_file["fps"]
                            )
                            for key in ("soma_joints", "soma_root_quat", "soma_transl"):
                                curr_motion[key][freeze_idx_new_fps:] = curr_motion[key][
                                    freeze_idx_new_fps : freeze_idx_new_fps + 1
                                ].clone()
                    else:
                        num_frames = curr_motion["global_translation"].shape[1]
                        n_soma = self.num_soma_joints
                        curr_motion["soma_joints"] = torch.zeros(num_frames, n_soma, 3).to(
                            curr_motion["global_translation"]
                        )
                        curr_motion["soma_root_quat"] = torch.zeros(num_frames, 4).to(
                            curr_motion["global_translation"]
                        )
                        curr_motion["soma_root_quat"][:, 0] = 1.0  # identity in wxyz
                        curr_motion["soma_transl"] = torch.zeros(num_frames, 3).to(
                            curr_motion["global_translation"]
                        )

                # Load object data if available
                if self.object_data is not None:
                    curr_object_data = object_data_list[f] if object_data_list is not None else None
                    if curr_object_data is not None:
                        if "path" in curr_object_data:
                            loaded = joblib.load(curr_object_data["path"])
                            curr_object_data, *_ = loaded.values()

                        if self.require_paired_reference_clock:
                            original_fps = self._validate_paired_runtime_payload(
                                curr_file, curr_object_data, int(curr_id)
                            )

                        num_frames = curr_motion["global_translation"].shape[1]
                        if not self.require_paired_reference_clock:
                            original_fps = curr_object_data.get(
                                "fps", curr_file["fps"]
                            )

                        # Retain declared object names so BPS rows can be
                        # resolved against per-object <motion_key>__<name> files.
                        name_candidates = []
                        declared_name = curr_object_data.get("object_name", None)
                        if declared_name is not None:
                            name_candidates.append(str(declared_name))
                        for name in curr_object_data.get("object_names", None) or []:
                            if str(name) not in name_candidates:
                                name_candidates.append(str(name))
                        curr_motion["object_name_candidates"] = name_candidates

                        if "root_pos" in curr_object_data:
                            curr_motion["object_root_pos"] = interpolate_translation_data(
                                curr_object_data["root_pos"][start:end],
                                source_fps=original_fps,
                                target_fps=self.target_fps,
                                num_frames=num_frames,
                                max_num_objects=self.max_num_objects,
                                pad_value=0.0,
                            )

                        if "root_quat" in curr_object_data:
                            curr_motion["object_root_quat"] = interpolate_quaternion_data(
                                curr_object_data["root_quat"][start:end],
                                source_fps=original_fps,
                                target_fps=self.target_fps,
                                num_frames=num_frames,
                                max_num_objects=self.max_num_objects,
                            )

                        # Compact per-clip provenance of the native binary hand labels, carried from the REAL
                        # object-record fields (OmniContact: joint_training_source_id="omnicontact",
                        # omnicontact_contact_info, object_contact_mask_provenance) plus an optional explicit flag.
                        # No scan, no new file; read by native_contact_label_authority in load_motions.
                        _prov = {}
                        for _k in (
                            "joint_training_source_id",
                            "object_contact_mask_provenance",
                            "contact_label_source",
                            "object_contact_mask_native_authoritative",
                        ):
                            if _k in curr_object_data:
                                _prov[_k] = curr_object_data[_k]
                        if "omnicontact_contact_info" in curr_object_data:
                            _prov["omnicontact_contact_info"] = True
                        if _prov:
                            curr_motion["_object_record_provenance"] = _prov
                        active_translation_keys = (
                            "active_articulation_root_pos",
                            "active_articulation_root_lin_vel",
                            "active_articulation_root_ang_vel",
                            "active_articulation_joint_pos",
                            "active_articulation_joint_vel",
                        )
                        for key in active_translation_keys:
                            if key not in curr_object_data:
                                continue
                            data = curr_object_data[key][start:end]
                            if data.ndim == 2:
                                data = data[:, None, :]
                            curr_motion[key] = interpolate_translation_data(
                                data,
                                source_fps=original_fps,
                                target_fps=self.target_fps,
                                num_frames=num_frames,
                                max_num_objects=1,
                                pad_value=0.0,
                            )[:, 0]

                        if "active_articulation_joint_valid_mask" in curr_object_data:
                            joint_valid_mask = curr_object_data[
                                "active_articulation_joint_valid_mask"
                            ][start:end]
                            if joint_valid_mask.ndim == 1:
                                joint_valid_mask = joint_valid_mask[:, None]
                            curr_motion["active_articulation_joint_valid_mask"] = (
                                interpolate_object_mask(
                                    joint_valid_mask,
                                    source_fps=original_fps,
                                    target_fps=self.target_fps,
                                    num_frames=num_frames,
                                    max_num_objects=joint_valid_mask.shape[1],
                                )
                            )

                        if "active_articulation_root_dynamic_mask" in curr_object_data:
                            root_dynamic_mask = curr_object_data[
                                "active_articulation_root_dynamic_mask"
                            ][start:end]
                            curr_motion["active_articulation_root_dynamic_mask"] = (
                                interpolate_object_mask(
                                    root_dynamic_mask,
                                    source_fps=original_fps,
                                    target_fps=self.target_fps,
                                    num_frames=num_frames,
                                    max_num_objects=1,
                                )
                            )

                        if "active_articulation_root_quat" in curr_object_data:
                            data = curr_object_data["active_articulation_root_quat"][start:end]
                            if data.ndim == 2:
                                data = data[:, None, :]
                            curr_motion["active_articulation_root_quat"] = (
                                interpolate_quaternion_data(
                                    data,
                                    source_fps=original_fps,
                                    target_fps=self.target_fps,
                                    num_frames=num_frames,
                                    max_num_objects=1,
                                )[:, 0]
                            )

                        for mask_key in (
                            "object_interaction_mask",
                            "object_motion_mask",
                            "object_contact_mask_left",
                            "object_contact_mask_right",
                            "object_relation_mask_left",
                            "object_relation_mask_right",
                            "object_wrist_relation_mask_left",
                            "object_wrist_relation_mask_right",
                        ):
                            if mask_key in curr_object_data:
                                curr_motion[mask_key] = interpolate_object_mask(
                                    curr_object_data[mask_key][start:end],
                                    source_fps=original_fps,
                                    target_fps=self.target_fps,
                                    num_frames=num_frames,
                                    max_num_objects=self.max_num_objects,
                                )

                        # Some source-only packs intentionally omit exact contact-point
                        # dictionaries. Preserve the standard command/reset ABI by
                        # deriving each hand's frame label from its validated per-object
                        # mask; contact-point data below still takes precedence when present.
                        for side in ("left", "right"):
                            mask_key = f"object_contact_mask_{side}"
                            contact_key = f"object_in_contact_{side}"
                            if mask_key in curr_motion and contact_key not in curr_motion:
                                curr_motion[contact_key] = (
                                    curr_motion[mask_key].reshape(num_frames, -1).amax(dim=-1)
                                )

                        for vector_key in (
                            "object_relation_vector_left",
                            "object_relation_vector_right",
                            "object_wrist_relation_vector_left",
                            "object_wrist_relation_vector_right",
                        ):
                            if vector_key in curr_object_data:
                                curr_motion[vector_key] = interpolate_translation_data(
                                    curr_object_data[vector_key][start:end],
                                    source_fps=original_fps,
                                    target_fps=self.target_fps,
                                    num_frames=num_frames,
                                    max_num_objects=self.max_num_objects,
                                    pad_value=0.0,
                                )

                        for distance_key in (
                            "object_wrist_surface_distance_left",
                            "object_wrist_surface_distance_right",
                        ):
                            if distance_key in curr_object_data:
                                distance = curr_object_data[distance_key][start:end]
                                if distance.ndim == 2:
                                    distance = distance[..., None]
                                curr_motion[distance_key] = interpolate_translation_data(
                                    distance,
                                    source_fps=original_fps,
                                    target_fps=self.target_fps,
                                    num_frames=num_frames,
                                    max_num_objects=self.max_num_objects,
                                    pad_value=0.0,
                                )[..., 0]

                        source_continuous_channel_keys = (
                            "rp2_source_relation_vector",
                            "rp2_source_surface_relation_vector",
                            "rp2_source_surface_point_local",
                            "rp2_source_surface_signed_distance",
                            "rp2_source_contact_probability",
                            "rp2_full_scene_source_relation_vector",
                            "rp2_full_scene_source_surface_relation_vector",
                            "rp2_full_scene_source_surface_point_local",
                            "rp2_full_scene_source_surface_signed_distance",
                            "rp2_full_scene_source_contact_probability",
                            "rp2_source_root_pos",
                            "rp2_source_root_heading_xy",
                        )
                        source_discrete_channel_keys = (
                            "rp2_source_relation_mask",
                            "rp2_source_surface_relation_mask",
                            "rp2_source_surface_penetration_mask",
                            "rp2_source_contact_mask",
                            "rp2_full_scene_source_relation_mask",
                            "rp2_full_scene_source_surface_relation_mask",
                            "rp2_full_scene_source_surface_penetration_mask",
                            "rp2_full_scene_source_contact_mask",
                        )
                        for key in source_continuous_channel_keys:
                            if key not in curr_object_data:
                                continue
                            data = torch.as_tensor(
                                curr_object_data[key][start:end], dtype=torch.float32
                            )
                            if original_fps != self.target_fps:
                                data = self._resample_continuous_tensor(
                                    data, original_fps, self.target_fps
                                )
                            if data.shape[0] != num_frames:
                                raise ValueError(
                                    f"RP2 source channel {key} has {data.shape[0]} frames, "
                                    f"expected {num_frames} after resampling"
                                )
                            curr_motion[key] = data
                        for key in source_discrete_channel_keys:
                            if key not in curr_object_data:
                                continue
                            data = torch.as_tensor(
                                curr_object_data[key][start:end],
                                dtype=torch.float32,
                            )
                            if (
                                data.shape[0] != num_frames
                                or original_fps != self.target_fps
                            ):
                                data = resample_discrete_frames(
                                    data,
                                    original_fps,
                                    self.target_fps,
                                    num_frames,
                                )
                            curr_motion[key] = (data > 0.5).float()

                        if "reference_contact_state" in curr_object_data:
                            contact_state = torch.as_tensor(
                                curr_object_data["reference_contact_state"][start:end],
                                dtype=torch.float32,
                            )
                            expected_tail = (
                                len(CONTACT_ENTITY_CLASSES),
                                len(CONTACT_BODY_GROUPS),
                            )
                            if contact_state.shape[1:] != expected_tail:
                                raise ValueError(
                                    "Expected reference_contact_state "
                                    f"[frames, {expected_tail[0]}, {expected_tail[1]}], "
                                    f"got {tuple(contact_state.shape)}"
                                )
                            if (
                                contact_state.shape[0] != num_frames
                                or original_fps != self.target_fps
                            ):
                                contact_state = resample_discrete_frames(
                                    contact_state,
                                    original_fps,
                                    self.target_fps,
                                    num_frames,
                                )
                            contact_state = contact_state.round()
                            unique_contact = set(
                                int(value)
                                for value in torch.unique(contact_state).cpu().tolist()
                            )
                            if not unique_contact.issubset({-1, 0, 1}):
                                raise ValueError(
                                    "reference_contact_state contains labels outside "
                                    f"{{-1, 0, +1}}: {sorted(unique_contact)}"
                                )
                            curr_motion["reference_contact_state"] = (
                                contact_state.to(dtype=torch.int8)
                            )

                        for key in (
                            "rp2_source_relation_robot_body_names",
                            "rp2_source_relation_soma_joint_names",
                            "rp2_full_scene_source_relation_robot_body_names",
                            "rp2_full_scene_source_relation_soma_joint_names",
                            "reference_contact_entity_classes",
                            "reference_contact_body_groups",
                        ):
                            if key in curr_object_data:
                                curr_motion[key] = [
                                    str(name) for name in curr_object_data[key]
                                ]

                        for key in (
                            "rp2_box29_target_object_index",
                            "rp2_box29_source_full_scene_object_count",
                        ):
                            if key in curr_object_data:
                                curr_motion[key] = int(curr_object_data[key])

                        retarget_init_linear_keys = (
                            "rp2_retarget_init_root_pos",
                            "rp2_retarget_init_root_lin_vel",
                            "rp2_retarget_init_root_ang_vel",
                            "rp2_retarget_init_dof",
                            "rp2_retarget_init_dof_vel",
                        )
                        for key in retarget_init_linear_keys:
                            if key not in curr_object_data:
                                continue
                            data = torch.as_tensor(
                                curr_object_data[key][start:end], dtype=torch.float32
                            )
                            if original_fps != self.target_fps:
                                data = self._resample_continuous_tensor(
                                    data, original_fps, self.target_fps
                                )
                            if data.shape[0] != num_frames:
                                raise ValueError(
                                    f"RP2 retarget initialization channel {key} has "
                                    f"{data.shape[0]} frames, expected {num_frames}"
                                )
                            curr_motion[key] = data

                        retarget_quat_key = "rp2_retarget_init_root_quat_wxyz"
                        if retarget_quat_key in curr_object_data:
                            data = curr_object_data[retarget_quat_key][start:end]
                            curr_motion[retarget_quat_key] = interpolate_quaternion_data(
                                data[:, None, :],
                                source_fps=original_fps,
                                target_fps=self.target_fps,
                                num_frames=num_frames,
                                max_num_objects=1,
                            )[:, 0]

                        # New RP2 packs store a dense, object-local surface target for
                        # each hand. Load it directly; legacy exact contact-point
                        # dictionaries below remain authoritative when populated.
                        for side in ("left", "right"):
                            center_key = f"object_contact_center_{side}"
                            if center_key not in curr_object_data:
                                continue
                            center = curr_object_data[center_key][start:end]
                            if center.ndim == 2:
                                center = center[:, None, :]
                            if center.ndim != 3 or center.shape[1:] != (1, 3):
                                raise ValueError(
                                    f"Expected {center_key} [frames, 3] or "
                                    f"[frames, 1, 3], got {tuple(center.shape)}"
                                )
                            curr_motion[center_key] = interpolate_translation_data(
                                center,
                                source_fps=original_fps,
                                target_fps=self.target_fps,
                                num_frames=num_frames,
                                max_num_objects=1,
                                pad_value=0.0,
                            )[:, 0]

                        # Exact points provide geometry, while validated dense masks own
                        # the contact phase whenever both representations are available.
                        for hand in ("left_hand", "right_hand"):
                            key = f"contact_points_{hand}"
                            side = hand.split("_")[0]  # "left" or "right"
                            if key in curr_object_data:
                                # Remap dict keys to [start:end] slice so contact frames
                                # align with the sliced root_pos/root_quat data
                                raw_dict = curr_object_data[key]
                                sliced_dict = {
                                    k - start: v for k, v in raw_dict.items() if start <= k < end
                                }
                                if sliced_dict:
                                    center, label = interpolate_contact_center(
                                        sliced_dict,
                                        source_fps=original_fps,
                                        target_fps=self.target_fps,
                                        num_frames=num_frames,
                                    )
                                    curr_motion[f"object_contact_center_{side}"] = center
                                    contact_key = f"object_in_contact_{side}"
                                    if contact_key not in curr_motion:
                                        curr_motion[contact_key] = label
                    else:
                        # Fill with zeros if no object data available
                        num_frames = curr_motion["global_translation"].shape[1]
                        curr_motion["object_root_pos"] = torch.zeros(
                            num_frames, self.max_num_objects, 3
                        ).to(curr_motion["global_translation"])
                        curr_motion["object_root_quat"] = torch.zeros(
                            num_frames, self.max_num_objects, 4
                        ).to(curr_motion["global_translation"])
                        curr_motion["object_root_quat"][
                            :, :, 0
                        ] = 1.0  # w=1 for identity quaternion

                curr_motion = easydict.EasyDict(
                    {
                        k: v.squeeze(dim=-1).squeeze(dim=0) if torch.is_tensor(v) else v
                        for k, v in curr_motion.items()
                    }
                )
                # add "action" to curr_motion
                if self.has_action:
                    curr_motion.action = to_torch(curr_file["action"]).clone()[start:end]

                # Extract hand DOFs if motion file has more than 29 DOFs
                hand_dof_count = self.m_cfg.get("hand_dof_count", 0)
                if hand_dof_count > 0 and "dof" in curr_file:
                    raw_dof = to_torch(curr_file["dof"]).clone()[start:end]
                    if raw_dof.shape[-1] > 29:
                        # Extract hand DOFs (indices 29 onwards) and interpolate to target FPS
                        hand_dof = raw_dof[:, 29 : 29 + hand_dof_count]
                        if curr_file["fps"] != self.target_fps:
                            # Simple linear interpolation for hand DOFs
                            num_target_frames = curr_motion["dof_pos"].shape[0]
                            hand_dof_interp = (
                                torch.nn.functional.interpolate(
                                    hand_dof.T.unsqueeze(0),  # (1, C, T)
                                    size=num_target_frames,
                                    mode="linear",
                                    align_corners=True,
                                )
                                .squeeze(0)
                                .T
                            )  # (T, C)
                            curr_motion.hand_dof_pos = hand_dof_interp
                        else:
                            curr_motion.hand_dof_pos = hand_dof

                if self.vid_smpl_pose is not None:  # for cross embodiment tracking
                    vid_smpl_pose = self.vid_smpl_pose[f]
                    vid_smpl_pose = self.mesh_parsers.interploate_pose(
                        None, vid_smpl_pose[None,], 30.0, self.target_fps
                    )[1][0]
                    if curr_motion["smpl_pose"].shape[0] < vid_smpl_pose.shape[0]:
                        for key in curr_motion.keys():  # noqa: SIM118
                            if isinstance(curr_motion[key], torch.Tensor):
                                curr_motion[key] = torch.cat(
                                    [
                                        curr_motion[key],
                                        torch.zeros(
                                            vid_smpl_pose.shape[0] - curr_motion[key].shape[0],
                                            *curr_motion[key].shape[1:],
                                        ).to(curr_motion[key]),
                                    ],
                                    dim=0,
                                )
                    curr_motion["smpl_pose"][:, :] = vid_smpl_pose[
                        : curr_motion["smpl_pose"].shape[0], :
                    ]
                feet_l, feet_r = self.foot_detect(curr_motion["global_translation"], 0.0005, 0.05)
                curr_motion["feet_l"] = feet_l
                curr_motion["feet_r"] = feet_r
                res[curr_id] = (curr_file, curr_motion)
            else:
                logger.error("No mesh parser found")

        if queue is not None:
            queue.put(res)
        else:
            return res

    def num_motions(self):
        return self._num_motions

    def get_total_length(self):
        return sum(self._motion_lengths)

    def get_motion_num_steps(self, motion_ids=None):
        if motion_ids is None:
            return (
                (self._motion_num_frames * self._sim_fps / self._motion_fps).floor().int()
            )  # don't use ceil as it will cause frames to be missed.
        else:
            return (
                (self._motion_num_frames[motion_ids] * self._sim_fps / self._motion_fps[motion_ids])
                .floor()
                .int()
            )

    def sample_time(self, motion_ids, truncate_time=None):
        n = len(motion_ids)  # noqa: F841
        phase = torch.rand(motion_ids.shape, device=self._device)
        motion_len = self._motion_lengths[motion_ids]
        if truncate_time is not None:
            assert truncate_time >= 0.0
            motion_len = torch.clamp(motion_len - truncate_time, min=0.0)

        motion_time = phase * motion_len
        return motion_time.to(self._device)

    def sample_time_steps(self, motion_ids, truncate_time=None):
        motion_time = self.sample_time(motion_ids, truncate_time)
        motion_time_steps = (motion_time * self._sim_fps).floor().int()
        return motion_time_steps

    def sample_motions(self, n):
        motion_ids = torch.multinomial(
            self._sampling_batch_prob, num_samples=n, replacement=True
        ).to(self._device)

        return motion_ids

    def current_motion_sampling_probabilities(self):
        """Return normalized probabilities for the currently loaded motions."""
        if self.use_adaptive_sampling:
            active_dataset_motion_ids = self.adp_samp_bins[
                self.adp_samp_active_motion_bins, 0
            ].long()
            active_local_motion_ids = self.orig_motion_id_to_motion_ids[
                active_dataset_motion_ids
            ]
            probabilities = torch.zeros(
                self._num_motions,
                dtype=self.adp_sampling_active_prob.dtype,
                device=self._device,
            )
            probabilities.scatter_add_(
                0,
                active_local_motion_ids,
                self.adp_sampling_active_prob,
            )
        else:
            probabilities = self._sampling_batch_prob.clone()
        if not torch.isfinite(probabilities).all() or probabilities.sum() <= 0:
            probabilities = torch.ones(
                self._num_motions,
                dtype=torch.float32,
                device=self._device,
            )
        return probabilities / probabilities.sum()

    def sample_motions_from_candidates(self, candidate_motion_ids, n):
        """Sample loaded motions without crossing an asset-compatibility set."""
        candidates = torch.as_tensor(
            candidate_motion_ids,
            dtype=torch.long,
            device=self._device,
        )
        if candidates.ndim != 1 or len(candidates) == 0:
            raise ValueError("candidate_motion_ids must be a non-empty vector")
        if (
            int(candidates.min().item()) < 0
            or int(candidates.max().item()) >= self._num_motions
        ):
            raise IndexError(
                "candidate motion ID is outside the loaded motion library"
            )
        probabilities = self.current_motion_sampling_probabilities()[candidates]
        if not torch.isfinite(probabilities).all() or probabilities.sum() <= 0:
            probabilities = torch.ones_like(probabilities)
        probabilities = probabilities / probabilities.sum()
        selected = torch.multinomial(
            probabilities,
            num_samples=int(n),
            replacement=True,
        )
        return candidates[selected]

    def get_motion_ids_in_dataset(self, motion_ids):
        return self._curr_motion_ids[motion_ids]

    def get_motion_length(self, motion_ids=None):
        if motion_ids is None:
            return self._motion_lengths
        else:
            return self._motion_lengths[motion_ids]

    def _calc_frame_blend(self, time, len, num_frames, dt):  # noqa: A002
        time = time.clone()
        phase = time / len
        phase = torch.clip(phase, 0.0, 1.0)  # clip time to be within motion length.
        time[time < 0] = 0

        frame_idx0 = (phase * (num_frames - 1)).long()
        frame_idx1 = torch.min(frame_idx0 + 1, num_frames - 1)
        blend = torch.clip(
            (time - frame_idx0 * dt) / dt, 0.0, 1.0
        )  # clip blend to be within 0 and 1

        return frame_idx0, frame_idx1, blend

    def _get_num_bodies(self):
        return self.num_bodies

    def _local_rotation_to_dof_smpl(self, local_rot):
        B, J, _ = local_rot.shape
        dof_pos = rotations.quat_to_exp_map(local_rot[:, 1:])
        return dof_pos.reshape(B, -1)

    def init_adaptive_sampling(self):
        """Initialize adaptive sampling data structures over all unique motions.

        Divides every motion clip into fixed-size bins (``bin_size`` frames each) and
        creates per-bin tracking tensors for failure rates and sampling probabilities.
        This enables fine-grained, time-segment-level curriculum learning: bins with
        higher failure rates are sampled more frequently during training.

        NOTE: This operates over ALL unique motions in the dataset (not just the
        currently loaded batch), so bin indices are stable across reloads.
        """
        self._adaptive_bins_pending = False
        self._adaptive_sequence_update_pending = False
        self._deferred_adaptive_attribute_names = frozenset()
        if not self._adaptive_frame_counts_known_now():
            # Directory entries without metadata ``length``/``fps``: the only way to know the frame
            # counts here would be to read every robot payload (and the loader reads them again, or
            # a reference tensor cache hit makes the read unnecessary). Build the bins from the loaded
            # reference data instead (``_finish_deferred_adaptive_init`` at the end of load_motions).
            self._adaptive_bins_pending = True
            self._init_adaptive_sampling_scalars()
            return
        self._build_adaptive_bins(self._adaptive_num_frames_from_entries())

    def _adaptive_frame_counts_known_now(self):
        """True when every motion entry carries its frame count and fps without a payload read."""
        for motion_key in self._motion_data_keys:
            motion_data = self._motion_data_load[motion_key]
            if motion_data.get("fps") is None:
                return False
            if "length" not in motion_data and "root_trans_offset" not in motion_data:
                return False
        return True

    def _adaptive_num_frames_from_entries(self, orig_motion_ids=None):
        """Target-fps frame count per motion from the entries (metadata, else the robot payload)."""
        num_frames_out = torch.zeros(
            self._num_unique_motions, device=self._device, dtype=torch.long
        )
        if orig_motion_ids is None:
            orig_motion_ids = range(self._num_unique_motions)
        # Compute motion lengths and frame counts for all unique motions using self._motion_data_keys
        for i in orig_motion_ids:
            motion_key = self._motion_data_keys[i]
            motion_data = self._motion_data_load[motion_key]

            # Compute motion length and frame count similar to how it's done in load_motions
            # Need to account for interpolation from original fps to target_fps
            original_fps = motion_data.get("fps")
            # Get frame count: prefer metadata 'length', else 'root_trans_offset' shape,
            # else lazy-load from pkl file (directory mode without metadata)
            if "length" in motion_data:
                original_num_frames = motion_data["length"]
            elif "root_trans_offset" in motion_data:
                original_num_frames = motion_data["root_trans_offset"].shape[0]
            elif "path" in motion_data:
                # Directory mode: lazy-load the pkl file to get frame count and fps
                loaded_data, *_ = joblib.load(motion_data["path"]).values()
                original_num_frames = loaded_data["root_trans_offset"].shape[0]
                if "fps" in loaded_data:
                    original_fps = loaded_data["fps"]
            else:
                raise KeyError(
                    f"Cannot determine frame count for motion '{motion_key}': no 'length', 'root_trans_offset', or 'path' key"  # noqa: E501
                )
            if original_fps is None and "path" in motion_data:
                loaded_data, *_ = joblib.load(motion_data["path"]).values()
                original_fps = loaded_data.get("fps")
            if original_fps is None:
                if self.require_paired_reference_clock:
                    raise KeyError(
                        "Joint multi-source robot reference "
                        f"{motion_key!r} is missing fps"
                    )
                original_fps = 30.0
            original_fps = float(original_fps)
            if not np.isfinite(original_fps) or original_fps <= 0.0:
                raise ValueError(
                    f"Motion {motion_key!r} has invalid fps {original_fps}"
                )
            motion_data["fps"] = original_fps

            original_duration = (original_num_frames - 1) / original_fps

            # Match fk_batch behavior: when fps == target_fps, interpolation is
            # skipped and raw frames are used. Otherwise use the canonical
            # interploate_pose formula (arange with exclusive end).
            if original_fps == self.target_fps:
                num_frames = original_num_frames
            else:
                num_frames = len(torch.arange(0, original_duration, 1 / self.target_fps))
            num_frames_out[i] = num_frames
        return num_frames_out

    def _init_adaptive_sampling_scalars(self):
        """Config-derived sampler scalars that do not depend on the bin layout."""
        init_num_failures = self.adaptive_sampling_cfg.get("init_num_failures", 1)
        self.adp_samp_initial_episode_prior = float(init_num_failures)
        self.adp_samp_failure_rate_max_over_mean = self.adaptive_sampling_cfg.get(
            "adp_samp_failure_rate_max_over_mean", 50.0
        )
        self.uniform_sampling_rate = self.adaptive_sampling_cfg.get("uniform_sampling_rate", 0.1)
        # Max probability constraints (None = skip, "auto" = use failure_rate_max_over_mean)
        # These prevent over-concentration on challenging motions. See update_adaptive_sampling_probabilities().
        self.max_prob_per_bin_cfg = self.adaptive_sampling_cfg.get("max_prob_per_bin", None)
        self.max_prob_per_motion_cfg = self.adaptive_sampling_cfg.get("max_prob_per_motion", None)
        keyed_sync = self.adaptive_sampling_cfg.get("distributed_keyed_sync")
        if keyed_sync is None:
            keyed_sync = self.use_joint_source_tracking
        self.use_joint_distributed_sampler = bool(
            self.use_joint_source_tracking and keyed_sync
        )
        if (
            self.use_joint_source_tracking
            and not self.use_joint_distributed_sampler
            and self._known_world_size() > 1
        ):
            raise ValueError(
                "adaptive_sampling.distributed_keyed_sync=false is only valid "
                "for a single process; this tracked run has world_size "
                f"{self._known_world_size()} (positional averaging would "
                "blend different motions across ranks)"
            )

    def _prepare_adaptive_motion_selection(self, *, random_sample, explicit_selection):
        """The motion-sequence probability update that precedes a load's motion selection.

        With deferred bins (path-only entries) an explicit or sequential selection does not consume
        the adaptive motion probabilities and draws no RNG, so the bins are built from the loaded
        reference data at the tail of the load and the sequence probabilities are updated there
        (same state as the constructor-time order, nothing sampled in between). An adaptive random
        draw needs the probabilities now: the bins are built first from the entries / payloads
        (the original computation and RNG consumption), then the update runs as before.
        """
        if getattr(self, "_adaptive_bins_pending", False):
            if explicit_selection or not random_sample:
                self._adaptive_sequence_update_pending = True
                return
            self._complete_deferred_adaptive_init(self._adaptive_num_frames_from_entries())
        self.update_adaptive_sampling_motion_sequences()

    def _complete_deferred_adaptive_init(self, num_frames):
        """Build the deferred bins and record the attributes it adds (sampler state, never reference
        tensors) so the tensor cache capture of this native load accounts for them."""
        before = frozenset(vars(self))
        self._build_adaptive_bins(num_frames.to(self._device))
        self._adaptive_bins_pending = False
        self._deferred_adaptive_attribute_names = (
            getattr(self, "_deferred_adaptive_attribute_names", frozenset()) | (set(vars(self)) - before)
        )

    def _finish_deferred_adaptive_init(self):
        """Build the deferred bins from the loaded batch: ``_motion_num_frames`` / ``_motion_fps`` of
        every loaded motion are exactly the loader's target-fps frame count and payload fps (the
        values the constructor path tried to predict and ``update_adaptive_sampling_motion_frames``
        asserts); motions outside a partial batch fall back to the entry/payload computation."""
        if not getattr(self, "_adaptive_bins_pending", False):
            return
        before = frozenset(vars(self))
        curr = self._curr_motion_ids.to("cpu", torch.long)
        num_frames = torch.zeros(self._num_unique_motions, dtype=torch.long)
        known = torch.zeros(self._num_unique_motions, dtype=torch.bool)
        num_frames[curr] = self._motion_num_frames.detach().to("cpu", torch.long)
        known[curr] = True
        source_fps = getattr(self, "_motion_source_fps", None)
        source_fps = None if source_fps is None else source_fps.detach().cpu().tolist()
        roster_fps = tuple(getattr(self, "joint_source_motion_sample_fps", ()) or ())
        for motion_id, orig in enumerate(curr.tolist()):
            key = self._motion_data_keys[orig]
            entry = self._motion_data_load[key]
            if entry.get("fps") is not None:
                continue
            if source_fps is not None:
                entry["fps"] = float(source_fps[motion_id])      # the payload fps read by the loader
            elif orig < len(roster_fps) and roster_fps[orig] is not None:
                entry["fps"] = float(roster_fps[orig])          # exact static roster fps (legacy bundle hit)
            else:
                self._motion_payload_fps(key)                  # legacy bundle without the fps tensor: one read
        if not bool(known.all()):
            missing = [int(i) for i in (~known).nonzero().flatten().tolist()]
            fallback = self._adaptive_num_frames_from_entries(missing).cpu()
            num_frames[missing] = fallback[missing]
        if self.use_joint_source_tracking and self.joint_source_motion_sample_fps:
            self.joint_source_motion_sample_fps = tuple(
                float(self._motion_data_load[key]["fps"]) if value is None else value
                for key, value in zip(
                    self._motion_data_keys, self.joint_source_motion_sample_fps, strict=True
                )
            )
        self._complete_deferred_adaptive_init(num_frames)
        self._deferred_adaptive_attribute_names = (
            self._deferred_adaptive_attribute_names | (set(vars(self)) - before)
        )

    def _build_adaptive_bins(self, num_frames):
        """Bin layout, per-bin counters and the keyed sampler from the per-motion frame counts."""
        self._init_adaptive_sampling_scalars()
        self.adp_samp_num_frames = num_frames.to(device=self._device, dtype=torch.long)
        # Compute length_starts similar to how it's done in load_motions (using num_frames, not lengths)
        lengths = self.adp_samp_num_frames
        lengths_shifted = lengths.roll(1)
        lengths_shifted[0] = 0
        self.adp_samp_length_starts = lengths_shifted.cumsum(0)
        self.adp_samp_total_frames = self.adp_samp_num_frames.sum()
        self.adp_samp_length_starts_mask = torch.zeros(
            self.adp_samp_total_frames, device=self._device, dtype=torch.bool
        )
        self.adp_samp_length_starts_mask[self.adp_samp_length_starts] = True

        # init bins - batch version
        self.adp_samp_bin_size = self.adaptive_sampling_cfg.get("bin_size", 50)
        self.adp_samp_frame_to_bin = torch.zeros(
            self.adp_samp_total_frames, device=self._device, dtype=torch.long
        )

        # Pre-compute all bin information in batch
        all_bins = []
        all_bin_motion_lengths = []
        all_bin_new_motion_masks = []
        all_num_peer_bins = []
        all_motion_to_bins = []

        cur_bin_idx = 0
        for orig_motion_id in range(self._num_unique_motions):
            num_frames = self.adp_samp_num_frames[orig_motion_id]
            frame_start = self.adp_samp_length_starts[orig_motion_id]
            frame_end = (
                self.adp_samp_length_starts[orig_motion_id + 1]
                if orig_motion_id < self._num_unique_motions - 1
                else self.adp_samp_total_frames
            )

            # Create bin starts and ends in batch
            bin_starts = torch.arange(
                0, num_frames, self.adp_samp_bin_size, device=self._device, dtype=torch.long
            )
            bin_ends = torch.minimum(bin_starts + self.adp_samp_bin_size, num_frames)
            num_bins = len(bin_starts)
            motion_ids = torch.full(
                (num_bins,), orig_motion_id, device=self._device, dtype=torch.long
            )
            motion_bins = torch.stack([motion_ids, bin_starts, bin_ends], dim=1)
            all_bins.append(motion_bins)

            # Calculate bin lengths
            bin_lengths = bin_ends - bin_starts
            all_bin_motion_lengths.append(bin_lengths)

            # Create new motion mask (first bin of each motion is True)
            new_motion_mask = torch.zeros(num_bins, device=self._device, dtype=torch.bool)
            new_motion_mask[0] = True
            all_bin_new_motion_masks.append(new_motion_mask)

            # Number of peer bins (same for all bins in this motion)
            peer_bins = torch.full((num_bins,), num_bins, device=self._device, dtype=torch.long)
            all_num_peer_bins.append(peer_bins)

            bin_ids = torch.zeros(num_frames, device=self._device, dtype=torch.long)
            bin_ids[bin_starts[1:]] = 1
            bin_ids = bin_ids.cumsum(0) + cur_bin_idx
            self.adp_samp_frame_to_bin[frame_start:frame_end] = bin_ids

            # Store motion to bins mapping
            motion_bin_indices = torch.arange(
                cur_bin_idx, cur_bin_idx + num_bins, device=self._device, dtype=torch.long
            )
            all_motion_to_bins.append(motion_bin_indices)

            cur_bin_idx += num_bins

        # Concatenate all batch results
        self.adp_samp_bins = torch.cat(all_bins, dim=0)
        self.adp_samp_bin_motion_length = torch.cat(all_bin_motion_lengths, dim=0)
        self.adp_samp_bin_new_motion_mask = torch.cat(all_bin_new_motion_masks, dim=0)
        self.adp_samp_num_peer_bins = torch.cat(all_num_peer_bins, dim=0)
        self.orig_motion_id_to_bins = all_motion_to_bins
        self.adp_samp_num_bins = len(self.adp_samp_bins)

        self.adp_samp_bin_weights = (
            self.adp_samp_bin_motion_length / self.adp_samp_bin_motion_length.float().mean()
        )
        # this will make sure each sequence is sampled equally.
        if self.adaptive_sampling_cfg.get("sequence_length_agnostic", True):
            self.adp_samp_bin_weights = self.adp_samp_bin_weights / self.adp_samp_num_peer_bins

        init_num_failures = self.adp_samp_initial_episode_prior
        self.adp_samp_num_failures = (
            torch.ones(self.adp_samp_num_bins, device=self._device, dtype=torch.float32)
            * init_num_failures
        )
        self.adp_samp_terminal_failure_counts = torch.zeros(
            self.adp_samp_num_bins, device=self._device, dtype=torch.float32
        )
        self.adp_samp_num_episodes = (
            torch.ones(self.adp_samp_num_bins, device=self._device, dtype=torch.float32)
            * init_num_failures
        )
        self.adp_samp_failure_rate = torch.ones(
            self.adp_samp_num_bins, device=self._device, dtype=torch.float32
        )
        self.adp_samp_failure_rate_raw = torch.ones(
            self.adp_samp_num_bins, device=self._device, dtype=torch.float32
        )
        self.adp_samp_previous_success_rate = torch.full(
            (self.adp_samp_num_bins,),
            0.5,
            device=self._device,
            dtype=torch.float32,
        )
        self.adp_samp_priority_score = torch.ones(
            self.adp_samp_num_bins, device=self._device, dtype=torch.float32
        )
        self.adp_sampling_prob = (
            torch.ones(self.adp_samp_num_bins, device=self._device, dtype=torch.float64)
            / self.adp_samp_num_bins
        )
        # Keyed (semantic-bin) distributed sync is the default whenever source
        # identities are tracked: heterogeneous unified ranks hold different
        # bins, so positional averaging would blend unrelated motions. An
        # explicit distributed_keyed_sync=false is honored only when the run
        # is a single process (legacy homogeneous behavior).
        if self.use_joint_distributed_sampler:
            self._init_joint_distributed_sampler()

    def _known_world_size(self, accelerator=None):
        """Largest world size known right now.

        Sources: an initialized ``torch.distributed`` group, the accelerator,
        the launcher's ``WORLD_SIZE`` variable (torchrun / accelerate set it in
        every worker before the process group exists) and the roster's pinned
        world size.
        """
        candidates = [1]
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            candidates.append(int(torch.distributed.get_world_size()))
        if accelerator is not None:
            candidates.append(int(getattr(accelerator, "num_processes", 1)))
        env_world_size = os.environ.get("WORLD_SIZE", "").strip()
        if env_world_size.isdigit():
            candidates.append(int(env_world_size))
        manifest_world_size = getattr(self, "joint_source_manifest_world_size", None)
        if manifest_world_size is not None:
            candidates.append(int(manifest_world_size))
        return max(candidates)

    def _check_effective_adaptive_tracking(self):
        """Refuse adaptive sampling without motion identities on a multi-rank run.

        Evaluates the EFFECTIVE configuration: ``adaptive_sampling.enable`` on
        while ``joint_source_manifest`` is None (a Hydra
        ``joint_source_tracking=false`` opt-out or a cohort without tracking)
        leaves no identities to key the cross-rank merge on, and positional
        averaging would blend unrelated motions across heterogeneous ranks.
        """
        if not self.use_adaptive_sampling or self.use_joint_source_tracking:
            return
        world_size = self._known_world_size()
        if world_size > 1:
            raise ValueError(
                "adaptive_sampling.enable=true on a run with world_size "
                f"{world_size} requires tracked motion identities "
                "(motion_lib_cfg.joint_source_tracking=true with a roster "
                "manifest); without them per-rank statistics could only be "
                "averaged by tensor position, which blends unrelated motions "
                "across ranks. Set joint_source_tracking=true or "
                "adaptive_sampling.enable=false"
            )

    def _adaptive_frame_counts_deferrable(self):
        """Adaptive sampling is enabled and at least one entry lacks metadata frame count or fps."""
        cfg = getattr(self, "adaptive_sampling_cfg", None) or {}
        return bool(cfg.get("enable", False)) and not self._adaptive_frame_counts_known_now()

    def _motion_payload_fps(self, motion_key):
        """The motion's own source fps, read exactly as ``init_adaptive_sampling`` reads it.

        ``_motion_data_load[key]["fps"]`` when already known, else the robot payload
        on disk (``path``); the value is cached back into the entry. Missing ->
        KeyError, non-finite / non-positive -> ValueError. Never a filled default.
        """
        motion_data = self._motion_data_load[motion_key]
        fps = motion_data.get("fps")
        if fps is None and "path" in motion_data:
            loaded_data, *_ = joblib.load(motion_data["path"]).values()
            fps = loaded_data.get("fps")
        if fps is None:
            raise KeyError(
                f"Joint-source roster record of {motion_key!r} carries no "
                "source_sample_fps_hz and the motion payload has no fps"
            )
        fps = float(fps)
        if not np.isfinite(fps) or fps <= 0.0:
            raise ValueError(f"Motion {motion_key!r} has invalid fps {fps}")
        motion_data["fps"] = fps
        return fps

    def init_joint_source_tracking(self):
        """Load stable source identities for a joint-training roster.

        Two identity sources, same validations: ``joint_source_manifest`` (a
        manifest FILE, unified cohorts) or ``joint_source_roster`` (an in-memory
        roster wrapped by ``gear_sonic.rp2.joint_source_roster`` from a view that
        carries ``identity_records``; its label is a directory that is never
        opened). Both set -> ValueError.
        """
        manifest_path = self.m_cfg.get("joint_source_manifest")
        memory_roster = self.m_cfg.get("joint_source_roster")
        if manifest_path and memory_roster:
            raise ValueError(
                "motion_lib_cfg.joint_source_manifest and "
                "motion_lib_cfg.joint_source_roster are both set; the identity "
                "source must be one manifest file or one in-memory roster"
            )
        self.use_joint_source_tracking = bool(manifest_path) or bool(memory_roster)
        self.use_joint_distributed_sampler = False
        self.joint_source_names = ()
        self.joint_source_manifest_path = None
        self.joint_source_manifest_world_size = None
        self.joint_source_motion_ids = None
        self.joint_source_motion_identity = ()
        self.joint_source_motion_aliases = ()
        self.joint_source_motion_sources = ()
        self.joint_source_motion_revisions = ()
        self.joint_source_motion_sample_fps = ()
        # identity -> {source, alias[, object_key][, asset_signature]} of the
        # motions this rank loads (local) and, after a keyed sync or a
        # checkpoint restore, of every rank's motions (global).
        self.joint_source_object_records = {}
        self.joint_source_object_records_global = {}
        if not self.use_joint_source_tracking:
            return

        if manifest_path:
            manifest_path = Path(str(manifest_path)).expanduser().resolve()
            if not manifest_path.is_file():
                raise FileNotFoundError(
                    f"Joint-source roster manifest does not exist: {manifest_path}"
                )
            manifest_bytes = manifest_path.read_bytes()
            manifest = json.loads(manifest_bytes.decode("utf-8"))
            fps_required = True
        else:
            # In-memory roster: validated mapping; its label (a directory for the
            # direct-directory roster) is kept as the roster path and never opened.
            manifest = normalize_joint_source_roster(memory_roster)
            manifest_path = str(manifest["label"])
            fps_required = False
        records = manifest.get("records", [])
        if not isinstance(records, list) or not records:
            raise ValueError(f"{manifest_path}: no roster records")
        by_alias = {}
        for record in records:
            alias = str(record.get("alias", ""))
            source = str(record.get("source", ""))
            source_motion_key = str(record.get("source_motion_key", ""))
            # promoted_revision is a provenance label only; it is recorded when
            # present but no longer required for identity tracking.
            promoted_revision = str(record.get("promoted_revision", "") or "")
            if not alias or not source or not source_motion_key:
                raise ValueError(f"{manifest_path}: incomplete roster record {record}")
            if alias in by_alias:
                raise ValueError(f"{manifest_path}: duplicate alias {alias}")
            raw_fps = record.get("source_sample_fps_hz")
            if raw_fps is None and not fps_required:
                # direct-directory record without meta: the real fps is read from
                # the robot payload below (never a filled default)
                pass
            else:
                source_sample_fps = float(
                    float("nan") if raw_fps is None else raw_fps
                )
                if (
                    not np.isfinite(source_sample_fps)
                    or source_sample_fps <= 0.0
                ):
                    raise ValueError(
                        f"{manifest_path}: {alias} has invalid source sample FPS"
                    )
            by_alias[alias] = dict(record)

        source_order = list((manifest.get("sampling_plan") or {}).get("sources", {}))
        if not source_order and manifest.get("sources"):
            source_order = [str(name) for name in manifest["sources"]]
        if not source_order:
            source_order = list(
                dict.fromkeys(
                    str(record["source"]) for record in by_alias.values()
                )
            )
        source_index = {name: index for index, name in enumerate(source_order)}
        missing = [key for key in self._motion_data_keys if key not in by_alias]
        extra = sorted(set(by_alias) - set(self._motion_data_keys))
        complete_roster = (
            manifest.get("contract")
            == "persistent_complete_asset_compatible_roster_v1"
            and manifest.get("persistent_complete_roster") is True
        )
        # Unified cohorts list every motion of the cohort while each rank
        # loads only its own slots' motions; records that other ranks own are
        # expected extras, not a roster mismatch. In-memory rosters say so
        # explicitly (direct_roster_identity_v1 records cover the whole layout).
        if manifest.get("contract") == "unified_no_vof_cohort_v1":
            complete_roster = True
        if not fps_required and manifest.get("complete_roster", True):
            complete_roster = True
        if missing or (extra and not complete_roster):
            raise ValueError(
                "Joint-source roster and motion library keys differ: "
                f"missing={missing[:8]}, extra={extra[:8]}"
            )

        identities = []
        aliases = []
        motion_sources = []
        revisions = []
        sample_fps_values = []
        source_ids = []
        object_records = {}
        for key in self._motion_data_keys:
            record = by_alias[key]
            source = str(record["source"])
            source_motion_key = str(record["source_motion_key"])
            promoted_revision = str(record.get("promoted_revision", "") or "")
            if source not in source_index:
                raise ValueError(f"{manifest_path}: source {source!r} has no allocation")
            source_ids.append(source_index[source])
            identity = f"{source}::{source_motion_key}"
            identities.append(identity)
            aliases.append(key)
            motion_sources.append(source)
            revisions.append(promoted_revision)
            if record.get("source_sample_fps_hz") is None:
                # Known once the motion is loaded (filled by _finish_deferred_adaptive_init from
                # the payload fps the loader reads anyway); read the payload now only when the
                # bins are built in the constructor.
                entry = self._motion_data_load[key]
                if entry.get("fps") is None and self._adaptive_frame_counts_deferrable():
                    sample_fps_values.append(None)
                else:
                    sample_fps_values.append(self._motion_payload_fps(key))
            else:
                sample_fps_values.append(
                    float(record["source_sample_fps_hz"])
                )
            # Identity -> object record from the manifest record itself
            # (unified cohorts carry object_key / asset_signature). Absent
            # fields stay absent; a record without any object field is not
            # recorded at all -- nothing is fabricated.
            object_record = {"source": source, "alias": str(key)}
            for field_name in ("object_key", "asset_signature"):
                value = record.get(field_name)
                if isinstance(value, str) and value.strip():
                    object_record[field_name] = value
            if len(object_record) > 2:
                previous = object_records.setdefault(identity, object_record)
                if any(
                    previous.get(field_name) != object_record.get(field_name)
                    for field_name in ("object_key", "asset_signature")
                ):
                    raise ValueError(
                        f"{manifest_path}: motion identity {identity} maps to "
                        f"different object records: {previous} != {object_record}"
                    )
        self.joint_source_names = tuple(source_order)
        self.joint_source_manifest_path = manifest_path
        declared_world_size = manifest.get("world_size")
        if declared_world_size is not None:
            declared_world_size = int(declared_world_size)
            if declared_world_size <= 0:
                raise ValueError(
                    f"{manifest_path}: world_size must be positive"
                )
        self.joint_source_manifest_world_size = declared_world_size
        self.joint_source_motion_ids = torch.tensor(
            source_ids, dtype=torch.long, device=self._device
        )
        self.joint_source_motion_identity = tuple(identities)
        self.joint_source_motion_aliases = tuple(aliases)
        self.joint_source_motion_sources = tuple(motion_sources)
        self.joint_source_motion_revisions = tuple(revisions)
        self.joint_source_motion_sample_fps = tuple(sample_fps_values)
        self.joint_source_object_records = normalize_object_records(object_records)
        self.joint_source_object_records_global = dict(
            self.joint_source_object_records
        )
        count = len(self.joint_source_names)
        self.joint_source_num_episodes = torch.zeros(
            count, dtype=torch.float32, device=self._device
        )
        self.joint_source_num_failures = torch.zeros(
            count, dtype=torch.float32, device=self._device
        )
        self.joint_source_previous_success_rate = torch.full(
            (count,), 0.5, dtype=torch.float32, device=self._device
        )

    def _init_joint_distributed_sampler(self):
        """Initialize semantic-bin delta tracking for heterogeneous ranks."""
        if self.joint_source_manifest_world_size is None:
            raise ValueError(
                "The joint roster must pin world_size when "
                "distributed_keyed_sync is enabled"
            )
        keys = []
        metadata = []
        seen = set()
        for raw_bin in self.adp_samp_bins.detach().cpu().tolist():
            motion_id, start_frame, end_frame = (int(value) for value in raw_bin)
            key = BinKey(
                source=self.joint_source_motion_sources[motion_id],
                motion_alias=self.joint_source_motion_aliases[motion_id],
                start_frame=start_frame,
                end_frame=end_frame,
            )
            if key in seen:
                raise ValueError(
                    f"Duplicate semantic adaptive-sampling bin {key}"
                )
            seen.add(key)
            motion_data = self._motion_data_load[
                self.joint_source_motion_aliases[motion_id]
            ]
            keys.append(key)
            metadata.append(
                {
                    MOTION_IDENTITY_METADATA_KEY: (
                        self.joint_source_motion_identity[motion_id]
                    ),
                    "promoted_revision": (
                        self.joint_source_motion_revisions[motion_id]
                    ),
                    "motion_num_frames": int(
                        self.adp_samp_num_frames[motion_id].item()
                    ),
                    "fps": float(motion_data["fps"]),
                }
            )
        self.joint_sampler_bin_keys = tuple(keys)
        self.joint_sampler_bin_metadata = tuple(metadata)
        self.joint_sampler_episode_baseline = (
            self.adp_samp_num_episodes.detach().clone()
        )
        self.joint_sampler_failure_baseline = (
            self.adp_samp_num_failures.detach().clone()
        )
        self.joint_sampler_terminal_failure_baseline = (
            self.adp_samp_terminal_failure_counts.detach().clone()
        )
        self.joint_sampler_global_aggregate = None
        self.joint_sampler_checkpoint = None
        # Per roster lifecycle: the static object records are exchanged once (first sync after
        # init / restore / neutral reset), the keyed sync serial counts completed syncs.
        self._joint_object_records_synced = False
        self.joint_sampler_sync_serial = 0
        self.joint_source_episode_baseline = (
            self.joint_source_num_episodes.detach().clone()
        )
        self.joint_source_failure_baseline = (
            self.joint_source_num_failures.detach().clone()
        )
        self.joint_source_global_num_episodes = (
            self.joint_source_num_episodes.detach().clone()
        )
        self.joint_source_global_num_failures = (
            self.joint_source_num_failures.detach().clone()
        )

    def _joint_sampler_rank_world_size(self, accelerator=None):
        rank = int(getattr(accelerator, "process_index", 0))
        world_size = int(getattr(accelerator, "num_processes", 1))
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            distributed_rank = torch.distributed.get_rank()
            distributed_world_size = torch.distributed.get_world_size()
            if accelerator is not None and (
                rank != distributed_rank
                or world_size != distributed_world_size
            ):
                raise ValueError(
                    "Accelerator and torch.distributed rank metadata disagree"
                )
            rank = distributed_rank
            world_size = distributed_world_size
        if world_size != self.joint_source_manifest_world_size:
            raise ValueError(
                "Joint roster world size differs from runtime: "
                f"{self.joint_source_manifest_world_size} != {world_size}"
            )
        if not 0 <= rank < world_size:
            raise ValueError(f"Invalid distributed rank {rank}/{world_size}")
        return rank, world_size

    def _joint_sampler_local_deltas(self):
        """Rank-local count increments since the last keyed sync: one ``[3, n_local]`` float64 CPU tensor
        (episodes, failures, terminal failures), clamped at zero after the ordinary negative-delta check."""
        deltas = (
            self.adp_samp_num_episodes - self.joint_sampler_episode_baseline,
            self.adp_samp_num_failures - self.joint_sampler_failure_baseline,
            self.adp_samp_terminal_failure_counts
            - self.joint_sampler_terminal_failure_baseline,
        )
        for name, values in zip(
            ("episode", "failure", "terminal failure"),
            deltas,
        ):
            if bool((values < -1.0e-5).any().item()):
                raise ValueError(
                    f"Joint adaptive sampler has negative {name} deltas"
                )
        rows = torch.stack(
            [values.clamp_min(0.0).detach() for values in deltas]
        ).to(dtype=torch.float64).cpu()
        if not bool(torch.isfinite(rows).all().item()):
            raise ValueError("Joint adaptive sampler deltas must be finite")
        return rows

    def _establish_joint_sampler_table(self, *, world_size):
        """Lifecycle start (fresh init or neutral reset): build the global key->slot table ONCE.

        One object gather of every rank's static ``(bin keys, bin metadata, local delta rows)``; the
        union of keys is sorted (BinKey order) into the slot table, each bin starts from the global
        prior exactly once and the ranks' deltas are added by slot with the exact merge
        (``apply_slot_deltas``). No per-bin record dicts, payload normalization or metadata
        fingerprints: the keys were validated when this rank built them, and a bin reported by several
        ranks keeps the lowest rank's metadata. After a checkpoint restore the table comes from the
        restored aggregate instead.
        """
        rows = self._joint_sampler_local_deltas()
        local = (
            list(self.joint_sampler_bin_keys),
            list(self.joint_sampler_bin_metadata),
            rows.tolist(),
        )
        if world_size == 1:
            gathered = [local]
        else:
            if (
                not torch.distributed.is_available()
                or not torch.distributed.is_initialized()
            ):
                raise RuntimeError(
                    "torch.distributed must be initialized for keyed sampler sync"
                )
            gathered = [None] * world_size
            torch.distributed.all_gather_object(gathered, local)
            if any(entry is None for entry in gathered):
                raise RuntimeError(
                    "all_gather_object did not populate every rank's sampler table"
                )
        table = {}
        for keys, metadata, _rows in gathered:
            for key, meta in zip(keys, metadata, strict=True):
                table.setdefault(key, meta)
        sorted_keys = sorted(table)
        index = {key: slot for slot, key in enumerate(sorted_keys)}
        prior = float(self.adp_samp_initial_episode_prior)
        size = len(sorted_keys)
        canonical = CanonicalAggregate(
            world_size,
            sorted_keys,
            [prior] * size,
            [prior] * size,
            [0.0] * size,
            [table[key] for key in sorted_keys],
        )
        return canonical.apply_slot_deltas(
            [
                ([index[key] for key in keys], rank_rows[0], rank_rows[1], rank_rows[2])
                for keys, _metadata, rank_rows in gathered
            ]
        )

    def _write_joint_sampler_counts(self, episodes, failures, terminal_failures):
        self.adp_samp_num_episodes.copy_(
            torch.tensor(
                episodes,
                dtype=self.adp_samp_num_episodes.dtype,
                device=self._device,
            )
        )
        self.adp_samp_num_failures.copy_(
            torch.tensor(
                failures,
                dtype=self.adp_samp_num_failures.dtype,
                device=self._device,
            )
        )
        self.adp_samp_terminal_failure_counts.copy_(
            torch.tensor(
                terminal_failures,
                dtype=self.adp_samp_terminal_failure_counts.dtype,
                device=self._device,
            )
        )
        self.joint_sampler_episode_baseline.copy_(
            self.adp_samp_num_episodes
        )
        self.joint_sampler_failure_baseline.copy_(
            self.adp_samp_num_failures
        )
        self.joint_sampler_terminal_failure_baseline.copy_(
            self.adp_samp_terminal_failure_counts
        )

    # ---- private canonical sampler state -------------------------------------------------
    # The keyed global table lives in a private ``CanonicalAggregate`` owned by this object
    # (sorted keys, owned canonical metadata, count lists). A sync applies only the delta's
    # bins to it; the public ``joint_sampler_global_aggregate`` / ``joint_sampler_checkpoint``
    # values are ordinary dicts exported from it (fresh records, independent nested metadata)
    # and are never trusted back by identity: assigning a Mapping goes through the full public
    # validation and replaces the private state, which also drops the cached apply positions.

    @property
    def joint_sampler_global_aggregate(self):
        """Ordinary aggregate payload of the keyed sampler state, or None (fresh export per read)."""
        canonical = getattr(self, "_joint_sampler_canonical", None)
        return None if canonical is None else canonical.export()

    @joint_sampler_global_aggregate.setter
    def joint_sampler_global_aggregate(self, value):
        # None or an ordinary Mapping (fully validated, privately copied); the sync assigns the
        # private state directly. Any public replacement drops the cached apply positions.
        if value is None:
            self._joint_sampler_canonical = None
        else:
            self._joint_sampler_canonical = CanonicalAggregate.from_payload(value)
        self._joint_sampler_apply_positions = None

    @property
    def joint_sampler_checkpoint(self):
        """Checkpoint-safe sampler state of the last sync/restore (ordinary dict), or None.

        After a sync it is materialized on the first read from the private state and the
        RNG states captured by that sync (fresh records, deep-copied RNG state) and kept
        until the next sync/restore/reset; an assigned dict is kept as given.
        """
        pending = getattr(self, "_joint_sampler_checkpoint_pending", None)
        if pending is not None:
            canonical, rng_state = pending
            value = serialize_canonical_checkpoint(canonical, rng_state=rng_state)
            # Compact history of THIS rank's bins by global slot (what the per-motion duplicate
            # used to carry for the saving rank); readers that do not know the field ignore it.
            value["previous_success_rate"] = self._joint_sampler_previous_success_by_slot(
                canonical
            )
            self._joint_sampler_checkpoint_value = value
            self._joint_sampler_checkpoint_pending = None
        return getattr(self, "_joint_sampler_checkpoint_value", None)

    @joint_sampler_checkpoint.setter
    def joint_sampler_checkpoint(self, value):
        self._joint_sampler_checkpoint_pending = None
        self._joint_sampler_checkpoint_value = value

    def _apply_joint_sampler_canonical(self, canonical):
        """Write the private state to the rank-local tensors; positions cached per key set."""
        cache = getattr(self, "_joint_sampler_apply_positions", None)
        if (
            cache is None
            or cache[0] is not canonical.keys
            or cache[1] is not self.joint_sampler_bin_keys
        ):
            index = canonical.index()
            missing = [
                key for key in self.joint_sampler_bin_keys if key not in index
            ]
            if missing:
                raise ValueError(
                    "Global adaptive-sampler state misses rank-local bins: "
                    f"{missing[:8]}"
                )
            slots = [index[key] for key in self.joint_sampler_bin_keys]
            cache = (
                canonical.keys,
                self.joint_sampler_bin_keys,
                slots,
                torch.tensor(slots, dtype=torch.long),
            )
            self._joint_sampler_apply_positions = cache
        positions = cache[2]
        self._write_joint_sampler_counts(
            [float(canonical.episodes[p]) for p in positions],
            [float(canonical.failures[p]) for p in positions],
            [float(canonical.terminals[p]) for p in positions],
        )

    def _joint_sampler_previous_success_by_slot(self, canonical):
        """``{"slots": [...], "values": [...]}``: this rank's ``adp_samp_previous_success_rate`` per
        global slot, with the per-identity mean over duplicate roster motions that the per-motion
        checkpoint form applied (one host transfer)."""
        slots = self._joint_sampler_local_slots(canonical)
        values = self.adp_samp_previous_success_rate.detach().to(torch.float32).cpu()
        identities = tuple(self.joint_source_motion_identity)
        grouped = {}
        for motion_id, identity in enumerate(identities):
            grouped.setdefault(identity, []).append(motion_id)
        for identity, motion_ids in grouped.items():
            if len(motion_ids) < 2:
                continue
            members = [self.orig_motion_id_to_bins[m].reshape(-1) for m in motion_ids]
            if len({int(b.numel()) for b in members}) != 1:
                raise ValueError(
                    f"Duplicate roster motion {identity} has inconsistent bin counts"
                )
            mean = torch.stack([values[b.cpu()] for b in members]).mean(0)
            for bins in members:
                values[bins.cpu()] = mean
        return {"slots": slots.tolist(), "values": values.tolist()}

    def _restore_joint_sampler_previous_success(self, checkpoint):
        """Overlay the compact ``previous_success_rate`` of a keyed checkpoint (new form) onto the
        rank-local bins it covers; old checkpoints without it keep the per-motion path."""
        compact = checkpoint.get("previous_success_rate") if isinstance(checkpoint, Mapping) else None
        if compact is None:
            return False
        if (
            not isinstance(compact, Mapping)
            or not isinstance(compact.get("slots"), (list, tuple))
            or not isinstance(compact.get("values"), (list, tuple))
            or len(compact["slots"]) != len(compact["values"])
        ):
            raise ValueError("Sampler checkpoint previous_success_rate is malformed")
        saved_bins = checkpoint["aggregate"]["bins"]
        # live bins by (semantic motion identity, frame interval): the per-motion form restored history
        # by identity, so every live copy of a saved identity (any alias) receives the value
        live_by_identity = {}
        for i, key in enumerate(self.joint_sampler_bin_keys):
            identity = self.joint_sampler_bin_metadata[i].get(MOTION_IDENTITY_METADATA_KEY)
            live_by_identity.setdefault((identity, key.start_frame, key.end_frame), []).append(i)
        targets, numbers = [], []
        for slot, value in zip(compact["slots"], compact["values"], strict=True):
            slot = int(slot)
            if not 0 <= slot < len(saved_bins):
                raise ValueError("Sampler checkpoint previous_success_rate slot is out of range")
            record = saved_bins[slot]
            metadata = record.get("metadata") or {}
            identity = metadata.get(MOTION_IDENTITY_METADATA_KEY)
            for local in live_by_identity.get(
                (identity, int(record["start_frame"]), int(record["end_frame"])), ()
            ):
                targets.append(local)
                numbers.append(float(value))
        if targets:
            self.adp_samp_previous_success_rate[torch.tensor(targets, dtype=torch.long)] = torch.tensor(
                numbers, dtype=self.adp_samp_previous_success_rate.dtype, device=self._device
            )
        return True

    def _joint_sampler_local_slots(self, canonical):
        """Global slot (position in the static key table) of every rank-local bin, as a CPU long tensor."""
        cache = getattr(self, "_joint_sampler_apply_positions", None)
        if (
            cache is None
            or cache[0] is not canonical.keys
            or cache[1] is not self.joint_sampler_bin_keys
        ):
            index = canonical.index()
            missing = [
                key for key in self.joint_sampler_bin_keys if key not in index
            ]
            if missing:
                raise ValueError(
                    "Global adaptive-sampler state misses rank-local bins: "
                    f"{missing[:8]}"
                )
            slots = [index[key] for key in self.joint_sampler_bin_keys]
            cache = (
                canonical.keys,
                self.joint_sampler_bin_keys,
                slots,
                torch.tensor(slots, dtype=torch.long),
            )
            self._joint_sampler_apply_positions = cache
        return cache[3]

    def _sync_joint_sampler_slot_deltas(self, canonical, *, world_size):
        """Steady-state keyed sync: exchange only the touched (slot, episode, failure, terminal) deltas.

        Two fixed-order tensor collectives in which every rank participates (an idle or zero-delta rank
        sends an empty, zero-padded contribution): the per-rank touched count, then the zero-padded
        ``[4, max_touched]`` float64 rows (slot, episodes, failures, terminals). Slots are positions in the
        static global key table of the current roster lifecycle, so rank-local bin order is irrelevant and
        no per-bin dictionaries, normalization or metadata comparison are needed. The merge is exact
        (``CanonicalAggregate.apply_slot_deltas``).
        """
        slots = self._joint_sampler_local_slots(canonical)
        rows = self._joint_sampler_local_deltas()
        touched = (rows > 0.0).any(dim=0)
        local = torch.cat(
            [slots[touched].to(torch.float64).unsqueeze(0), rows[:, touched]], dim=0
        )
        if world_size == 1:
            contributions = [local]
        else:
            if (
                not torch.distributed.is_available()
                or not torch.distributed.is_initialized()
            ):
                raise RuntimeError(
                    "torch.distributed must be initialized for keyed sampler sync"
                )
            device = torch.device(self._device)
            count = torch.tensor([local.shape[1]], dtype=torch.long, device=device)
            counts = [torch.zeros_like(count) for _ in range(world_size)]
            torch.distributed.all_gather(counts, count)
            widths = [int(value.item()) for value in counts]
            width = max(1, max(widths))
            padded = torch.zeros((4, width), dtype=torch.float64, device=device)
            padded[:, : local.shape[1]] = local.to(device)
            gathered = [torch.zeros_like(padded) for _ in range(world_size)]
            torch.distributed.all_gather(gathered, padded)
            contributions = [
                block[:, :width_r].cpu() for block, width_r in zip(gathered, widths)
            ]
        return canonical.apply_slot_deltas(
            [
                (
                    block[0].to(torch.long).tolist(),
                    block[1].tolist(),
                    block[2].tolist(),
                    block[3].tolist(),
                )
                for block in contributions
            ]
        )

    def _capture_joint_sampler_rng_state(self):
        state = {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch_cpu": torch.get_rng_state(),
        }
        device = torch.device(self._device)
        if device.type == "cuda" and torch.cuda.is_available():
            state["torch_cuda"] = torch.cuda.get_rng_state(device)
        return state

    def _restore_joint_sampler_rng_state(self, rng_state, *, rank, world_size):
        if not isinstance(rng_state, dict):
            raise TypeError("Joint sampler RNG state must be a mapping")
        if int(rng_state.get("world_size", -1)) != world_size:
            raise ValueError(
                "Checkpoint sampler RNG world size differs from runtime"
            )
        rank_states = rng_state.get("rank_states")
        if not isinstance(rank_states, list) or len(rank_states) != world_size:
            raise ValueError(
                "Checkpoint sampler RNG state does not cover every rank"
            )
        local = rank_states[rank]
        if not isinstance(local, dict):
            raise TypeError(f"Checkpoint RNG state for rank {rank} is invalid")
        random.setstate(local["python"])
        np.random.set_state(tuple(local["numpy"]))
        torch.set_rng_state(local["torch_cpu"].cpu())
        if "torch_cuda" in local:
            device = torch.device(self._device)
            if device.type != "cuda" or not torch.cuda.is_available():
                raise ValueError(
                    "Checkpoint has CUDA RNG state but runtime is not CUDA"
                )
            torch.cuda.set_rng_state(local["torch_cuda"].cpu(), device)

    def _sync_joint_source_count_deltas(self, accelerator, world_size):
        source_delta = torch.cat(
            [
                self.joint_source_num_episodes
                - self.joint_source_episode_baseline,
                self.joint_source_num_failures
                - self.joint_source_failure_baseline,
            ]
        )
        if bool((source_delta < -1.0e-5).any().item()):
            raise ValueError("Joint source counters have negative deltas")
        source_delta = source_delta.clamp_min(0.0)
        if world_size > 1:
            if accelerator is None:
                raise ValueError(
                    "Accelerator is required for distributed source sync"
                )
            source_delta = accelerator.gather(source_delta).reshape(
                world_size, -1
            ).sum(dim=0)
        episodes_delta, failures_delta = source_delta.chunk(2)
        self.joint_source_global_num_episodes += episodes_delta
        self.joint_source_global_num_failures += failures_delta
        self.joint_source_num_episodes.copy_(
            self.joint_source_global_num_episodes
        )
        self.joint_source_num_failures.copy_(
            self.joint_source_global_num_failures
        )
        self.joint_source_episode_baseline.copy_(
            self.joint_source_num_episodes
        )
        self.joint_source_failure_baseline.copy_(
            self.joint_source_num_failures
        )

    def _sync_joint_distributed_sampler(self, accelerator=None):
        rank, world_size = self._joint_sampler_rank_world_size(accelerator)
        canonical = getattr(self, "_joint_sampler_canonical", None)
        if canonical is None:
            # Lifecycle start (fresh init or neutral reset): one gather of every rank's static keys,
            # metadata and deltas establishes the global key->slot table with the prior once per bin.
            # After a checkpoint restore the table comes from the restored aggregate.
            canonical = self._establish_joint_sampler_table(world_size=world_size)
        else:
            # Steady state: compact slot deltas only (no record dicts, normalization or fingerprints).
            canonical = self._sync_joint_sampler_slot_deltas(
                canonical, world_size=world_size
            )
        self._joint_sampler_canonical = canonical
        self._apply_joint_sampler_canonical(canonical)
        self._sync_joint_source_count_deltas(accelerator, world_size)
        # Identity -> object records of every rank's motions are static for the roster lifecycle:
        # gathered once per lifecycle (first sync after init / restore / neutral reset), then reused.
        if not getattr(self, "_joint_object_records_synced", False):
            local_object_records = dict(self.joint_source_object_records)
            if world_size == 1:
                gathered_object_records = [local_object_records]
            else:
                gathered_object_records = [None] * world_size
                torch.distributed.all_gather_object(
                    gathered_object_records,
                    local_object_records,
                )
            self.joint_source_object_records_global = (
                merge_object_records_by_identity(gathered_object_records)
            )
            self._joint_object_records_synced = True

        local_rng_state = self._capture_joint_sampler_rng_state()
        if world_size == 1:
            rank_rng_states = [local_rng_state]
        else:
            rank_rng_states = [None] * world_size
            torch.distributed.all_gather_object(
                rank_rng_states,
                local_rng_state,
            )
        # Materialized (fresh records, deep-copied RNG state) on the first read, i.e. at
        # checkpoint time, not on every synchronization.
        self.joint_sampler_checkpoint = None
        self._joint_sampler_checkpoint_pending = (
            canonical,
            {"world_size": world_size, "rank_states": rank_rng_states},
        )
        self.joint_sampler_sync_serial = getattr(self, "joint_sampler_sync_serial", 0) + 1

    def _parent_motion_identity_by_alias(self, source_state):
        """Alias -> semantic motion identity of the parent roster, from the checkpoint.

        Uses the identity map embedded in the joint-source checkpoint state when
        present, otherwise the parent roster manifest's records (read by path).
        """
        if not isinstance(source_state, Mapping):
            raise ValueError(
                "Roster transition requires joint-source checkpoint metadata"
            )

        embedded = source_state.get("motion_identity_by_alias")
        if isinstance(embedded, Mapping):
            normalized = {
                str(alias): str(identity)
                for alias, identity in embedded.items()
            }
            if normalized and all(
                alias and identity for alias, identity in normalized.items()
            ):
                return normalized
            raise ValueError(
                "Checkpoint motion_identity_by_alias contains empty values"
            )

        manifest_value = source_state.get("manifest")
        if not isinstance(manifest_value, str) or not manifest_value:
            raise ValueError(
                "Legacy joint-source checkpoint has no parent roster path"
            )
        manifest_path = Path(manifest_value).expanduser().resolve()
        if not manifest_path.is_file():
            raise FileNotFoundError(
                f"Parent roster manifest is unavailable: {manifest_path}"
            )
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        records = manifest.get("records")
        if not isinstance(records, list) or not records:
            raise ValueError(f"{manifest_path}: parent roster records are empty")
        identities = {}
        for record in records:
            if not isinstance(record, Mapping):
                raise TypeError(f"{manifest_path}: invalid parent roster record")
            alias = str(record.get("alias", ""))
            source = str(record.get("source", ""))
            source_motion_key = str(record.get("source_motion_key", ""))
            if not alias or not source or not source_motion_key:
                raise ValueError(
                    f"{manifest_path}: incomplete parent roster record"
                )
            identity = f"{source}::{source_motion_key}"
            previous = identities.setdefault(alias, identity)
            if previous != identity:
                raise ValueError(
                    f"{manifest_path}: alias {alias!r} has multiple identities"
                )
        return identities

    def _joint_sampler_candidate_catalog(self, *, world_size):
        local_bins = [
            BinStatistics(
                key=key,
                episode_count=self.adp_samp_initial_episode_prior,
                failure_count=self.adp_samp_initial_episode_prior,
                terminal_failure_count=0.0,
                metadata=self.joint_sampler_bin_metadata[index],
            ).to_record()
            for index, key in enumerate(self.joint_sampler_bin_keys)
        ]
        if world_size == 1:
            return local_bins
        if (
            not torch.distributed.is_available()
            or not torch.distributed.is_initialized()
        ):
            raise RuntimeError(
                "torch.distributed must be initialized to collect the candidate "
                "roster sampler catalog"
            )
        gathered = [None] * world_size
        torch.distributed.all_gather_object(gathered, local_bins)
        if any(not isinstance(value, list) for value in gathered):
            raise RuntimeError(
                "Candidate roster sampler catalog did not cover every rank"
            )
        return [
            record
            for rank_records in gathered
            for record in rank_records
        ]

    @staticmethod
    def _sampler_bins_match_catalog(canonical, catalog_records):
        """Does the restored keyed state cover exactly this lifecycle's roster?

        Compared on actual values: the full set of bin keys (source, alias,
        frame interval) and each bin's canonical metadata (motion identity,
        revision label, frame count, fps). The aggregate side is the private
        canonical state built once from the deserialized checkpoint (keys +
        canonical metadata); the catalog side is the gathered candidate records
        every rank produced from its own validated bins. Metadata equality is
        the strict JSON-fingerprint equality (``canonical_metadata_equal``)
        without a fingerprint string per bin, and without re-validating
        records that were validated when they were produced.
        """
        catalog = {}
        for record in catalog_records:
            catalog[BinKey.from_record(record)] = record.get("metadata", {})
        if len(catalog) != len(canonical.keys):
            return False
        missing = object()
        for key, metadata in zip(canonical.keys, canonical.metadata, strict=True):
            other = catalog.get(key, missing)
            if other is missing or not canonical_metadata_equal(metadata, other):
                return False
        return True

    def _restore_joint_distributed_sampler(
        self, checkpoint, source_state=None, restore_plan=None
    ):
        rank, world_size = self._joint_sampler_rank_world_size()
        # Ordinary validation first (contract, shapes, finite counts); an exact
        # prior/multiplier rebase then moves every saved bin into the live
        # semantics BEFORE any roster migration adds live-prior bins.
        restored = rebase_sampler_checkpoint_counts(
            checkpoint, plan=restore_plan or {"action": RESTORE_RETAIN}
        )
        saved_checkpoint = checkpoint  # raw dict: keeps the compact previous_success_rate
        checkpoint = restored
        # One private state from the payload that deserialize/rebase just normalized and validated: it serves the
        # roster comparison below and becomes the live state. No second normalization through the public setter
        # and no per-bin JSON fingerprints (measured at 706,802 bins: those two passes were
        # ~47 of the ~64 s of pure-Python restore work).
        canonical = CanonicalAggregate.from_normalized(restored["aggregate"])
        candidate_catalog = self._joint_sampler_candidate_catalog(
            world_size=world_size
        )
        roster_transitioned = not self._sampler_bins_match_catalog(
            canonical, candidate_catalog
        )
        if roster_transitioned:
            if not self.allow_roster_transition:
                raise ValueError(
                    "Sampler checkpoint bins do not match the active roster "
                    "(motion identity, alias, frame-bin layout or bin metadata "
                    "changed); set adaptive_sampling.allow_roster_transition=true "
                    "to migrate keyed statistics across the roster change"
                )
            parent_identity_by_alias = self._parent_motion_identity_by_alias(
                source_state
            )
            restored = migrate_sampler_checkpoint_roster_transition(
                checkpoint,
                candidate_bins=candidate_catalog,
                parent_motion_identity_by_alias=parent_identity_by_alias,
                initial_episode_prior=self.adp_samp_initial_episode_prior,
                initial_failure_prior=self.adp_samp_initial_episode_prior,
                replacement_motion_identities=(
                    self.roster_replacement_motion_identities
                ),
            )
            transition = restored.get("transition", {})
            logger.info(
                "[JointSampler] roster transition applied: added={} replaced={}",
                transition.get("added_motion_identities", [])[:8],
                transition.get("replaced_motion_identities", [])[:8],
            )
            # the migration output keeps its full public validation (rare path)
            canonical = CanonicalAggregate.from_payload(restored["aggregate"])
        if canonical.world_size != world_size:
            raise ValueError(
                "Checkpoint sampler world size differs from runtime"
            )
        # same effect as the public setter (private state replaced, cached apply positions dropped)
        # without its second full validation of a payload validated a moment ago
        self._joint_sampler_canonical = canonical
        self._joint_sampler_apply_positions = None
        self._apply_joint_sampler_canonical(canonical)
        self._joint_object_records_synced = False
        # Every rank restores the globally keyed counts, while only rank zero
        # writes the outer trainer checkpoint. Seed learning-progress history
        # from those counts so nonzero ranks do not see a synthetic progress
        # spike on the first update after a four-hour autoresume.
        restored_success_rate = (
            1.0
            - self.adp_samp_num_failures
            / self.adp_samp_num_episodes.clamp_min(1.0)
        ).clamp(0.0, 1.0)
        self.adp_samp_previous_success_rate.copy_(
            restored_success_rate.float()
        )
        self._restore_joint_sampler_previous_success(saved_checkpoint)
        # The next save materializes (collective-free) from the live private state, the restored
        # rank RNG states and the live previous_success_rate, so a save before the first
        # post-restore sync still carries the compact history.
        self.joint_sampler_checkpoint = None
        self._joint_sampler_checkpoint_pending = (
            self._joint_sampler_canonical,
            restored["rng_state"],
        )
        self._restore_joint_sampler_rng_state(
            restored["rng_state"],
            rank=rank,
            world_size=world_size,
        )
        return roster_transitioned

    def _update_joint_source_tracking(self, failure, dataset_motion_ids):
        if not self.use_joint_source_tracking or len(dataset_motion_ids) == 0:
            return
        source_ids = self.joint_source_motion_ids[dataset_motion_ids]
        counts = torch.bincount(
            source_ids, minlength=len(self.joint_source_names)
        ).to(self.joint_source_num_episodes)
        self.joint_source_num_episodes += counts
        if failure.any():
            failure_counts = torch.bincount(
                source_ids[failure], minlength=len(self.joint_source_names)
            ).to(self.joint_source_num_failures)
            self.joint_source_num_failures += failure_counts

    def get_joint_source_statistics(self):
        """Return source-level completed-episode statistics for roster rotation."""
        if not self.use_joint_source_tracking:
            return {}
        observed_success = 1.0 - (
            self.joint_source_num_failures
            / self.joint_source_num_episodes.clamp_min(1.0)
        )
        success = torch.where(
            self.joint_source_num_episodes > 0,
            observed_success,
            self.joint_source_previous_success_rate,
        )
        return {
            name: {
                "episode_count": float(
                    self.joint_source_num_episodes[index].item()
                ),
                "failure_count": float(
                    self.joint_source_num_failures[index].item()
                ),
                "success_rate": float(success[index].item()),
                "previous_success_rate": float(
                    self.joint_source_previous_success_rate[index].item()
                ),
            }
            for index, name in enumerate(self.joint_source_names)
        }

    def get_joint_source_statistics_tensor(self):
        """Return one on-device ``(num_sources, 3)`` snapshot of source metrics.

        Backs ``runtime_sync_cleanup.batch_joint_source_metrics``: the same episode,
        failure and success-rate values as :meth:`get_joint_source_statistics`
        without per-source scalar host synchronizations. The snapshot is newly
        allocated and does not alias the live counters.
        """
        if not self.use_joint_source_tracking:
            return (), None
        from gear_sonic.utils.runtime_sync_cleanup import joint_source_metric_snapshot

        values = joint_source_metric_snapshot(
            self.joint_source_num_episodes,
            self.joint_source_num_failures,
            self.joint_source_previous_success_rate,
        )
        return self.joint_source_names, values

    def _adaptive_state_by_motion(self):
        """Serialize bin statistics by stable source/original-motion identity.

        The three per-bin arrays are gathered for every rank-local motion with ONE
        indexed read each and moved to the host once (three device-to-host
        transfers in total instead of three per motion). The per-motion slices,
        their order, dtypes, shapes, the duplicate-identity consistency check and
        the host-side mean over duplicate roster motions are exactly those of the
        per-motion form: the checkpoint entry is state-content equivalent (same
        keys, order, values, dtypes, shapes and host device; verified by the
        container equivalence test, not by serialized-byte identity).
        """
        if not self.use_adaptive_sampling or not self.use_joint_source_tracking:
            return {}
        identities = tuple(self.joint_source_motion_identity)
        if not identities:
            return {}
        bin_lists = [
            self.orig_motion_id_to_bins[motion_id].reshape(-1)
            for motion_id in range(len(identities))
        ]
        lengths = [int(bins.numel()) for bins in bin_lists]
        all_bins = torch.cat(bin_lists) if sum(lengths) > 0 else None

        def gathered(values):
            if all_bins is None:
                return values[:0].detach().cpu()
            return values[all_bins].detach().cpu()

        episodes = torch.split(gathered(self.adp_samp_num_episodes), lengths)
        failures = torch.split(gathered(self.adp_samp_num_failures), lengths)
        previous = torch.split(gathered(self.adp_samp_previous_success_rate), lengths)
        grouped = {}
        for motion_id, identity in enumerate(identities):
            grouped.setdefault(identity, []).append(
                (episodes[motion_id], failures[motion_id], previous[motion_id])
            )
        output = {}
        for identity, values in grouped.items():
            lengths = {len(item[0]) for item in values}
            if len(lengths) != 1:
                raise ValueError(
                    f"Duplicate roster motion {identity} has inconsistent bin counts"
                )
            output[identity] = {
                "num_episodes": torch.stack([item[0] for item in values]).mean(0),
                "num_failures": torch.stack([item[1] for item in values]).mean(0),
                "previous_success_rate": torch.stack(
                    [item[2] for item in values]
                ).mean(0),
            }
        return output

    def get_object_records(self):
        """Return ``{identity: {source, alias[, object_key][, asset_signature]}}``.

        Identities are ``"<source>::<source_motion_key>"``. After a keyed sync
        (or a checkpoint restore) the mapping covers every rank's motions;
        before that it covers the motions this rank loads. Empty when source
        identities are not tracked. The returned dict is a copy.
        """
        if not self.use_joint_source_tracking:
            return {}
        return {
            identity: dict(record)
            for identity, record in self.joint_source_object_records_global.items()
        }

    def get_state_dict(self):
        """Return a serializable state dict for checkpointing adaptive sampling stats.

        Returns:
            Dict containing ``adp_samp_num_episodes`` and ``adp_samp_num_failures``
            tensors if adaptive sampling is enabled, otherwise an empty dict.
        """
        state_dict = {}
        if self.use_adaptive_sampling:
            if (
                self.use_joint_distributed_sampler
                and self.joint_sampler_checkpoint is None
            ):
                if self.joint_source_manifest_world_size != 1:
                    raise RuntimeError(
                        "The distributed adaptive sampler has not completed a "
                        "global sync; refusing an incomplete checkpoint"
                    )
                self._sync_joint_distributed_sampler()
            state_dict.update(
                {
                    "adp_samp_num_episodes": self.adp_samp_num_episodes,
                    "adp_samp_num_failures": self.adp_samp_num_failures,
                    "adp_samp_terminal_failure_counts": (
                        self.adp_samp_terminal_failure_counts
                    ),
                    "adp_samp_previous_success_rate": self.adp_samp_previous_success_rate,
                }
            )
            if not self.use_joint_distributed_sampler:
                # Without the keyed global table the per-motion form is the identity-keyed
                # representation; with it the keyed checkpoint (counts by key + compact
                # previous_success_rate by slot) is the single representation.
                state_dict["adaptive_sampling_by_motion"] = (
                    self._adaptive_state_by_motion()
                )
            # Ordinary provenance of the saved counts (optional for readers, no collective, no
            # sidecar): how episodes were counted and which prior/weighting produced them, so a
            # later leg or an offline consumer never interprets old statistics with a newer
            # default. Absent in older checkpoints -> consumers treat the mode as unknown. The
            # counts saved here are live-semantics counts by construction: load_state_dict
            # retains, exactly rebases or neutralizes saved history (``last_restore``).
            state_dict[PROVENANCE_KEY] = {
                **self._adaptive_sampling_provenance(),
                "last_restore": str(
                    getattr(self, "adaptive_last_restore", "none")
                ),
            }
            if self.use_joint_distributed_sampler:
                state_dict["joint_distributed_sampler"] = (
                    self.joint_sampler_checkpoint
                )
            if not self.use_joint_source_tracking:
                # Ordered motion keys: the only identity an untracked roster
                # has, required to accept the positional flat arrays later.
                state_dict["adp_samp_motion_keys"] = [
                    str(key) for key in self._motion_data_keys
                ]
        if self.use_joint_source_tracking:
            state_dict["joint_source_sampling"] = {
                "source_names": list(self.joint_source_names),
                "num_episodes": self.joint_source_num_episodes,
                "num_failures": self.joint_source_num_failures,
                "previous_success_rate": self.joint_source_previous_success_rate,
                "manifest": str(self.joint_source_manifest_path),
                "world_size": self.joint_source_manifest_world_size,
                "motion_identity_by_alias": dict(
                    zip(
                        self.joint_source_motion_aliases,
                        self.joint_source_motion_identity,
                        strict=True,
                    )
                ),
                # Additive: identity -> object record of every rank's motions
                # (gathered by the keyed sync). Old checkpoints lack it.
                "object_key_by_identity": self.get_object_records(),
            }
        return state_dict

    def _adaptive_sampling_provenance(self):
        """The live counting semantics of the per-bin/source counts (primitives only)."""
        cfg = self.adaptive_sampling_cfg
        episode_only = bool(cfg.get("update_on_episode_end_only", False))
        return {
            "counting_mode": "episode" if episode_only else "step",
            "update_on_episode_end_only": episode_only,
            "credit_sampled_start_bin": bool(cfg.get("credit_sampled_start_bin", False)),
            "init_num_failures": float(self.adp_samp_initial_episode_prior),
            "failure_counts_multiplier": float(cfg.get("failure_counts_multiplier", 1)),
            "priority_mode": str(cfg.get("priority_mode", "failure_rate")),
            "sequence_length_agnostic": bool(cfg.get("sequence_length_agnostic", True)),
            "bin_size": int(cfg.get("bin_size", 50)),
            "uniform_sampling_rate": float(cfg.get("uniform_sampling_rate", 0.1)),
            "distributed_keyed_sync": bool(self.use_joint_distributed_sampler),
            "joint_source_tracking": bool(self.use_joint_source_tracking),
        }

    def _plan_adaptive_restore(self, state_dict):
        """Compare the checkpoint's recorded counting semantics with the live ones.

        Pure function of the checkpoint and the live config (identical on every
        rank, no collective). See ``distributed_sampler.plan_adaptive_restore``.
        """
        plan = plan_adaptive_restore(
            state_dict.get(PROVENANCE_KEY),
            self._adaptive_sampling_provenance(),
            saved_counts_present=any(
                state_dict.get(key) is not None
                for key in ADAPTIVE_COUNT_STATE_KEYS
            ),
        )
        if plan["action"] == RESTORE_NEUTRAL:
            logger.info(
                "adaptive sampler: saved statistics are not comparable with "
                "the live counting semantics ({}); starting from the neutral "
                "prior (per-bin counts, previous_success_rate and source-level "
                "counts are not carried)",
                plan["reason"],
            )
        elif plan["action"] == RESTORE_REBASE:
            logger.info(
                "adaptive sampler: {} (saved mode {}); the saved history is "
                "kept in the live semantics",
                plan["reason"],
                plan["saved_mode"],
            )
        elif plan["saved_mode"] == "unknown":
            logger.info("adaptive sampler: {}", plan["reason"])
        return plan

    def _rebase_saved_count_tensors(self, episodes, failures, plan):
        """Saved (episode, failure) count tensors moved into the live prior/multiplier."""
        if plan["action"] != RESTORE_REBASE:
            return episodes, failures
        episodes = episodes.to(self._device, dtype=torch.float32)
        failures = failures.to(self._device, dtype=torch.float32)
        saved_prior = float(plan["saved_prior"])
        live_prior = float(plan["live_prior"])
        rebased_episodes = (episodes + float(plan["prior_delta"])).clamp_min(0.0)
        rebased_failures = live_prior + (failures - saved_prior).clamp_min(0.0) * float(
            plan["failure_scale"]
        )
        return rebased_episodes, rebased_failures

    def _reset_joint_sampler_to_neutral_prior(self):
        """Keyed sampler state of a fresh start: no aggregate, baselines = live.

        The live per-bin tensors are not modified (a true completed episode is
        never dropped); with the baselines equal to them the first keyed sync
        sends every bin once and merges only post-load local deltas, and
        the lifecycle-start table adds the global prior once per bin.
        """
        self.joint_sampler_global_aggregate = None
        self.joint_sampler_checkpoint = None
        self._joint_object_records_synced = False
        self.joint_sampler_episode_baseline.copy_(self.adp_samp_num_episodes)
        self.joint_sampler_failure_baseline.copy_(self.adp_samp_num_failures)
        self.joint_sampler_terminal_failure_baseline.copy_(
            self.adp_samp_terminal_failure_counts
        )

    def _restore_object_records(self, source_state):
        """Merge a checkpoint's ``object_key_by_identity`` (optional) with the live records.

        Old checkpoints without the key are accepted. A present but malformed
        mapping raises the ordinary errors. Live manifest records win for the
        motions this rank loads; the next keyed sync replaces the mapping with
        the gathered union of every rank's live records.
        """
        saved = source_state.get("object_key_by_identity")
        restored = {} if saved is None else normalize_object_records(saved)
        self.joint_source_object_records_global = {
            **restored,
            **self.joint_source_object_records,
        }

    def _positional_adaptive_restore_refusal(self, state_dict, source_state):
        """Why the positional flat-array restore is refused, or None if allowed.

        Rule: the checkpoint's ordered alias/identity list must equal the live
        roster's (tracked: ``joint_source_sampling.motion_identity_by_alias``,
        or the ``adaptive_sampling_by_motion`` identities together with their
        per-motion bin counts; untracked: ``adp_samp_motion_keys``). Equal
        tensor length alone is never identity.
        """
        saved_episodes = state_dict.get("adp_samp_num_episodes")
        saved_failures = state_dict.get("adp_samp_num_failures")
        if (
            saved_episodes is None
            or saved_failures is None
            or len(saved_episodes) != len(self.adp_samp_num_episodes)
            or len(saved_failures) != len(self.adp_samp_num_failures)
        ):
            return "checkpoint bin count differs from the live roster"
        if self.use_joint_source_tracking:
            live = list(
                zip(
                    self.joint_source_motion_aliases,
                    self.joint_source_motion_identity,
                    strict=True,
                )
            )
            saved_map = (
                source_state.get("motion_identity_by_alias")
                if isinstance(source_state, Mapping)
                else None
            )
            if isinstance(saved_map, Mapping):
                saved = [
                    (str(alias), str(identity))
                    for alias, identity in saved_map.items()
                ]
                if saved == live:
                    return None
                return (
                    "checkpoint alias/identity list differs from the live "
                    "roster (equal bin count is not identity)"
                )
            by_motion = state_dict.get("adaptive_sampling_by_motion")
            if isinstance(by_motion, Mapping) and by_motion:
                live_identities = list(
                    dict.fromkeys(self.joint_source_motion_identity)
                )
                if [str(key) for key in by_motion] != live_identities:
                    return (
                        "checkpoint motion identities differ from the live "
                        "roster (equal bin count is not identity)"
                    )
                for motion_id, identity in enumerate(
                    self.joint_source_motion_identity
                ):
                    value = by_motion.get(identity)
                    if not isinstance(value, Mapping) or len(
                        value.get("num_episodes", [])
                    ) != len(self.orig_motion_id_to_bins[motion_id]):
                        return (
                            f"checkpoint bin layout of {identity} differs "
                            "from the live roster"
                        )
                return None
            return "checkpoint carries no alias/identity list"
        saved_keys = state_dict.get("adp_samp_motion_keys")
        if saved_keys is None:
            return "checkpoint carries no ordered motion key list"
        live_keys = [str(key) for key in self._motion_data_keys]
        if [str(key) for key in saved_keys] == live_keys:
            return None
        return (
            "checkpoint motion key list differs from the live motion set "
            "(equal bin count is not identity)"
        )

    def load_state_dict(self, state_dict):
        """Restore adaptive sampling statistics from a checkpoint.

        Per-bin statistics are restored only through semantic identity: the
        keyed global aggregate (``joint_distributed_sampler``), else the
        per-motion identity map (``adaptive_sampling_by_motion``), else the
        positional flat arrays when the checkpoint's ordered alias/identity
        list equals the live roster's. Equal tensor length alone never allows
        a positional restore; the live prior is kept (one log line).

        Counting semantics: the checkpoint's ``adaptive_sampling_provenance``
        (optional) is compared with the live config first. Equal semantics
        retain the counts; a changed prior/failure multiplier is rebased
        exactly; a different or unknown counting mode under live episode
        counting, a changed bin credit or bin size start from the neutral
        prior (one diagnostic line). Policy/optimizer state and the ordinary
        load/shape errors are untouched.

        Args:
            state_dict: Dict previously returned by ``get_state_dict()``.
        """
        source_state = state_dict.get("joint_source_sampling")
        distributed_state = state_dict.get("joint_distributed_sampler")
        restored_distributed = False
        roster_transitioned = False
        # Counting semantics of the saved counts vs the live config (optional
        # provenance written with the counts; no admission gate): retain,
        # exact rebase, or neutral start. Decided before anything is copied.
        if self.use_adaptive_sampling:
            restore_plan = self._plan_adaptive_restore(state_dict)
        else:
            restore_plan = {"action": RESTORE_RETAIN, "saved_mode": "unknown"}
        semantics_neutral = restore_plan["action"] == RESTORE_NEUTRAL
        self.adaptive_last_restore = (
            f"{restore_plan['action']}:{restore_plan.get('saved_mode', 'unknown')}"
        )
        # Keyed sampler active but the checkpoint has no keyed aggregate (an
        # earlier non-adaptive or tracked-but-non-keyed run): the per-bin
        # statistics start from the neutral prior exactly like a fresh start.
        # Nothing per-bin is copied from such a checkpoint -- neither counts
        # (the keyed sync would treat them as post-load local deltas and add
        # replicated history once per rank) nor previous_success_rate (a
        # carried history would fake learning progress). Source-level counts
        # (restored as global values with matching baselines) are kept.
        keyed_neutral_prior = (
            self.use_joint_distributed_sampler and distributed_state is None
        )
        if (
            self.use_joint_distributed_sampler
            and distributed_state is not None
            and not semantics_neutral
        ):
            # Present keyed state must be valid: a malformed contract, shape
            # or nonfinite count raises the ordinary errors.
            roster_transitioned = self._restore_joint_distributed_sampler(
                distributed_state,
                source_state=source_state,
                restore_plan=restore_plan,
            )
            restored_distributed = True
        elif self.use_joint_distributed_sampler and semantics_neutral:
            # A present keyed state is still validated (ordinary errors) but
            # its counts are not expressible in the live semantics: fresh
            # keyed start (no aggregate, no RNG continuation of the sampler).
            if distributed_state is not None:
                deserialize_sampler_checkpoint(distributed_state)
            self._reset_joint_sampler_to_neutral_prior()
        elif keyed_neutral_prior:
            # Runs BEFORE any per-motion or positional restore; both are
            # skipped below.
            self._reset_joint_sampler_to_neutral_prior()
            logger.info(
                "adaptive sampler: no keyed statistics in checkpoint; "
                "starting from the neutral prior (per-bin counts and "
                "previous_success_rate are not carried)"
            )
        per_bin_neutral = keyed_neutral_prior or semantics_neutral

        if self.use_joint_source_tracking and isinstance(source_state, dict):
            old_index = {
                str(name): index
                for index, name in enumerate(source_state.get("source_names", []))
            }
            # Roster rotation is detected from the actual data: a saved
            # alias -> identity map that disagrees with the runtime roster on a
            # shared alias, or (legacy checkpoints) a different manifest path.
            saved_identity_map = source_state.get("motion_identity_by_alias")
            if isinstance(saved_identity_map, Mapping):
                runtime_identity_map = dict(
                    zip(
                        self.joint_source_motion_aliases,
                        self.joint_source_motion_identity,
                        strict=True,
                    )
                )
                roster_rotated = roster_transitioned or any(
                    runtime_identity_map.get(str(alias), str(identity))
                    != str(identity)
                    for alias, identity in saved_identity_map.items()
                )
            else:
                roster_rotated = roster_transitioned or (
                    str(source_state.get("manifest", ""))
                    != str(self.joint_source_manifest_path)
                )
            removed_sources = sorted(
                set(old_index) - set(self.joint_source_names)
            )
            if roster_rotated and removed_sources:
                raise ValueError(
                    "Roster transition removed checkpoint "
                    f"sources: {removed_sources}"
                )
            for current_index, name in enumerate(self.joint_source_names):
                previous_index = old_index.get(name)
                if previous_index is None or semantics_neutral:
                    # neutral: source-level counts share the saved counting
                    # semantics and are not carried either (roster checks
                    # above and the object records below still apply)
                    continue
                previous_episodes = source_state["num_episodes"][
                    previous_index
                ].to(self._device)
                previous_failures = source_state["num_failures"][
                    previous_index
                ].to(self._device)
                self.joint_source_num_episodes[
                    current_index
                ] = previous_episodes
                self.joint_source_num_failures[
                    current_index
                ] = previous_failures
                if "previous_success_rate" in source_state:
                    self.joint_source_previous_success_rate[current_index] = source_state[
                        "previous_success_rate"
                    ][previous_index].to(self._device)
                elif float(previous_episodes.item()) > 0.0:
                    self.joint_source_previous_success_rate[
                        current_index
                    ] = 1.0 - (
                        previous_failures / previous_episodes.clamp_min(1.0)
                    )
            if self.use_joint_distributed_sampler:
                self.joint_source_global_num_episodes.copy_(
                    self.joint_source_num_episodes
                )
                self.joint_source_global_num_failures.copy_(
                    self.joint_source_num_failures
                )
                self.joint_source_episode_baseline.copy_(
                    self.joint_source_num_episodes
                )
                self.joint_source_failure_baseline.copy_(
                    self.joint_source_num_failures
                )
            self._restore_object_records(source_state)

        restored_by_motion = restored_distributed
        by_motion = state_dict.get("adaptive_sampling_by_motion")
        if (
            self.use_adaptive_sampling
            and self.use_joint_source_tracking
            and not per_bin_neutral
            and isinstance(by_motion, dict)
        ):
            for motion_id, identity in enumerate(self.joint_source_motion_identity):
                value = by_motion.get(identity)
                if not isinstance(value, dict):
                    continue
                bin_ids = self.orig_motion_id_to_bins[motion_id]
                if len(value.get("num_episodes", [])) != len(bin_ids):
                    continue
                if not restored_distributed:
                    episodes, failures = self._rebase_saved_count_tensors(
                        value["num_episodes"], value["num_failures"], restore_plan
                    )
                    self.adp_samp_num_episodes[bin_ids] = episodes.to(self._device)
                    self.adp_samp_num_failures[bin_ids] = failures.to(self._device)
                if "previous_success_rate" in value:
                    self.adp_samp_previous_success_rate[bin_ids] = value[
                        "previous_success_rate"
                    ].to(self._device)
                if not restored_distributed:
                    restored_by_motion = True

        if (
            self.use_adaptive_sampling
            and not restored_by_motion
            and not per_bin_neutral
            and "adp_samp_num_episodes" in state_dict
        ):
            refusal = self._positional_adaptive_restore_refusal(
                state_dict, source_state
            )
            if refusal is None:
                episodes, failures = self._rebase_saved_count_tensors(
                    state_dict["adp_samp_num_episodes"],
                    state_dict["adp_samp_num_failures"],
                    restore_plan,
                )
                self.adp_samp_num_episodes[:] = episodes.to(self._device)
                self.adp_samp_num_failures[:] = failures.to(self._device)
                if "adp_samp_terminal_failure_counts" in state_dict:
                    self.adp_samp_terminal_failure_counts[:] = state_dict[
                        "adp_samp_terminal_failure_counts"
                    ].to(self._device)
                if "adp_samp_previous_success_rate" in state_dict:
                    self.adp_samp_previous_success_rate[:] = state_dict[
                        "adp_samp_previous_success_rate"
                    ].to(self._device)
            else:
                logger.info(
                    "adaptive sampler: positional per-bin restore refused "
                    "({}); keeping the live prior and the source-level state",
                    refusal,
                )
        if self.use_adaptive_sampling:
            self.sync_and_compute_adaptive_sampling(sync_across_gpus=False)
        return

    def update_adaptive_sampling(
        self,
        failure,
        motion_ids,
        motion_time_steps,
        sampled_start_time_steps=None,
    ):
        """Update adaptive sampling statistics based on training outcomes.

        Increments episode counts for all sampled bins, and failure counts for bins
        where the policy terminated early. Uses bincount for efficient batched updates
        when multiple environments hit the same bin.

        Args:
            failure: Boolean tensor of shape ``(N,)`` indicating which environments
                terminated due to failure (not timeout).
            motion_ids: Tensor of shape ``(N,)`` with batch-local motion indices.
            motion_time_steps: Tensor of shape ``(N,)`` with the simulation time step
                at which the episode ended (or was sampled).
            sampled_start_time_steps: Optional ``(N,)`` tensor with the sampled
                episode-start step. The opt-in joint curriculum credits this bin
                for the episode outcome while retaining terminal bins separately
                for diagnostics.
        """
        # Convert motion_ids to dataset motion ids if needed
        dataset_motion_ids = self.get_motion_ids_in_dataset(motion_ids)
        self._update_joint_source_tracking(failure, dataset_motion_ids)

        motion_fps = self._motion_fps[motion_ids]
        dataset_frame_steps = (
            motion_time_steps.float() * motion_fps / float(self._sim_fps)
        ).floor().long()
        dataset_frame_steps = torch.minimum(
            dataset_frame_steps,
            self.adp_samp_num_frames[dataset_motion_ids] - 1,
        ).clamp_min(0)
        terminal_time_steps = (
            self.adp_samp_length_starts[dataset_motion_ids] + dataset_frame_steps
        )
        credit_time_steps = terminal_time_steps
        if (
            sampled_start_time_steps is not None
            and self.adaptive_sampling_cfg.get("credit_sampled_start_bin", False)
        ):
            start_frame_steps = (
                sampled_start_time_steps.float()
                * motion_fps
                / float(self._sim_fps)
            ).floor().long()
            start_frame_steps = torch.minimum(
                start_frame_steps,
                self.adp_samp_num_frames[dataset_motion_ids] - 1,
            ).clamp_min(0)
            credit_time_steps = (
                self.adp_samp_length_starts[dataset_motion_ids] + start_frame_steps
            )

        # Handle non-unique dataset_motion_ids by counting occurrences
        if len(credit_time_steps) > 0:
            # Use bincount to count occurrences of each unique ID
            bin_ids = self.adp_samp_frame_to_bin[credit_time_steps]
            # These are episode counts, not frame-density estimates. Dividing
            # only this denominator by bin length made failure rates exceed 1
            # and biased the curriculum toward longer bins.
            counts = torch.bincount(bin_ids, minlength=self.adp_samp_num_bins)
            self.adp_samp_num_episodes += counts

        # Update failure counts for failed motions
        if failure.any():
            failed_time_steps = credit_time_steps[failure]
            # Handle non-unique failed motion IDs by counting occurrences
            if len(failed_time_steps) > 0:
                bin_ids = self.adp_samp_frame_to_bin[failed_time_steps]
                failure_counts = torch.bincount(bin_ids, minlength=self.adp_samp_num_bins)
                failure_counts_multiplier = self.adaptive_sampling_cfg.get(
                    "failure_counts_multiplier", 1
                )
                self.adp_samp_num_failures += failure_counts * failure_counts_multiplier
            terminal_failed_steps = terminal_time_steps[failure]
            if len(terminal_failed_steps) > 0:
                terminal_bin_ids = self.adp_samp_frame_to_bin[
                    terminal_failed_steps
                ]
                self.adp_samp_terminal_failure_counts += torch.bincount(
                    terminal_bin_ids, minlength=self.adp_samp_num_bins
                )

    def sync_and_compute_adaptive_sampling(self, accelerator=None, sync_across_gpus=False):
        """Synchronize adaptive sampling stats across GPUs and recompute probabilities.

        In multi-GPU training, averages episode/failure counts across all processes
        before recomputing the per-bin sampling distribution. Optionally applies
        failure-rate decay to propagate difficulty information to preceding bins.

        Args:
            accelerator: HuggingFace Accelerator instance for multi-GPU gather.
                Required when ``sync_across_gpus=True``.
            sync_across_gpus: Whether to synchronize statistics across distributed
                processes before computing probabilities.
        """
        if not self.use_adaptive_sampling:
            return

        if sync_across_gpus:
            with common.Timer("sync_adaptive_sampling_across_gpus"):
                if self.use_joint_distributed_sampler:
                    self._sync_joint_distributed_sampler(accelerator)
                else:
                    world_size = self._known_world_size(accelerator)
                    if world_size > 1 and self.use_joint_source_tracking:
                        raise ValueError(
                            "Tracked multi-rank adaptive sampling requires "
                            "adaptive_sampling.distributed_keyed_sync=true "
                            "(the keyed sampler); positional averaging is "
                            "not valid for heterogeneous ranks"
                        )
                    if world_size > 1 and bool(
                        self.adaptive_sampling_cfg.get("require_tracked_identities", False)
                    ):
                        raise ValueError(
                            "adaptive_sampling.enable=true on a run with "
                            f"world_size {world_size} requires tracked motion "
                            "identities (motion_lib_cfg.joint_source_tracking="
                            "true); positional averaging across ranks is not "
                            "valid. Set joint_source_tracking=true or "
                            "adaptive_sampling.enable=false"
                        )
                    # Legacy positional path: average counts across ranks
                    # (identity on world_size 1). Ranks loading heterogeneous
                    # motion slices can opt into the strict check with
                    # adaptive_sampling.require_tracked_identities=true.
                    adp_samp_stats = torch.cat(
                        [
                            self.adp_samp_num_episodes,
                            self.adp_samp_num_failures,
                            self.adp_samp_terminal_failure_counts,
                        ],
                        dim=-1,
                    )
                    adp_samp_stats_all = accelerator.gather(
                        adp_samp_stats
                    ).reshape(-1, *adp_samp_stats.shape)
                    adp_samp_stats_all = adp_samp_stats_all.mean(dim=0)
                    (
                        self.adp_samp_num_episodes,
                        self.adp_samp_num_failures,
                        self.adp_samp_terminal_failure_counts,
                    ) = adp_samp_stats_all.chunk(3, dim=-1)
                if (
                    self.use_joint_source_tracking
                    and not self.use_joint_distributed_sampler
                ):
                    source_stats = torch.cat(
                        [
                            self.joint_source_num_episodes,
                            self.joint_source_num_failures,
                        ],
                        dim=-1,
                    )
                    source_stats_all = accelerator.gather(source_stats).reshape(
                        -1, *source_stats.shape
                    ).mean(dim=0)
                    (
                        self.joint_source_num_episodes,
                        self.joint_source_num_failures,
                    ) = source_stats_all.chunk(2, dim=-1)

        with common.Timer("compute_sampling_prob"):
            failure_rate = self.adp_samp_num_failures / self.adp_samp_num_episodes.clamp_min(1.0)
            self.adp_samp_failure_rate_raw = failure_rate.clone()
            self.adp_samp_failure_rate = failure_rate
            # This is to compute the failure rate with decay.
            # However, this is very slow and not necessary. We can just sample an offset before the failure happens.  # noqa: E501
            if self.adaptive_sampling_cfg.get("use_failure_rate_decay", False):
                gamma = self.adaptive_sampling_cfg.get("decay_gamma", 0.99)
                num_steps = self.adp_samp_num_episodes.shape[0]
                failure_rate_w_decay = torch.zeros_like(failure_rate)
                for step in reversed(range(num_steps)):
                    if step == num_steps - 1:
                        next_failure_rate = 0
                        next_is_not_terminal = 0.0
                    else:
                        next_failure_rate = failure_rate_w_decay[step + 1]
                        next_is_not_terminal = (
                            1.0 - self.adp_samp_bin_new_motion_mask[step + 1].float()
                        )
                    failure_rate_w_decay[step] = (
                        failure_rate[step] + next_is_not_terminal * gamma * next_failure_rate
                    )
                self.adp_samp_failure_rate = failure_rate_w_decay

            self._update_adaptive_sampling_priority()
            # Compute the sampling probability based on the failure rate
            self.update_adaptive_sampling_probabilities()
        return

    def _update_adaptive_sampling_priority(self):
        """Build the opt-in learning-progress curriculum signal.

        Raw failure rate over-samples both impossible data and sources that are
        merely numerous. The learning-progress mode instead prioritizes the
        frontier (success near 50%), recent absolute progress, and under-observed
        bins. A positive floor keeps mastered and currently hard bins represented.
        """
        mode = self.adaptive_sampling_cfg.get("priority_mode", "failure_rate")
        if mode == "failure_rate":
            self.adp_samp_priority_score = self.adp_samp_failure_rate.float()
            return
        if mode == "beta_capped_difficulty":
            success_rate = (1.0 - self.adp_samp_failure_rate_raw).clamp(
                0.0, 1.0
            )
            observed_episodes = (
                self.adp_samp_num_episodes
                - self.adp_samp_initial_episode_prior
            ).clamp_min(0.0)
            warmup = float(
                self.adaptive_sampling_cfg.get(
                    "difficulty_warmup_completed_episodes", 100.0
                )
            )
            if float(observed_episodes.sum().item()) < warmup:
                self.adp_samp_priority_score = torch.ones_like(success_rate)
            else:
                soft_floor = float(
                    self.adaptive_sampling_cfg.get(
                        "difficulty_soft_success_floor", 0.10
                    )
                )
                if not 0.0 < soft_floor <= 1.0:
                    raise ValueError(
                        "difficulty_soft_success_floor must be within (0, 1]"
                    )
                learnability = torch.clamp(
                    success_rate / soft_floor, max=1.0
                )
                inverse_difficulty = 1.0 - success_rate
                progress = (
                    success_rate - self.adp_samp_previous_success_rate
                ).abs()
                uncertainty = torch.rsqrt(observed_episodes + 1.0)
                self.adp_samp_priority_score = (
                    float(
                        self.adaptive_sampling_cfg.get(
                            "difficulty_min_weight", 0.05
                        )
                    )
                    + inverse_difficulty * learnability
                    + float(
                        self.adaptive_sampling_cfg.get(
                            "learning_progress_weight", 1.0
                        )
                    )
                    * progress
                    + float(
                        self.adaptive_sampling_cfg.get(
                            "uncertainty_weight", 0.25
                        )
                    )
                    * uncertainty
                ).float()
            self.adp_samp_previous_success_rate.copy_(success_rate.float())
            return
        if mode != "learning_progress":
            raise ValueError(f"Unknown adaptive sampling priority_mode: {mode}")

        success_rate = (1.0 - self.adp_samp_failure_rate_raw).clamp(0.0, 1.0)
        progress = (success_rate - self.adp_samp_previous_success_rate).abs()
        frontier = 4.0 * success_rate * (1.0 - success_rate)
        uncertainty_prior = float(
            self.adaptive_sampling_cfg.get("uncertainty_prior_episodes", 20.0)
        )
        uncertainty = uncertainty_prior / (
            uncertainty_prior + self.adp_samp_num_episodes.clamp_min(0.0)
        )
        min_priority = float(self.adaptive_sampling_cfg.get("min_priority", 0.05))
        self.adp_samp_priority_score = (
            min_priority
            + float(self.adaptive_sampling_cfg.get("frontier_weight", 1.0)) * frontier
            + float(
                self.adaptive_sampling_cfg.get("learning_progress_weight", 1.0)
            )
            * progress
            + float(self.adaptive_sampling_cfg.get("uncertainty_weight", 0.25))
            * uncertainty
        ).float()
        self.adp_samp_previous_success_rate.copy_(success_rate.float())

    def _adaptive_sampling_signal(self):
        if self.adaptive_sampling_cfg.get("priority_mode", "failure_rate") in {
            "learning_progress",
            "beta_capped_difficulty",
        }:
            return self.adp_samp_priority_score
        return self.adp_samp_failure_rate

    def _normalize_adaptive_scores(self, scores):
        """Normalize positive scores with the opt-in InterMimic-style beta cap."""
        scores = scores.double().clamp_min(torch.finfo(torch.float64).eps)
        if (
            self.adaptive_sampling_cfg.get("priority_mode", "failure_rate")
            != "beta_capped_difficulty"
        ):
            return scores / scores.sum()

        beta = float(
            self.adaptive_sampling_cfg.get("difficulty_beta_cap", 3.0)
        )
        if beta < 1.0:
            raise ValueError("difficulty_beta_cap must be at least one")
        count = len(scores)
        probability_cap = min(1.0, beta / max(count, 1))
        probability = torch.zeros_like(scores)
        active = torch.ones_like(scores, dtype=torch.bool)
        remaining = 1.0
        tolerance = torch.finfo(scores.dtype).eps * 16.0
        while active.any() and remaining > tolerance:
            active_scores = torch.where(active, scores, 0.0)
            proposal = remaining * active_scores / active_scores.sum()
            saturated = active & (proposal > probability_cap)
            if not saturated.any():
                probability[active] = proposal[active]
                remaining = 0.0
                break
            probability[saturated] = probability_cap
            active[saturated] = False
            remaining = 1.0 - float(probability.sum().item())
        if remaining > tolerance:
            probability += remaining / count
        return probability / probability.sum()

    def update_adaptive_sampling_probabilities(self):
        """Recompute per-bin sampling probabilities for the currently loaded motion batch.

        Blends failure-rate-based probabilities with a uniform baseline (controlled by
        ``uniform_sampling_rate``), then applies optional max-probability constraints
        per bin and per motion to prevent over-concentration on outlier sequences.
        See the inline comments for detailed rationale on the constraint design.
        """
        adaptive_signal = self._adaptive_sampling_signal().double()
        self.adp_samp_active_failure_rate = adaptive_signal[
            self.adp_samp_active_motion_bins
        ]
        adp_samp_failure_rate_upper_bound = (
            self.adp_samp_active_failure_rate.mean() * self.adp_samp_failure_rate_max_over_mean
        )
        adp_samp_active_failure_rate_clipped = torch.clip(
            self.adp_samp_active_failure_rate, 0.0, adp_samp_failure_rate_upper_bound
        )
        failure_based_sampling_prob = self._normalize_adaptive_scores(
            adp_samp_active_failure_rate_clipped
        )
        uniform_sampling_prob = torch.ones_like(failure_based_sampling_prob) / len(
            failure_based_sampling_prob
        )
        self.adp_sampling_active_prob = (
            failure_based_sampling_prob * (1 - self.uniform_sampling_rate)
            + uniform_sampling_prob * self.uniform_sampling_rate
        )
        self.adp_sampling_active_prob = self._normalize_adaptive_scores(
            self.adp_sampling_active_prob
            * self.adp_samp_bin_weights[self.adp_samp_active_motion_bins]
        )

        # ==========================================================================
        # MAX PROBABILITY CONSTRAINTS: Prevent over-concentration on challenging motions
        # ==========================================================================
        # WHY THESE CONSTRAINTS EXIST:
        # ---------------------------
        # Adaptive sampling focuses training on motions with higher failure rates.
        # Without constraints, this can cause several problems:
        #
        # 1. CATASTROPHIC FORGETTING: If one motion has 90% failure rate while others
        #    have 10%, it could dominate sampling → policy forgets "easy" motions.
        #
        # 2. TRAINING INSTABILITY: Narrow sample distribution causes high gradient
        #    variance, leading to unstable training dynamics.
        #
        # 3. OVERFITTING TO OUTLIERS: Some motions may be impossible (bad mocap data,
        #    kinematic infeasibility) but still get sampled heavily, wasting compute.
        #
        # 4. DIVERSITY LOSS: For policies that need to generalize across many motions
        #    (e.g., CHIP_token compliance training with 18k+ clips), diversity is critical.
        #
        # EFFECTS OF THESE CONSTRAINTS:
        # -----------------------------
        # - max_prob_per_bin: No single time-segment can exceed N× its fair share.
        #   Prevents over-sampling one specific "hard moment" in a motion.
        #
        # - max_prob_per_motion: No single motion clip can exceed N× its fair share.
        #   Prevents a single broken/impossible motion from dominating training.
        #
        # CONFIGURATION:
        # --------------
        # - "auto": Uses adp_samp_failure_rate_max_over_mean to set the multiplier
        #           (e.g., if failure_rate_max=2, then max_prob = 2x uniform)
        # - null/not set: SKIP these constraints entirely (for legacy configs)
        # - 0: Explicitly disable the constraint
        # - float value: Set exact max probability threshold
        #
        # For CHIP_token compliance training, we use conservative values (2x) to maintain
        # motion diversity. For other training, higher values (50x+) may be acceptable.
        # ==========================================================================

        # Skip all max_prob constraints if neither is configured (legacy behavior)
        if self.max_prob_per_bin_cfg is None and self.max_prob_per_motion_cfg is None:
            self.adp_sampling_active_prob = self.adp_sampling_active_prob.float()
            assert (self.adp_sampling_active_prob >= 0).all()
            return

        num_active_bins = len(self.adp_samp_active_motion_bins)
        active_orig_motion_ids = self.adp_samp_bins[self.adp_samp_active_motion_bins, 0]
        num_active_motions = len(active_orig_motion_ids.unique())

        # 1. Max probability per bin: no single bin can exceed this fraction of total samples
        if self.max_prob_per_bin_cfg is not None:
            if self.max_prob_per_bin_cfg == "auto":
                # Auto: use adp_samp_failure_rate_max_over_mean as multiplier
                multiplier = self.adp_samp_failure_rate_max_over_mean
                max_prob_per_bin = multiplier / num_active_bins if num_active_bins > 0 else 1.0
            else:
                max_prob_per_bin = (
                    float(self.max_prob_per_bin_cfg) if self.max_prob_per_bin_cfg else 0.0
                )

            # Only apply if constraint is meaningful (more bins than 1/max_prob)
            if max_prob_per_bin > 0 and num_active_bins > 1.0 / max_prob_per_bin:
                self.adp_sampling_active_prob = torch.clamp(
                    self.adp_sampling_active_prob, max=max_prob_per_bin
                )
                self.adp_sampling_active_prob = (
                    self.adp_sampling_active_prob / self.adp_sampling_active_prob.sum()
                )

        # 2. Max probability per motion: aggregate bins per motion and cap total
        if self.max_prob_per_motion_cfg is not None:
            if self.max_prob_per_motion_cfg == "auto":
                # Auto: use adp_samp_failure_rate_max_over_mean as multiplier
                multiplier = self.adp_samp_failure_rate_max_over_mean
                max_prob_per_motion = (
                    multiplier / num_active_motions if num_active_motions > 0 else 1.0
                )
            else:
                max_prob_per_motion = (
                    float(self.max_prob_per_motion_cfg) if self.max_prob_per_motion_cfg else 0.0
                )

            # Only apply if constraint is meaningful (more motions than 1/max_prob)
            if max_prob_per_motion > 0 and num_active_motions > 1.0 / max_prob_per_motion:
                unique_motions = active_orig_motion_ids.unique()

                for motion_id in unique_motions:
                    motion_mask = active_orig_motion_ids == motion_id
                    motion_total_prob = self.adp_sampling_active_prob[motion_mask].sum()

                    if motion_total_prob > max_prob_per_motion:
                        # Scale down all bins belonging to this motion
                        scale_factor = max_prob_per_motion / motion_total_prob
                        self.adp_sampling_active_prob[motion_mask] *= scale_factor

                # Re-normalize after capping
                self.adp_sampling_active_prob = (
                    self.adp_sampling_active_prob / self.adp_sampling_active_prob.sum()
                )
        # ==========================================================================

        self.adp_sampling_active_prob = self.adp_sampling_active_prob.float()
        assert (self.adp_sampling_active_prob >= 0).all()

    def update_adaptive_sampling_motion_sequences(self):
        """Recompute global (full-dataset) motion-level sampling probabilities.

        Called before ``load_motions()`` to determine which motions to load next.
        Aggregates per-bin failure rates into per-motion probabilities and applies
        the same max-probability constraints as the batch-level update.
        """
        adaptive_signal = self._adaptive_sampling_signal().double()
        adp_samp_failure_rate_upper_bound = (
            adaptive_signal.mean() * self.adp_samp_failure_rate_max_over_mean
        )
        adp_samp_failure_rate_clipped = torch.clip(
            adaptive_signal, 0.0, adp_samp_failure_rate_upper_bound
        )
        failure_based_sampling_prob = self._normalize_adaptive_scores(
            adp_samp_failure_rate_clipped
        )
        uniform_sampling_prob = torch.ones_like(failure_based_sampling_prob) / len(
            failure_based_sampling_prob
        )
        self.adp_sampling_prob = (
            failure_based_sampling_prob * (1 - self.uniform_sampling_rate)
            + uniform_sampling_prob * self.uniform_sampling_rate
        )
        self.adp_sampling_prob = self._normalize_adaptive_scores(
            self.adp_sampling_prob * self.adp_samp_bin_weights
        )

        # ==========================================================================
        # MAX PROBABILITY CONSTRAINTS (applied to global bin probabilities)
        # See update_adaptive_sampling_probabilities() for detailed explanation.
        # Skip if neither constraint is configured (legacy behavior).
        # ==========================================================================
        if self.max_prob_per_bin_cfg is None and self.max_prob_per_motion_cfg is None:
            # Sum up the adp_sampling_prob for each motion's frames (no constraints)
            motion_sampling_probs = torch.zeros(self._num_unique_motions, device=self._device)
            for orig_motion_id in range(self._num_unique_motions):
                motion_sampling_probs[orig_motion_id] = self.adp_sampling_prob[
                    self.orig_motion_id_to_bins[orig_motion_id]
                ].sum()
            self._sampling_prob = motion_sampling_probs / motion_sampling_probs.sum()
            return

        num_bins = self.adp_samp_num_bins
        num_motions = self._num_unique_motions

        # Apply max_prob_per_bin constraint if configured
        if self.max_prob_per_bin_cfg is not None:
            if self.max_prob_per_bin_cfg == "auto":
                # Auto: use adp_samp_failure_rate_max_over_mean as multiplier
                multiplier = self.adp_samp_failure_rate_max_over_mean
                max_prob_per_bin = multiplier / num_bins if num_bins > 0 else 1.0
            else:
                max_prob_per_bin = (
                    float(self.max_prob_per_bin_cfg) if self.max_prob_per_bin_cfg else 0.0
                )

            # Only apply if constraint is meaningful (more bins than 1/max_prob)
            if max_prob_per_bin > 0 and num_bins > 1.0 / max_prob_per_bin:
                self.adp_sampling_prob = torch.clamp(self.adp_sampling_prob, max=max_prob_per_bin)
                self.adp_sampling_prob = self.adp_sampling_prob / self.adp_sampling_prob.sum()

        # Sum up the adp_sampling_prob for each motion's frames
        motion_sampling_probs = torch.zeros(self._num_unique_motions, device=self._device)
        for orig_motion_id in range(self._num_unique_motions):
            motion_sampling_probs[orig_motion_id] = self.adp_sampling_prob[
                self.orig_motion_id_to_bins[orig_motion_id]
            ].sum()

        # Apply max_prob_per_motion constraint if configured
        if self.max_prob_per_motion_cfg is not None:
            if self.max_prob_per_motion_cfg == "auto":
                # Auto: use adp_samp_failure_rate_max_over_mean as multiplier
                multiplier = self.adp_samp_failure_rate_max_over_mean
                max_prob_per_motion = multiplier / num_motions if num_motions > 0 else 1.0
            else:
                max_prob_per_motion = (
                    float(self.max_prob_per_motion_cfg) if self.max_prob_per_motion_cfg else 0.0
                )

            # Only apply if constraint is meaningful (more motions than 1/max_prob)
            if max_prob_per_motion > 0 and num_motions > 1.0 / max_prob_per_motion:
                motion_sampling_probs = torch.clamp(motion_sampling_probs, max=max_prob_per_motion)

        self._sampling_prob = motion_sampling_probs / motion_sampling_probs.sum()

    def update_adaptive_sampling_motion_frames(self):
        """Build the active-bin index for the currently loaded motion batch.

        Maps each loaded motion to its corresponding global bins, creating
        ``adp_samp_active_motion_bins`` which is used by
        ``sample_motion_ids_and_time_steps()`` and
        ``update_adaptive_sampling_probabilities()`` to sample and update only
        the bins that correspond to currently loaded motions.
        """
        self.adp_samp_active_motion_bins = []
        self.orig_motion_id_to_motion_ids = torch.zeros(
            self._num_unique_motions, device=self._device, dtype=torch.long
        )
        for motion_id, orig_motion_id in enumerate(self._curr_motion_ids):
            bins = self.orig_motion_id_to_bins[orig_motion_id.item()]
            self.adp_samp_active_motion_bins.append(bins)
            self.orig_motion_id_to_motion_ids[orig_motion_id.item()] = motion_id

            # Validate adaptive sampling frame count matches actual loaded frames
            adp_frames = self.adp_samp_num_frames[orig_motion_id].item()
            loaded_frames = self._motion_num_frames[motion_id].item()
            assert adp_frames == loaded_frames, (
                f"Adaptive sampling frame count mismatch for motion "
                f"{orig_motion_id.item()} (key={self._motion_data_keys[orig_motion_id]}): "
                f"adp_samp={adp_frames}, loaded={loaded_frames}. "
                f"This means init_adaptive_sampling computed a different frame count "
                f"than fk_batch produced at load time."
            )

        self.adp_samp_active_motion_bins = torch.cat(self.adp_samp_active_motion_bins, dim=0)
        self.update_adaptive_sampling_probabilities()

    def sample_motion_ids_and_time_steps(self, n):
        """Sample motion IDs and time steps using adaptive sampling probabilities.

        Draws bins from the active-bin distribution, then samples a random frame
        within each selected bin. Optionally shifts the sampled frame backward by
        a random offset (``pre_failure_sample_window``) so the policy starts
        practicing before the difficult segment.

        Args:
            n: Number of (motion_id, time_step) pairs to sample.

        Returns:
            Tuple of (motion_ids, motion_time_steps) where:
                - motion_ids: ``(n,)`` long tensor with batch-local motion indices.
                - motion_time_steps: ``(n,)`` int tensor with frame indices.
        """
        sampled_bin_ids = torch.multinomial(
            self.adp_sampling_active_prob, num_samples=n, replacement=True
        ).to(self._device)
        bin_ids = self.adp_samp_active_motion_bins[sampled_bin_ids]
        bins = self.adp_samp_bins[bin_ids]
        orig_motion_ids, bin_start, bin_end = bins[:, 0], bins[:, 1], bins[:, 2]
        motion_ids = self.orig_motion_id_to_motion_ids[orig_motion_ids]

        motion_time_steps = (
            torch.rand(len(bin_start), device=bin_start.device) * (bin_end - bin_start)
        ).floor().long() + bin_start
        # Sample motion time steps before failures makes more sense since we need to take actions before the failure happens.  # noqa: E501
        pre_failure_sample_window = self.adaptive_sampling_cfg.get("pre_failure_sample_window", 0)
        if pre_failure_sample_window > 0:
            offset = torch.randint(pre_failure_sample_window, (n,), device=self._device)
            motion_time_steps = (motion_time_steps - offset).clamp_min(0)
        return motion_ids, motion_time_steps.int()

    def sample_time_steps_for_motion_ids_adaptive(self, motion_ids, truncate_time=None):
        """Sample adaptive start bins while preserving already-paired motion IDs.

        Paired motion/USD environments cannot change motion identity at reset.
        This method conditions the adaptive distribution on each assigned motion
        and samples only starts that leave the requested truncated horizon.
        """
        if not self.use_adaptive_sampling:
            return self.sample_time_steps(motion_ids, truncate_time=truncate_time)
        if len(motion_ids) == 0:
            return torch.empty_like(motion_ids, dtype=torch.int)

        sampled_time_steps = torch.zeros_like(motion_ids, dtype=torch.long)
        dataset_motion_ids = self.get_motion_ids_in_dataset(motion_ids)
        adaptive_signal = self._adaptive_sampling_signal().double()
        pre_failure_window = int(
            self.adaptive_sampling_cfg.get("pre_failure_sample_window", 0)
        )

        for motion_id in motion_ids.unique():
            assignment = motion_ids == motion_id
            count = int(assignment.sum().item())
            local_motion_id = int(motion_id.item())
            dataset_motion_id = int(dataset_motion_ids[assignment][0].item())
            bin_ids = self.orig_motion_id_to_bins[dataset_motion_id]
            bins = self.adp_samp_bins[bin_ids]

            available_seconds = self._motion_lengths[local_motion_id]
            if truncate_time is not None:
                available_seconds = torch.clamp(
                    available_seconds - float(truncate_time), min=0.0
                )
            motion_fps = float(self._motion_fps[local_motion_id].item())
            max_frame_exclusive = int(
                torch.floor(available_seconds * motion_fps).item()
            )
            if max_frame_exclusive <= 0:
                sampled_time_steps[assignment] = 0
                continue

            valid_bin_end = torch.minimum(
                bins[:, 2],
                torch.full_like(bins[:, 2], max_frame_exclusive),
            )
            valid_bin_length = (valid_bin_end - bins[:, 1]).clamp_min(0)
            valid = valid_bin_length > 0
            if not valid.any():
                sampled_time_steps[assignment] = 0
                continue

            valid_bin_ids = bin_ids[valid]
            valid_bins = bins[valid]
            valid_bin_end = valid_bin_end[valid]
            valid_bin_length = valid_bin_length[valid]
            weights = adaptive_signal[valid_bin_ids].clamp_min(0.0)
            weights *= self.adp_samp_bin_weights[valid_bin_ids].double()
            weights *= valid_bin_length.double() / (
                valid_bins[:, 2] - valid_bins[:, 1]
            ).clamp_min(1).double()
            if not torch.isfinite(weights).all() or weights.sum() <= 0:
                weights = torch.ones_like(weights)
            weights = self._normalize_adaptive_scores(weights)

            sampled_local_bins = torch.multinomial(
                weights, num_samples=count, replacement=True
            )
            sampled_bins = valid_bins[sampled_local_bins]
            sampled_ends = valid_bin_end[sampled_local_bins]
            sampled_frames = (
                torch.rand(count, device=self._device)
                * (sampled_ends - sampled_bins[:, 1]).float()
            ).floor().long() + sampled_bins[:, 1]
            if pre_failure_window > 0:
                sampled_frames = (
                    sampled_frames
                    - torch.randint(
                        pre_failure_window,
                        (count,),
                        device=self._device,
                    )
                ).clamp_min(0)
            sampled_time_steps[assignment] = (
                sampled_frames.float() * float(self._sim_fps) / motion_fps
            ).floor().long()

        return sampled_time_steps.int()
