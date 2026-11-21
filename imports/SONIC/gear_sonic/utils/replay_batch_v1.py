"""Run several deterministic reference replays inside one live Isaac app.

The environment and authored object are constructed once.  Every task then
selects one already-resident motion, updates the fixed camera and reward-audit
destination, and calls ``run_replay``.  This is intentionally small and
write-once so review rendering cannot silently mix incomplete artifacts.
"""

from __future__ import annotations

from contextlib import nullcontext
import json
import math
from pathlib import Path
import time
from typing import Any, Mapping

from omegaconf import OmegaConf, open_dict


SCHEMA = "general_v4_presence_one_isaac_app_case_v1"
RECEIPT_SCHEMA = "general_v4_presence_one_isaac_app_receipt_v1"
TASK_ORDER = (
    ("full_body", "source"),
    ("full_body", "v6"),
    ("contact_close_up", "source"),
    ("contact_close_up", "v6"),
)


class ReplayBatchError(RuntimeError):
    """One-app replay evidence is malformed or incomplete."""


def _json(path: Path) -> tuple[Path, Mapping[str, Any]]:
    source = Path(path).expanduser().resolve(strict=True)
    value = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ReplayBatchError(f"batch manifest is not one object: {source}")
    return source, value


def _vec3(value: Any, label: str) -> list[float]:
    if not isinstance(value, list) or len(value) != 3:
        raise ReplayBatchError(f"{label} must be a finite vec3")
    result = [float(item) for item in value]
    if any(not math.isfinite(item) for item in result):
        raise ReplayBatchError(f"{label} must be a finite vec3")
    return result


def _path(value: Any, label: str, *, must_not_exist: bool = False) -> Path:
    if not isinstance(value, str) or not value:
        raise ReplayBatchError(f"{label} path is missing")
    path = Path(value).expanduser().resolve(strict=False)
    if must_not_exist and (path.exists() or path.is_symlink()):
        raise ReplayBatchError(f"write-once {label} already exists: {path}")
    return path


def validate_manifest(path: Path) -> tuple[Path, Mapping[str, Any]]:
    source, raw = _json(path)
    motion_keys = raw.get("motion_keys")
    tasks = raw.get("tasks")
    if (
        raw.get("schema") != SCHEMA
        or not isinstance(raw.get("case_id"), str)
        or not isinstance(motion_keys, list)
        or len(motion_keys) != 2
        or len(set(motion_keys)) != 2
        or any(not isinstance(key, str) or not key for key in motion_keys)
        or not isinstance(tasks, list)
        or len(tasks) != 4
    ):
        raise ReplayBatchError("one-app manifest identity/motion/task roster differs")
    observed_order = []
    normalized_tasks = []
    for index, task in enumerate(tasks):
        if not isinstance(task, Mapping) or task.get("task_index") != index:
            raise ReplayBatchError("one-app task ordering differs")
        view_id, column = str(task.get("view_id")), str(task.get("column"))
        observed_order.append((view_id, column))
        motion_key = str(task.get("motion_key"))
        if motion_key not in motion_keys:
            raise ReplayBatchError(f"task motion is not resident: {motion_key}")
        normalized_tasks.append(
            {
                **dict(task),
                "camera_offset": _vec3(task.get("camera_offset"), "camera offset"),
                "camera_target_offset": _vec3(
                    task.get("camera_target_offset"), "camera target offset"
                ),
                "video_mp4": str(_path(task.get("video_mp4"), "video", must_not_exist=True)),
                "reward_audit": str(
                    _path(task.get("reward_audit"), "reward audit", must_not_exist=True)
                ),
            }
        )
    if tuple(observed_order) != TASK_ORDER:
        raise ReplayBatchError(f"one-app task roster differs: {observed_order}")
    receipt = _path(raw.get("receipt"), "receipt", must_not_exist=True)
    return source, {**dict(raw), "tasks": normalized_tasks, "receipt": str(receipt)}


def _set_config(config: Any, values: Mapping[str, Any]) -> None:
    context = open_dict(config) if OmegaConf.is_config(config) else nullcontext()
    with context:
        for key, value in values.items():
            if isinstance(config, Mapping):
                config[key] = value
            else:
                setattr(config, key, value)


def run_replay_batch(
    env: Any,
    manifest_path: Path,
    *,
    process_start_monotonic: float,
) -> Mapping[str, Any]:
    manifest_file, manifest = validate_manifest(manifest_path)
    resident = [str(value) for value in env._motion_lib.curr_motion_keys]  # noqa: SLF001
    expected = [str(value) for value in manifest["motion_keys"]]
    if resident != expected:
        raise ReplayBatchError(f"resident motion order differs: {resident} != {expected}")
    motion_ids = {key: index for index, key in enumerate(resident)}
    startup_s = time.monotonic() - float(process_start_monotonic)
    results = []
    for task in manifest["tasks"]:
        video = Path(str(task["video_mp4"]))
        audit = Path(str(task["reward_audit"]))
        video.parent.mkdir(parents=True, exist_ok=True)
        audit.parent.mkdir(parents=True, exist_ok=True)
        _set_config(
            env.config,
            {
                "eval_camera_offset": list(task["camera_offset"]),
                "eval_camera_target_offset": list(task["camera_target_offset"]),
                "replay_reward_audit_path": str(audit),
            },
        )
        started = time.monotonic()
        env.run_replay(
            motion_id=motion_ids[str(task["motion_key"])],
            start_time_step=-1,
            loop=1,
            target_fps=50,
            grid_spacing=2.0,
            save_video_path=str(video),
        )
        duration = time.monotonic() - started
        if not video.is_file() or video.stat().st_size <= 0:
            raise ReplayBatchError(f"replay produced no video: {video}")
        if not audit.is_file() or audit.stat().st_size <= 0:
            raise ReplayBatchError(f"replay produced no reward audit: {audit}")
        results.append(
            {
                "task_index": int(task["task_index"]),
                "view_id": str(task["view_id"]),
                "column": str(task["column"]),
                "motion_key": str(task["motion_key"]),
                "video_mp4": str(video),
                "reward_audit": str(audit),
                "render_duration_s": duration,
            }
        )
    render_s = sum(float(row["render_duration_s"]) for row in results)
    receipt = {
        "schema": RECEIPT_SCHEMA,
        "status": "PASS",
        "case_id": str(manifest["case_id"]),
        "manifest": str(manifest_file),
        "isaac_app_start_count": 1,
        "resident_motion_keys": resident,
        "startup_s": startup_s,
        "render_s": render_s,
        "startup_fraction": startup_s / (startup_s + render_s),
        "tasks": results,
    }
    receipt_path = Path(str(manifest["receipt"]))
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    receipt_path.write_text(
        json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return receipt
