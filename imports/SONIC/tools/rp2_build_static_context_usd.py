#!/usr/bin/env python3
"""Build per-motion static context USDs for RP2 single-object training.

Each output USD references the inactive objects from a full-scene object list
and bakes their first-frame poses into the scene. Slot 0 is always active;
linked target parts and actor-owned co-manipulated trajectories are excluded as
well so they are not duplicated as static colliders.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys
from typing import Any

import joblib
import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from gear_sonic.rp2.static_context import (  # noqa: E402
    enforce_static_context_kinematics,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--motion-lib-dir", type=Path, required=True)
    parser.add_argument("--objects-dir", type=Path, required=True)
    parser.add_argument("--full-scene-usd-dir", type=Path, required=True)
    parser.add_argument("--output-usd-dir", type=Path, required=True)
    parser.add_argument("--active-index", type=int, default=0)
    parser.add_argument(
        "--exclude-name-regex",
        action="append",
        default=[],
        help="Also exclude object names matching this regex; may be repeated.",
    )
    parser.add_argument("--motion-list", type=Path, default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _safe_prim_name(text: str) -> str:
    text = re.sub(r"[^A-Za-z0-9_]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    if not text:
        text = "Object"
    if text[0].isdigit():
        text = f"Object_{text}"
    return text


def _asset_path(path: Path) -> str:
    return str(path.resolve()).replace("\\", "/").replace("@", "%40")


def _load_entry(path: Path, key: str) -> dict[str, Any]:
    data = joblib.load(path)
    if isinstance(data, dict) and key in data:
        return data[key]
    if isinstance(data, dict):
        return next(iter(data.values()))
    raise TypeError(f"Expected dict joblib at {path}")


def _load_meta(path: Path, key: str) -> dict[str, Any]:
    data = joblib.load(path)
    if isinstance(data, dict) and key in data and isinstance(data[key], dict):
        return data[key]
    if isinstance(data, dict):
        return data
    raise TypeError(f"Expected metadata dict at {path}")


def _animated_object_indices(objects: dict[str, Any], tolerance: float = 1.0e-7) -> set[int]:
    root_pos = np.asarray(objects["root_pos"])
    root_quat = np.asarray(objects["root_quat"])
    if root_pos.ndim != 3 or root_quat.ndim != 3:
        raise ValueError(
            f"Expected root_pos/root_quat [T,O,D], got {root_pos.shape} and {root_quat.shape}"
        )
    if root_pos.shape[:2] != root_quat.shape[:2]:
        raise ValueError(
            f"root_pos/root_quat frame-object mismatch: {root_pos.shape} vs {root_quat.shape}"
        )
    pos_change = np.max(np.abs(root_pos - root_pos[:1]), axis=(0, 2))
    quat_change = np.max(np.abs(root_quat - root_quat[:1]), axis=(0, 2))
    return {
        int(index)
        for index in np.flatnonzero((pos_change > tolerance) | (quat_change > tolerance))
    }


def _metadata_animated_indices(
    meta: dict[str, Any], object_names: list[str]
) -> set[int]:
    indices = {
        int(value)
        for value in meta.get("rp2_animated_clip_object_indices", [])
        if 0 <= int(value) < len(object_names)
    }
    dynamic_names = {
        str(value) for value in meta.get("rp2_dynamic_source_object_names", [])
    }
    indices.update(
        index for index, name in enumerate(object_names) if str(name) in dynamic_names
    )
    return indices


def _load_dynamic_role_stems(motion_lib_dir: Path) -> set[str]:
    dynamic: set[str] = set()
    for path in sorted(motion_lib_dir.glob("object_roles*.json")):
        payload = json.loads(path.read_text())
        for stem, role in (payload.get("objects") or {}).items():
            if bool((role or {}).get("dynamic", False)):
                dynamic.add(str(stem))
    return dynamic


def _motion_keys(motion_lib_dir: Path, motion_list: Path | None) -> list[str]:
    if motion_list is not None:
        return [line.strip() for line in motion_list.read_text().splitlines() if line.strip()]
    return sorted(path.stem for path in (motion_lib_dir / "robot").glob("*.pkl"))


def _read_usd_list(path: Path) -> list[Path]:
    return [Path(line.strip()) for line in path.read_text().splitlines() if line.strip()]


def _fmt_tuple(values: np.ndarray | list[float] | tuple[float, ...]) -> str:
    return ", ".join(f"{float(v):.9g}" for v in values)


def _write_static_context_usda(
    path: Path,
    *,
    motion_key: str,
    usd_paths: list[Path],
    root_pos: np.ndarray,
    root_quat: np.ndarray,
    excluded_indices: set[int],
) -> int:
    lines = [
        "#usda 1.0",
        "(",
        '    defaultPrim = "StaticContext"',
        "    metersPerUnit = 1",
        '    upAxis = "Z"',
        ")",
        "",
        'def Xform "StaticContext"',
        "{",
    ]

    written = 0
    for obj_idx, usd_path in enumerate(usd_paths):
        if obj_idx in excluded_indices:
            continue
        if obj_idx >= root_pos.shape[0] or obj_idx >= root_quat.shape[0]:
            continue
        prim_name = _safe_prim_name(f"Context_{obj_idx:02d}_{usd_path.stem}")
        pos = root_pos[obj_idx]
        quat = root_quat[obj_idx]
        lines.extend(
            [
                f'    def Xform "{prim_name}" (',
                f"        prepend references = @{_asset_path(usd_path)}@",
                "    )",
                "    {",
                f"        double3 xformOp:translate = ({_fmt_tuple(pos)})",
                f"        quatf xformOp:orient = ({_fmt_tuple(quat)})",
                '        uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:orient"]',
                "    }",
                "",
            ]
        )
        written += 1

    lines.extend(
        [
            "}",
            "",
            f"# motion_key = {motion_key}",
            f"# inactive_context_objects = {written}",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")
    enforce_static_context_kinematics(path)
    return written


def main() -> int:
    args = _parse_args()
    motion_lib_dir = args.motion_lib_dir.resolve()
    objects_dir = args.objects_dir.resolve()
    full_scene_usd_dir = args.full_scene_usd_dir.resolve()
    output_usd_dir = args.output_usd_dir.resolve()

    for path in (motion_lib_dir / "robot", objects_dir, full_scene_usd_dir):
        if not path.exists():
            raise FileNotFoundError(path)
    output_usd_dir.mkdir(parents=True, exist_ok=True)
    dynamic_role_stems = _load_dynamic_role_stems(motion_lib_dir)

    total = 0
    total_context_objects = 0
    total_excluded_animated_objects = 0
    for motion_key in _motion_keys(motion_lib_dir, args.motion_list):
        out_path = output_usd_dir / f"{motion_key}.usda"
        if out_path.exists() and not args.overwrite:
            total += 1
            continue

        objects = _load_entry(objects_dir / f"{motion_key}.pkl", motion_key)
        root_pos_all = np.asarray(objects["root_pos"])
        root_quat_all = np.asarray(objects["root_quat"])
        root_pos = root_pos_all[0]
        root_quat = root_quat_all[0]
        excluded_indices = {args.active_index}
        object_names = list(objects.get("object_names", []))
        animated_indices = _animated_object_indices(objects)
        meta_path = motion_lib_dir / "meta" / f"{motion_key}.pkl"
        if meta_path.is_file():
            meta = _load_meta(meta_path, motion_key)
            animated_indices.update(_metadata_animated_indices(meta, object_names))
        exclude_patterns = [re.compile(pattern) for pattern in args.exclude_name_regex]
        for object_index, object_name in enumerate(object_names):
            if any(pattern.search(str(object_name)) for pattern in exclude_patterns):
                excluded_indices.add(object_index)
        usd_paths = _read_usd_list(full_scene_usd_dir / f"{motion_key}.txt")
        animated_indices.update(
            index
            for index, usd_path in enumerate(usd_paths)
            if usd_path.stem in dynamic_role_stems
        )
        excluded_indices.update(animated_indices)
        total_excluded_animated_objects += len(animated_indices - {args.active_index})
        context_count = _write_static_context_usda(
            out_path,
            motion_key=motion_key,
            usd_paths=usd_paths,
            root_pos=root_pos,
            root_quat=root_quat,
            excluded_indices=excluded_indices,
        )
        total += 1
        total_context_objects += context_count

    print(
        f"wrote {total} static context USDs to {output_usd_dir} "
        f"with {total_context_objects} inactive object references; "
        f"excluded {total_excluded_animated_objects} additional animated objects"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
