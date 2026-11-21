"""Pure helpers shared by render recorders and their regression tests."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch


CAMERA_SOURCE_ORIENTATION_TRACE_SCHEMA = (
    "thin_hold_source_camera_orientation_trace_v1"
)
CAMERA_RUNTIME_TRACE_SCHEMA = "thin_hold_runtime_camera_trace_v2"


def load_camera_source_orientation_trace(
    path: str | Path,
    *,
    expected_motion_key: str,
    expected_frame_count: int,
    expected_replay_fps: float,
    device: torch.device | str,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Load one source-object WXYZ orientation trace and validate its data.

    Validation is on the actual content: schema version, motion key, frame
    count, replay/source timing consistency, quaternion tensor shape,
    finiteness and norm.  No file fingerprint is required or compared; legacy
    traces that carry extra fingerprint metadata load normally and those
    fields are ignored.
    """
    path = Path(path).resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("source camera orientation trace is not valid JSON") from exc
    required_keys = {
        "schema_version",
        "motion_key",
        "object_index",
        "source_fps",
        "source_frame_count",
        "source_duration_s",
        "replay_fps",
        "frame_count",
        "frame_index_semantics",
        "sampling_method",
        "quaternion_convention",
        "object_quat_wxyz",
    }
    if not isinstance(payload, dict) or not required_keys <= set(payload):
        raise ValueError("source camera orientation trace schema mismatch")
    if payload["schema_version"] != CAMERA_SOURCE_ORIENTATION_TRACE_SCHEMA:
        raise ValueError("source camera orientation trace version mismatch")
    if payload["motion_key"] != expected_motion_key:
        raise ValueError(
            "source camera orientation trace motion mismatch: "
            f"expected={expected_motion_key!r}, actual={payload['motion_key']!r}"
        )
    if payload["frame_index_semantics"] != "zero_based_replay_step_unit_stride":
        raise ValueError("source camera orientation frame semantics mismatch")
    if payload["sampling_method"] != "shortest_path_slerp_at_physical_replay_time":
        raise ValueError("source camera orientation sampling method mismatch")
    if payload["quaternion_convention"] != "wxyz":
        raise ValueError("source camera orientation trace must be explicit WXYZ")
    if payload["object_index"] != 0:
        raise ValueError("source camera orientation trace must bind object index zero")
    if payload["frame_count"] != expected_frame_count:
        raise ValueError(
            "source camera orientation trace frame-count mismatch: "
            f"expected={expected_frame_count}, actual={payload['frame_count']}"
        )
    timing_values = (
        payload["source_fps"], payload["source_duration_s"], payload["replay_fps"]
    )
    if not all(
        isinstance(value, (int, float))
        and torch.isfinite(torch.tensor(float(value))).item()
        for value in timing_values
    ):
        raise ValueError("source camera orientation trace timing is invalid")
    if (
        payload["source_fps"] <= 0
        or payload["source_duration_s"] < 0
        or payload["replay_fps"] <= 0
        or not isinstance(payload["source_frame_count"], int)
        or payload["source_frame_count"] <= 0
    ):
        raise ValueError("source camera orientation trace timing is invalid")
    expected_duration_s = (
        (payload["source_frame_count"] - 1) / float(payload["source_fps"])
    )
    if abs(float(payload["source_duration_s"]) - expected_duration_s) > 1.0e-12:
        raise ValueError("source camera orientation trace duration mismatch")
    if abs(float(payload["replay_fps"]) - float(expected_replay_fps)) > 1.0e-12:
        raise ValueError(
            "source camera orientation replay FPS mismatch: "
            f"expected={expected_replay_fps}, actual={payload['replay_fps']}"
        )
    derived_frame_count = (
        int(expected_duration_s * float(expected_replay_fps) + 1.0e-9) + 1
    )
    if derived_frame_count != expected_frame_count:
        raise ValueError(
            "source/replay duration does not match runtime frame count: "
            f"derived={derived_frame_count}, runtime={expected_frame_count}"
        )
    quaternions = payload["object_quat_wxyz"]
    tensor = torch.as_tensor(quaternions, device=device, dtype=dtype)
    if tensor.shape != (expected_frame_count, 4):
        raise ValueError(
            "source camera orientation tensor shape mismatch: "
            f"expected={(expected_frame_count, 4)}, actual={tuple(tensor.shape)}"
        )
    if not torch.isfinite(tensor).all():
        raise ValueError("source camera orientation trace contains non-finite values")
    norms = torch.linalg.vector_norm(tensor, dim=-1)
    if (norms <= 1.0e-6).any():
        raise ValueError("source camera orientation trace contains a zero quaternion")
    return tensor, {
        "path": str(path),
        "content": payload,
    }


def resolve_camera_visibility_gprim_policy(
    body_names: list[str] | tuple[str, ...],
    requested_keep_body_names: list[str] | tuple[str, ...],
    gprim_paths: list[str] | tuple[str, ...],
    *,
    robot_root_path: str,
    gprim_initial_visibilities: list[str] | tuple[str, ...],
    gprim_effective_purposes: list[str] | tuple[str, ...],
) -> dict[str, Any]:
    """Resolve exact robot geometry visibility without assuming USD hierarchy.

    Callers supply only concrete ``UsdGeom.Gprim`` paths.  The deepest exact
    body-name path component owns each geometry, so hiding one body's geometry
    cannot accidentally hide a protected descendant articulation subtree.
    """
    keep_indices = resolve_camera_focus_body_indices(
        body_names, requested_keep_body_names
    )
    resolved_keep = [body_names[index] for index in keep_indices]
    root = robot_root_path.rstrip("/")
    if not root.startswith("/") or not root:
        raise ValueError("camera visibility robot root must be an absolute path")
    paths = list(gprim_paths)
    if not paths or len(paths) != len(set(paths)):
        raise ValueError("camera visibility Gprim roster must be non-empty and unique")
    visibilities = list(gprim_initial_visibilities)
    purposes = list(gprim_effective_purposes)
    if len(visibilities) != len(paths) or len(purposes) != len(paths):
        raise ValueError("camera visibility Gprim state roster length mismatch")
    body_set = set(body_names)
    keep_set = set(resolved_keep)
    owned_visible_visuals: dict[str, list[str]] = {
        name: [] for name in resolved_keep
    }
    kept_paths: list[str] = []
    hidden_paths: list[str] = []
    hidden_unmapped_paths: list[str] = []
    initially_invisible_paths: list[str] = []
    nonvisual_purpose_paths: list[str] = []
    initial_states: list[dict[str, str]] = []
    for path, visibility, purpose in zip(paths, visibilities, purposes):
        if not isinstance(path, str) or not path.startswith(root + "/"):
            raise ValueError(
                f"camera visibility Gprim is outside robot root: {path!r}"
            )
        if visibility not in {"inherited", "invisible"}:
            raise ValueError(
                f"camera visibility Gprim has invalid initial visibility: {path!r}"
            )
        if purpose not in {"default", "render", "guide", "proxy"}:
            raise ValueError(
                f"camera visibility Gprim has invalid effective purpose: {path!r}"
            )
        initial_states.append(
            {
                "path": path,
                "computed_visibility": visibility,
                "effective_purpose": purpose,
            }
        )
        relative_components = path[len(root) + 1 :].split("/")
        body_components = [name for name in relative_components if name in body_set]
        owner = body_components[-1] if body_components else None
        if visibility == "invisible":
            initially_invisible_paths.append(path)
        elif purpose in {"guide", "proxy"}:
            nonvisual_purpose_paths.append(path)
        elif owner in keep_set:
            kept_paths.append(path)
            owned_visible_visuals[owner].append(path)
        else:
            hidden_paths.append(path)
            if owner is None:
                hidden_unmapped_paths.append(path)
    missing_geometry = [
        name for name, records in owned_visible_visuals.items() if not records
    ]
    if missing_geometry:
        raise ValueError(
            "camera visibility protected bodies have no initially-visible "
            "default/render geometry: "
            f"{missing_geometry}"
        )
    if not hidden_paths:
        raise ValueError("camera visibility isolation did not select any geometry to hide")
    return {
        "mode": "isolate_robot_keep_exact_bodies",
        "robot_root_path": root,
        "requested_keep_body_names": list(requested_keep_body_names),
        "resolved_keep_body_names": resolved_keep,
        "gprim_initial_states": sorted(
            initial_states, key=lambda record: record["path"]
        ),
        "kept_gprim_paths": sorted(kept_paths),
        "hidden_gprim_paths": sorted(hidden_paths),
        "hidden_unmapped_gprim_paths": sorted(hidden_unmapped_paths),
        "initially_invisible_gprim_paths": sorted(initially_invisible_paths),
        "nonvisual_purpose_gprim_paths": sorted(nonvisual_purpose_paths),
    }


def write_camera_runtime_trace(
    path: str | Path,
    *,
    view_id: str,
    angle: str,
    motion_key: str,
    source_orientation_trace: dict[str, Any],
    visibility_policy: dict[str, Any],
    frames: list[dict[str, Any]],
) -> dict[str, Any]:
    """Persist per-frame camera pose commands associated with one render.

    These are cached ``set_world_poses`` inputs, not camera, USD, or sensor
    pose readback. The trace is an optional diagnostic record; nothing is
    required to consume it.
    """
    path = Path(path).resolve()
    if path.exists():
        raise FileExistsError(path)
    if not frames:
        raise ValueError("camera-command trace must contain at least one frame")
    frame_keys = {
        "frame_ordinal",
        "replay_frame_index",
        "source_object_quat_wxyz",
        "eye_world",
        "target_world",
        "rotation_opengl_world",
    }
    for ordinal, frame in enumerate(frames):
        if not isinstance(frame, dict) or set(frame) != frame_keys:
            raise ValueError("runtime camera frame schema mismatch")
        if frame["frame_ordinal"] != ordinal or frame["replay_frame_index"] != ordinal:
            raise ValueError("runtime camera frames are not zero-based unit-stride")
        for key, shape in (
            ("source_object_quat_wxyz", (4,)),
            ("eye_world", (3,)),
            ("target_world", (3,)),
            ("rotation_opengl_world", (3, 3)),
        ):
            value = torch.as_tensor(frame[key], dtype=torch.float64)
            if tuple(value.shape) != shape or not torch.isfinite(value).all():
                raise ValueError(f"runtime camera frame {key} is invalid")
        rotation = torch.as_tensor(
            frame["rotation_opengl_world"], dtype=torch.float64
        )
        identity = torch.eye(3, dtype=torch.float64)
        if not torch.allclose(
            rotation.transpose(-1, -2) @ rotation,
            identity,
            atol=1.0e-5,
            rtol=0.0,
        ) or not torch.allclose(
            torch.linalg.det(rotation),
            torch.tensor(1.0, dtype=torch.float64),
            atol=1.0e-5,
            rtol=0.0,
        ):
            raise ValueError("runtime camera rotation is not a proper frame")
    source_record = {
        "path": str((source_orientation_trace or {}).get("path", "")),
    }
    payload = {
        "schema_version": CAMERA_RUNTIME_TRACE_SCHEMA,
        "view_id": view_id,
        "angle": angle,
        "motion_key": motion_key,
        "frame_count": len(frames),
        "frame_index_semantics": "zero_based_replay_step_unit_stride",
        "object_quaternion_convention": "wxyz",
        "camera_quaternion_convention": "opengl_wxyz",
        "translation_policy": "per_view_exact_hand_body_centroid",
        "source_orientation_trace": source_record,
        "visibility_policy": visibility_policy,
        "frames": frames,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    return payload


def resolve_camera_focus_body_indices(
    body_names: list[str] | tuple[str, ...],
    requested_names: list[str] | tuple[str, ...],
) -> list[int]:
    """Resolve an exact, non-empty robot body roster without fallback."""
    requested_names = list(requested_names)
    if not requested_names or not all(
        isinstance(name, str) and name for name in requested_names
    ):
        raise ValueError(
            "eval camera focus body roster must contain non-empty strings"
        )
    if len(set(requested_names)) != len(requested_names):
        raise ValueError("eval camera focus body roster contains duplicates")
    indices = []
    for requested_name in requested_names:
        matches = [
            index for index, body_name in enumerate(body_names)
            if body_name == requested_name
        ]
        if len(matches) != 1:
            raise ValueError(
                "eval camera focus body must resolve exactly once: "
                f"requested={requested_name!r}, matches={len(matches)}, "
                f"available={list(body_names)}"
            )
        indices.append(matches[0])
    return indices


def resolve_camera_focus_body_index(
    body_names: list[str] | tuple[str, ...],
    requested_name: str,
) -> int:
    """Backward-compatible singular exact body resolver."""
    return resolve_camera_focus_body_indices(body_names, [requested_name])[0]


def camera_pose_radial_from_root(
    root_pos: torch.Tensor,
    focus_pos: torch.Tensor,
    *,
    radial_distance_m: float,
    height_m: float,
    tangent_offset_m: float = 0.0,
    target_offset: torch.Tensor | None = None,
    fallback_xy: torch.Tensor | tuple[float, float] = (1.0, -1.0),
    radial_epsilon_m: float = 1.0e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Place a camera outward from root through a hand-centered focus.

    The torso then stays behind the hand from the camera's perspective.  If
    root and focus have indistinguishable XY coordinates, ``fallback_xy`` is
    normalized and used deterministically; a zero fallback fails closed.
    """
    focus_pos = torch.as_tensor(
        focus_pos, device=root_pos.device, dtype=root_pos.dtype
    )
    if focus_pos.shape != root_pos.shape:
        raise ValueError(
            "radial camera focus must match root positions, got "
            f"{focus_pos.shape} versus {root_pos.shape}"
        )
    if (
        radial_distance_m <= 0
        or height_m < 0
        or not torch.isfinite(torch.tensor(float(tangent_offset_m)))
        or radial_epsilon_m <= 0
    ):
        raise ValueError(
            "radial camera distance/epsilon must be positive, height must "
            "be non-negative, and tangent offset must be finite"
        )
    fallback_xy = torch.as_tensor(
        fallback_xy, device=root_pos.device, dtype=root_pos.dtype
    )
    if fallback_xy.shape != (2,):
        raise ValueError(
            f"radial camera fallback must have shape (2,), got {fallback_xy.shape}"
        )
    fallback_norm = torch.linalg.vector_norm(fallback_xy)
    if not torch.isfinite(fallback_norm) or fallback_norm <= radial_epsilon_m:
        raise ValueError("radial camera fallback direction is zero or non-finite")
    fallback_unit = fallback_xy / fallback_norm

    radial_xy = focus_pos[:, :2] - root_pos[:, :2]
    radial_norm = torch.linalg.vector_norm(radial_xy, dim=-1, keepdim=True)
    radial_unit = radial_xy / radial_norm.clamp_min(radial_epsilon_m)
    radial_unit = torch.where(
        (radial_norm > radial_epsilon_m).expand_as(radial_unit),
        radial_unit,
        fallback_unit.unsqueeze(0).expand_as(radial_unit),
    )
    target = focus_pos.clone()
    if target_offset is not None:
        target_offset = torch.as_tensor(
            target_offset, device=root_pos.device, dtype=root_pos.dtype
        )
        if target_offset.shape != (3,):
            raise ValueError(
                "radial camera target offset must have shape (3,), got "
                f"{target_offset.shape}"
            )
        target = target + target_offset
    eye = target.clone()
    eye[:, :2] += radial_unit * float(radial_distance_m)
    tangent_unit = torch.stack((-radial_unit[:, 1], radial_unit[:, 0]), dim=-1)
    eye[:, :2] += tangent_unit * float(tangent_offset_m)
    eye[:, 2] += float(height_m)
    return eye, target


def camera_pose_from_object_axis(
    focus_pos: torch.Tensor,
    object_quat_wxyz: torch.Tensor,
    *,
    local_axis_index: int,
    local_axis_sign: float,
    local_up_axis_index: int | None = None,
    local_up_axis_sign: float = 1.0,
    distance_m: float,
    target_offset: torch.Tensor | None = None,
    quaternion_convention: str = "wxyz",
    singularity_cosine: float = 0.999,
    epsilon: float = 1.0e-6,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Follow an object-local axis while looking at a hand-centered focus.

    ``object_quat_wxyz`` is explicitly an active local-to-world rotation in
    Isaac's ``(w, x, y, z)`` convention.  The returned rotation matrices map
    OpenGL camera-local axes ``(+X right, +Y up, -Z forward)`` into world
    coordinates and can therefore be converted to ``wxyz`` quaternions and
    passed to ``Camera.set_world_poses(..., convention="opengl")``.

    When ``local_up_axis_index`` is supplied, both the eye direction and image
    up direction are transported by the same object orientation.  This is the
    comparison-safe mode: all views can consume one frozen SOURCE trace with
    aperture ``eye=-Y, up=+Z`` and face ``eye=+Z, up=+Y``.  The legacy world-up
    fallback remains available only when no local up axis is requested.
    """
    focus_pos = torch.as_tensor(focus_pos)
    object_quat_wxyz = torch.as_tensor(
        object_quat_wxyz, device=focus_pos.device, dtype=focus_pos.dtype
    )
    if focus_pos.ndim != 2 or focus_pos.shape[-1] != 3:
        raise ValueError(
            "object-axis camera focus must have shape (num_envs, 3), got "
            f"{focus_pos.shape}"
        )
    if object_quat_wxyz.shape != (focus_pos.shape[0], 4):
        raise ValueError(
            "object-axis camera quaternion must have shape (num_envs, 4), "
            f"got {object_quat_wxyz.shape}"
        )
    if quaternion_convention != "wxyz":
        raise ValueError(
            "object-axis camera only accepts explicit wxyz quaternions, got "
            f"{quaternion_convention!r}"
        )
    if local_axis_index not in (0, 1, 2):
        raise ValueError(
            f"object-axis camera local axis must be 0, 1, or 2, got {local_axis_index}"
        )
    if float(local_axis_sign) not in (-1.0, 1.0):
        raise ValueError(
            "object-axis camera local axis sign must be exactly -1 or +1, "
            f"got {local_axis_sign}"
        )
    if local_up_axis_index is not None:
        if local_up_axis_index not in (0, 1, 2):
            raise ValueError(
                "object-axis camera local up axis must be 0, 1, or 2, got "
                f"{local_up_axis_index}"
            )
        if local_up_axis_index == local_axis_index:
            raise ValueError("object-axis camera eye and up axes must be distinct")
        if float(local_up_axis_sign) not in (-1.0, 1.0):
            raise ValueError(
                "object-axis camera local up sign must be exactly -1 or +1, "
                f"got {local_up_axis_sign}"
            )
    scalar_values = torch.tensor(
        [float(distance_m), float(singularity_cosine), float(epsilon)],
        device=focus_pos.device,
        dtype=focus_pos.dtype,
    )
    if (
        not torch.isfinite(scalar_values).all()
        or distance_m <= 0
        or epsilon <= 0
        or not 0 < singularity_cosine < 1
    ):
        raise ValueError(
            "object-axis camera distance/epsilon must be positive and finite, "
            "and singularity cosine must be in (0, 1)"
        )
    if not torch.isfinite(focus_pos).all() or not torch.isfinite(
        object_quat_wxyz
    ).all():
        raise ValueError("object-axis camera inputs contain non-finite values")

    quat_norm = torch.linalg.vector_norm(
        object_quat_wxyz, dim=-1, keepdim=True
    )
    if (quat_norm <= epsilon).any():
        raise ValueError("object-axis camera quaternion has near-zero norm")
    quat = object_quat_wxyz / quat_norm

    def rotate_local_axis(axis_index: int, axis_sign: float) -> torch.Tensor:
        local_axis = torch.zeros(
            (focus_pos.shape[0], 3),
            device=focus_pos.device,
            dtype=focus_pos.dtype,
        )
        local_axis[:, axis_index] = float(axis_sign)
        twice_cross = 2.0 * torch.cross(q_vec, local_axis, dim=-1)
        return (
            local_axis
            + quat[:, :1] * twice_cross
            + torch.cross(q_vec, twice_cross, dim=-1)
        )

    # Active quaternion rotation, q * [0, v] * conjugate(q), for wxyz q.
    q_vec = quat[:, 1:]
    object_axis_world = rotate_local_axis(local_axis_index, local_axis_sign)
    axis_norm = torch.linalg.vector_norm(
        object_axis_world, dim=-1, keepdim=True
    )
    if (axis_norm <= epsilon).any() or not torch.isfinite(axis_norm).all():
        raise ValueError("object-axis camera produced an invalid world axis")
    camera_back = object_axis_world / axis_norm

    target = focus_pos.clone()
    if target_offset is not None:
        target_offset = torch.as_tensor(
            target_offset, device=focus_pos.device, dtype=focus_pos.dtype
        )
        if target_offset.shape != (3,) or not torch.isfinite(
            target_offset
        ).all():
            raise ValueError(
                "object-axis camera target offset must be a finite shape-(3,) vector"
            )
        target = target + target_offset
    eye = target + camera_back * float(distance_m)

    if local_up_axis_index is None:
        world_z = torch.tensor(
            [0.0, 0.0, 1.0], device=focus_pos.device, dtype=focus_pos.dtype
        ).expand_as(camera_back)
        world_y = torch.tensor(
            [0.0, 1.0, 0.0], device=focus_pos.device, dtype=focus_pos.dtype
        ).expand_as(camera_back)
        use_fallback = (
            torch.abs(torch.sum(camera_back * world_z, dim=-1, keepdim=True))
            >= float(singularity_cosine)
        )
        up_reference = torch.where(use_fallback, world_y, world_z)
    else:
        up_reference = rotate_local_axis(
            local_up_axis_index, local_up_axis_sign
        )
    camera_right = torch.cross(up_reference, camera_back, dim=-1)
    right_norm = torch.linalg.vector_norm(camera_right, dim=-1, keepdim=True)
    if (right_norm <= epsilon).any() or not torch.isfinite(right_norm).all():
        raise ValueError("object-axis camera up fallback is singular")
    camera_right = camera_right / right_norm
    camera_up = torch.cross(camera_back, camera_right, dim=-1)
    camera_up = camera_up / torch.linalg.vector_norm(
        camera_up, dim=-1, keepdim=True
    ).clamp_min(epsilon)
    rotation_opengl_world = torch.stack(
        (camera_right, camera_up, camera_back), dim=-1
    )
    if not torch.isfinite(rotation_opengl_world).all():
        raise ValueError("object-axis camera rotation is non-finite")
    return eye, target, rotation_opengl_world


def should_write_render_frame(
    frame_id: int,
    *,
    frame_skip: int,
    skip_first_frame: bool,
) -> bool:
    """Return whether a post-step state belongs in the encoded video."""
    if frame_skip <= 0:
        raise ValueError(f"render frame skip must be positive, got {frame_skip}")
    return frame_id % frame_skip == 0 and (
        not skip_first_frame or frame_id >= 1
    )


def camera_pose_with_focus(
    root_pos: torch.Tensor,
    camera_offset: torch.Tensor,
    *,
    focus_pos: torch.Tensor | None = None,
    focus_margin_m: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Frame the robot and optional interaction points with adaptive range.

    The legacy path (no ``focus_pos``) targets the robot root. Multiple focus
    points use their bounding-box center and back the camera away as the
    subjects separate.
    """
    camera_offset = torch.as_tensor(
        camera_offset, device=root_pos.device, dtype=root_pos.dtype
    )
    if camera_offset.shape != (3,):
        raise ValueError(
            f"eval camera offset must have shape (3,), got {camera_offset.shape}"
        )
    if focus_pos is None:
        return root_pos + camera_offset, root_pos
    if focus_pos.ndim == 2:
        focus_pos = focus_pos.unsqueeze(1)
    if (
        focus_pos.ndim != 3
        or focus_pos.shape[0] != root_pos.shape[0]
        or focus_pos.shape[2] != 3
    ):
        raise ValueError(
            "camera focus must have shape (num_envs, 3) or "
            f"(num_envs, num_focus, 3), got {focus_pos.shape}"
        )
    if focus_margin_m < 0:
        raise ValueError(
            f"camera focus margin must be non-negative, got {focus_margin_m}"
        )

    subjects = torch.cat((root_pos.unsqueeze(1), focus_pos), dim=1)
    subject_min = subjects.amin(dim=1)
    subject_max = subjects.amax(dim=1)
    target = 0.5 * (subject_min + subject_max)
    target[:, 2] = subject_max[:, 2]
    base_distance = torch.linalg.vector_norm(camera_offset).clamp_min(1.0e-6)
    subject_span = torch.linalg.vector_norm(subject_max - subject_min, dim=-1)
    required_distance = subject_span + 2.0 * float(focus_margin_m)
    distance_scale = torch.maximum(
        torch.ones_like(required_distance), required_distance / base_distance
    )
    eye = target + camera_offset.unsqueeze(0) * distance_scale.unsqueeze(-1)
    return eye, target


def camera_pose_with_target_offset(
    root_pos: torch.Tensor,
    camera_offset: torch.Tensor,
    *,
    target_offset: torch.Tensor | None = None,
    focus_pos: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return a replay camera pose around an explicit interaction target.

    With no focus or target offset this is exactly the legacy replay pose:
    ``eye = root_pos + camera_offset`` and ``target = root_pos``.  A focus
    entity can replace the root as the target, while ``target_offset`` applies
    a final world-axis adjustment to either target.
    """
    camera_offset = torch.as_tensor(
        camera_offset, device=root_pos.device, dtype=root_pos.dtype
    )
    if camera_offset.shape != (3,):
        raise ValueError(
            f"eval camera offset must have shape (3,), got {camera_offset.shape}"
        )
    target = root_pos if focus_pos is None else torch.as_tensor(
        focus_pos, device=root_pos.device, dtype=root_pos.dtype
    )
    if target.shape != root_pos.shape:
        raise ValueError(
            "eval camera focus must match root positions, got "
            f"{target.shape} versus {root_pos.shape}"
        )
    if target_offset is not None:
        target_offset = torch.as_tensor(
            target_offset, device=root_pos.device, dtype=root_pos.dtype
        )
        if target_offset.shape != (3,):
            raise ValueError(
                "eval camera target offset must have shape (3,), got "
                f"{target_offset.shape}"
            )
        target = target + target_offset
    return target + camera_offset, target
