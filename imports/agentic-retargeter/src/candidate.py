"""Reset native edit state without changing upstream input bindings.

The state/cache contract matches the standalone contact edit kit. Keep this
module NumPy-only so the installed backend does not depend on source-tree
scripts or Blender. Undeclared custom caches are retained but invalid whenever
``fk_cache_valid`` is false; consumers must recompute them before use.
"""

import numpy as np


INVALIDATED_FIELDS = frozenset({
    "body_pos", "body_quat_wxyz", "geom_pos", "geom_R",
    "qvel", "qacc", "qacc_warmstart",
    "body_R", "body_rot", "body_rotmat", "body_quat", "body_positions",
    "body_world_pos", "body_world_R", "body_world_quat", "body_world_transforms",
    "joint_world_pos", "joint_world_R", "joint_world_quat", "joint_world_transforms",
    "xpos", "xmat", "xquat", "xipos", "ximat", "xanchor", "xaxis",
    "geom_xpos", "geom_xmat", "site_xpos", "site_xmat", "cam_xpos", "cam_xmat",
    "light_xpos", "light_xdir", "subtree_com", "cvel", "cacc",
    "native_task_errors", "native_task_errors_status",
    "parent_native_task_errors", "parent_native_task_errors_status",
    "parent_native_task_errors_scope", "solver_diagnostics",
})
CANDIDATE_STATE_FIELDS = frozenset({
    "status", "native_status", "candidate_status", "review_status", "approval_status",
    "selection_status", "validation_status", "acceptance_status", "selected", "accepted",
    "reviewed", "selection", "export_kind", "contact_edit_review_status",
    "contact_edit_provenance_json", "pose_edit_parent_sha256",
    "pose_edit_config_canonical_sha256", "pose_edit_config_file_sha256",
    "refinement_parent_sha256", "refinement_config_sha256", "refinement_provenance_json",
})
PROTECTED_FIELDS = frozenset({
    "qpos", "fps", "timestamps", "dof_names", "qposadr", "joint_names",
    "joint_qposadr", "joint_dofadr", "joint_ranges", "joint_limited",
    "hinge_qposadr", "hinge_dofadr", "finger_seed", "finger_qposadr",
    "body_names", "native_wrist_targets", "focal_length_px", "frame_width",
    "frame_height", "scene_scale", "height_offset", "height_scale",
    "pelvis_rotation", "descriptor_sha256", "raw_native_sha256",
    "fk_cache_valid", "export_kind", "native_status",
})
PROTECTED_PREFIXES = (
    "source_", "object_", "obj_", "human_", "camera_", "contact_",
    "material_", "model_", "compiled_model_", "target_", "depth_",
)


def validate_derived_fields(fields):
    """Validate optional custom caches without allowing input metadata removal."""
    if not isinstance(fields, list) or any(not isinstance(name, str) or not name for name in fields):
        raise ValueError("derived_fields must be a list of nonempty field names")
    if len(fields) != len(set(fields)):
        raise ValueError("derived_fields must not contain duplicates")
    for name in fields:
        if (
            name in PROTECTED_FIELDS
            or name in CANDIDATE_STATE_FIELDS
            or name.startswith(PROTECTED_PREFIXES)
            or name.endswith("_targets")
        ):
            raise ValueError(f"Cannot invalidate protected source field {name!r}")
    return fields


def prepare_candidate(source, qpos, *, derived_fields=None):
    """Return an independent UNREVIEWED native archive and its metadata changes.

    Prior approval fields move to unused ``parent_*`` names. In particular,
    ``source_sha256`` continues to identify the upstream human/object input;
    it must never be replaced with the hash of the parent robot trajectory.
    """
    declared = validate_derived_fields([] if derived_fields is None else derived_fields)
    baseline, qpos = np.asarray(source["qpos"]), np.asarray(qpos)
    if qpos.shape != baseline.shape or qpos.ndim != 2 or not qpos.size:
        raise ValueError("Candidate qpos must match the source trajectory shape")
    if qpos.dtype.kind != "f" or not np.isfinite(qpos).all():
        raise ValueError("Candidate qpos must contain finite floating samples")
    changed = bool(np.any(qpos != baseline))
    invalidated = sorted(set(source) & (INVALIDATED_FIELDS | set(declared))) if changed else []
    candidate = {
        key: np.array(value, copy=True) for key, value in source.items() if key not in invalidated
    }
    candidate["qpos"] = qpos.copy()
    prior_validity = np.asarray(source.get("fk_cache_valid", []))
    cache_validity = (
        bool(prior_validity) if prior_validity.shape == () and prior_validity.dtype.kind == "b" else None
    )
    if changed:
        candidate["fk_cache_valid"] = np.asarray(False)
        cache_validity = False
    parent_fields = {}
    for key in sorted(CANDIDATE_STATE_FIELDS & set(candidate)):
        parent_key = "parent_" + key
        while parent_key in candidate:
            parent_key = "parent_" + parent_key
        candidate[parent_key] = candidate.pop(key)
        parent_fields[key] = parent_key
    candidate["status"] = np.asarray("UNREVIEWED")
    candidate["review_status"] = np.asarray("UNREVIEWED")
    candidate["selected"] = np.asarray(False)
    candidate["export_kind"] = np.asarray("native_robot_refinement_UNREVIEWED")
    metadata = {
        "status": "UNREVIEWED",
        "review_status": "UNREVIEWED",
        "selected": False,
        "actual_qpos_changed": changed,
        "changed_values": int(np.count_nonzero(qpos != baseline)),
        "invalidated_fields": invalidated,
        "declared_derived_fields": declared,
        "parent_provenance_fields": parent_fields,
        "fk_cache_valid": cache_validity,
        "unknown_cached_data_status": (
            "All motion-derived caches are invalid after this edit, including custom arrays "
            "retained because they were not declared in derived_fields. Recompute before use."
            if changed else "Qpos unchanged; existing cache validity was not established or changed."
        ),
    }
    return candidate, metadata
