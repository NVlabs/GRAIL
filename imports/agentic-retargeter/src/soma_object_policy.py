"""Whole-source object policy shared by SOMA contact and motion objectives."""
import numpy as np


def source_object_is_static(rotations, positions):
    """Use exact full-trajectory constancy, including rotation-only motion."""
    rotations, positions = np.asarray(rotations), np.asarray(positions)
    if (positions.ndim != 2 or positions.shape[1:] != (3,) or not len(positions)
            or rotations.shape != (len(positions), 3, 3)
            or not np.isfinite(positions).all() or not np.isfinite(rotations).all()):
        raise ValueError("Static object policy requires finite full-source poses")
    return bool(np.all(positions == positions[0]) and np.all(rotations == rotations[0]))


def contact_loss_policy(schedule):
    """Disable attraction to static objects; retain contact arrays for diagnostics.

    Recompute from immutable reference poses even for previously compiled NPZs.
    Legacy hand-built schedules without reference poses remain contact-enabled:
    target-point constancy alone cannot establish that the object is static.
    """
    keys = ("object_reference_rotations", "object_reference_positions")
    present = [key in schedule for key in keys]
    if any(present) and not all(present):
        raise ValueError("Contact policy requires both reference rotations and positions")
    static = source_object_is_static(*(schedule[key] for key in keys)) if all(present) else None
    return dict(enabled=static is not True, source_is_static=static,
                reason=("source_trajectory_unavailable" if static is None else
                        "static_source_object" if static else "moving_source_object"),
                detection="exact_full_source_translation_and_rotation")
