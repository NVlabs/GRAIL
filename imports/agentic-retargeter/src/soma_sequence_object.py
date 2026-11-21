"""Sparse object reference/temporal rows for a simultaneous SOMA trajectory solve.

The squared residual norm is the existing ``soma_object`` energy: translation
and geodesic rotation reference costs, plus SI-scaled finite differences of
actual positions and rotation matrices. Rotation matrix differences are the
existing surrogate, not physical angular acceleration or jerk. Contact and
collision rows belong to the coupled runtime and are deliberately not included
here. Tangent columns are six per unique pose, ordered world translation then
left rotation: ``t_new = t + dp``, ``R_new = Exp(dtheta) @ R``.

Moving sources share one optimizable prefix pose. Exact full-source static
inputs have no object columns and are copied unchanged. Optional hard corridors
are rejected for moving sources because these least-squares rows alone cannot
enforce them; silently treating them as soft terms would change the contract.
"""

from dataclasses import dataclass
from math import comb

import numpy as np
from scipy import sparse
from scipy.spatial.transform import Rotation

try:
    from .soma_object import _poses, _timeline, resolve_object_policy, validate_object_config
except ImportError:  # The external SOMA interpreter loads sibling modules.
    from soma_object import _poses, _timeline, resolve_object_policy, validate_object_config


_ORDERS = ("velocity", "acceleration", "jerk")
_HARD_LIMITS = frozenset({
    "max_translation_m", "max_rotation_rad", "max_translation_offset_rate_m_s",
    "max_rotation_offset_rate_rad_s",
})
_ROTATION_LOG_STEP = 1e-6
_GENERATORS = np.array([
    [[0., 0., 0.], [0., 0., -1.], [0., 1., 0.]],
    [[0., 0., 1.], [0., 0., 0.], [-1., 0., 0.]],
    [[0., -1., 0.], [1., 0., 0.], [0., 0., 0.]],
])


@dataclass(frozen=True)
class ObjectSequenceLayout:
    """Immutable original poses and their exact variable-sharing incidence."""

    frame_to_block: np.ndarray
    reference_rotations: np.ndarray
    reference_positions: np.ndarray
    contact_frame: int
    stationary_reference_frame: int
    block_count: int
    is_static: bool

    @property
    def frame_count(self):
        return len(self.frame_to_block)

    @property
    def variable_count(self):
        return 6 * self.block_count


def build_object_layout(reference_R, reference_t, cfg):
    """Bind original-source policy and six-column unique-pose blocks.

    With a nonempty prefix, block zero represents *all* frames strictly before
    ``contact_frame``. At contact and later, each frame has its own block. With
    contact at zero all frames are independent. Static arrays have only -1
    indices, and no optimization variables, irrespective of contact timing.
    """
    cfg = validate_object_config(cfg)
    reference_R, reference_t = _poses(reference_R, reference_t, "reference object")
    policy = resolve_object_policy(reference_R, reference_t, cfg)
    is_static = policy["source_is_static"]
    unsupported = sorted(set(cfg) & _HARD_LIMITS)
    if unsupported and not is_static:
        raise ValueError("simultaneous object least-squares does not enforce optional hard limits: "
                         + ", ".join(unsupported))
    frames, stop = len(reference_t), cfg["contact_frame"]
    if is_static:
        indices = np.full(frames, -1, dtype=np.int64)
        count = 0
    elif stop:
        indices = np.r_[np.zeros(stop, dtype=np.int64), np.arange(1, frames - stop + 1)]
        count = frames - stop + 1
    else:
        indices, count = np.arange(frames, dtype=np.int64), frames
    arrays = [indices, reference_R.copy(), reference_t.copy()]
    for value in arrays:
        value.setflags(write=False)
    return ObjectSequenceLayout(*arrays, stop, cfg["stationary_reference_frame"], count, is_static)


def initialize_object_poses(layout):
    """Copy original poses, flattening only the moving source's shared prefix."""
    rotations = layout.reference_rotations.copy()
    positions = layout.reference_positions.copy()
    if not layout.is_static and layout.contact_frame:
        anchor = layout.stationary_reference_frame
        rotations[:layout.contact_frame] = rotations[anchor]
        positions[:layout.contact_frame] = positions[anchor]
    return rotations, positions


def expand_object_poses(block_R, block_t, layout):
    """Expand unique absolute poses without a numeric stationarity penalty."""
    block_R, block_t = np.asarray(block_R), np.asarray(block_t)
    if block_R.shape != (layout.block_count, 3, 3) or block_t.shape != (layout.block_count, 3):
        raise ValueError("object blocks must have shapes [blocks,3,3] and [blocks,3]")
    if layout.is_static:
        return layout.reference_rotations.copy(), layout.reference_positions.copy()
    _poses(block_R, block_t, "object blocks")
    return block_R[layout.frame_to_block].copy(), block_t[layout.frame_to_block].copy()


def _checked_frame_poses(frame_R, frame_t, layout):
    frame_R, frame_t = _poses(frame_R, frame_t, "working object")
    if len(frame_t) != layout.frame_count:
        raise ValueError("working object and variable layout timelines differ")
    if layout.is_static:
        if not (np.array_equal(frame_R, layout.reference_rotations)
                and np.array_equal(frame_t, layout.reference_positions)):
            raise ValueError("static object poses must equal the original trajectory exactly")
    elif layout.contact_frame:
        stop = layout.contact_frame
        if not (np.all(frame_R[:stop] == frame_R[0]) and np.all(frame_t[:stop] == frame_t[0])):
            raise ValueError("precontact object frames must share one exactly constant pose")
    return frame_R, frame_t


def retract_object_poses(frame_R, frame_t, increment, layout):
    """Apply one simultaneous left-tangent increment per unique object pose."""
    frame_R, frame_t = _checked_frame_poses(frame_R, frame_t, layout)
    increment = np.asarray(increment, dtype=np.float64)
    if increment.shape != (layout.variable_count,) or not np.isfinite(increment).all():
        raise ValueError("object increment must match the finite layout tangent vector")
    if layout.is_static:
        return frame_R.copy(), frame_t.copy()
    first = np.r_[0, np.flatnonzero(np.diff(layout.frame_to_block)) + 1]
    step = increment.reshape(layout.block_count, 6)
    rotations = Rotation.from_rotvec(step[:, 3:]).as_matrix() @ frame_R[first]
    positions = frame_t[first] + step[:, :3]
    return expand_object_poses(rotations, positions, layout)


def _difference_matrix(frames, order):
    rows = max(0, frames - order)
    coefficients = [(-1.) ** (order - k) * comb(order, k) for k in range(order + 1)]
    return sparse.diags([np.full(rows, x) for x in coefficients], np.arange(order + 1),
                        shape=(rows, frames), format="csr")


def _frame_jacobian(blocks, layout, column_start):
    """Place [frames,residual_dimension,3] tangent blocks in object columns."""
    dimension = blocks.shape[1]
    if not layout.variable_count:
        return sparse.csr_matrix((layout.frame_count * dimension, 0))
    row = np.broadcast_to(np.arange(layout.frame_count * dimension).reshape(-1, dimension, 1),
                          blocks.shape)
    col = np.broadcast_to(layout.frame_to_block[:, None, None] * 6
                          + np.arange(column_start, column_start + 3), blocks.shape)
    result = sparse.coo_matrix((blocks.ravel(), (row.ravel(), col.ravel())),
                               shape=(layout.frame_count * dimension, layout.variable_count)).tocsr()
    result.eliminate_zeros()
    return result


def object_sequence_residual_jacobian(frame_R, frame_t, reference_R, reference_t, fps, cfg, layout):
    """Return original object-only residuals and sparse left-tangent Jacobian.

    The objective is ``residual @ residual`` (no 1/2), and each global temporal
    stencil is included exactly once. Rows are translation references, rotation
    references, then translation/rotation temporal rows in derivative order.
    Three-component rotation references are differentiated centrally in a
    vectorized batch. This follows SciPy's actual rotation-log normalization for
    original matrices rounded to their upstream precision; matrix temporal rows
    use analytic derivatives of the unmodified stored matrices.

    Static rows have no object columns. Their immutable-source retention is
    enforced before evaluation, rather than approximated by a large weight.
    """
    cfg = validate_object_config(cfg)
    frame_R, frame_t = _checked_frame_poses(frame_R, frame_t, layout)
    reference_R, reference_t = _poses(reference_R, reference_t, "reference object")
    fps = _timeline(fps, layout.frame_count, cfg)
    if not (np.array_equal(reference_R, layout.reference_rotations)
            and np.array_equal(reference_t, layout.reference_positions)):
        raise ValueError("object references differ from the immutable layout originals")
    if (cfg["contact_frame"] != layout.contact_frame
            or cfg["stationary_reference_frame"] != layout.stationary_reference_frame):
        raise ValueError("object config and immutable variable layout differ")
    unsupported = sorted(set(cfg) & _HARD_LIMITS)
    if unsupported and not layout.is_static:
        raise ValueError("simultaneous object least-squares does not enforce optional hard limits: "
                         + ", ".join(unsupported))
    n = layout.frame_count
    translation_blocks = np.broadcast_to(np.eye(3), (n, 3, 3))
    translation_jacobian = _frame_jacobian(translation_blocks, layout, 0)
    matrix_blocks = np.einsum("aij,tjk->tika", _GENERATORS, frame_R).reshape(n, 9, 3)
    matrix_jacobian = _frame_jacobian(matrix_blocks, layout, 3)
    relative = frame_R @ reference_R.transpose(0, 2, 1)
    reference_rotation = Rotation.from_matrix(relative).as_rotvec()
    log_blocks = np.empty((n, 3, 3))
    for axis in range(3):
        delta = np.eye(3)[axis] * _ROTATION_LOG_STEP
        plus = Rotation.from_rotvec(delta).as_matrix() @ relative
        minus = Rotation.from_rotvec(-delta).as_matrix() @ relative
        log_blocks[:, :, axis] = (Rotation.from_matrix(plus).as_rotvec()
                                 - Rotation.from_matrix(minus).as_rotvec()) / (2 * _ROTATION_LOG_STEP)
    rotation_reference_jacobian = _frame_jacobian(log_blocks, layout, 3)
    wt, wr = np.sqrt(cfg["reference_translation_weight"]), np.sqrt(cfg["reference_rotation_weight"])
    residuals = [wt * (frame_t - reference_t).ravel(), wr * reference_rotation.ravel()]
    jacobians = [wt * translation_jacobian, wr * rotation_reference_jacobian]
    for order, label in enumerate(_ORDERS, 1):
        for kind, values, base_jacobian, width in (
                ("translation", frame_t, translation_jacobian, 3),
                ("rotation", frame_R, matrix_jacobian, 9)):
            weight = cfg[f"{kind}_{label}_weight"]
            if not weight or n <= order:
                continue
            gain = np.sqrt(weight) * fps ** order
            residuals.append(gain * np.diff(values, n=order, axis=0).ravel())
            operator = sparse.kron(_difference_matrix(n, order), sparse.eye(width), format="csr")
            jacobians.append(gain * (operator @ base_jacobian))
    residual = np.concatenate(residuals)
    jacobian = sparse.vstack(jacobians, format="csr")
    jacobian.eliminate_zeros()
    if not (np.isfinite(residual).all() and np.isfinite(jacobian.data).all()):
        raise ValueError("nonfinite object sequence residual or Jacobian")
    return residual, jacobian
