"""Hash-bound warm starts for free-hand banks; rejection never becomes approval."""

import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from .contact_edit.pose import _keys
from .grasp_io import rigid


def load_bank_seed(spec, directory, ctx):
    """Read one immutable bank and require the original native/model binding."""
    _keys(spec, {"path", "sha256"}, set(), "warm_start_bank")
    if not isinstance(spec["path"], str) or not spec["path"]:
        raise ValueError("warm_start_bank.path must be a nonempty string")
    path = (Path(directory) / spec["path"]).resolve(strict=True)
    raw = path.read_bytes()
    actual = hashlib.sha256(raw).hexdigest()
    if spec["sha256"] != actual:
        raise ValueError("Warm-start bank SHA256 mismatch")
    bank = json.loads(raw)
    if bank.get("schema_version") != 1 or bank.get("binding") != ctx.binding:
        raise ValueError("Warm-start bank must bind exact original native/model/object inputs")
    if not isinstance(bank.get("candidates"), dict) or set(bank["candidates"]) != set(ctx.hands):
        raise ValueError("Warm-start bank must contain exactly the configured hands")
    return bank, dict(path=str(path), sha256=actual, exact_original_binding=True,
                     source_prior_unchanged=True, motion_accepted=False)


def bank_seed_deltas(bank, side, hand, frames, starts, source_positions,
                     source_rotations, source_fingers, joint_limits, wrist_limits):
    """Validate all rows before returning free-wrist/finger offsets.

    Rejected poses may seed a repair, but their complete parent bank remains
    bound by hash and their rejection is carried into every child row. They do
    not confer geometry, reachability or motion acceptance on the new result.
    """
    rows = bank["candidates"][side]
    if not isinstance(rows, list) or len(rows) != len(frames):
        raise ValueError("Warm-start row count mismatch")
    low, high = np.asarray(joint_limits).T
    deltas, reviews = [], []
    for i, (frame, row) in enumerate(zip(frames, rows)):
        if (not isinstance(row, dict) or row.get("seed_frame") != frame
                or row.get("finger_joints") != hand["finger_joints"]):
            raise ValueError("Warm-start source frame or named joint mapping mismatch")
        if starts is not None and row.get("candidate_id") != starts[i]["candidate_id"]:
            raise ValueError("Warm-start multistart identity mismatch")
        if type(row.get("rejected")) is not bool or type(row.get("motion_accepted")) is not bool:
            raise ValueError("Warm-start must declare rejection and motion acceptance status")
        p = np.asarray(row.get("position"), dtype=float)
        q = np.asarray(row.get("finger_qpos"), dtype=float)
        r = rigid(row.get("rotation"), "warm-start rotation")
        if (p.shape != (3,) or q.shape != (len(hand["finger_joints"]),)
                or not np.isfinite(p).all() or not np.isfinite(q).all()):
            raise ValueError("Warm-start position and fingers must be finite and correctly shaped")
        # Small tolerance allows a float32 export at a native hinge bound.
        if np.any(q < low - 1e-6) or np.any(q > high + 1e-6):
            raise ValueError("Warm-start fingers exceed original joint limits")
        delta = np.concatenate((p - source_positions[i],
                                Rotation.from_matrix(r @ source_rotations[i].T).as_rotvec(),
                                q - source_fingers[i]))
        if (np.max(np.abs(delta[:3])) > wrist_limits[0] + 1e-6
                or np.max(np.abs(delta[3:6])) > wrist_limits[1] + 1e-6):
            raise ValueError("Warm-start exceeds original free-wrist bounds")
        deltas.append(delta)
        reviews.append(dict(candidate_index=i, candidate_id=row.get("candidate_id"),
                            seed_frame=frame, rejected=row["rejected"],
                            rejection_reasons=row.get("rejection_reasons", []),
                            parent_motion_accepted=row["motion_accepted"],
                            inherited_acceptance=False))
    return np.asarray(deltas), reviews
