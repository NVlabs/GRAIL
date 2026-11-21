"""Reproducible initial deltas for the existing batched free-wrist objective.

Only the starting iterate changes. The source wrist/finger pose remains the
center of the objective's priors and its original bounds for every start.
"""
import hashlib

import numpy as np


MAX_CANDIDATES_PER_HAND = 128


def validate_multistart(config, translation_limit_m, rotation_limit_rad):
    """Validate an opt-in configuration; absence preserves the legacy seed."""
    if config is None:
        return None
    required = {"count_per_frame", "seed", "translation_radius_m",
                "rotation_radius_rad", "finger_radius_rad"}
    if not isinstance(config, dict) or set(config) != required:
        raise ValueError(f"multistart must contain exactly {sorted(required)}")
    count, seed = config["count_per_frame"], config["seed"]
    if type(count) is not int or not 1 <= count <= MAX_CANDIDATES_PER_HAND:
        raise ValueError(f"multistart count_per_frame must be in [1, {MAX_CANDIDATES_PER_HAND}]")
    if type(seed) is not int or not 0 <= seed < 2**64:
        raise ValueError("multistart seed must be an integer in [0, 2**64)")
    bounds = {"translation_radius_m": translation_limit_m,
              "rotation_radius_rad": rotation_limit_rad, "finger_radius_rad": np.pi}
    for name, upper in bounds.items():
        value = config[name]
        if (type(value) not in (int, float) or not np.isfinite(value)
                or not 0 <= value <= upper):
            raise ValueError(f"multistart {name} must be finite and in [0, {upper}]")
    return dict(config)


def _ball(rng, radius):
    # Uniform volume in an object-coordinate 3D ball (axis-angle for rotation).
    direction = rng.normal(size=3)
    norm = np.linalg.norm(direction)
    return direction * (radius * rng.random() ** (1/3) / norm) if norm else np.zeros(3)


def make_initial_deltas(config, hand, frames, finger_qpos, finger_low, finger_high):
    """Return source row indices, float32 deltas, and stable per-candidate IDs.

    Randomness is keyed by hand, frame and start index, so adding/reordering
    source frames or changing the batch size does not change existing starts.
    Start zero is the unperturbed source. Finger radii are per joint; translation
    and rotation radii bound vector norms. Joint clipping uses original limits.
    ``config`` must first pass :func:`validate_multistart`.
    """
    q = np.asarray(finger_qpos, dtype=np.float32)
    low, high = np.asarray(finger_low, dtype=np.float32), np.asarray(finger_high, dtype=np.float32)
    if (not isinstance(hand, str) or not hand or not isinstance(frames, list)
            or not frames or any(type(f) is not int or f < 0 for f in frames)
            or len(set(frames)) != len(frames)):
        raise ValueError("multistart needs a hand name and unique nonnegative source frames")
    if (q.ndim != 2 or q.shape[0] != len(frames) or q.shape[1] == 0
            or low.shape != q.shape[1:] or high.shape != low.shape
            or not all(np.isfinite(v).all() for v in (q, low, high))
            or np.any(low > high)):
        raise ValueError("multistart finger poses and original joint limits must be finite and compatible")
    count = 1 if config is None else config["count_per_frame"]
    if config is not None and len(frames)*count > MAX_CANDIDATES_PER_HAND:
        raise ValueError(f"multistart exceeds {MAX_CANDIDATES_PER_HAND} candidates per hand")
    rows = np.repeat(np.arange(len(frames)), count)
    delta = np.zeros((len(rows), 6+q.shape[1]), dtype=np.float32)
    records = []
    hand_key = list(np.frombuffer(hashlib.sha256(hand.encode()).digest(), dtype="<u4"))
    for i, row in enumerate(rows):
        frame, start = frames[row], i % count
        if start:
            rng = np.random.default_rng(np.random.SeedSequence(
                [config["seed"], *map(int, hand_key), frame, start]))
            delta[i, :3] = _ball(rng, config["translation_radius_m"])
            delta[i, 3:6] = _ball(rng, config["rotation_radius_rad"])
            radius = config["finger_radius_rad"]
            finger = rng.uniform(-radius, radius, q.shape[1])
            delta[i, 6:] = np.clip(q[row]+finger, low, high)-q[row]
        records.append(dict(candidate_id=f"{hand}/frame_{frame}/start_{start}",
                            seed_frame=frame, start_index=start,
                            initial_delta=delta[i].tolist()))
    return rows, delta, records
