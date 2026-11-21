"""Time-weighted tolerance for brief contact gap and normal excursions.

This classifies existing measurements. It does not measure clearance, alter
contact targets, waive collisions, or accept a complete motion.
"""

from __future__ import annotations

import copy
import math

VISUAL_GAP_TOLERANCE = {
    "nominal_gap_m": 0.015,
    "hard_gap_m": 0.040,
    "max_outlier_fraction": 0.05,
    "max_outlier_duration_s": 0.20,
}
VISUAL_NORMAL_TOLERANCE = {
    "nominal_normal_deg": 60.0,
    "hard_normal_deg": 180.0,
    "max_outlier_fraction": 0.05,
    "max_outlier_duration_s": 0.20,
}
_METRICS = {
    "gap": ("gap_m", "nominal_gap_m", "hard_gap_m", VISUAL_GAP_TOLERANCE),
    "normal": ("normal_deg", "nominal_normal_deg", "hard_normal_deg", VISUAL_NORMAL_TOLERANCE),
}


def validate_policy(policy, metric="gap"):
    field, nominal, hard, profile = _METRICS[metric]
    if not isinstance(policy, dict) or set(policy) != set(profile):
        raise ValueError("Contact tolerance policy needs all four documented fields")
    for key, value in policy.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError("Invalid contact tolerance policy value: " + key)
    if not 0 < policy[nominal] <= policy[hard]:
        raise ValueError("Need 0 < nominal <= hard tolerance")
    if metric == "normal" and policy[hard] > 180:
        raise ValueError("Normal angle cannot exceed 180 degrees")
    if not 0 <= policy["max_outlier_fraction"] <= 1 or policy["max_outlier_duration_s"] < 0:
        raise ValueError("Invalid contact gap outlier fraction/duration")
    return copy.deepcopy(policy)


def evaluate_contact_tolerance(rows, *, policy, fps, required_frames, contacts, metric="gap"):
    """Evaluate each contact separately on its complete declared sample grid.

    Contacts contain unique ``name`` and inclusive ``active_frames``. Rows contain
    ``source_frame`` and ``contacts`` with ``name`` / ``gap_m`` or ``normal_deg``. Midpoint cells
    clipped to the first/last required sample estimate time spent beyond nominal;
    duplicate samples cannot dilute a fraction. Missing samples fail coverage.
    A single static sample permits no transient exception. Shards must be merged
    before using this to judge a longer interval.
    """
    policy = validate_policy(policy, metric)
    field, nominal, hard, _ = _METRICS[metric]
    if isinstance(fps, bool) or not isinstance(fps, (float, int)) or not math.isfinite(fps) or fps <= 0:
        raise ValueError("fps must be positive and finite")
    frames = [float(f) for f in required_frames]
    if not frames or any(not math.isfinite(f) or f < 0 for f in frames) or len(set(frames)) != len(frames):
        raise ValueError("required_frames must be finite, nonnegative and unique")
    frames.sort()
    definitions = {}
    for c in contacts:
        name = c["name"]
        active = c["active_frames"]
        if not isinstance(name, str) or not name or name in definitions:
            raise ValueError("Contact names must be nonempty and unique")
        if len(active) != 2 or any(not math.isfinite(v) for v in active) or not 0 <= active[0] <= active[1]:
            raise ValueError("Invalid active_frames")
        definitions[name] = active
    observed = {}
    for row in rows:
        frame = float(row["source_frame"])
        if not math.isfinite(frame) or frame < 0 or frame in observed:
            raise ValueError("Observed frames must be finite, nonnegative and unique")
        observed[frame] = {}
        for c in row["contacts"]:
            name, gap = c["name"], c[field]
            if name not in definitions or name in observed[frame]:
                raise ValueError("Unknown or duplicate contact measurement")
            if isinstance(gap, bool) or not isinstance(gap, (int, float)) or not math.isfinite(gap) or gap < 0:
                raise ValueError("Invalid " + field)
            if metric == "normal" and gap > 180:
                raise ValueError("Normal angle cannot exceed 180 degrees")
            observed[frame][name] = float(gap)
    # Extra probes remain binding rather than being silently discarded.
    grid = sorted(set(frames) | set(observed))
    reports = []
    for name, active in definitions.items():
        samples = [t for t in grid if active[0] <= t <= active[1]]
        if not samples:
            continue
        missing = [t for t in samples if name not in observed.get(t, {})]
        if missing:
            reports.append(dict(name=name, passed=False, complete=False,
                                missing_frames=missing, violations=[], tolerated_frames=[]))
            continue
        gaps = [observed[t][name] for t in samples]
        edges = [samples[0]] + [(a+b)/2 for a,b in zip(samples,samples[1:])] + [samples[-1]]
        weights = [(b-a)/fps for a,b in zip(edges,edges[1:])]
        duration = sum(weights)
        bad = [g > policy[nominal] + 1e-10 for g in gaps]
        bad_seconds = sum(w for w,b in zip(weights,bad) if b)
        fraction = bad_seconds / duration if duration else float(any(bad))
        runs = []
        start = None
        for i,b in enumerate(bad + [False]):
            if b and start is None:
                start = i
            if not b and start is not None:
                runs.append(dict(first_frame=samples[start], last_frame=samples[i-1],
                                 duration_s=(edges[i]-edges[start])/fps,
                                 source_frames=samples[start:i]))
                start = None
        long_frames = {t for r in runs if r["duration_s"] > policy["max_outlier_duration_s"] + 1e-10
                       for t in r["source_frames"]}
        violations, tolerated = [], []
        for t,g,b in zip(samples,gaps,bad):
            reasons = []
            if g > policy[hard] + 1e-10:
                reasons.append("hard_" + metric)
            if b and fraction > policy["max_outlier_fraction"] + 1e-10:
                reasons.append("outlier_fraction")
            if b and duration == 0:
                reasons.append("static_sample")
            if t in long_frames:
                reasons.append("outlier_duration")
            if reasons:
                violations.append(dict(source_frame=t, **{field:g}, reasons=reasons))
            elif b:
                tolerated.append(t)
        reports.append(dict(name=name, passed=not violations, complete=True,
                            sample_count=len(samples), **{"max_"+field:max(gaps)},
                            active_sampled_duration_s=duration, outlier_samples=sum(bad),
                            outlier_fraction=fraction, outlier_duration_s=bad_seconds,
                            longest_outlier_duration_s=max((r["duration_s"] for r in runs), default=0),
                            outlier_runs=runs, violations=violations, tolerated_frames=tolerated))
    return dict(policy=policy, passed=bool(reports) and all(r["passed"] for r in reports),
                complete=all(r["complete"] for r in reports), contacts=reports,
                metric=("native_pad_to_assigned_object_target" if metric == "gap"
                        else "native_pad_opposition_angle_to_assigned_object_normal"),
                scope_frames=[grid[0],grid[-1]],
                duration_method="sample midpoint cells, clipped to active sampled span; finite samples only")


def evaluate_gap_policy(rows, **kwargs):
    return evaluate_contact_tolerance(rows, metric="gap", **kwargs)
