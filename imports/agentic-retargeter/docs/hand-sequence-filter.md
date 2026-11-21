# Offline hand-trajectory filter comparison

[`hand_sequence_filter.py`](../src/hand_sequence_filter.py) creates a new
**UNREVIEWED free-hand target sequence** from an existing hand initializer.
It changes wrist poses and the seven native finger hinges per configured hand;
it does not mount the hands on a robot or run contact/collision optimization.
There is no GRAIL runtime dependency or separate finger-pad refinement stage.

The default [shared hand policy](hand-shared-policy.md) uses this module between
its base paired solve and contact-aware temporal refinement. The standalone
filter still produces an unreviewed intermediate; downstream hand targets come
from the final refined selection, not directly from the filtered output.

The default is a centered Savitzky–Golay filter with nine samples and polynomial
order three. Gaussian filtering is optional (`filter_kind: "gaussian"`), with
`sigma_frames: 1.5` and `truncate: 3.0`. Window sizes are source-frame counts,
not a change to the source clock. Reflected sample indices handle sequence ends.

This is an SG-family comparison inspired by the GRAIL filtering approach, not
an exact copy of its quaternion-component filtering. At each output frame, the
rotation samples are mapped to the center rotation's SO(3) tangent space,
weighted there, then mapped back with the exponential map. Euler coordinates
and raw quaternion components are never averaged. This handles quaternion sign
equivalence; principal-log ambiguity near a relative half-turn is still reported.

Each complete stencil uses object coordinates when its center has active
`hold`, `grasp`, `stable_hold`, `support` or `slide` contact. Other centers use
world coordinates. Neighbors are converted into that same coordinate system,
including neighbors across a phase boundary. Strict shared-hold intervals and
frame zero remain byte-exact. The tool requires an already exact original first
frame and constant declared holds; it does not silently repair those contracts.

## API

```python
from agentic_retargeter.hand_sequence_filter import run

report = run(
    candidate_path="paired/hand_sequence.npz",
    source_path="source/source.npz",  # immutable raw-source hand snapshot
    output="filter_comparison",      # must not exist
    joint_limits=original_native_limits,  # {side: array of seven [low, high] rows}
    config={
        "filter_kind": "savgol",
        "window_length": 9,
        "polyorder": 3,
        "translation_limit_m": 0.15,
        "rotation_limit_rad": 1.5,
        "strict_hold_intervals": paired_config["strict_hold_parameterization"]["intervals"],
    },
)
```

The interval mapping uses half-open `start` / `end_exclusive` records and
optional `grasp_id`. Provide every existing shared-control interval. Derive
limits from the bound original model in native finger-address order. The array
API is `filter_arrays(candidate, source, *, joint_limits, config=None)` and
returns copied arrays plus a report without editing either input.

The equivalent CLI accepts `--candidate`, `--source`, `--joint-limits` (JSON
mapping), `--config` (JSON) and `--out`. Outputs are
`hand_sequence_UNREVIEWED.npz` and `report.json`, with input/output/code hashes.
Body qpos, object transforms/scale, model/joint metadata, phases and timestamps
remain exact. Derived collision/contact diagnostics and parameterization
controls are removed and enumerated in the report because they are stale.

## Bounds and review

SG can overshoot. Proposed finger values are clipped to the original native
limits and every clipped value is counted. One correction fraction per hand is
then backtracked, checking the actual export dtype against the same immutable
source-relative translation and principal-rotation-vector component boxes.
Defaults are ±0.15 m and ±1.5 rad, with a 1e-6 numerical feasibility tolerance.
Frozen values remain exact throughout; no bound is relaxed. An infeasible input
is rejected, and a zero correction fraction is reported as no movement.

Centered interior stencils introduce no causal delay. Phase changes, exact
frozen keys, clipping and bound backtracking do **not** guarantee globally
zero-phase behavior, smooth joins or C2 native replay. Filtering can increase
contact gaps, change clearance or retain a bad held grasp. Recompute contact,
collision, source-deviation, velocity, acceleration and jerk metrics over the
exported sequence and its joins before selecting it. Filtering may instead be
used as a warm start for a separate contact-aware coupled solve.

The upstream focused tests (not shipped in this snapshot) covered SO(3)
branch/sign behavior, interior timing, strict-hold and first-frame preservation,
native/source bounds, phase coordinate consistency, immutable arrays and export
provenance. The initial implementation passed 17 tests and real 121-frame Frypan/Hammer
CPU export checks. Those checks establish tool execution and preservation,
not contact or motion acceptance.
