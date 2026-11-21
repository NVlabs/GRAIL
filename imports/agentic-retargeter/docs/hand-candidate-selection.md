# Explicit bank selection into hand sequence seeds

`agentic_retargeter.hand_candidate_selection` connects the existing Topo/pad
multi-start banks to the phase-aware sequence helpers. It performs no solve,
changes no geometry, and accepts no motion. The caller supplies the exact bank
SHA256 and original `GraspInputs.binding` when loading a bank.

```python
from agentic_retargeter.hand_candidate_selection import (
    load_bound_bank, shortlist_candidates, build_seed_plans)
from agentic_retargeter.hand_sequence_seed import build_seed
from agentic_retargeter.hand_sequence_path_seed import build_path_seed

banks = {"pad": load_bound_bank(
    pad_path, sha256=pad_sha256, expected_binding=ctx.binding, stage="pad")}
shortlist = shortlist_candidates(banks, "right", source_arrays=raw_source_arrays)

# Example source-frame interval only: derive actual intervals from reviewed
# source phase labels. start/end are inclusive; contact_phases uses end-exclusive.
selection = {"right": [{
    "start": 30, "end": 90, "role": "hold",
    "candidate": {"stage": "pad", "candidate_id": "right/frame_60/start_7",
                  "diagnostic_reason": "Test this rejected seed; keep its collision evidence"}
}]}
hold_plan, path_plan, receipt = build_seed_plans(
    banks, sequence_config, selection, native_joint_limits)
held, hold_receipt = build_seed(raw_source_arrays, sequence_config, hold_plan)
seed, path_receipt = build_path_seed(
    held, sequence_config, path_plan, native_joint_limits)
```

Each selection explicitly names `stage: topo|pad` and `candidate_id`. Multiple
phase-specific banks can use explicit `bank_id`, passed to `load_bound_bank`
and each choice. Candidate names, native joint order/limits, proper rotations,
phase roles, contact masks, source frame indices and overlapping intervals are
checked. Candidate position, rotation and fingers are transferred unchanged.
Every key carries the bank hash, original binding, raw rejection reasons,
inherited parent Topo evidence, saved screen and patch policy. A rejected row
requires a nonempty diagnostic reason. That reason never removes its rejection.

The shortlist exposes lexicographic columns: incomplete declared coverage,
missing declared points, sampled penetration, then source deviation when known.
Unknown evidence sorts last. Source deviation is translation/0.05 m plus
rotation/0.5 rad plus finger RMS/0.5 rad; its raw components are also returned.
There is no hidden pass score. Original first-witness screen failures remain
visible and require independent geometry review before any use as accepted data.
Comparing banks with different regions or coverage policies is not an equal
contract comparison. Saved coverage thresholds are never changed by selection.

Strict `hold`, `grasp` or `stable_hold` intervals select one object-relative pose
for `build_seed`. This creates a constant warm seed, not a guarantee that later
optimization preserves it. Use the pair solver's explicit shared-hold
parameterization to retain exact relative wrist/finger constancy. Check the
candidate against the source-relative bounds at **every** frame of its hold.

Support/sliding phases instead provide explicit ordered key layers:

```json
{"start": 91, "end": 120, "role": "slide", "key_layers": [
  {"frame": 95, "candidates": [
    {"stage": "pad", "candidate_id": "right/frame_95/start_2",
     "diagnostic_reason": "Diagnostic path candidate, not accepted"}]},
  {"frame": 120, "candidates": [
    {"stage": "pad", "candidate_id": "right/frame_120/start_4",
     "diagnostic_reason": "Diagnostic path candidate, not accepted"}]}
]}
```

The existing path helper performs dynamic programming over these alternatives,
then PCHIP position/fingers and SO(3) Slerp at the original timestamps. It retains
the preceding/following warm poses as endpoints; a phase at a sequence boundary
must provide that endpoint key explicitly. It rejects a constant support/slide
path. Hold seeds should be applied first. No fixed material anchors are created.

The path DP consumes a field named `five_mm_spaced_coverage`, so this adapter
requires the saved patch policy to use at most 5 mm gap and at least 3 mm spacing.
Older looser banks are rejected for this path bridge, not relabeled as tighter
evidence. No anchor search or threshold relaxation occurs here. Unvisited pairs
in a first-witness screen remain unresolved, including retained self crossings.

First-frame preservation remains an explicit sequence optimization constraint;
this bridge does not imply it. Retain raw source frame zero and evaluate entry/
exit joins, both-hand interactions, original-mesh/subframe collisions and source
deviation after sequence optimization. Free-hand keys do not establish mounted
arm reachability, scene clearance, force closure or final motion acceptance.
