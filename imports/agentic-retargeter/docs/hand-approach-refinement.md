# Approach-only hand refinement

`hand_sequence_approach.compile_approach_refinement` builds an optional repair of
an existing pre-SOMA free-hand sequence when acquisition motion is concentrated
in the final few frames. It does not rerun Topo, pad refinement or SOMA and does
not change the original frame clock, object motion, contact masks or role labels.

The shared recipe opens at most 1.5 seconds before each original contact onset or
stable-hold start. A preceding protected pose clips that window. Frame zero, all
stable-hold poses, original protected frames and frames outside the window stay
exactly equal to the selected warm sequence. The interval entering a frozen
endpoint remains in the objective. Empty scopes pass through without solving.

All existing contact, collision, source-pose and derivative costs remain. The
repair uses per-frame parameters and CUDA Graph evaluation for 900 Adam steps;
it does not reoptimize shared grip controls. This avoids a grip on a source-box
boundary stalling every approach variable through a common backtracking factor.
Direct object-relative wrist excess-speed costs use 0.5 m/s and 3 rad/s, weight
0.1 and normalization scales 0.2 m/s and 1 rad/s. Native finger speed uses the
existing soft 3 rad/s objective, weight 1. These are soft animation objectives,
not hard guarantees or hardware speed limits. Fixed endpoints and original
contact timing can make the nominal limits infeasible.

```python
from agentic_retargeter.hand_sequence_approach import compile_approach_refinement
from agentic_retargeter.hand_sequence_pair import solve

# arrays comes from the exact selected trajectory; paths in base are resolved.
config, scope = compile_approach_refinement(base, arrays)
config['warm_start_npz'] = str(selected_sequence)
if scope['needs_solve']:
    solve(config, directory=case_directory, output_directory=new_output)
```

Save the compiled configuration, scope and selected input hash beside the new
UNREVIEWED output. Never overwrite the selected input. Compare actual wrist and
finger speed at original times, approach-window and whole-sequence derivatives,
source deviations, per-digit contact coverage and original-mesh collision
samples at frames and midpoints. Render the full before/after sequence with a
matched camera at its original playback speed. Isolated-hand tests do not verify
mounted whole-body reachability or environment clearance.

Frozen poses enter every FK, objective and export as the exact warm states.
Their original source-box deviations are measured separately in float64; even
inherited float32 serialization errors are reported. Editable controls retain
the existing 1e-6 feasibility tolerance. `feasible` is explicitly scoped to the
editable variables when a frozen overlay exists; inspect
`all_frames_within_bounds` and `frozen_reference_bounds` for the full scope.
A good numerical objective or successful export does not accept motion quality.

The normal initialization policy remains versioned separately. This repair must
be explicitly selected and reviewed before it becomes a downstream SOMA target.

## Phase-derived quintic warm start

For a remaining late approach detour, the reusable
`hand_sequence_approach_seed.seed_approach_arrays` builds an initialization for
the same 900-step refinement. It is not a post-optimization filter or a replacement
for the native contact/collision objective. The chosen shared recipe is
`phase_quintic_22_over_24s_v001`: the same 22/24-second horizon and feasibility
policy for every case, without case names, hand-picked frame numbers or new loss
weights. A stable-hold entry at frame 50 on a 24 fps source therefore selects
frame 28 as the bridge start. The input is the currently selected hand sequence,
not a fresh Topo solution.

Each continuous acquisition followed by a hold produces one endpoint at the
original hold entry. An onset with no following hold, or with an intervening
support/slide/unknown role, stays a separate event. Quintic Hermite curves act
on object-relative wrist position, native finger angles, and a local SO(3)
tangent chart. The incoming velocity and acceleration come from a quadratic fit
to five original samples at their actual timestamps. A numerically constant
stable hold (at most `1e-6` m/rad variation across the existing hold) uses zero
outgoing derivatives; every actual hold sample is still preserved exactly.
Varying holds and non-hold endpoints use a quadratic fit to up to five outgoing
samples, with at least three required. They are never silently converted into
rigid grasps.

Only bridge interiors change. The existing approach edit mask remains the
optimizer's mask; the seed does not enlarge it. Protected boundaries shorten
the bridge, and an early start is advanced to frame 4 to provide five incoming
samples. Fewer than one interior sample or too little outgoing evidence gives
an explicit unchanged-input fallback. A discontinuous principal SO(3) chart
also retains the original. First poses, endpoints, stable holds, outside-window
poses, original source/body/object arrays, phase labels, timing and all other
metadata remain byte-identical.

The full proposal is checked against the original source position/rotation
boxes and native joint limits **after conversion to the output dtype**. If it
fails, a common deterministic sequence of blend factors `1, 1/2, ..., 2^-20`
mixes it toward the selected input: linear position/native fingers and an SO(3)
geodesic for orientation. No geometry or pose component is clipped. The first
bound-valid proposal is recorded, including its actual blend factor; if none
passes, the input stays unchanged. A tiny factor can yield very little benefit.
Feasible warm starts are not necessarily collision-free or smoother, and the
subsequent solve may change their temporal behavior.

```python
import numpy as np

from agentic_retargeter.hand_sequence_approach_seed import seed_approach_arrays

# Native model joint ranges must match inputs.hands[side].finger_joints order.
config, scope = compile_approach_refinement(original_config, selected_arrays)
seed_arrays, seed_receipt = seed_approach_arrays(
    original_config, selected_arrays, joint_limits=native_joint_limits)
if scope['needs_solve']:
    # Exclusive creation prevents overwriting an existing seed or selected input.
    with new_seed_path.open('xb') as stream:
        np.savez_compressed(stream, **seed_arrays)
    config['warm_start_npz'] = str(new_seed_path.resolve())
    config['paired_learning_rate'] = 0.001
    config['learning_rate_schedule'] = {
        'kind': 'cosine', 'minimum_rate': 0.00002}
    # Save config, scope and seed_receipt beside this new candidate.
    solve(config, directory=case_directory, output_directory=new_output)
```

The validated `quintic_seed_cosine_900_v001` recipe combines this seed with the
900-step refinement and a cosine learning rate from `0.001` to `0.00002`.
Set the schedule explicitly as above: `compile_approach_refinement` inherits the
base learning-rate settings, so compiling alone does not select this recipe.
All original loss weights, contact assignments and protected phases remain.

The warm-contract NPZ preserves inherited contact, collision and speed diagnostic
arrays solely for input compatibility. The receipt explicitly lists them as
**stale input evidence**; they do not describe the new seed. The optimizer
recomputes its output diagnostics. Save the seed receipt, actual configuration,
input/output hashes and full original-clock video comparison together. Continue
to report inherited frozen-bound exceptions separately from editable feasibility.
