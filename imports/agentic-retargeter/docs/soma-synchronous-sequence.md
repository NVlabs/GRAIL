# Simultaneous SOMA trajectory optimization

This explicit converter option runs the original forward sequential robot IK,
then solves one sparse nonlinear problem containing every source-frame robot
tangent coordinate and every editable object pose. The original human,
sampled-clearance and native joint-limit residuals remain. Material-contact
residuals apply to moving-source objects only, as described below.
Robot scalar-hinge and object temporal residuals use the original SI-scaled
velocity, acceleration and jerk energies. Free-root coordinates are variables,
but this option does not silently add new free-root temporal penalties.

Each linearization stacks the original Newton frame residuals and robot
Jacobians, analytic material/contact and grouped-clearance object derivatives,
and the actual cross-frame sparse temporal rows. A whole-trajectory LM/LSMR
increment is accepted only when the complete nonlinear energy decreases.
Serial residual evaluation does not turn the update into independent frame IK.
The native upstream integration and limit clamp apply to each candidate.
The original native limit-filter residual is unchanged; its constant upstream
Jacobian approximation is replaced with its exact derivative in this mode.

In `temporal_optimization`:

```json
{
  "method": "synchronous_sparse",
  "sweeps": 1,
  "velocity_weight": 0.002,
  "acceleration_weight": 0.0001,
  "jerk_weight": 0.0000001,
  "max_seconds": 900,
  "synchronous_solver": {"max_iterations": 20, "max_seconds": 900}
}
```

The legacy required `sweeps` setting does not execute conditional sweeps in this
mode. Actual reports retain `completed_sweeps=0` and report accepted
`completed_global_iterations` instead. G20 is a separate research seed and is
not automatically combined with this mode. Existing sweep-mode configurations
remain unchanged when this new method is absent.

Moving objects have one shared optimizable precontact SE(3) pose, guaranteeing
an exactly stationary prefix while permitting a global shift and rotation.
Postcontact frames have independent object variables. Exactly static full-source
trajectories are copied unchanged with zero object variables. Original-reference
object costs are soft. Optional hard displacement/rate corridors are currently
rejected rather than silently ignored by the least-squares model.

For a static full-source object, both material-position and normal-alignment
contact losses are omitted from the forward IK and synchronous solve. This uses
the same exact whole-source translation-and-rotation test as static passthrough;
a stationary precontact prefix of a moving object still retains its authored
contact phases. Collision avoidance against the object and support scene,
self-collision, human targets, joint limits and robot temporal costs remain.
Authored anchors and phase gains are retained for diagnostics, with zero effective
contact objective counts and an explicit `contact_loss_policy` in the report.
Static cases may specify `contacts: []`; moving cases still require contacts.
Older compiled contact NPZ inputs are classified from their original reference
poses at runtime, so they follow this policy without rewriting archived inputs.

Saved evidence includes the initial CSR Jacobian and residual, full native
initial/final states, shared-pose incidence, accepted-energy history, source
hashes, and an independently validated CSV/native projection. A complete step
is not a convergence proof. Estimated contacts and incomplete collision query
coverage remain unaccepted; original geometry and material IDs are retained.

Tests cover sparse/dense parity, whole-system descent/rejection/deadlines,
actual Newton free-root integration, material and original-mesh collision
derivatives, exact temporal energies, static/shared object invariants, original
CSV semantics, and rejection of falsely relabeled sweep results. Representative
full-input pilots and complete playable videos are required before cohort
release. Preserve previous experiment outputs and cumulative per-case ledgers.

## CUDA forward initialization and synchronous evaluation

With `method: "synchronous_sparse"`, an omitted converter `device` now selects
the CUDA device used by synchronous evaluation. Omitted evaluation settings
are equivalent to:

```json
{
  "evaluation_backend": "cuda_graph",
  "evaluation_device": "cuda:0"
}
```

For a CPU-only deployment or a CPU comparison, explicitly set converter
`"device": "cpu"`, set `"evaluation_backend": "serial"`, and omit
`evaluation_device`. An omitted forward device also resolves to CPU when the
evaluator is explicitly `serial`. To compare only the forward stage while
retaining CUDA global evaluation, set `"device": "cpu"` and retain the default
`cuda_graph` evaluator. Legacy
alternating-sweep configurations and configurations without the synchronous
method retain their existing defaults. This changes future synchronous runs;
it does not rewrite archived configurations, outputs or audit evidence.
The direct contact-bridge CLI uses the same forward-device resolver. An explicit
converter `device` or CLI `--device` always takes precedence, including when
the forward and global-evaluation devices differ. An invalid evaluator setting
or an unavailable CUDA device raises; neither stage silently falls back to CPU.
Existing configurations that explicitly contain `"device": "cpu"` continue to
use CPU forward initialization until that setting is changed or omitted.

CUDA forward retains the upstream sequential warm start, objectives, LM
accept/reject rule and iteration count. Its bounded-memory linear algebra
forms the normal matrix and gradient on the device, then solves the small
joint-space system. The residual count no longer enlarges a fused LM tile
beyond per-block shared-memory capacity. This does not make sequential frames
independent or reduce contact/collision coverage to save time. CUDA graph
execution keeps each frame's iterative solve on the device. Float32 execution
can still change the resulting trajectory, so execution tests and native-output
comparisons remain separate from motion acceptance.
The adapter is limited to at most 64 native tangent DOFs and checks the
pinned Newton LM method and buffer contract. Larger systems and unsupported
runtime changes raise an error rather than selecting an unverified fallback.

The synchronous evaluator rebuilds matching model/objective buffers after the
forward pass, retaining the original source targets, weights, topology,
material contacts and scene. It never executes the dense upstream Newton LM
kernel. Two captured graphs evaluate the original frame
residuals and analytic Jacobians, with staged targets and bulk output transfers.
Collision failures are checked over the complete evaluated sequence.

The second stage remains a **hybrid backend**: global temporal/object-reference rows, CSR assembly,
native limit derivative correction and sparse LM/LSMR still run on CPU. Inspect
`global_problem.execution_devices`, `device_setup_seconds` and
`evaluation_telemetry` to verify actual execution and distinguish setup/capture
from warm evaluation time. Compare forward setup and execution time separately
from global optimization, export, diagnostic checks and rendering; a shorter
kernel runtime alone does not establish an end-to-end speedup.
Initial compilation and graph capture can dominate the first process using a
new kernel/device cache. A later process can reuse compiled files while still
building its own CUDA graph. Report that ordering when comparing cases or
variants; a faster second process does not prove its objective is cheaper.

Upstream source-human FK and effector scaling also run on the requested forward
device. Identical BVH, scale and mapping settings therefore need not produce
byte-identical float32 target arrays across CPU and CUDA. Validate that
preprocessing separately from robot optimization: capture the target arrays
before IK, match their hashes to the actual run receipts, and measure position
and orientation differences. A source hash remaining fixed *within* one solve
does not establish equality *between* two device runs.

CPU and CUDA float32 collision queries can produce different normals near zero
distance or nearest-face transitions. Strict graph-versus-direct-CUDA checks
and separate CPU/CUDA component/output comparisons are required when validating
this backend; a graph match or speedup does not establish CPU-equivalent
motion, complete collision coverage, convergence or physical acceptance.
