# Current agentic retargeting workflow

The selected initialization route is **Topo-inspired hand keys → paired hand
sequence → phase-aware SG initial guess → contact-aware temporal hand refinement
→ hand trajectory targets → fresh SOMA forward IK → synchronous whole-body
optimization → native export and review**. The agent identifies source phases
and reviews the results;
the tools execute explicit configurations. No tool automatically accepts a
motion or converts a rejected pose into a reviewed grasp.

## Bind the case and support scene

Keep the original source clock, human motion, model geometry/limits, object
scale and full object trajectory bound by hashes. Reuse HOI contact evidence to
author hand roles, contact intervals and eligible original object regions.
Human-thumb regions guide the Dex3 thumb; the remaining fingers can supply a
shared region for the two long fingers, with separate coverage checks per digit.
This contact policy is separate from the anatomical graph correspondence.

For the authorized tabletop setting, use the
[floating support-block tool](support-block.md). Its default frame-zero recipe
centers a static block below the object, with world-XY dimensions at least
**0.60 × 0.50 m**, or the object footprint plus **0.03 m per side** if larger,
and **0.08 m** thickness. Preserve the original support in an archive. Bind the
replacement to SOMA collision inputs, native export, audits and renders; a new
mesh in a render alone does not change the solver scene. Other cases retain
their original scene unless replacement is explicitly authorized.

Support preparation changes no robot/object pose, including frame zero. For
subsequent collision-key repair, protect the selected original first-frame
pose and keep editable intervals after frame zero. A global soft pose target
does **not** hard-pin frame zero: use the declared refinement scope when exact
preservation is required and check exported equality. Record deviations from
the original first-frame reference separately from equality to a refinement
parent; preserving an already changed parent does not restore the source.

## Search and optimize hands

The routine [shared hand policy](hand-shared-policy.md) uses one versioned
numerical profile for every case. The agent identifies source-video intervals
and their grasp identities/roles; it does not choose case-specific loss weights,
transition lengths, collision sampling or representation branches. Observed
stable grips share an object-relative wrist/finger pose within each interval;
support, sliding, uncertain motion and transitions retain per-frame variables.

The reusable [Topo candidate tool](topo-hand-initialization.md) runs a GPU-batched
multi-initialization structure objective. The standalone distal-pad correction
is off by default and remains available only for explicit ablations. Contact
surface objectives and collision checks remain in the sequence optimizer.
Wrist position/orientation and native finger joints are free; source body, arms
and object remain reference data. It preserves candidate identities, parent
hashes, failed screens and full collision geometry. The graph is inspired by
TopoRetarget; this is not a full paper reproduction.

The shared-policy entrypoint records deterministic candidate ranking and path
selection, including rejected seed evidence, before the sequence solve. It
does not accept the selected seeds as collision-free grasps. The existing
`hand_sequence_seed` and `hand_sequence_path_seed` utilities create declared
hold or varying-path seeds. Then use `hand_sequence` or `hand_sequence_pair`
for all original frames, with phase-aware temporal terms, source priors and
native limits. The direct all-frame Topo/temporal route remains available without a
key bank. Hand-only quality does not prove arm reachability or body clearance.
See [configuration and execution](hand-initialization-workflow.md).

The default shared profile retains the original 300-step paired solve, then
applies SG9/poly3 filtering and another 300-step paired solve. The final solve
uses contact weight 256, four spatially separated contact samples per digit
(3 mm spacing), and 100 times the base temporal weights. These settings are
identical across cases. The filter preserves raw frame zero and each strict
hold pose; the following optimizer may adjust a hold coherently while keeping
one pose within its interval. The original source bounds, geometry, source
priors and collision terms remain. This is the tested additional-refinement
route; skipping the base paired solve has not been validated. The final
hash-bound selection, rather than the intermediate paired output, supplies
SOMA's hand targets.

An object is static only if its **entire original** rotation and translation
are exactly constant. The hand solvers infer this from the source; a supplied
`is_static_obj` flag must agree. Static cases skip contact-key search and omit
hand-object contact attraction/material-position/normal losses in the hand and
SOMA stages. Collision, source and temporal objectives remain. A moving
object's stationary prefix is a different policy and does not disable contact
for the sequence.

Review object-relative wrist motion as well as world motion, fingers, velocity,
acceleration/jerk, contact acquisition and source deviation. A low jerk score
with lost contact or reduced motion amplitude is a tradeoff, not an automatic
improvement. Keep strict holds, support, sliding and release distinct.

## Hand targets and two SOMA stages

Use `tools/retarget/hand_targets.py` to export an explicitly selected sequence.
It validates solver/input/checkpoint hashes, source timing, original trajectory
and scale, named native joints, rotations and phase masks. It loads the output
through the actual SOMA target consumer before publishing a new directory.
No automatic selection or whole-body solve occurs in this exporter.

Wrist and finger poses are the primary hand objectives. Sparse fixed material
anchors are optional additional evidence: author only sufficiently close,
properly spaced pairs with appropriate phase coverage. Preserve missing
coverage and failed candidate records. The exporter saves anchor evidence
separately and never silently installs it as a contact loss. A pose-only
experiment is explicitly labeled; soft pose targets do not guarantee no slip.

Run both fresh SOMA stages with the bound hand targets:

1. Sequential forward whole-body IK on CUDA.
2. All-frame synchronous optimization with CUDA Graph residual/Jacobian
   evaluation and CPU sparse LM/LSMR solving.

Keep original body tracking, object-reference and collision terms and the
declared temporal settings. Enabled hand targets may explicitly replace the
source hand tracking contribution. The source-defined precontact prefix of a
moving object shares one adjustable but strictly stationary pose; contact
quality does not control when this optimization is permitted. Export the exact
solver-bound native robot/object state. See
[handoff and execution](hand-initialization-workflow.md#hand-target-handoff)
and [synchronous solver](soma-synchronous-sequence.md).

## Optional collision-key refinement

Inspect the final full robot with the same support scene. For hand/block or
hand/body collisions, choose affected frames and native joints, author bounded
pose corrections, and retain the unchanged first frame and guard intervals.
Save versioned native/Blender state and review the mounted result; moving an
isolated wrist is not a complete robot repair.

Feed reviewed key targets into synchronous refinement using the bound prior
global state and explicit `refinement_scope`. Restrict editable columns to the
authorized joints/windows when the rest must remain fixed. Recheck the actual
export, including original meshes, held-out fractional times, interpolation
joins, contact, source posture and temporal regressions. A local clearance gain
does not erase inherited failures elsewhere. Keep unsuccessful branches as
diagnostics and preserve the best reviewed partial. See
[refinement loop](hand-soma-refinement-workflow.md) and
[scope contract](soma-refinement-scope.md).

## Execute many cases

For many cases, run one case per GPU process with your own scheduler; keep inputs
read-only and give each case its own output directory and ledger. Each agent runs
its planning, tools, diagnostics and rendering for its own case. Preserve the same
case ledger across retries; scheduling changes do not reset it.

Provision the external SOMA/Blender runtimes, original assets and all
case-specific inputs before starting a case. A generic one-GPU process does not
make a historical multi-case script compatible without checking its bindings.
