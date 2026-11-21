# Timed case workflows

`run.py` executes an explicit list of foreground commands per case, with independent
case directories and bounded concurrency. It avoids a separate assistant turn
between already specified fitting, screening, repair and rendering commands.
It provides orchestration and timing, not automatic grasp planning or acceptance.

Each case is bounded by the current 30-minute budget. Under a case agent, the
runner inherits `RETARGET_AGENT_DEADLINE_UNIX` and `RETARGET_STOP_REPAIRS_UNIX`;
plan or stage environment overrides cannot extend them. Standalone runs keep
`case_budget.json` per case and reuse it on `--resume`. Preserve this ledger
across retries; a new plan/output directory is not permission for a new budget.

The final 4 minutes are reserved for delivery. Mark known delivery stages with
`category: "audit"`, `"validation"`, `"render"`, `"export"` or `"report"`.
Solver/setup/unspecified stages stop at the reserve boundary; do not relabel a
solver as delivery to evade it. Each command's timeout is capped by its remaining
allowance. `DELIVERY_RESERVE` returns control to the agent to author a delivery
plan from already saved checkpoints; `BUDGET_EXHAUSTED` ends computation.
Neither status implies motion acceptance or automatically exports a partial.
Save useful native files, review evidence and `best_checkpoint.json` throughout.

In the default per-case workflow, one agent owns each case and invokes this
runner with `--workers 1`; for many cases, run one case per GPU process with your
own scheduler. The same agent reviews numerical results and images at decision
boundaries, handles failed gates and revises the plan; the runner does not start
or replace that agent.

```bash
python tools/batch/run.py --plan plan.json --output runs/new_attempt --workers 4
```

The output directory must be new unless `--resume` is supplied. Each case gets a log for every attempted stage
and `execution.json`; the root report includes wall time and all case reports.
Arguments are passed directly to a subprocess without a shell. Cases may have a
`cwd`; optional plan/case `env` values override the inherited environment. Keep
credentials out of plans, logs and output directories. Do not run GPU workers in
parallel without explicit device ownership; CPU case concurrency is independent
of GPU allocation. No agent, model API or scheduled job is launched.

```json
{
  "schema_version": 1,
  "cases": [{
    "name": "case_a",
    "cwd": "/absolute/repository",
    "stages": [{
      "name": "screen",
      "argv": ["/path/python", "/path/screen.py", "--out", "{case_output}/screen"],
      "gate": {
        "path": "{case_output}/screen/review.json",
        "field": "required_scope_passed",
        "equals": true
      }
    }, {
      "name": "propagate",
      "argv": ["/path/python", "/path/propagate.py", "--out", "{case_output}/trajectory"]
    }]
  }]
}
```

`{case_output}` and `{run_output}` are absolute directory substitutions. A command
failure, missing/malformed review or a gate mismatch stops that case. Other cases
continue and every attempted stage retains timing, logs and the gate file hash.
No failed candidate is silently propagated. Gate values are type-sensitive:
`1`, `null`, a missing field and `true` are distinct. Bind the review to the actual
candidate and all required checks in the screening command; the runner cannot
infer omitted motion checks. An executed command and a complete workflow are
always reported separately from motion acceptance.

Use the grasp-first skill to author the stage plan. Batch a small explicitly
declared contact bank, include difficult orientations and short transitions in
the screening stage, and propagate only after its required scope passes. For an
intentional rejected diagnostic, make a separate diagnostic plan and retain that
status rather than disabling the normal propagation gate. Reuse original geometry
and compiled runtimes within workers; repair changed intervals and joins, then
render the selected full trajectory once. Keep raw source evidence, hypotheses,
numeric reviews and visual selection distinct.

Record development/setup, command execution, solving, audit and rendering
separately. Concurrent case/stage durations overlap; their sum is not elapsed
wall time. A rerun from raw initialization must not use a previous repaired
trajectory as fitting input or a seed. Historical trajectories may be read for
explicitly labeled comparisons; record that separately from solver inputs. Disclose reused source caches and fixed planning
inputs; such reuse is not a cold full reconstruction benchmark.

For an already reviewed bank of native key poses, `src/pose_path.py` provides
`select_pose_path(poses, costs, times, valid=..., scales=...)`. It minimizes supplied
pose costs and normalized hinge velocity across the complete bank instead of
choosing each key independently. Mark failed candidates invalid; an infeasible
key raises rather than selecting the least bad pose. Native limited hinges are
not wrapped by 2π. This chooses a discrete path only: actual interpolated intervals,
collision, source and motion derivatives still require independent review. During
propagation, avoid applying raw finger-closure changes a second time to an already
fitted hold; bind the intended hand-shape/timing prior explicitly.

The Python helper `affected_samples(before_qpos, after_qpos, times)` returns all
samples whose interpolation endpoints changed, including interval joins. It is
not cache authorization: object motion, geometry, contact targets, timebase and
audit policy must also match. Preserve cached rows outside that support and bind
the merged report to both the unchanged evidence and the new candidate.

Use an explicit stage `gate` after static-grasp / short-transition review. A zero
process exit code only means the check executed; it must not permit propagation
of a rejected grasp. A repair branch can then consume the rejected checkpoint and
its witnesses. Keep source-defined phases and solver settings unchanged when
benchmarking execution speed.

## Fixed execution and interrupted runs

For a reviewed case recipe, keep its objectives, contact hypotheses, phase keys,
weights and audit density fixed. Execute the authored stages in this order:

1. Bound input/topology checks and the cached raw diagnostic audit.
2. Declared static grasp candidates, then a grasp review gate.
3. Difficult short transitions, then their review gate.
4. Phase propagation and declared smoothing/local repairs.
5. Full declared-scope audit (unchanged sample evidence reused).
6. One hardware-rendered full-duration preview and scene export/reopen checks.

Use existing commands for each phase; the runner does not replace the solver or
invent a new phase plan. A local correction retains reviewed grasp/phase gates
and runs only its specified branch. The [local repair example](local_repair_plan.example.json)
shows this shorter route. Fill runtime, input and cache paths explicitly; its
three example render frames are a draft, so replace them with the actual
full-duration frame list for delivery. Never
substitute historical refined motion for raw merely to make a rerun faster.

```bash
python tools/batch/run.py --plan plan.json --output runs/fixed_case --workers 4
python tools/batch/run.py --plan plan.json --output runs/fixed_case --resume --workers 4
```

New stage fields:

- `inputs`: explicit files **or directories** whose content binds execution.
  Include scripts/configs, original model with referenced meshes, motion,
  contact/source definitions and the pinned runtime manifest. Directories are
  recursively hashed; avoid an entire dataset when a case bundle suffices.
- `outputs`: files/directories required for reuse. Include images/native/scene
  artifacts, not just a report claiming they exist. Missing, changed or damaged
  outputs force re-execution. Use a child directory, not the stage root itself.
- `timeout_seconds`: optional bound for this stage and its child process group.
- `category`: timing label such as `solver`, `audit`, `render` or `export`.
- `json_files`: JSON templates written inside the stage directory; string values
  support the same substitutions as command arguments.

`{stage_output}` is a unique directory for this attempt of a stage.
`{stage:fit}` refers to the output of a preceding stage named `fit`.
Tools requiring new output directories should use a child: `{stage_output}/fit`.
The old `{case_output}` / `{run_output}` substitutions remain supported.

Resume skips only a successful prefix with explicit inputs **and** outputs,
matching content/command/code/executable/environment overrides, and a still-valid
type-sensitive gate. The first invalid stage and its downstream stages get new
attempt directories. Numerical and image caches still reuse unchanged samples.
Failed attempts and logs remain; no failed gate is converted to a pass. Concurrent
writers to the same batch output are rejected with a lock. Runtime dependency
manifests must be included in inputs; the generic runner cannot discover arbitrary
child commands' external packages or undeclared data dependencies.

`agent_request.json` contains failed cases and their failed stage/log/gate.
Give that file to the agent instead of requesting another complete open-ended
case run. Hypothesis/recipe changes remain explicit revisions; `--resume`
revalidates the unchanged prefix. Missing source/visual reviews are not supplied
by numeric gates. All execution completion remains UNREVIEWED.
