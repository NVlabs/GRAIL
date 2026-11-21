# Bounded case exploration and stopping rules

The default per-case operational retargeting limit is **30 minutes (1800 seconds)** of
cumulative active work, including solvers, audits, rendering and export. Stop at
the deadline even if the motion remains partial; this is not an infeasibility
certificate or a guarantee that every case can meet every quality gate.

## Selected time budget and optional iteration bounds

For a new ledger, declare `cumulative_case_budget_seconds` in the manifest or
use the 1800-second default. An explicit user cap takes precedence. Declare the
delivery reserve too (default 240 seconds); the complete v003 hand-plus-SOMA
cohort uses 3600 seconds and at least 180 seconds for final review. The separate
`timeout_seconds` bounds one attempt and never replenishes its ledger. Smaller
smoke-test timeouts also apply. Scheduler queue time is separate; a job's walltime
includes container startup and may leave less working time. Provision shared
runtimes before execution. These are operational bounds, not an invented
user-task stopping deadline; preserve partials and diagnose technical failures
without resetting accounting or silently abandoning an authorized full run.

Optional bounds of 6 outer rounds, 3 hypotheses, or 2 stalled reviews remain
proposals, not newly enabled caps. Record any chosen iteration limits separately.
Neither retries nor a new output directory authorizes resetting the case ledger.
The declared policy applies to new case ledgers. Historical ledgers retain their
recorded limits and spending; changing this default does not rewrite them or
restart an exhausted case. The ledger rejects a mismatched limit.
An explicitly authorized extension must preserve consumed time and attempt
history; do not delete the ledger or invent a fresh case identity to extend it.

One outer round is one explicit proposed change (or a declared finite candidate
bank), bounded computation, affected-scope checks and a review decision. Solver
iterations, frames and subprocess calls are not outer rounds. Declare a bank's
size before executing it; do not hide unbounded search inside one round. A
multi-step initialization recipe with known dependencies is one declared plan,
not a fresh hypothesis at every internal command. Rejected proposals count too.

Count hypotheses only when changing contact regions, pad correspondence, mounted
palm/arm branch or phase roles. Renaming a file or adjusting a weight does not
reset a branch's stagnation counter. Changing editable scope requires its existing
authorization and an explicit revision; it does not reset case spending.

## Three stop levels

| Level | Trigger | Required response |
|---|---|---|
| Candidate | A confirmed violation of a required gate, including a new forbidden crossing, frozen-field change or native limit violation | Reject immediately; retain the witness. Skip remaining acceptance checks and full rendering for this rejected candidate. A separately declared repair branch may consume it as diagnostic input; it is not an accepted propagation seed. |
| Branch | An explicitly selected branch-stagnation limit is reached, or the same failed trial has no new evidence-supported change to test | Stop repeating that branch. Retain the best reviewed checkpoint and move to another evidence-supported edit within the case budget. A branch stop alone does not terminate the case or enable an unselected two-review cap. Do not blindly add solver iterations or widen bounds. |
| Case | Required numerical and source/visual checks pass; the agreed round/hypothesis/time limit is reached; or a prerequisite remains blocked after its allowed recovery | Stop search. Deliver the checked result, or the best scoped partial/raw fallback with unresolved checks and the stop reason. No automatic restart to evade a cap. |

Successful completion ends search even if some acceptable metric could be made
smaller. A budget stop is `PARTIAL_BUDGET`, stagnation is `PARTIAL_STALLED`, and an
environment prerequisite failure is `BLOCKED_ENVIRONMENT` in the case decision.
These labels are a decision-record convention, not newly implemented runner
status enums. Keep the runner's process status and quality status separate.

## What counts as progress

Compare the same frames, original geometry, contact revision, metric definitions
and frozen reference. Establish acceptance thresholds and numerical tolerances
before the trial. Report contact gaps/slip, forbidden collision evidence,
source deviation and joint/wrist temporal measures separately; decreasing a
weighted solver loss alone is not progress. Contact-normal alignment is disabled
in the current workflow; its diagnostic angle is neither a rejection reason nor
a reason to continue search. Preserve collision geometry normals and the
source-supported object-relative palm orientation constraints.

A useful improvement removes a target failure or improves the declared primary
defect beyond its measurement tolerance without introducing a new hard-gate
failure or violating an already satisfied source/smoothness bound. Source/visual
review is still required. Once contact/geometry requirements are satisfied,
prefer the smoother candidate within the remaining gates, consistent with the
user's selection preference. An explicit scoped tradeoff may remain a partial;
do not relabel it as a universally better or accepted motion.

Unknown or incomplete required checks cannot establish a reviewed scoped pass
or full acceptance. Preserve measured improvements as diagnostics while finishing
the needed check, or stop with the unknowns when the budget is spent.
Unchanged frozen-body collisions, retained native seams and out-of-scope defects
remain separately reported; they do not justify repeated active-hand solves.
When changing a contact hypothesis, retain the old review and rebind affected
checks; distances to two different targets are not directly comparable progress.

Retain the best reviewed shared grasp when an approach-only or optional
secondary repair fails. Reducing a total intersection count does not justify a
new forbidden body pair; discard that proposal instead of merging it into the
delivered motion. Once the requested contact/slip defect passes its checks, an
unchanged inherited defect outside that scope is reported separately rather
than becoming an automatic new search. Optional cleanup cannot consume the
delivery reserve.

## Time and recovery accounting

Include agent deliberation, tools, failed branches, audits, rendering and export
in case-active time. Agent wall time already contains nested tool time: do not
add the two. Across retries/continuations, sum each attempt once; report scheduler
queue/container startup and elapsed calendar time separately. Concurrent sibling
cases do not consume each other's budget.

Before each long stage, compare its measured/estimated duration with remaining
search time. If it would consume the delivery reserve, stop proposing repairs.
Persist each reviewed checkpoint and audit chunk immediately. At the reserve
boundary, finalize the best retained result. Prefer cached coverage and images;
if final checks cannot finish, disclose missing coverage rather than reducing
sampling or declaring a pass. A retry or delivery-only continuation consumes
the same case budget; an extension must be explicit.

Tool failures are not motion failures. Retry a transient failure once only if
the dependency and saved-state checks support it. Repeated environment failures
end the affected case with evidence; independent cases can continue. Tool fixes
and recipe reconstruction belong in development accounting, not repeated inside
every case job.

## Decision record and current enforcement

Keep a case decision record with the policy, raw/input/recipe hashes, attempt
lineage and accumulated active time. For every round retain hypothesis ID,
parent and proposal hashes, changed scope, evidence paths, measured differences,
decision, branch stagnation count and best-checkpoint identity. Do not overwrite
the best checkpoint when a later proposal is rejected.

The batch runner enforces the declared cumulative limit (1800 seconds by default)
using an exclusive durable ledger (`case_budget.json` in the case output). All
retries/continuations must reuse the same ledger, kept separately from disposable
attempt workspaces. It preserves completed attempt time and refuses new work after
exhaustion. A stale unfinished attempt is charged conservatively through recovery
time, capped at the total limit. Report that recovery condition rather than
silently granting more time. Do not delete or relocate the ledger to extend work.

Shared batch stages inherit the case deadline, reject new solver/setup
stages at the delivery reserve, and cap each command by remaining time. Audits,
rendering and export may use the reserve. A batch running without the agent uses
its own persistent per-case ledger, reused by `--resume`. The owning GPU worker
also checks the hard deadline. The agent's choice to finalize remains prompt
guidance; the hard case deadline and any scheduler walltime bound execution.

Use [`tools/repair/progress.py`](../../../../tools/repair/README.md#preserve-reviewed-progress)
to retain the baseline, `best_scoped_reviewed_partial`, `best_diagnostic` and
explicit `full_accepted` separately. Each record binds an independent audit to
the exact candidate/config/assets/code and coverage; a reviewed partial also
needs a visual receipt bound to the candidate, scope and contact revision.
Screening reports and solver summaries cannot promote it. The helper ranks
improvements only within one fixed policy/revision by its declared lower-is-better
metric; it does not choose a new grasp hypothesis or replace visual judgment.
Missing source evidence remains unknown, including for an authored generated-hand
correction; do not omit a required source gate to claim full acceptance.

Call `progress.py decision` with the ledger's actual remaining case time and a
measured/estimated next-step duration. `CONTINUE`, `FINALIZE` and `FINISH` are
advice for the owning agent; this command neither runs another solver nor resumes
an agent or scheduler session. A normal process exit or saved media is not a quality pass.
The operational limit is a cap, not a requirement to consume all time after success.

At timeout, retain the ledger's evidence and finalize the best usable scoped
result or original-input fallback with unresolved checks. The batch runner retains
saved artifacts and budget-stop evidence; it does not invent exports, invoke the
progress helper or select a motion automatically. Automatic stagnation counters
and automatic quality-driven agent continuation are not implemented. No scheduled
restart is authorized by this policy.
