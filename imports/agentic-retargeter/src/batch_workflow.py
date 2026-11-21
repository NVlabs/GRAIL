"""Run explicit case stages without a new agent round between local commands.

Commands run in foreground children, with independent logs and monotonic timing.
This runner records execution, never motion acceptance. A numeric review gate
can stop propagation while preserving the complete failed stage and its inputs.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import fcntl
import json
import os
from pathlib import Path
import re
import subprocess
import shutil
import uuid
import signal
import time
from .case_budget import CaseBudget, DELIVERY_RESERVE_SECONDS, seconds_left


class BudgetStop(RuntimeError):
    pass


def _now():
    return datetime.now(timezone.utc).isoformat()


def _write(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def _safe_name(value):
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", value):
        raise ValueError("Case and stage names must be simple unique identifiers")
    return value


def validate_plan(plan):
    if plan.get("schema_version") != 1 or not plan.get("cases"):
        raise ValueError("Expected schema_version 1 and nonempty cases")
    seen = set()
    for case in plan["cases"]:
        name = _safe_name(case["name"])
        if name in seen or not case.get("stages"):
            raise ValueError("Duplicate case or empty stages")
        seen.add(name)
        stages = set()
        for stage in case["stages"]:
            name = _safe_name(stage["name"])
            if name in stages:
                raise ValueError("Duplicate stage")
            stages.add(name)
            argv = stage.get("argv")
            if not isinstance(argv, list) or not argv or any(not isinstance(x, str) or not x for x in argv):
                raise ValueError("argv must be a nonempty list of strings; no shell is used")
            timeout = stage.get("timeout_seconds")
            if timeout is not None and (type(timeout) is not int or timeout <= 0):
                raise ValueError("timeout_seconds must be a positive integer")
            for key in ("inputs", "outputs"):
                if key in stage and (not isinstance(stage[key], list) or not stage[key]
                                     or any(not isinstance(v, str) or not v for v in stage[key])):
                    raise ValueError(key + " must be a nonempty list of file paths")
            for name in stage.get("json_files", {}):
                if Path(name).name != name or not name.endswith(".json"):
                    raise ValueError("json_files keys must be plain JSON filenames")
            if stage.get("gate") is not None:
                gate = stage["gate"]
                if not isinstance(gate, dict) or set(gate) != {"path", "field", "equals"}:
                    raise ValueError("gate requires path, field and equals")
    return plan


def _file_hash(path):
    path = Path(path)
    if path.is_dir():
        return hashlib.sha256(json.dumps(
            [(str(p.relative_to(path)), _file_hash(p)) for p in sorted(path.rglob("*")) if p.is_file()],
            separators=(",", ":")).encode()).hexdigest()
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _bound_files(paths, expand, cwd):
    return {str(p := (Path(cwd) / expand(name)).resolve(strict=True)): _file_hash(p)
            for name in paths}


def _gate(stage, expand, cwd):
    if not stage.get("gate"):
        return None
    gate = stage["gate"]
    path = (Path(cwd) / expand(gate["path"])).resolve(strict=True)
    evidence = json.loads(path.read_text())
    observed = evidence
    for part in gate["field"].split("."):
        observed = observed[part]
    return dict(path=str(path), sha256=_file_hash(path), observed=observed,
                expected=gate["equals"],
                passed=type(observed) is type(gate["equals"]) and observed == gate["equals"])


def _command(argv, cwd, env, stream, timeout):
    process = subprocess.Popen(argv, cwd=cwd, env=env, stdin=subprocess.DEVNULL,
                               stdout=stream, stderr=subprocess.STDOUT, process_group=0)
    try:
        return process.wait(timeout=timeout)
    finally:
        # Keep nested Blender/solver processes owned by this foreground stage.
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait()


def run_plan(plan, output, *, workers=1, resume=False):
    validate_plan(plan)
    if type(workers) is not int or not 1 <= workers <= 32:
        raise ValueError("workers must be an integer between 1 and 32")
    output = Path(output).resolve()
    if resume:
        if not (output / "plan.json").is_file():
            raise ValueError("Resume requires an existing batch output")
    else:
        output.mkdir(parents=True, exist_ok=False)
        _write(output / "plan.json", plan)
    with (output / ".run.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return _execute_plan(plan, output, workers=workers, resume=resume)


def _execute_plan(plan, output, *, workers, resume):
    from .render_sequence import code_fingerprint
    implementation = code_fingerprint()
    start, started = time.monotonic(), _now()
    invocation = uuid.uuid4().hex
    _write(output / ("plan_" + invocation + ".json"), plan)

    def case_run(case):
        directory = output / case["name"]
        directory.mkdir(exist_ok=resume)
        if "RETARGET_AGENT_DEADLINE_UNIX" in os.environ:
            deadline = float(os.environ["RETARGET_AGENT_DEADLINE_UNIX"])
            seconds_left(deadline)
            stop = min(deadline,float(os.environ.get("RETARGET_STOP_REPAIRS_UNIX",deadline)))
            seconds_left(stop)
            return case_steps(case,directory,deadline,stop)
        with CaseBudget(directory / "case_budget.json",case["name"]) as budget:
            deadline = time.time()+budget.remaining_seconds
            if "RETARGET_ALLOCATION_DEADLINE_UNIX" in os.environ:
                deadline = min(deadline,float(os.environ["RETARGET_ALLOCATION_DEADLINE_UNIX"]))
                seconds_left(deadline)
            stop = time.time()+max(0.,budget.remaining_seconds-DELIVERY_RESERVE_SECONDS)
            stop = min(stop,deadline- min(DELIVERY_RESERVE_SECONDS,seconds_left(deadline)*.2))
            report = case_steps(case,directory,deadline,stop)
            budget.outcome = report["status"]
            report["case_budget"] = budget.snapshot()
            _write(directory / "execution.json", report)
            return report

    def case_steps(case, directory, deadline, stop_repairs):
        report = {"case": case["name"], "started_utc": _now(), "stages": [], "motion_accepted": False}
        report["deadlines"] = dict(hard_unix=deadline,stop_new_repairs_unix=stop_repairs)
        tick = time.monotonic()
        replacements = {"{case_output}": str(directory), "{run_output}": str(output)}
        cwd = str(Path(case.get("cwd", ".")).resolve())
        dirty = not resume
        env = os.environ.copy()
        env.update(plan.get("env", {}))
        env.update(case.get("env", {}))
        # Stage/plan overrides cannot extend the owning supervisor's deadline.
        env.update(RETARGET_AGENT_DEADLINE_UNIX=str(deadline),RETARGET_STOP_REPAIRS_UNIX=str(stop_repairs))

        def expand(text):
            for key, value in replacements.items():
                text = text.replace(key, value)
            return text

        def expand_json(value):
            if isinstance(value, str):
                return expand(value)
            if isinstance(value, list):
                return [expand_json(v) for v in value]
            if isinstance(value, dict):
                return {k: expand_json(v) for k, v in value.items()}
            return value

        for stage in case["stages"]:
            st = time.monotonic()
            stage_root = directory / "stages" / stage["name"]
            stage_root.mkdir(parents=True, exist_ok=True)
            receipt = stage_root / "receipt.json"
            row = dict(name=stage["name"], started_utc=_now(), category=stage.get("category", "unspecified"))
            log = None
            budget_limited = False
            delivery = stage.get("category") in {"audit", "render", "export", "report", "validation"}
            stage_deadline = deadline if delivery else min(deadline,stop_repairs)
            try:
                if seconds_left(deadline) <= 0:
                    raise BudgetStop("BUDGET_EXHAUSTED")
                previous = json.loads(receipt.read_text()) if receipt.exists() else {}
                # Exact-prefix reuse. Any rerun invalidates downstream execution
                # receipts; persistent numerical/image caches still reuse data.
                old_output = previous.get("output", "")
                replacements["{stage_output}"] = old_output
                inputs = _bound_files(stage.get("inputs", []), expand, cwd)
                executable = shutil.which(expand(stage["argv"][0]), path=env.get("PATH"))
                if not executable:
                    raise FileNotFoundError(stage["argv"][0])
                binding = dict(stage=stage, inputs=inputs, cwd=cwd,
                               environment={**plan.get("env", {}), **case.get("env", {})},
                               executable=_file_hash(executable), implementation=implementation,
                               dependencies={k: v for k, v in replacements.items()
                                             if k.startswith("{stage:")})
                fingerprint = hashlib.sha256(json.dumps(binding, sort_keys=True).encode()).hexdigest()
                reusable = (not dirty and stage.get("inputs") and stage.get("outputs")
                            and previous.get("status") == "EXECUTED"
                            and previous.get("fingerprint") == fingerprint)
                if reusable:
                    try:
                        reusable = (previous["output_hashes"] == _bound_files(stage["outputs"], expand, cwd))
                        gate = _gate(stage, expand, cwd)
                        reusable = reusable and (gate is None or gate["passed"])
                        if gate is not None:
                            reusable = reusable and previous.get("gate") == gate
                    except (OSError, ValueError, KeyError, TypeError):
                        reusable = False
                if reusable:
                    row.update(status="REUSED", output=old_output, fingerprint=fingerprint,
                               reused_receipt=str(receipt), original_seconds=previous["seconds"],
                               gate=previous.get("gate"), output_hashes=previous["output_hashes"])
                else:
                    if seconds_left(stage_deadline) <= 0:
                        raise BudgetStop("BUDGET_EXHAUSTED" if seconds_left(deadline) <= 0 else "DELIVERY_RESERVE")
                    dirty = True
                    attempt = stage_root / ("attempt_" + uuid.uuid4().hex)
                    attempt.mkdir()
                    replacements["{stage_output}"] = str(attempt)
                    for name, value in stage.get("json_files", {}).items():
                        _write(attempt / name, expand_json(value))
                    # Inputs may not depend on an as-yet-unwritten output of this stage.
                    inputs_now = _bound_files(stage.get("inputs", []), expand, cwd)
                    if inputs_now != inputs:
                        raise ValueError("Stage inputs may not refer to its own stage_output")
                    row.update(argv=[expand(x) for x in stage["argv"]], output=str(attempt),
                               fingerprint=fingerprint, inputs=inputs)
                    log = attempt / "command.log"
                    remaining = seconds_left(stage_deadline)
                    if remaining <= 0:
                        raise BudgetStop("BUDGET_EXHAUSTED" if seconds_left(deadline) <= 0 else "DELIVERY_RESERVE")
                    budget_limited = stage.get("timeout_seconds",float('inf')) >= remaining
                    timeout = min(stage.get("timeout_seconds",remaining),remaining)
                    row["effective_timeout_seconds"] = timeout
                    with log.open("w") as stream:
                        returncode = _command(row["argv"], cwd, env, stream, timeout)
                    row["returncode"] = returncode
                    row["status"] = "EXECUTED" if returncode == 0 else "COMMAND_FAILED"
                    if inputs != _bound_files(stage.get("inputs", []), expand, cwd):
                        raise ValueError("Bound stage input changed during execution")
                    if code_fingerprint() != implementation or _file_hash(executable) != binding["executable"]:
                        raise ValueError("Implementation or executable changed during execution")
                    if returncode == 0:
                        row["output_hashes"] = _bound_files(stage.get("outputs", []), expand, cwd)
                        row["gate"] = _gate(stage, expand, cwd)
                        if row["gate"] is not None and not row["gate"]["passed"]:
                            row["status"] = "GATE_NOT_PASSED"
            except BudgetStop as error:
                row.update(status=str(error))
            except subprocess.TimeoutExpired:
                row.update(status=("BUDGET_EXHAUSTED" if delivery or seconds_left(deadline) <= 0 else "DELIVERY_RESERVE")
                           if budget_limited else "TIMED_OUT")
            except Exception as error:
                row.update(status="ERROR", error=f"{type(error).__name__}: {error}")
            row.update(seconds=time.monotonic() - st, completed_utc=_now(), log=str(log) if log else None)
            if row["status"] != "REUSED":
                _write(receipt, row)
                if row.get("output") and Path(row["output"]).is_dir():
                    _write(Path(row["output"]) / "execution.json", row)
            report["stages"].append(row)
            report.update(seconds=time.monotonic() - tick, status=row["status"])
            if row["status"] in {"BUDGET_EXHAUSTED", "DELIVERY_RESERVE"}:
                report["quality_status"] = "PARTIAL_BUDGET"
            _write(directory / "execution.json", report)
            print(json.dumps({"case": case["name"], "stage": row["name"], "status": row["status"],
                              "seconds": round(row["seconds"], 3)}), flush=True)
            if row["status"] not in ("EXECUTED", "REUSED"):
                break
            replacements["{stage:" + stage["name"] + "}"] = row["output"]
        else:
            report["status"] = "EXECUTION_COMPLETE_UNREVIEWED"
        report.update(seconds=time.monotonic() - tick, completed_utc=_now())
        _write(directory / "execution.json", report)
        return report

    with ThreadPoolExecutor(max_workers=workers) as pool:
        cases = list(pool.map(case_run, plan["cases"]))
    report = dict(started_utc=started, completed_utc=_now(), wall_seconds=time.monotonic() - start,
                  workers=workers, cases=cases, motion_accepted=False,
                  timing_note="Case/stage elapsed intervals overlap across workers; do not sum as wall time.")
    _write(output / "execution.json", report)
    _write(output / ("execution_" + invocation + ".json"), report)
    _write(output / "agent_request.json", dict(motion_accepted=False, cases=[
        dict(case=c["case"], status=c["status"], failed_stage=c["stages"][-1],
             instruction=("Stop new repairs; finalize the best saved partial/fallback within the remaining deadline. Do not reset the case budget."
                          if c["status"] in {"BUDGET_EXHAUSTED", "DELIVERY_RESERVE"}
                          else "Inspect the failed scope; preserve completed stages. Resume the explicit plan after repair within the remaining case budget."))
        for c in cases if c["status"] != "EXECUTION_COMPLETE_UNREVIEWED"]))
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--case-index", type=int, help="Run one case, e.g. one scheduler array task")
    args = parser.parse_args(argv)
    plan = json.loads(Path(args.plan).read_text())
    validate_plan(plan)
    if args.case_index is not None:
        if not 0 <= args.case_index < len(plan["cases"]):
            raise ValueError("case-index outside plan")
        plan = dict(plan, cases=[plan["cases"][args.case_index]])
    report = run_plan(plan, args.output, workers=args.workers, resume=args.resume)
    return int(any(case["status"] != "EXECUTION_COMPLETE_UNREVIEWED" for case in report["cases"]))


def affected_samples(before, after, times):
    """Return samples touched by changed native interpolation endpoints.

    This identifies native-qpos support only, not a cache authorization. Model,
    geometry, contacts, object trajectory, timebase and policy must be unchanged
    independently. Linear hinges / endpoint-based quaternion interpolation both
    depend on the same endpoints. No numerical tolerance hides small edits.
    """
    import numpy as np
    before, after = np.asarray(before), np.asarray(after)
    times = np.asarray(times, dtype=float)
    if (before.ndim != 2 or before.shape != after.shape or not len(before)
            or not np.isfinite(before).all() or not np.isfinite(after).all()
            or times.ndim != 1 or not np.isfinite(times).all()
            or np.any(times < 0) or np.any(times > len(before) - 1)):
        raise ValueError("Expected matching finite native matrices and valid sample times")
    changed = np.any(before != after, axis=1)
    lo, hi = np.floor(times).astype(int), np.ceil(times).astype(int)
    return times[changed[lo] | changed[hi]].tolist()


if __name__ == "__main__":
    raise SystemExit(main())
