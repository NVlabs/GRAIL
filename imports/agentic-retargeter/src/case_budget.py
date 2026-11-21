"""Persistent per-case active-time budget; foreground owners hold an exclusive lock."""
from __future__ import annotations

import fcntl
import json
import math
import os
from pathlib import Path
import time

CASE_LIMIT_SECONDS = 30 * 60
DELIVERY_RESERVE_SECONDS = 4 * 60


def seconds_left(deadline):
    value = float(deadline)
    if not math.isfinite(value):
        raise ValueError("Deadline must be finite")
    return max(0., value - time.time())


class CaseBudget:
    """Count active attempts once; stopped gaps do not refill or spend the cap.

    A stale, unfinished attempt is charged through recovery time, capped at the
    limit. This deliberately fails closed after an uncatchable kill. Persist the
    same file across retries; changing output directories does not create time.
    A fresh ledger may use an explicitly configured positive finite limit;
    reopening an existing ledger always requires its unchanged original limit.
    """
    def __init__(self, path, case_id, *, limit_seconds=CASE_LIMIT_SECONDS):
        if type(limit_seconds) not in (int, float) or not math.isfinite(limit_seconds) or limit_seconds <= 0:
            raise ValueError("Case budget must be a positive finite number of seconds")
        self.path = Path(path).resolve()
        self.case_id, self.limit = case_id, float(limit_seconds)
        self.lock = None
        self.outcome = "INTERRUPTED"

    def _write(self):
        temporary = self.path.with_name(self.path.name + ".tmp")
        temporary.write_text(json.dumps(self.state, indent=2, allow_nan=False) + "\n")
        temporary.chmod(0o600)
        temporary.replace(self.path)

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.lock = self.path.with_name(self.path.name + ".lock").open("a")
        try:
            fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            if self.path.exists():
                self.state = json.loads(self.path.read_text())
                if (self.state["case_id"] != self.case_id
                        or type(self.state["limit_seconds"]) not in (int, float)
                        or self.state["limit_seconds"] != self.limit
                        or not math.isfinite(self.state["consumed_seconds"])
                        or not 0 <= self.state["consumed_seconds"] <= self.limit):
                    raise ValueError("Case budget identity/limit/accounting mismatch")
            else:
                self.state = dict(schema_version=1, case_id=self.case_id, limit_seconds=self.limit,
                                  consumed_seconds=0., attempts=0, active_since_unix=None)
            stale = self.state["active_since_unix"]
            if stale is not None:
                if not isinstance(stale, (int,float)) or not math.isfinite(stale) or stale > time.time():
                    raise ValueError("Invalid unfinished case budget timestamp")
                self.state["consumed_seconds"] = min(self.limit, self.state["consumed_seconds"] + time.time()-stale)
                self.state["recovered_unfinished_attempt"] = True
            self.started_monotonic = time.monotonic()
            self.started_unix = time.time()
            self.state.update(active_since_unix=self.started_unix, owner_pid=os.getpid(),
                              attempts=self.state["attempts"]+1)
            self._write()
            return self
        except BaseException:
            self.lock.close()
            raise

    @property
    def remaining_seconds(self):
        return max(0., self.limit-self.state["consumed_seconds"]-(time.monotonic()-self.started_monotonic))

    def snapshot(self):
        remaining = self.remaining_seconds
        return dict(path=str(self.path), limit_seconds=self.limit,
                    consumed_seconds=self.limit-remaining, remaining_seconds=remaining,
                    attempt=self.state["attempts"],
                    recovered_unfinished_attempt=self.state.get("recovered_unfinished_attempt",False))

    def __exit__(self, *exc):
        try:
            self.state.update(consumed_seconds=self.limit-self.remaining_seconds,
                              active_since_unix=None, finished_unix=time.time(),
                              last_outcome=self.outcome)
            self._write()
        finally:
            self.lock.close()
