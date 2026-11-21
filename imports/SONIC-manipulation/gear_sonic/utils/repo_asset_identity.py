"""Pin-independent identity of assets that ship INSIDE the code checkout (``gear_sonic/data/...``).

Every detached pinned worktree holds the same robot USD / MJCF skeleton at a different absolute path with a fresh
checkout mtime, so a path+mtime record misses on every version switch. The identity used for cache COMPATIBILITY is
instead an ordinary, readable git fact plus the byte size:

    {"repo_relative": "gear_sonic/data/robots/g1/x.usda", "size_bytes": 2392,
     "asset_revision": "<last commit that touched the asset's directory tree>"}

* relocation of an unchanged checkout (new pin, new mtime) -> identical record;
* an edit of the asset, even with equal byte length, or of any sibling/referenced layer under the same directory
  tree -> a different last-touch commit -> not compatible (ordinary rebuild, native fallback);
* no git / not a repository / detached pin removed -> ``None`` and the caller keeps the raw path+mtime record,
  which never produces a false hit on its own; the authored-stage comparison then gives one exact second chance
  (``repo_asset_trees_identical``): two raw records of the same repo-relative asset in two different checkouts are
  compatible when the recorded file is unchanged since it was recorded (live size/mtime) and the asset's whole
  directory tree is byte-identical in both checkouts (read with filecmp; no digest). Generalist 36593474 rank 1
  (2026-10-14 20:34Z) lost 2604 s to a native rebuild because the git lookup failed once in that process while the
  seven other ranks - and the same process 55 min later - resolved it; a failure is now logged with its reason and
  never remembered.
No content digest is computed here; the commit id is git's own history metadata (the same id the pin is named by).

Scope (stated precisely): this identity is meant for CLEAN DETACHED PINS, which is what the boundary adopter and the
published-source follow create. An uncommitted edit inside a dirty checkout that keeps the byte size and leaves history
untouched does not change the identity - an applicability limitation of running from an editable tree, not a failure
of the pinned production path. Assets referenced from OUTSIDE the asset's directory tree (e.g. base-stage layers in
other roots) are not tracked here; nothing has shown them to enter the cached spawn subtree, and no dependency graph
scan or gate is added for a hypothetical case.
"""
from __future__ import annotations

import os
import contextlib
import filecmp
import logging
import subprocess
from collections.abc import Mapping
from typing import Any

logger = logging.getLogger(__name__)

REPO_ASSET_MARKER = "/gear_sonic/data/"


def split_repo_asset_path(path: str) -> tuple[str, str] | None:
    """(checkout_root, repo_relative) for a path under a code checkout's ``gear_sonic/data``, else None."""
    text = str(path or "")
    index = text.find(REPO_ASSET_MARKER)
    if index < 0:
        return None
    return text[:index], text[index + 1:]


_LAST_TOUCH_COMMITS: dict[tuple[str, str], str] = {}  # successes: remembered for the process
_FAILED_LOOKUPS: dict[tuple[str, str], str] = {}  # failures: remembered only inside one decision scope
_DECISION_DEPTH = 0


@contextlib.contextmanager
def decision_scope():
    """One cache-compatibility decision. Several comparison branches view the same asset record (exact view, mtime-flip
    tolerance, alias equivalence, byte second chance); inside the scope a lookup that already failed is not repeated,
    so a failed git call costs its attempts once per checkout per decision. The failure memory is dropped when the
    outermost scope ends: a transient failure is never carried into the next decision (no permanent poisoning)."""
    global _DECISION_DEPTH
    _DECISION_DEPTH += 1
    try:
        yield
    finally:
        _DECISION_DEPTH -= 1
        if _DECISION_DEPTH <= 0:
            _DECISION_DEPTH = 0
            _FAILED_LOOKUPS.clear()


def _last_touch_commit(checkout_root: str, repo_relative_dir: str) -> str | None:
    key = (checkout_root, repo_relative_dir)
    cached = _LAST_TOUCH_COMMITS.get(key)
    if cached is not None:
        return cached
    if _DECISION_DEPTH > 0 and key in _FAILED_LOOKUPS:
        return None
    argv = ["git", "-C", checkout_root, "log", "-1", "--format=%H", "--", repo_relative_dir]
    failure = ""
    for attempt in (1, 2):  # one immediate second attempt after a FAST failure only (36593474 rank 1 was transient)
        try:
            out = subprocess.run(argv, capture_output=True, text=True, timeout=20, check=False)
        except subprocess.TimeoutExpired as exc:
            failure = f"{type(exc).__name__}: {exc}"
            break  # a 20 s timeout is not a fast transient failure: no second attempt
        except (OSError, subprocess.SubprocessError) as exc:
            failure = f"{type(exc).__name__}: {exc}"
            continue
        commit = out.stdout.strip()
        if out.returncode == 0 and len(commit) >= 7:
            if attempt == 2:
                logger.warning(
                    "repo_asset_identity: git history lookup for %s (%s) succeeded on the second attempt (first: %s)",
                    checkout_root, repo_relative_dir, failure,
                )
            _LAST_TOUCH_COMMITS[key] = commit
            return commit
        failure = f"rc={out.returncode} stderr={(out.stderr or '').strip()[:300]!r} stdout={commit[:80]!r}"
    logger.warning(
        "repo_asset_identity: git history lookup unavailable for %s (%s): %s",
        checkout_root, repo_relative_dir, failure,
    )
    if _DECISION_DEPTH > 0:
        _FAILED_LOOKUPS[key] = failure
    return None


def repo_asset_identity(path: str, size_bytes: Any) -> dict[str, Any] | None:
    """Pin-independent record for a repository-internal asset, or None when it cannot be established."""
    parts = split_repo_asset_path(path)
    if parts is None:
        return None
    checkout_root, repo_relative = parts
    if not os.path.isdir(checkout_root):
        return None
    revision = _last_touch_commit(checkout_root, os.path.dirname(repo_relative) or ".")
    if revision is None:
        return None
    return {"repo_relative": repo_relative, "size_bytes": size_bytes, "asset_revision": revision}


def comparable_asset_record(record: Any) -> Any:
    """Record view for compatibility: repository-internal assets pin-independent, everything else unchanged."""
    if not isinstance(record, Mapping):
        return record
    identity = repo_asset_identity(str(record.get("path") or ""), record.get("size_bytes"))
    return identity if identity is not None else record


def _relative_files(root: str) -> set[str]:
    files: set[str] = set()
    for base, _dirs, names in os.walk(root):
        for name in names:
            files.add(os.path.relpath(os.path.join(base, name), root))
    return files


_TREE_COMPARISONS: dict[tuple[str, str], bool] = {}  # per process: pinned checkouts do not change while it runs


def _asset_trees_identical(dir_a: str, dir_b: str) -> bool:
    """Same relative file set, equal sizes, identical bytes (filecmp, no digest); traversed once per directory pair
    per process, so several records of the same asset directory never repeat the reads."""
    key = (dir_a, dir_b)
    known = _TREE_COMPARISONS.get(key)
    if known is not None:
        return known
    result = False
    if os.path.isdir(dir_a) and os.path.isdir(dir_b):
        files = _relative_files(dir_a)
        result = bool(files) and files == _relative_files(dir_b) and all(
            os.path.getsize(os.path.join(dir_a, rel)) == os.path.getsize(os.path.join(dir_b, rel))
            and filecmp.cmp(os.path.join(dir_a, rel), os.path.join(dir_b, rel), shallow=False)
            for rel in sorted(files)
        )
    _TREE_COMPARISONS[key] = result
    return result


def repo_asset_trees_identical(recorded: Any, current: Any) -> bool:
    """Exact second chance for a repository-internal asset record pair whose git identity could not be established:
    True when both records name the same repo-relative asset with the same byte size in two DIFFERENT checkouts,
    each record still matches its file's live size and mtime (the recorded file was not rewritten since the receipt),
    and the asset's whole directory tree is the same in both checkouts (same relative file set, equal sizes,
    identical bytes). Reads the files (gear_sonic/data/robots/g1: 2 files, 175 MB per checkout) only on this failure
    path; computes no digest; never raises; False on any doubt (ordinary miss)."""
    if not isinstance(recorded, Mapping) or not isinstance(current, Mapping):
        return False
    parts_a = split_repo_asset_path(str(recorded.get("path") or ""))
    parts_b = split_repo_asset_path(str(current.get("path") or ""))
    if parts_a is None or parts_b is None or parts_a[1] != parts_b[1] or parts_a[0] == parts_b[0]:
        return False
    if recorded.get("size_bytes") is None or recorded.get("size_bytes") != current.get("size_bytes"):
        return False
    rel_dir = os.path.dirname(parts_a[1]) or "."
    dir_a, dir_b = os.path.join(parts_a[0], rel_dir), os.path.join(parts_b[0], rel_dir)
    try:
        for record, parts in ((recorded, parts_a), (current, parts_b)):
            live = os.stat(os.path.join(parts[0], parts[1]))
            if live.st_size != record.get("size_bytes") or (
                record.get("mtime_ns") is not None and live.st_mtime_ns != record.get("mtime_ns")
            ):
                return False
        if not _asset_trees_identical(dir_a, dir_b):
            return False
    except OSError as exc:
        logger.warning("repo_asset_identity: byte comparison of %s and %s failed: %s", dir_a, dir_b, exc)
        return False
    logger.info(
        "repo_asset_identity: %s byte-identical in %s and %s (git identity unavailable on at least one side)",
        rel_dir, parts_a[0], parts_b[0],
    )
    return True
