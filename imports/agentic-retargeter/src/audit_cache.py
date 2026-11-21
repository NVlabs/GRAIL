"""Durable exact sample evidence, shared across audit processes and attempts.

Only numerical rows are reused. Coverage, invariants and trajectory derivatives
are always assembled again by AuditContext. A cache hit never accepts a motion.
"""
from __future__ import annotations

import hashlib
from importlib.metadata import version, PackageNotFoundError
import json
from pathlib import Path
import platform
import uuid


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False,
                                    separators=(",", ":")).encode()).hexdigest()


def runtime_binding():
    result = {"python": platform.python_version(), "machine": platform.machine()}
    for name in ("numpy", "scipy", "mujoco", "numba", "warp-lang"):
        try:
            result[name] = version(name)
        except PackageNotFoundError:
            result[name] = None
    return result


class AuditCache:
    def __init__(self, directory, binding):
        self.namespace = digest(binding)
        self.directory = Path(directory).resolve() / self.namespace
        self.directory.mkdir(parents=True, exist_ok=True)
        self.hits = self.misses = self.invalid = 0

    def _path(self, key):
        return self.directory / key[:2] / (key + ".json")

    def get(self, key):
        path = self._path(key)
        try:
            record = json.loads(path.read_text())
            if (record["namespace"] != self.namespace or record["key"] != key
                    or digest(record["row"]) != record["row_sha256"]):
                raise ValueError("Cache record mismatch")
            self.hits += 1
            return record["row"]
        except FileNotFoundError:
            pass
        except (OSError, ValueError, KeyError, TypeError):
            self.invalid += 1
        self.misses += 1
        return None

    def put(self, key, row):
        path = self._path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        record = dict(namespace=self.namespace, key=key, row=row, row_sha256=digest(row))
        temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
        temporary.write_text(json.dumps(record, allow_nan=False) + "\n")
        temporary.replace(path)
