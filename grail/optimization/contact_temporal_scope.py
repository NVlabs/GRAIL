"""Load source-reviewed temporal scopes without baking case frames into defaults."""

import copy
import hashlib
import json
from pathlib import Path

SCHEMA = "grail.contact_temporal_scope.v1"
SCOPE_FIELDS = {"fps", "hold_interval_ids", "windows"}


def load_temporal_scope(path, *, contact_map_sha256, defaults):
    """Bind a reviewed scope to the exact map consumed by the current solve.

    Case-specific FPS, hold IDs and windows come only from the sidecar. Scalar
    loss settings remain in the run config. Native geometry/FPS/region validation
    is performed by PhaseContactTemporalLoss before any optimizer update.
    """
    if not isinstance(path, str) or not path or "{" in path:
        raise ValueError("Temporal scope requires a resolved scope_path before Stage 2")
    if not isinstance(defaults, dict) or SCOPE_FIELDS.intersection(defaults):
        raise ValueError("With scope_path, FPS/hold IDs/windows must come only from the scope")
    try:
        payload = Path(path).read_bytes()
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"Temporal scope is missing: {path}. Before Stage 2, review the source "
            "holds and transition windows; see docs/fixed_contact_points.md."
        ) from exc
    spec = json.loads(payload)
    fields = {"schema", "contact_map_sha256", "reviewed_by", "options"}
    if not isinstance(spec, dict) or set(spec) != fields or spec["schema"] != SCHEMA:
        raise ValueError(f"Temporal scope requires schema {SCHEMA} and fields {sorted(fields)}")
    if not isinstance(spec["reviewed_by"], str) or not spec["reviewed_by"].strip():
        raise ValueError("Temporal scope requires reviewed_by")
    digest = spec["contact_map_sha256"]
    if (not isinstance(digest, str) or len(digest) != 64
            or any(c not in "0123456789abcdef" for c in digest)
            or digest != contact_map_sha256):
        raise ValueError("Temporal scope contact_map_sha256 differs from the consumed contact map")
    if not isinstance(spec["options"], dict) or set(spec["options"]) != SCOPE_FIELDS:
        raise ValueError("Temporal scope options must declare exactly fps, hold_interval_ids, windows")
    options = copy.deepcopy(defaults)
    options.update(copy.deepcopy(spec["options"]))
    provenance = {
        "path": str(Path(path).resolve()),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "scope": copy.deepcopy(spec),
    }
    return options, provenance
