"""Source-specific temporal scopes cannot silently drift from their contact map."""

import copy
import hashlib
import json

import pytest

from grail.optimization.contact_temporal_scope import SCHEMA, load_temporal_scope


@pytest.fixture
def scope(tmp_path):
    spec = {
        "schema": SCHEMA,
        "contact_map_sha256": "a" * 64,
        "reviewed_by": "source review",
        "options": {
            "fps": 24.,
            "hold_interval_ids": {"left": ["hold"], "right": []},
            "windows": [{"id": "join", "frames": [1, 7], "ramp_frames": 2,
                         "regions": ["left/hold/pad"], "object": True}],
        },
    }
    path = tmp_path / "scope.json"
    path.write_text(json.dumps(spec))
    return path, spec


def test_load_preserves_authored_scope_and_separate_scalar_settings(scope):
    path, spec = scope
    defaults = {"hold_ramp_frames": 4, "window_speed_tolerance_mps": .35}
    options, provenance = load_temporal_scope(str(path), contact_map_sha256="a" * 64,
                                             defaults=defaults)
    assert options == dict(defaults, **spec["options"])
    assert provenance["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert provenance["scope"] == spec
    options["hold_interval_ids"]["left"].clear()
    assert provenance["scope"]["options"]["hold_interval_ids"]["left"] == ["hold"]
    assert defaults == {"hold_ramp_frames": 4, "window_speed_tolerance_mps": .35}


@pytest.mark.parametrize("change,match", [
    ({"contact_map_sha256": "b" * 64}, "consumed contact map"),
    ({"schema": "unsupported"}, "schema"),
    ({"reviewed_by": " "}, "reviewed_by"),
    ({"invent_holds": True}, "schema"),
    ({"options": {"fps": 24.}}, "exactly fps"),
])
def test_reject_stale_unreviewed_or_partial_scope(scope, change, match):
    path, spec = scope
    bad = copy.deepcopy(spec)
    bad.update(change)
    path.write_text(json.dumps(bad))
    with pytest.raises(ValueError, match=match):
        load_temporal_scope(str(path), contact_map_sha256="a" * 64, defaults={})


@pytest.mark.parametrize("field", ["fps", "hold_interval_ids", "windows"])
def test_no_ambiguous_inline_and_sidecar_scope(scope, field):
    path, spec = scope
    with pytest.raises(ValueError, match="only from the scope"):
        load_temporal_scope(str(path), contact_map_sha256="a" * 64,
                            defaults={field: spec["options"][field]})


def test_missing_scope_explains_stage2_prerequisite(tmp_path):
    with pytest.raises(FileNotFoundError, match="Before Stage 2"):
        load_temporal_scope(str(tmp_path / "missing.json"),
                            contact_map_sha256="a" * 64, defaults={})


def test_unresolved_path_is_not_read():
    with pytest.raises(ValueError, match="resolved scope_path"):
        load_temporal_scope("{cache_dir}/scope.json", contact_map_sha256="a" * 64,
                            defaults={})
