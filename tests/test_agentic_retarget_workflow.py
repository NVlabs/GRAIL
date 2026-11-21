"""Thin delegation must preserve real routes, failures and unreviewed status."""

import hashlib
import json
import sys
from pathlib import Path

import pytest
import yaml

from grail.retargeting import agentic

RETARGET_SCRIPT = """import argparse, hashlib, json
from pathlib import Path
p=argparse.ArgumentParser()
p.add_argument("action")
p.add_argument("--source-model")
p.add_argument("--config")
p.add_argument("--output")
p.add_argument("--timeout")
a=p.parse_args()
c=json.loads(Path(a.config).read_text())
if a.action == "preflight":
    print(json.dumps({"ok": not c.get("blocked", False)}))
    raise SystemExit(2 if c.get("blocked") else 0)
if a.action == "plan":
    print(json.dumps({"source_model": a.source_model, "status":"PLANNED_UNREVIEWED", "backend_plan": c}))
else:
    out=Path(a.output); out.mkdir()
    native=out / "raw/robot/clip.pkl.native.npz"
    native.parent.mkdir(parents=True); native.write_bytes(b"native")
    result={"source_model":a.source_model, "status":"CONVERTED_UNREVIEWED", "motion_status":"UNREVIEWED", "selected":False}
    (out / "retarget_result.json").write_text(json.dumps(result))
    (out / "run.json").write_text(json.dumps({"outputs": {"raw/robot/clip.pkl.native.npz": {"path": str(native), "sha256":hashlib.sha256(native.read_bytes()).hexdigest()}}}))
    print(json.dumps(result))
"""


@pytest.fixture
def configured(tmp_path):
    checkout = tmp_path / "external"
    (checkout / "src").mkdir(parents=True)
    (checkout / "src/backend.py").write_text("# immutable backend\n")
    for name in ("tools/retarget/run.py", "tools/retarget/native.py", "tools/batch/run.py"):
        path = checkout / name
        path.parent.mkdir(parents=True, exist_ok=True)
        if name.endswith("retarget/run.py"):
            path.write_text(RETARGET_SCRIPT)
        elif name.endswith("retarget/native.py"):
            path.write_text("import sys; from pathlib import Path\np = Path(sys.argv[sys.argv.index('--output')+1]); p.write_bytes(b'test native'); Path(str(p)+'.json').write_text('{}')\n")
        else:
            path.write_text("")
    path = tmp_path / "case.yaml"
    config = {
        "schema_version": 1,
        "source_model": "soma03",
        "runtime": {"checkout": str(checkout), "python": sys.executable},
        "backend": {
            "source": "native_source.bvh",
        },
    }
    path.write_text(yaml.safe_dump(config))
    return path, config, checkout


def write_config(path, cfg):
    path.write_text(yaml.safe_dump(cfg))


def test_public_vendored_revision_is_bound_and_checked(configured):
    path, cfg, checkout = configured
    revision = "41264543b9ba220f952f4e2088d419000e8e5b0f"
    (checkout / "VENDORED_REVISION").write_text(revision + "\n")
    cfg["runtime"]["revision"] = revision
    write_config(path, cfg)
    assert agentic.load_config(path)["runtime"]["revision"] == revision
    cfg["runtime"]["revision"] = "0" * 40
    write_config(path, cfg)
    with pytest.raises(ValueError, match="revision differs"):
        agentic.load_config(path)


def test_explicit_model_route(configured, tmp_path):
    path, _, _ = configured
    result = agentic.execute(path, tmp_path / "plan", action="plan")
    assert result["backend_result"]["source_model"] == "soma"
    assert result["backend_result"]["backend_plan"]["human_scale_ratio"] == 1.0
    assert result["selected"] is False
    assert result["motion_status"] == "UNREVIEWED"
    assert result["status"] == "PLANNED_UNREVIEWED"
    assert not (tmp_path / "plan/retarget").exists()


@pytest.mark.parametrize("model", ["soma", "mhr", "auto", "smplx", "g1_smplx"])
def test_never_guesses_unknown_model(configured, model):
    path, cfg, _ = configured
    cfg["source_model"] = model
    write_config(path, cfg)
    with pytest.raises(ValueError, match="source_model"):
        agentic.load_config(path)


def test_soma_custom_robot_paths_reach_backend_relative_to_yaml(configured, tmp_path):
    path, cfg, _ = configured
    cfg["source_model"] = "soma03"
    cfg["backend"] = {
        "source": "source.bvh",
        "robot": "unitree_g1_dex3",
        "extra_robot_paths": ["targets/dex3"],
    }
    write_config(path, cfg)
    result = agentic.execute(path, tmp_path / "custom_plan", action="plan")
    backend = result["backend_result"]["backend_plan"]
    assert result["backend"] == "soma"
    assert backend["robot"] == "unitree_g1_dex3"
    assert backend["extra_robot_paths"] == [str(path.parent / "targets/dex3")]


def test_soma_contact_constraints_resolve_before_backend_config_relocation(configured, tmp_path):
    path, cfg, _ = configured
    cfg["source_model"] = "soma03"
    cfg["backend"] = {
        "source": "source.bvh",
        "contact_constraints": "inputs/contacts.json",
    }
    write_config(path, cfg)
    output = tmp_path / "separate_receipt"
    result = agentic.execute(path, output, action="plan")
    expected = str(path.parent / "inputs/contacts.json")
    assert result["backend_result"]["backend_plan"]["contact_constraints"] == expected
    assert json.loads((output / "backend.json").read_text())["contact_constraints"] == expected


@pytest.mark.parametrize("value", [None, "", 42, []])
def test_soma_contact_constraints_require_a_path(configured, value):
    path, cfg, _ = configured
    cfg["source_model"] = "soma03"
    cfg["backend"]["contact_constraints"] = value
    write_config(path, cfg)
    with pytest.raises(ValueError, match="nonempty path string"):
        agentic.load_config(path)


@pytest.mark.parametrize(
    "robot", ["unitree_g1", "unitree_h2", "booster_t1", "agibot_x2ultra", "agibot-a3t3"]
)
@pytest.mark.parametrize("ratio", [None, 0.91])
def test_soma_human_scale_reaches_backend_and_request(configured, tmp_path, robot, ratio):
    path, cfg, _ = configured
    cfg["source_model"] = "soma03"
    cfg["backend"] = {"source": "source.bvh", "robot": robot}
    if ratio is not None:
        cfg["backend"]["human_scale_ratio"] = ratio
    write_config(path, cfg)
    output = tmp_path / "scale_plan"
    result = agentic.execute(path, output, action="plan")
    expected = 1.0 if ratio is None else ratio
    assert result["status"] == "PLANNED_UNREVIEWED"
    assert result["backend_result"]["backend_plan"]["human_scale_ratio"] == expected
    assert json.loads((output / "backend.json").read_text())["human_scale_ratio"] == expected
    request = json.loads((output / "request.json").read_text())
    assert request["backend"]["human_scale_ratio"] == expected


@pytest.mark.parametrize("ratio", [None, True, "1.0", 0, -1, float("inf"), float("nan")])
def test_invalid_soma_human_scale_fails_before_launch(configured, tmp_path, ratio):
    path, cfg, _ = configured
    cfg["source_model"] = "soma03"
    cfg["backend"]["human_scale_ratio"] = ratio
    write_config(path, cfg)
    output = tmp_path / "invalid_scale"
    with pytest.raises(ValueError, match="backend.human_scale_ratio must be positive and finite"):
        agentic.execute(path, output, action="plan")
    assert not output.exists()


@pytest.mark.parametrize("paths", ["targets/dex3", [], [None], [""]])
def test_invalid_custom_robot_paths_fail_before_launch(configured, paths):
    path, cfg, _ = configured
    cfg["source_model"] = "soma03"
    cfg["backend"]["extra_robot_paths"] = paths
    write_config(path, cfg)
    with pytest.raises(ValueError):
        agentic.load_config(path)


def test_run_delegates_preflight_then_conversion_and_binds_native(configured, tmp_path):
    path, _, _ = configured
    result = agentic.execute(path, tmp_path / "actual")
    assert [s["name"] for s in result["stages"]] == ["preflight", "run", "native"]
    assert result["status"] == "CONVERTED_UNREVIEWED"
    assert Path(result["outputs"]["native"]["path"]).read_bytes() == b"test native"
    assert (
        result["outputs"]["run"]["sha256"]
        == hashlib.sha256(Path(result["outputs"]["run"]["path"]).read_bytes()).hexdigest()
    )
    assert all(s["seconds"] >= 0 for s in result["stages"])
    assert result["selected"] is False


def test_failed_preflight_does_not_convert(configured, tmp_path):
    path, cfg, _ = configured
    cfg["backend"]["blocked"] = True
    write_config(path, cfg)
    result = agentic.execute(path, tmp_path / "failed")
    assert result["status"] == "FAILED_UNREVIEWED"
    assert len(result["stages"]) == 1
    assert not (tmp_path / "failed/retarget").exists()
    assert (tmp_path / "failed/bridge_result.json").is_file()


def test_existing_output_is_not_overwritten(configured, tmp_path):
    path, _, _ = configured
    out = tmp_path / "existing"
    out.mkdir()
    with pytest.raises(FileExistsError):
        agentic.execute(path, out)


def test_pending_native_soma_export_does_not_claim_preflight_passed(configured, tmp_path):
    path, cfg, _ = configured
    source = tmp_path / "human.npz"
    source.write_bytes(b"source")
    cfg.update(
        source_model="soma03",
        source={
            "path": str(source),
            "python": sys.executable,
            "model_config": {"model_type": "soma03"},
        },
        backend={"soma_root": str(tmp_path / "soma")},
    )
    write_config(path, cfg)
    result = agentic.execute(path, tmp_path / "pending", action="preflight")
    assert result["status"] == "SOURCE_EXPORT_REQUIRED_UNREVIEWED"
    assert result["ready"] is False
    assert result["stages"] == []
    assert "grail.retargeting.soma_export" in result["source_export"]["argv"]
    assert (
        agentic.main(
            ["preflight", "--config", str(path), "--output", str(tmp_path / "pending_cli")]
        )
        == 2
    )


def test_checkout_mutation_is_recorded_as_failure(configured, tmp_path, monkeypatch):
    path, _, root = configured
    original = agentic._invoke

    def mutate(*args, **kwargs):
        result = original(*args, **kwargs)
        (root / "src/backend.py").write_text("# changed\n")
        return result

    monkeypatch.setattr(agentic, "_invoke", mutate)
    result = agentic.execute(path, tmp_path / "mutation", action="plan")
    assert result["status"] == "FAILED_UNREVIEWED"
    assert "changed" in result["error"]


def test_workflow_retains_failed_gate_report(configured, tmp_path):
    path, cfg, checkout = configured
    (checkout / "tools/batch/run.py").write_text(
        """import argparse,json
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument("--plan");p.add_argument("--output");p.add_argument("--workers")
a=p.parse_args();out=Path(a.output);out.mkdir()
(out/"execution.json").write_text(json.dumps({"cases":[{"name":"demo","status":"GATE_FAILED"}],"motion_accepted":False}))
raise SystemExit(1)
"""
    )
    plan = tmp_path / "workflow.yaml"
    plan.write_text(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "cases": [
                    {
                        "name": "demo",
                        "stages": [{"name": "gate", "argv": [sys.executable, "-c", "pass"]}],
                    }
                ],
            }
        )
    )
    cfg["workflow"] = {"plan": str(plan), "workers": 1}
    write_config(path, cfg)
    result = agentic.execute(path, tmp_path / "workflow_run", action="workflow")
    assert result["status"] == "INCOMPLETE_UNREVIEWED"
    assert result["workflow_result"]["cases"][0]["status"] == "GATE_FAILED"
    assert result["selected"] is False
    assert result["stages"][0]["returncode"] == 1


def test_native_model_paths_are_relative_to_yaml(configured):
    path, cfg, _ = configured
    source = path.parent / "human.npz"
    source.write_bytes(b"source")
    cfg.update(
        source_model="soma03",
        source={
            "path": str(source),
            "python": sys.executable,
            "model_config": {
                "soma_runtime": "gem_x",
                "gem_x_soma_assets": "assets/soma",
                "soma_identity_params_path": "identity.npz",
            },
        },
        backend={"soma_root": "official"},
    )
    write_config(path, cfg)
    model = agentic.load_config(path)["source"]["model_config"]
    assert model["soma_runtime"] == "gem_x"
    assert model["gem_x_soma_assets"] == str(path.parent / "assets/soma")
    assert model["soma_identity_params_path"] == str(path.parent / "identity.npz")


def test_workflow_resume_keeps_existing_budget_and_passes_flag(configured, tmp_path):
    path, cfg, checkout = configured
    (checkout / "src/case_budget.py").write_text(
        "CASE_LIMIT_SECONDS = 20 * 60\nDELIVERY_RESERVE_SECONDS = 4 * 60\n"
    )
    (checkout / "tools/batch/run.py").write_text(
        """import argparse,json
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument("--plan");p.add_argument("--output");p.add_argument("--workers");p.add_argument("--resume",action="store_true")
a=p.parse_args();assert a.resume
out=Path(a.output);assert (out/"demo/case_budget.json").is_file()
(out/"execution.json").write_text(json.dumps({"cases":[{"name":"demo","status":"EXECUTION_COMPLETE_UNREVIEWED"}],"motion_accepted":False}))
"""
    )
    plan = tmp_path / "flow.yaml"
    plan.write_text(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "cases": [
                    {
                        "name": "demo",
                        "stages": [{"name": "gate", "argv": [sys.executable, "-c", "pass"]}],
                    }
                ],
            }
        )
    )
    previous = tmp_path / "saved_batch"
    ledger = previous / "demo/case_budget.json"
    ledger.parent.mkdir(parents=True)
    ledger.write_text('{"consumed_seconds": 100}')
    cfg["workflow"] = {"plan": str(plan), "workers": 1, "resume": True, "output": str(previous)}
    write_config(path, cfg)
    result = agentic.execute(path, tmp_path / "new_resume_receipt", action="workflow")
    assert result["status"] == "EXECUTION_COMPLETE_UNREVIEWED"
    assert result["workflow_budget_policy"]["case_seconds"] == 1200
    assert result["workflow_budget_policy"]["delivery_reserve_seconds"] == 240
    assert str(ledger) in result["resumed_budget_bindings"]
    assert json.loads(ledger.read_text())["consumed_seconds"] == 100
    assert result["selected"] is False


@pytest.mark.parametrize("with_camera", [False, True])
def test_soma_run_exports_native_source_and_object_sidecar(
    configured, tmp_path, monkeypatch, with_camera
):
    path, cfg, _ = configured
    source = tmp_path / "human.npz"
    source.write_bytes(b"source")
    cfg.update(
        source_model="soma03",
        source={
            "path": str(source),
            "python": sys.executable,
            "model_config": {"model_type": "soma03"},
        },
        backend={"soma_root": str(tmp_path / "official"), "python": sys.executable},
        native={"object_mesh": str(tmp_path / "mesh.obj"), "object_scale": 0.5},
    )
    if with_camera:
        camera = tmp_path / "camera.json"
        camera.write_text(
            json.dumps(
                {
                    "convention": "world_to_camera",
                    "R": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                    "t": [0, 0, 1],
                    "focal_length": 500,
                }
            )
        )
        original_camera = camera.read_bytes()
        cfg["backend"]["bindings"] = {"camera": str(camera)}
    write_config(path, cfg)
    original = agentic._invoke

    def fake_export(argv, output, name, **kwargs):
        if name == "source_export":
            folder = Path(argv[argv.index("--output") + 1])
            folder.mkdir()
            (folder / "source.bvh").write_text("test bvh")
            (folder / "manifest.json").write_text(
                json.dumps(
                    {
                        "source_units": "cm",
                        "source_facing_direction": "Mujoco",
                        "fps": 24,
                        "world_offset": [0, 0, -0.1],
                        "grounding": {"enabled": True},
                    }
                )
            )
            assert "--upstream-python" in argv
            assert "--ground-initial-frame" in argv
        elif name == "object_sidecar":
            request = json.loads(Path(argv[-1]).read_text())
            Path(request["output_path"]).write_bytes(b"bound-object")
            assert request["object_scale"] == 0.5
        elif name == "native":
            destination = Path(argv[argv.index("--output") + 1])
            destination.write_bytes(b"native")
            Path(str(destination) + ".json").write_text(
                json.dumps({"status": "UNREVIEWED", "missing_scene_inputs": ["camera"]})
            )
            assert "--object-sidecar" in argv
        else:
            return original(argv, output, name, **kwargs)
        return {"name": name, "status": "EXECUTED", "seconds": 0, "returncode": 0}

    monkeypatch.setattr(agentic, "_invoke", fake_export)
    result = agentic.execute(path, tmp_path / "soma_run")
    assert result["status"] == "CONVERTED_UNREVIEWED"
    assert [stage["name"] for stage in result["stages"]] == [
        "source_export",
        "object_sidecar",
        "preflight",
        "run",
        "native",
    ]
    assert result["native_result"]["missing_scene_inputs"] == ["camera"]
    backend = json.loads((tmp_path / "soma_run/backend.json").read_text())
    assert backend["source_units"] == "cm"
    assert backend["human_scale_ratio"] == 1.0
    assert Path(backend["bindings"]["object"]).is_file()
    if with_camera:
        shifted = json.loads(Path(backend["bindings"]["camera"]).read_text())
        assert shifted["t"] == [0, 0, 1.1]
        assert shifted["focal_length"] == 500
        assert shifted["grounding_parent_camera"]["path"] == str(camera)
        assert camera.read_bytes() == original_camera


@pytest.mark.parametrize("enabled", [True, False])
def test_raw_soma_grounding_is_explicit_in_export_command(configured, tmp_path, enabled):
    path, cfg, _ = configured
    source = tmp_path / "source.npz"
    source.write_bytes(b"source")
    cfg.update(
        source_model="soma03",
        source=dict(
            path=str(source),
            python=sys.executable,
            model_config={"model_type": "soma03"},
            ground_initial_frame=enabled,
        ),
        backend={"soma_root": str(tmp_path / "official")},
    )
    write_config(path, cfg)
    result = agentic.execute(path, tmp_path / "plan_grounding", action="plan")
    assert ("--ground-initial-frame" in result["source_export"]["argv"]) == enabled


@pytest.mark.parametrize(
    "field",
    [
        "contact_constraints",
        "robot_hand_targets",
        "pose_keyframes",
        "wrist_keyframes",
        "prior_global_seed",
    ],
)
def test_grounding_rejects_precomputed_world_targets(configured, tmp_path, field):
    path, cfg, _ = configured
    source = tmp_path / "source.npz"
    source.write_bytes(b"source")
    cfg.update(
        source_model="soma03",
        source=dict(path=str(source), python=sys.executable, model_config={"model_type": "soma03"}),
        backend={"soma_root": str(tmp_path / "official"), field: "old_targets.json"},
    )
    write_config(path, cfg)
    with pytest.raises(ValueError, match="Ground the raw source before preparing"):
        agentic.load_config(path)


def test_timeout_reaps_solver_child_processes(tmp_path):
    import time

    child = tmp_path / "child.py"
    child.write_text(
        "import pathlib,time,sys\np=pathlib.Path(sys.argv[1])\nwhile True:\n p.write_text(str(time.monotonic()))\n time.sleep(.02)\n"
    )
    heartbeat = tmp_path / "heartbeat"
    parent = tmp_path / "router.py"
    parent.write_text(
        "import subprocess,time,sys\nsubprocess.Popen([sys.executable,sys.argv[1],sys.argv[2]])\ntime.sleep(30)\n"
    )
    row = agentic._invoke(
        [sys.executable, str(parent), str(child), str(heartbeat)],
        tmp_path,
        "timeout",
        cwd=tmp_path,
        timeout=0.3,
    )
    assert row["status"] == "TIMED_OUT"
    value = heartbeat.read_text()
    time.sleep(0.1)
    assert heartbeat.read_text() == value
