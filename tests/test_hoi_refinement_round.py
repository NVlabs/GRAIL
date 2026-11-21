"""Round orchestration preserves native data and records failures without workers."""

import copy
import json
import sys
from pathlib import Path

import numpy as np
import pytest

from grail.agentic_kinematic.hoi_refinement import blender_bridge, review_worker
from grail.agentic_kinematic.hoi_refinement import round as round_module
from grail.agentic_kinematic.hoi_refinement.contracts import (
    binding,
    load_npz,
    read_case,
    read_json,
    save_npz,
    validate_scene,
)
from grail.agentic_kinematic.hoi_refinement.interpolation import (
    apply_edits,
    sample_motion,
    validate_plan,
)
from grail.agentic_kinematic.hoi_refinement.native import DERIVED_FIELDS, load_native


def _motion(model_type="soma03"):
    count = 5
    value = {
        "model_type": np.asarray(model_type),
        "poses": np.zeros((count, 77, 3), dtype=np.float32),
        "trans": np.zeros((count, 3), dtype=np.float32),
        "fps": np.asarray(30.0),
        "obj_R": np.repeat(np.eye(3)[None], count, axis=0),
        "obj_t": np.zeros((count, 3)),
        "obj_scale": np.asarray(0.4),
    }
    value.update(identity_coeffs=np.zeros((1, 128)), scale_params=np.ones((1, 57)))
    return value


def _plan():
    return {
        "schema_version": 1,
        "windows": [
            {
                "joint": 12,
                "start": 0,
                "end": 4,
                "max_angle_rad": 0.5,
                "keys": [
                    {"frame": 0, "delta_rotvec": [0, 0, 0]},
                    {"frame": 2.5, "delta_rotvec": [0.2, 0.0, 0.0]},
                    {"frame": 4, "delta_rotvec": [0, 0, 0]},
                ],
            }
        ],
    }


def _scene():
    return {
        "object_vertices": np.array([[2.0, 0.0, 2.0], [3.0, 0.0, 2.0], [2.0, 1.0, 2.0]]),
        "object_faces": np.array([[0, 1, 2]]),
        "object_scale": np.asarray(0.4),
        "camera_matrix_world": np.eye(4),
        "camera_intrinsics": np.array([[100.0, 0.0, 50.0], [0.0, 100.0, 50.0], [0.0, 0.0, 1.0]]),
        "camera_convention": np.asarray("opencv_camera_to_world"),
        "image_width": np.asarray(100),
        "image_height": np.asarray(100),
    }


@pytest.mark.parametrize("model_type", ["soma03"])
def test_saved_parent_allows_body_edits_and_ignores_stale_derived_caches(model_type):
    base = _motion(model_type)
    base["vertices"] = np.ones((5, 3, 3))
    parent = apply_edits(base, _plan())
    parent.pop("vertices")
    parent["contact_cache"] = np.zeros(5)
    round_module._check_parent(base, parent)


@pytest.mark.parametrize("field", ["trans", "obj_t", "obj_scale", "fps", "identity_coeffs", "scale_params"])
def test_saved_parent_rejects_frozen_native_field_changes(field):
    base, parent = _motion(), _motion()
    parent[field].flat[0] += 0.01
    with pytest.raises(ValueError, match="frozen field"):
        round_module._check_parent(base, parent)


@pytest.mark.parametrize("joint", [0, 22, 25, 54])
def test_saved_parent_rejects_root_face_and_finger_rotation_changes(joint):
    base, parent = _motion(), _motion()
    parent["poses"][2, joint, 1] += 0.1
    with pytest.raises(ValueError, match="frozen root, finger or face"):
        round_module._check_parent(base, parent)


def test_saved_parent_compares_dtype_shape_signed_zero_and_field_set_exactly():
    base = _motion()
    parent = copy.deepcopy(base)
    parent["poses"] = parent["poses"].astype(np.float64)
    with pytest.raises(ValueError, match="shape/dtype"):
        round_module._check_parent(base, parent)
    parent = copy.deepcopy(base)
    parent["poses"] = parent["poses"].reshape(5, 231)
    with pytest.raises(ValueError, match="shape/dtype"):
        round_module._check_parent(base, parent)
    parent = copy.deepcopy(base)
    parent["trans"][0, 0] = -0.0
    with pytest.raises(ValueError, match="frozen field"):
        round_module._check_parent(base, parent)
    parent = copy.deepcopy(base)
    parent["unbound_shape_parameter"] = np.asarray(1.0)
    with pytest.raises(ValueError, match="different canonical fields"):
        round_module._check_parent(base, parent)


def test_without_caches_keeps_native_metadata_without_mutating_source():
    source = _motion()
    source["native_note"] = np.asarray("keep")
    for key in DERIVED_FIELDS:
        source[key] = np.arange(3)
    filtered = round_module._without_caches(source)
    assert not set(DERIVED_FIELDS) & filtered.keys()
    assert filtered["native_note"].item() == "keep"
    assert set(DERIVED_FIELDS) <= source.keys()
    np.testing.assert_array_equal(filtered["poses"], source["poses"])


def test_edit_operations_replace_and_delete_explicit_keys_without_mutating_plan():
    plan = _plan()
    original = copy.deepcopy(plan)
    result = round_module.edit_plan(
        plan,
        [
            {"op": "set_key", "window": 0, "frame": 2.5, "delta_rotvec": [0.0, 0.1, 0.0]},
            {"op": "set_key", "window": 0, "frame": 1.0, "delta_rotvec": [0.0, 0.2, 0.0]},
            {"op": "delete_key", "window": 0, "frame": 2.5},
        ],
    )
    assert plan == original
    assert [key["frame"] for key in result["windows"][0]["keys"]] == [0, 1.0, 4]
    assert result["windows"][0]["keys"][1]["delta_rotvec"] == [0.0, 0.2, 0.0]
    validate_plan(_motion(), result)


@pytest.mark.parametrize(
    "operation",
    [
        {"op": "set_key", "window": 0, "frame": 0, "delta_rotvec": [0, 0, 0]},
        {"op": "delete_key", "window": 0, "frame": 4},
        {"op": "delete_key", "window": 0, "frame": 1},
        {"op": "delete_key", "window": True, "frame": 2.5},
        {"op": "delete_key", "window": 1, "frame": 2.5},
        {"op": "set_key", "window": 0, "frame": float("nan"), "delta_rotvec": [0, 0, 0]},
        {"op": "set_key", "window": 0, "frame": 2, "delta_rotvec": [0, 0, 0], "scale": 2},
        {"op": "move_mesh", "window": 0, "frame": 2},
    ],
)
def test_edit_operation_guards_protect_scope_and_endpoints(operation):
    plan = _plan()
    original = copy.deepcopy(plan)
    with pytest.raises(ValueError):
        round_module.edit_plan(plan, [operation])
    assert plan == original


def test_operation_delta_is_validated_against_declared_motion_bound():
    plan = round_module.edit_plan(
        _plan(),
        [
            {"op": "set_key", "window": 0, "frame": 2.5, "delta_rotvec": [0.0, 1.0, 0.0]},
        ],
    )
    with pytest.raises(ValueError, match="max_angle_rad"):
        validate_plan(_motion(), plan)


@pytest.fixture
def case_files(tmp_path):
    native = _motion()
    native["vertices"] = np.zeros((5, 3, 3))
    native["eval_data"] = np.asarray([99.0])
    save_npz(tmp_path / "source.npz", native)
    save_npz(tmp_path / "scene.npz", _scene())
    (tmp_path / "plan.json").write_text(json.dumps(_plan()))
    (tmp_path / "source.mp4").write_bytes(b"synthetic source binding, not decoded in mocked worker")
    (tmp_path / "model.npz").write_bytes(b"model fixture asset")
    case = {
        "schema_version": 1,
        "name": "round-test",
        "input_native": "source.npz",
        "model": {"model_type": "soma03", "gem_x_soma_assets": "model.npz"},
        "runtime": {"native_python": sys.executable, "blender": sys.executable, "timeout_s": 20},
        "scene": "scene.npz",
        "plan": "plan.json",
        "source_video": "source.mp4",
        "observation_frames": [0, 2],
        "review_frames": [0, 2, 4],
        "render": False,
    }
    path = tmp_path / "case.json"
    path.write_text(json.dumps(case))
    return path, case


def test_case_resolves_paths_against_config_and_keeps_subframes(case_files):
    path, case = case_files
    case["observation_frames"] = [0, 2.5]
    case["review_frames"] = [0, 2.5, 4]
    path.write_text(json.dumps(case))
    loaded = read_case(path)
    for key in ("input_native", "scene", "plan", "source_video"):
        assert Path(loaded[key]).parent == path.parent
    assert loaded["model"]["gem_x_soma_assets"] == str(path.parent / "model.npz")
    assert loaded["review_frames"] == [0, 2.5, 4]
    assert loaded["model"]["backend"] == "native"


@pytest.mark.parametrize("runtime_kind", ["native", "newton"])
def test_case_preserves_virtualenv_python_symlink_invocation_path(case_files, runtime_kind):
    path, case = case_files
    environment = path.parent / "virtual environment"
    (environment / "bin").mkdir(parents=True)
    executable = environment / "bin" / "python"
    executable.symlink_to(sys.executable)
    (environment / "pyvenv.cfg").write_text("include-system-site-packages = false\n")
    relative = str(executable.relative_to(path.parent))
    if runtime_kind == "native":
        case["runtime"]["native_python"] = relative
    else:
        repo = path.parent / "newton checkout"
        repo.mkdir()
        case["newton"] = {"python": relative, "repo": str(repo), "device": "cpu"}
    path.write_text(json.dumps(case))
    loaded = read_case(path)
    invoked = (
        loaded["runtime"]["native_python"]
        if runtime_kind == "native"
        else loaded["newton"]["python"]
    )
    # Python locates pyvenv.cfg from the invocation path. Resolving the link
    # would silently select the base interpreter and lose model dependencies.
    assert invoked == str(executable.absolute())
    assert Path(invoked).is_symlink()
    assert Path(invoked).resolve() == Path(sys.executable).resolve()


@pytest.mark.parametrize(
    "mutation",
    [
        lambda value: value.update(schema_version=True),
        lambda value: value.update(unknown_option=True),
        lambda value: value.update(observation_frames=[0, 0]),
        lambda value: value.update(review_frames=[4, 2, 0]),
        lambda value: value.update(review_frames=[0, 4]),
        lambda value: value.update(observation_frames=[True]),
        lambda value: value["runtime"].update(timeout_s=0),
        lambda value: value["runtime"].update(unbound_worker="ignored"),
        lambda value: value["model"].update(model_type="soma"),
    ],
)
def test_case_rejects_ambiguous_timeline_or_runtime_contract(case_files, mutation):
    path, case = case_files
    mutation(case)
    path.write_text(json.dumps(case))
    with pytest.raises(ValueError):
        read_case(path)


@pytest.mark.parametrize(
    "mutation",
    [
        lambda value: value.update(camera_convention=np.asarray("blender")),
        lambda value: value["camera_matrix_world"].__setitem__((0, 0), 2),
        lambda value: value["camera_matrix_world"].__setitem__((3, 0), 1),
        lambda value: value["camera_intrinsics"].__setitem__((0, 1), 0.01),
        lambda value: value["object_faces"].__setitem__((0, 0), 99),
        lambda value: value["object_vertices"].__setitem__((0, 0), np.nan),
        lambda value: value.update(image_width=np.asarray(100.0)),
    ],
)
def test_scene_rejects_changed_coordinate_convention_or_invalid_original_geometry(mutation):
    scene = _scene()
    assert validate_scene(scene) is scene
    mutation(scene)
    with pytest.raises(ValueError):
        validate_scene(scene)


@pytest.fixture
def mock_workers(case_files, monkeypatch):
    path, _ = case_files
    calls = {"geometry": [], "blender": [], "fault": None}
    monkeypatch.setattr(
        round_module, "_model_bindings", lambda model: [binding(model["gem_x_soma_assets"])]
    )

    def execute(command, log, timeout):
        Path(log).write_text("mocked worker; no runtime process launched\n")
        if "grail.agentic_kinematic.hoi_refinement.review_worker" in command:
            review_worker.main(["--request", command[command.index("--request") + 1]])
            return
        assert "grail.agentic_kinematic.hoi_refinement.native_worker" in command
        source = Path(command[command.index("--motion") + 1])
        motion = load_npz(source)
        model_path = Path(command[command.index("--model-config") + 1])
        configured = read_json(model_path.parent / "model.json")
        assert read_json(model_path) == {**configured, "device": "cpu"}
        assert not set(DERIVED_FIELDS) & motion.keys()
        frames = read_json(command[command.index("--frames") + 1])
        sampled = sample_motion(motion, round_module.EMPTY_PLAN, frames)
        # A deterministic geometry stub isolates orchestration from FK. The
        # native export/reload and actual review worker still execute normally.
        pose = sampled["poses"].reshape(len(frames), -1, 3)[:, 12]
        vertices = np.array([[0.0, 0.0, 2.0], [1.0, 0.0, 2.0], [0.0, 1.0, 2.0]])[None]
        vertices = vertices + pose[:, None]
        faces = np.array([[0, 1, 2]])
        if source.name == "reopened.npz":
            if calls["fault"] == "reopened_vertices":
                vertices += 1e-3
            elif calls["fault"] == "reopened_faces":
                faces = faces[:, ::-1]
        if calls["fault"] == "wrong_frames":
            frames = [0, 1, 2, 3]
        result = dict(
            vertices=vertices,
            joints=vertices[:, :2].copy(),
            faces=faces,
            frames=np.asarray(frames, dtype=float),
            obj_R=sampled["obj_R"],
            obj_t=sampled["obj_t"],
            obj_scale=sampled["obj_scale"],
        )
        save_npz(command[command.index("--output") + 1], result)
        calls["geometry"].append((source.name, list(frames), motion))

    def blender(request, output, executable, timeout):
        calls["blender"].append(request)
        if request["mode"] == "inspect":
            return {
                "bindings": request["bindings"],
                "plan": calls["saved_plan"],
                "parent_motion": binding(calls["saved_parent"]),
            }
        output = Path(output)
        output.mkdir()
        checkpoint = output / "checkpoint.blend"
        checkpoint.write_bytes(b"synthetic Blender checkpoint")
        if calls["fault"] == "source_changed":
            (path.parent / "source.mp4").write_bytes(b"changed during worker")
        return {"checkpoint": str(checkpoint), "status": "UNREVIEWED"}

    monkeypatch.setattr(round_module, "_execute", execute)
    monkeypatch.setattr(blender_bridge, "run_blender", blender)
    return calls


@pytest.mark.parametrize("source_format", ["npz", "pickle"])
def test_round_exports_reloads_and_reviews_without_acceptance(
    case_files, mock_workers, source_format
):
    path, case = case_files
    case["model"]["device"] = "cuda"
    path.write_text(json.dumps(case))
    if source_format == "pickle":
        joblib = pytest.importorskip("joblib")
        source = _motion()
        human = {
            key: value
            for key, value in source.items()
            if key not in ("obj_R", "obj_t", "obj_scale", "fps", "human_scale")
        }
        human.update(
            mocap_frame_rate=source["fps"],
            scale=1.0,
            vertices=np.ones((5, 3, 3)),
            preserved_native_note={"keep": [1, 2]},
        )
        payload = {
            "human_data": human,
            "obj_data": {key: source[key] for key in ("obj_R", "obj_t", "obj_scale")},
            "eval_data": {"stale": True},
            "unknown_metadata": {"keep": "yes"},
        }
        joblib.dump(payload, path.parent / "source.pickle")
        case["input_native"] = "source.pickle"
        path.write_text(json.dumps(case))
    source_path = path.parent / case["input_native"]
    original_bytes = source_path.read_bytes()
    output = path.parent / "round"
    result = round_module.run(path, output)
    assert result["status"] == "EXPORTED"
    assert result["motion_status"] == "UNREVIEWED"
    assert result["selected"] is False and result["api_calls"] == 0
    assert source_path.read_bytes() == original_bytes
    assert [name for name, _, _ in mock_workers["geometry"]] == [
        "parent.npz",
        "candidate.npz",
        "reopened.npz",
    ]
    assert all(frames == [0, 2, 2.5, 4] for _, frames, _ in mock_workers["geometry"])
    receipt = read_json(result["iteration"])
    assert receipt["roundtrip"]["canonical_arrays_equal"]
    assert receipt["roundtrip"]["geometry_device"] == "cpu"
    assert receipt["roundtrip"]["max_vertex_difference_m"] == 0
    assert receipt["roundtrip"]["max_joint_difference_m"] == 0
    assert read_json(output / "review.json")["status"] == "UNREVIEWED"
    assert receipt["native_export"]["removed_derived_fields"]
    candidate = load_native(result["native"], "soma03")
    expected = apply_edits(load_native(source_path, "soma03"), _plan())
    assert not set(DERIVED_FIELDS) & candidate.keys()
    np.testing.assert_array_equal(candidate["poses"], expected["poses"])
    for key in ("trans", "obj_R", "obj_t", "scale_params", "identity_coeffs"):
        assert candidate[key].tobytes() == expected[key].tobytes()
    assert mock_workers["blender"][-1]["review"]["status"] == "UNREVIEWED"
    if source_format == "pickle":
        assert Path(result["native"]).suffix == ".pkl"
        exported = joblib.load(result["native"])
        assert exported["unknown_metadata"] == {"keep": "yes"}
        assert exported["human_data"]["preserved_native_note"] == {"keep": [1, 2]}
        assert "eval_data" not in exported and "vertices" not in exported["human_data"]
    with pytest.raises(FileExistsError):
        round_module.run(path, output)


def test_local_repair_preserves_original_and_seed_with_complete_replay_review(
    case_files, mock_workers, monkeypatch
):
    path, case = case_files
    contact_map = path.parent / "map.json"
    contact_map.write_text(
        "{}"
    )  # Isolates orchestration; solver map validation is tested separately.
    case["repair"] = {"contact_map": "map.json"}
    case["iteration"] = {"wrist_joint_ids": [1]}
    path.write_text(json.dumps(case))
    execute = round_module._execute

    def worker(command, log, timeout):
        if "grail.agentic_kinematic.hoi_refinement.repair_worker" not in command:
            return execute(command, log, timeout)
        request = read_json(command[command.index("--request") + 1])
        out = Path(request["output_dir"])
        out.mkdir()
        plan = read_json(request["seed_plan"])
        plan["windows"][0]["keys"][1]["delta_rotvec"] = [0.1, 0, 0]
        (out / "proposal_plan.json").write_text(json.dumps(plan))
        (out / "report.json").write_text(
            json.dumps(
                {
                    "status": "UNREVIEWED",
                    "terms_before": {"contact": 1.0},
                    "terms_after": {"contact": 0.5},
                }
            )
        )

    monkeypatch.setattr(round_module, "_execute", worker)
    output = path.parent / "local_repair"
    result = round_module.run(path, output, repair=True)
    assert result["motion_status"] == "UNREVIEWED" and not result["selected"]
    original = load_npz(output / "original.npz")
    seed = load_npz(output / "parent.npz")
    candidate = load_npz(output / "candidate.npz")
    expected_seed = apply_edits(original, _plan())
    np.testing.assert_array_equal(seed["poses"], expected_seed["poses"])
    assert not np.array_equal(seed["poses"], candidate["poses"])
    for name in ("trans", "obj_t", "scale_params", "identity_coeffs"):
        assert original[name].tobytes() == candidate[name].tobytes()
    review = read_json(output / "review.json")
    assert review["evaluated_frames"] == list(np.arange(0, 4.25, 0.5))
    assert review["checks"]["original_mesh_intersections"]["status"] == "PERFORMED"
    assert review["checks"]["iteration"]["passed"] is True
    assert review["checks"]["dense_contact_map_objective"]["after"] == 0.5


@pytest.mark.parametrize(
    "fault,match",
    [
        ("wrong_frames", "different frame batch"),
        ("reopened_vertices", "Reopened native geometry"),
        ("reopened_faces", "topology|faces"),
    ],
)
def test_failed_geometry_or_export_reload_records_failure_and_never_builds_scene(
    case_files, mock_workers, fault, match
):
    path, _ = case_files
    mock_workers["fault"] = fault
    output = path.parent / fault
    with pytest.raises(ValueError, match=match):
        round_module.run(path, output)
    receipt = read_json(output / "iteration.json")
    assert receipt["status"] == "FAILED"
    assert receipt["motion_status"] == "UNREVIEWED"
    assert not mock_workers["blender"]


def test_input_change_during_worker_is_recorded_as_changed_not_success(case_files, mock_workers):
    path, _ = case_files
    mock_workers["fault"] = "source_changed"
    output = path.parent / "changed"
    with pytest.raises(ValueError, match="Bound input changed"):
        round_module.run(path, output)
    receipt = read_json(output / "iteration.json")
    assert receipt["status"] == "INPUT_CHANGED"
    assert receipt["selected"] is False


def test_saved_scene_edits_replay_against_original_instead_of_accumulating(
    case_files, mock_workers
):
    path, _ = case_files
    original = load_native(path.parent / "source.npz", "soma03")
    previous = apply_edits(original, _plan())
    parent_path = path.parent / "previous.npz"
    save_npz(parent_path, round_module._without_caches(previous))
    mock_workers["saved_parent"] = parent_path
    mock_workers["saved_plan"] = _plan()
    blend = path.parent / "previous.blend"
    blend.write_bytes(b"saved scene fixture")
    operations = path.parent / "operations.json"
    operations.write_text(
        json.dumps(
            [
                {"op": "set_key", "window": 0, "frame": 2.5, "delta_rotvec": [0.1, 0, 0]},
            ]
        )
    )
    result = round_module.run(
        path, path.parent / "next", input_blend=blend, operations_path=operations
    )
    candidate = load_native(result["native"], "soma03")
    replaced = _plan()
    replaced["windows"][0]["keys"][1]["delta_rotvec"] = [0.1, 0, 0]
    np.testing.assert_array_equal(candidate["poses"], apply_edits(original, replaced)["poses"])
    saved_before = load_npz(path.parent / "next" / "parent.npz")
    np.testing.assert_array_equal(saved_before["poses"], previous["poses"])


def test_contact_review_cannot_run_without_bound_source_video(case_files, mock_workers):
    path, case = case_files
    case.pop("source_video")
    (path.parent / "contacts.json").write_text("{}")
    case["contact_review"] = "contacts.json"
    path.write_text(json.dumps(case))
    with pytest.raises(ValueError, match="source_video|source video"):
        round_module.run(path, path.parent / "unbound")
    assert not mock_workers["geometry"]


def test_source_hash_mismatch_and_unsupported_keypoints_fail_before_workers(
    case_files, mock_workers
):
    path, case = case_files
    case["evidence"] = {"source_video_sha256": "f" * 64}
    path.write_text(json.dumps(case))
    with pytest.raises(ValueError, match="Source evidence hash"):
        round_module.run(path, path.parent / "bad_hash")
    case["evidence"] = {"source_keypoints": {"unbound_observations": []}}
    path.write_text(json.dumps(case))
    with pytest.raises(ValueError, match="not implemented"):
        round_module.run(path, path.parent / "unimplemented")
    assert not mock_workers["geometry"] and not mock_workers["blender"]


def test_contact_overlay_uses_active_fixed_original_material_points_and_normals():
    frames = np.array([0.0, 1.0, 1.5, 2.0])
    rotation = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    scene = {
        "object_vertices": np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
        "object_faces": np.array([[0, 1, 2]]),
        "object_scale": np.array([2.0, 3.0, 1.0]),
    }
    target = np.array([8.5, 20.5, 2.0])
    triangle = np.array([target, target + [1, 0, 0], target + [0, 1, 0]])
    after = {
        "frames": frames,
        "vertices": np.repeat(triangle[None], len(frames), axis=0),
        "faces": np.array([[0, 2, 1]]),
        "obj_R": np.repeat(rotation[None], len(frames), axis=0),
        "obj_t": np.tile([10.0, 20.0, 2.0], (len(frames), 1)),
    }
    before = copy.deepcopy(after)
    before["vertices"][:, :, 2] += 0.1
    report = {
        "checks": {
            "fixed_material_contacts": {
                "pairs": [
                    {
                        "id": "right:hold:palm",
                        "human_vertex_id": 0,
                        "object_face_id": 0,
                        "object_barycentric": [0.25, 0.25, 0.5],
                        "evaluated_active_frames": [1.0, 1.5],
                    },
                ]
            }
        }
    }
    overlay = round_module._contact_overlay(report, before, after, scene)
    assert len(overlay["samples"]) == 4
    for sample in overlay["samples"]:
        assert sample["source_frame"] in [1.0, 1.5]
        assert sample["label"] == "right:hold:palm"
        np.testing.assert_allclose(sample["object_point"], target, atol=1e-12)
        gap = np.linalg.norm(np.asarray(sample["human_point"]) - sample["object_point"])
        assert gap == pytest.approx(0.1 if sample["phase"] == "before" else 0.0)
        np.testing.assert_allclose(sample["human_normal"], [0.0, 0.0, -1.0])
        np.testing.assert_allclose(sample["object_normal"], [0.0, 0.0, 1.0])
    # An original but degenerate human surface cannot invent a normal marker.
    after["vertices"][:, 1:] = after["vertices"][:, :1]
    degenerate = round_module._contact_overlay(report, before, after, scene)
    assert all(
        "human_normal" not in row for row in degenerate["samples"] if row["phase"] == "after"
    )


def test_vertex_contact_overlay_transforms_authored_vertex_and_incident_normals():
    rotation = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    scene = dict(
        object_vertices=np.array(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
        ),
        object_faces=np.array([[0, 1, 2], [0, 3, 1]]),
        object_scale=2.0,
    )
    geometry = dict(
        frames=np.array([0.0, 1.0]),
        vertices=np.array([[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]] * 2),
        faces=np.array([[0, 1, 2]]),
        obj_R=np.array([rotation, rotation]),
        obj_t=np.array([[10.0, 20.0, 2.0]] * 2),
    )
    report = {
        "checks": {
            "fixed_material_contacts": {
                "pairs": [
                    dict(
                        id="right:grip:pad",
                        human_vertex_id=0,
                        object_vertex_id=1,
                        evaluated_active_frames=[1.0],
                    )
                ]
            }
        }
    }
    result = round_module._contact_overlay(report, geometry, geometry, scene)
    assert len(result["samples"]) == 2
    for sample in result["samples"]:
        assert sample["source_frame"] == 1
        np.testing.assert_allclose(sample["object_point"], [10.0, 22.0, 2.0])
        np.testing.assert_allclose(sample["object_normal"], np.array([-1.0, 0.0, 1.0]) / np.sqrt(2))
