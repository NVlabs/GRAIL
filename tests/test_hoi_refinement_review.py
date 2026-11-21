"""Synthetic native-geometry review with fixed source/material/ray bindings."""

import copy
import hashlib
import json

import numpy as np
import pytest

from grail.agentic_kinematic.hoi_refinement.review import review_candidate
from grail.optimization.fixed_contact import build_binding


@pytest.fixture
def case():
    vertices = np.array([[0.0, 0.0, 2.0], [1.0, 0.0, 2.0], [0.0, 1.0, 2.0]])
    faces = np.array([[0, 1, 2]])
    parent = {
        "model_type": "soma03",
        "poses": np.zeros((4, 56, 3)),
        "trans": np.zeros((4, 3)),
        "obj_R": np.repeat(np.eye(3)[None], 4, axis=0),
        "obj_t": np.zeros((4, 3)),
        "fps": 30.0,
    }
    candidate = copy.deepcopy(parent)
    before = {
        "vertices": np.repeat((vertices + [0, 0, 0.1])[None], 4, axis=0),
        "joints": np.repeat(vertices[None], 4, axis=0),
        "faces": faces,
        "frames": np.arange(4),
    }
    after = copy.deepcopy(before)
    after["vertices"][1:3, :, 2] -= 0.1
    scene = {
        "object_vertices": vertices,
        "object_faces": faces[:, ::-1],
        "object_scale": 1.0,
        "camera_matrix_world": np.eye(4),
        "camera_intrinsics": np.array([[100.0, 0, 50], [0, 100.0, 50], [0, 0, 1]]),
        "image_width": 100,
        "image_height": 100,
        "_sha256": "c" * 64,
    }
    evidence = {"sequence": "synthetic", "source_video_sha256": "a" * 64}
    binding = build_binding(
        sequence=evidence["sequence"],
        source_video_sha256=evidence["source_video_sha256"],
        frame_count=4,
        fps=parent["fps"],
        human_model_type=parent["model_type"],
        human_vertex_count=3,
        human_faces=faces,
        object_vertices=vertices,
        object_faces=scene["object_faces"],
    )
    spec = {
        "schema": "grail.fixed_contact.v1",
        "binding": binding,
        "prediction": {"method": "synthetic", "evidence": "test fixture"},
        "contacts": [
            {
                "id": "right-hand-contact",
                "body_part": "R_Hand",
                "human_vertex_id": 0,
                "object_face_id": 0,
                "object_barycentric": [0.0, 0.0, 1.0],
                "active_frames": [1, 3],
                "confidence": 0.8,
            }
        ],
    }
    return dict(
        parent=parent,
        candidate=candidate,
        before=before,
        after=after,
        scene=scene,
        contact_spec=spec,
        evidence=evidence,
    )


def test_planted_contact_retains_material_pair_and_opposing_normals(case):
    report = review_candidate(**case)
    assert report["status"] == "UNREVIEWED"
    pair = report["checks"]["fixed_material_contacts"]["pairs"][0]
    assert pair["evaluated_active_frames"] == [1, 2]
    assert pair["before"]["fixed_material_gap_m"]["mean"] == pytest.approx(0.1)
    assert pair["after"]["fixed_material_gap_m"]["mean"] == pytest.approx(0.0)
    assert pair["gap_change_m"]["mean"] == pytest.approx(-0.1)
    assert pair["after"]["normal"]["per_frame_dot"] == [-1.0, -1.0]
    assert pair["authored_confidence"] == 0.8
    assert report["checks"]["original_mesh_intersections"]["status"] == "NOT_PERFORMED"
    assert report["checks"]["fixed_ray_depth"]["status"] == "NOT_PERFORMED"
    json.dumps(report, allow_nan=False)


def test_vertex_pair_review_uses_original_vertex_and_vertex_normals(case):
    spec = case["contact_spec"]
    spec["schema"] = "grail.fixed_contact.v2"
    pair = spec["contacts"][0]
    pair.pop("object_face_id")
    pair.pop("object_barycentric")
    pair["object_vertex_id"] = 0
    report = review_candidate(**case)
    pair = report["checks"]["fixed_material_contacts"]["pairs"][0]
    assert pair["object_vertex_id"] == 0
    assert "object_face_id" not in pair
    assert pair["after"]["fixed_material_gap_m"]["max"] == pytest.approx(0)
    assert pair["after"]["normal"]["per_frame_dot"] == [-1.0, -1.0]


def test_reviewed_vertex_pairs_switch_at_source_interval_boundary(case):
    def interval(name, frames, state, contacts):
        return dict(
            id=name,
            frames=frames,
            state=state,
            description="Source-reviewed change",
            evidence_frames=[frames[0]],
            contacts=contacts,
        )

    def point(index):
        return dict(
            id="pad",
            human_vertex_id=index,
            object_vertex_id=index,
            confidence=0.8,
            hand_surface="Selected hand vertex",
            object_surface="Selected object vertex",
        )

    case["contact_spec"] = dict(
        schema="grail.contact_review.v2",
        binding=case["contact_spec"]["binding"],
        reviewed_by="synthetic fixture",
        hands=dict(
            left_hand=[interval("unknown", [0, 4], "unknown", [])],
            right_hand=[
                interval("approach", [0, 1], "no_contact", []),
                interval("touch", [1, 2], "contact", [point(0)]),
                interval("grip", [2, 3], "contact", [point(1)]),
                interval("release", [3, 4], "no_contact", []),
            ],
        ),
    )
    # Only the inactive old pair is displaced; the new interval must use vertex1.
    case["after"]["vertices"][2, 0, 2] += 2
    result = review_candidate(**case)["checks"]["fixed_material_contacts"]
    assert [p["evaluated_active_frames"] for p in result["pairs"]] == [[1], [2]]
    assert [p["object_vertex_id"] for p in result["pairs"]] == [0, 1]
    assert all(
        p["after"]["fixed_material_gap_m"]["max"] == pytest.approx(0) for p in result["pairs"]
    )


def test_nearest_other_material_does_not_replace_authored_contact(case):
    case["after"]["vertices"][:, 0] = case["scene"]["object_vertices"][1]
    report = review_candidate(**case)
    pair = report["checks"]["fixed_material_contacts"]["pairs"][0]
    assert pair["after"]["fixed_material_gap_m"]["mean"] == pytest.approx(1.0)
    # Duplicating vertex one degenerates the human triangle. Gap is still real;
    # normals cannot be silently replaced by an arbitrary unit axis.
    assert pair["after"]["normal"]["status"] == "NOT_PERFORMED"


def test_sparse_frame_mask_never_fills_missing_contact_or_velocity_samples(case):
    for name in ("before", "after"):
        case[name]["frames"] = np.array([0, 2])
        for field in ("vertices", "joints"):
            case[name][field] = case[name][field][[0, 2]]
    report = review_candidate(**case)
    pair = report["checks"]["fixed_material_contacts"]["pairs"][0]
    assert pair["evaluated_active_frames"] == [2]
    assert pair["missing_active_frame_count"] == 1
    assert pair["after"]["contact_relative_velocity"]["status"] == "NOT_PERFORMED"
    assert report["checks"]["temporal"]["after_joint_velocity"]["status"] == "NOT_PERFORMED"
    assert report["unevaluated_frame_count"] == 2


@pytest.mark.parametrize(
    "field,value", [("source_video_sha256", "b" * 64), ("sequence", "another")]
)
def test_independent_source_binding_mismatch_rejected(case, field, value):
    case["evidence"][field] = value
    with pytest.raises(ValueError, match="binding differs"):
        review_candidate(**case)


def test_missing_independent_source_hash_does_not_trust_contact_self_binding(case):
    case["evidence"].pop("source_video_sha256")
    report = review_candidate(**case)
    assert report["checks"]["fixed_material_contacts"]["status"] == "NOT_PERFORMED"


def test_changed_original_topology_and_wrong_frame_masks_rejected(case):
    case["after"]["faces"] = case["after"]["faces"][:, ::-1]
    with pytest.raises(ValueError, match="original human topology"):
        review_candidate(**case)
    case["after"]["faces"] = case["before"]["faces"]
    case["after"]["frames"] = np.array([0, 1, 3, 2])
    with pytest.raises(ValueError, match="strictly increasing"):
        review_candidate(**case)


def test_invalid_collision_query_does_not_silently_resample(case):
    case["evidence"]["collision_frames"] = [4]
    with pytest.raises(ValueError, match="subset"):
        review_candidate(**case)


def test_original_triangle_crossing_detected_and_open_mesh_not_certified(case):
    case["evidence"]["collision_frames"] = [1]
    case["after"]["vertices"][1] = np.array([[0.2, 0.2, 1.0], [0.3, 0.2, 3.0], [0.2, 0.3, 3.0]])
    report = review_candidate(**case)
    result = report["checks"]["original_mesh_intersections"]
    assert result["queried_frames"] == [1]
    assert result["frames_with_new_crossings"] == [1]
    assert result["frames_with_crossings_before_and_after"] == []
    candidate = result["frames"][0]["after"]
    assert candidate["human_edges_to_object"]["intersection_locations"] > 0
    assert candidate["human_edges_to_object"]["source_edges_tested"] == 3
    assert candidate["object_edges_to_human"]["target_faces_tested"] == 1
    assert candidate["human_topology"]["solid_state"] == "UNKNOWN_OPEN_GEOMETRY"


def test_inherited_crossings_reported_separately(case):
    case["evidence"]["collision_frames"] = [1]
    crossing = np.array([[0.2, 0.2, 1.0], [0.3, 0.2, 3.0], [0.2, 0.3, 3.0]])
    case["before"]["vertices"][1] = crossing
    case["after"]["vertices"][1] = crossing
    result = review_candidate(**case)["checks"]["original_mesh_intersections"]
    assert result["frames_with_crossings_before_and_after"] == [1]
    assert result["frames_with_new_crossings"] == []


def test_projection_measures_parent_movement_without_rgb_confidence_claim(case):
    case["after"]["joints"][:, :, 0] += 0.2
    report = review_candidate(**case)
    projection = report["checks"]["source_projection"]
    assert projection["joint_displacement_px"]["mean"] == pytest.approx(10.0)
    assert "confidence" not in projection
    assert "not an observed RGB keypoint error" in projection["limitations"]


def test_existing_review_schema_compiles_independent_hand_intervals(case):
    fixed = case["contact_spec"]
    pair = {
        k: v for k, v in fixed["contacts"][0].items() if k not in ("body_part", "active_frames")
    }
    pair.update(hand_surface="palm", object_surface="original triangle")

    def interval(name, frames, state, contacts):
        return dict(
            id=name,
            frames=frames,
            state=state,
            description="Synthetic source observation",
            evidence_frames=[frames[0]],
            contacts=contacts,
        )

    case["contact_spec"] = {
        "schema": "grail.contact_review.v1",
        "binding": fixed["binding"],
        "reviewed_by": "fixture",
        "hands": {
            "left_hand": [interval("unresolved", [0, 4], "unknown", [])],
            "right_hand": [
                interval("approach", [0, 1], "no_contact", []),
                interval("hold", [1, 3], "contact", [pair]),
                interval("release", [3, 4], "no_contact", []),
            ],
        },
    }
    result = review_candidate(**case)["checks"]["fixed_material_contacts"]
    assert result["video_intervals_present"]
    assert result["pairs"][0]["evaluated_active_frames"] == [1, 2]
    assert result["reviewed_hand_intervals"]["left_hand"][0]["state"] == "unknown"


def _depth_case(case, tmp_path):
    case["contact_spec"] = None
    case["scene"]["object_vertices"] += [20, 0, 0]
    human = np.array([[-1.0, -1.0, 2.0], [1.0, -1.0, 2.0], [0.0, 1.0, 2.0]])
    case["before"]["vertices"][:] = human
    case["after"]["vertices"][:] = human + [0, 0, 1]
    path = tmp_path / "fixed-rays.npz"
    directions = np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 1.0]])
    directions /= np.linalg.norm(directions, axis=-1, keepdims=True)
    np.savez(
        path,
        frames=np.array([1, 1]),
        origins=np.zeros((2, 3)),
        directions=directions,
        target_distance=np.array([2.0, 2.0]),
        group=np.array(["human", "missing"]),
    )
    case["evidence"]["fixed_depth"] = {
        "path": str(path),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "source_video_sha256": case["evidence"]["source_video_sha256"],
        "scene_sha256": case["scene"]["_sha256"],
        "calibration_sha256": "d" * 64,
        "calibration": "background_gt",
        "units": "m",
        "depth_kind": "ray_distance",
    }
    return case


def test_frozen_pixel_rays_measure_exact_original_surface_hits(case, tmp_path):
    report = review_candidate(**_depth_case(case, tmp_path))
    depth = report["checks"]["fixed_ray_depth"]
    assert depth["status"] == "PERFORMED"
    assert depth["common_hit_count"] == 1
    assert depth["before"]["coverage"] == 0.5
    assert depth["before"]["common_hit_absolute_error_m"]["mean"] == pytest.approx(0)
    assert depth["after"]["common_hit_absolute_error_m"]["mean"] == pytest.approx(1)
    assert depth["rows"][0]["before"] == {
        "surface": "human",
        "original_face_id": 0,
        "distance_m": 2.0,
    }
    assert depth["rows"][1]["before"]["distance_m"] is None
    json.dumps(report, allow_nan=False)


def test_frozen_rays_count_lost_coverage_instead_of_changing_rays(case, tmp_path):
    _depth_case(case, tmp_path)
    case["after"]["vertices"][:, :, 0] += 10
    depth = review_candidate(**case)["checks"]["fixed_ray_depth"]
    assert depth["new_missing_count"] == 1
    assert depth["inherited_missing_count"] == 1
    assert depth["common_hit_count"] == 0
    assert depth["after"]["common_hit_absolute_error_m"]["mean"] is None


@pytest.mark.parametrize("field", ["sha256", "source_video_sha256", "scene_sha256"])
def test_frozen_ray_cache_rejects_hash_mismatch(case, tmp_path, field):
    _depth_case(case, tmp_path)
    case["evidence"]["fixed_depth"][field] = "f" * 64
    with pytest.raises(ValueError, match="mismatch|differs"):
        review_candidate(**case)


def test_frozen_ray_rows_outside_evaluated_frames_stay_missing(case, tmp_path):
    _depth_case(case, tmp_path)
    for name in ("before", "after"):
        case[name]["frames"] = np.array([0, 2])
        for field in ("vertices", "joints"):
            case[name][field] = case[name][field][[0, 2]]
    depth = review_candidate(**case)["checks"]["fixed_ray_depth"]
    assert depth["status"] == "NOT_PERFORMED"
    assert depth["unevaluated_ray_count"] == 2


@pytest.mark.parametrize("model_type,pose_shape", [("soma03", (4, 56, 3)), ("smplx", (4, 165))])
def test_review_is_native_model_neutral(case, model_type, pose_shape):
    for name in ("parent", "candidate"):
        case[name]["model_type"] = model_type
        case[name]["poses"] = np.zeros(pose_shape)
    case["contact_spec"]["binding"]["human_model_type"] = model_type
    assert review_candidate(**case)["status"] == "UNREVIEWED"


def test_canonical_npz_scalars_and_native_subframes_are_preserved(case):
    for name in ("parent", "candidate"):
        case[name]["model_type"] = np.asarray("soma03")
        case[name]["fps"] = np.asarray(30.0)
    for name in ("before", "after"):
        geometry = case[name]
        geometry["frames"] = np.array([0.0, 1.0, 1.5, 2.0])
        geometry["obj_R"] = case["parent"]["obj_R"].copy()
        geometry["obj_t"] = case["parent"]["obj_t"].copy()
        # 1 cm / half a source frame => 0.6 m/s, evaluated natively.
        geometry["joints"][2:, :, 0] += 0.01
    case["evidence"]["collision_frames"] = [1.5]
    report = review_candidate(**case)
    assert report["evaluated_frames"] == [0, 1, 1.5, 2]
    assert report["evaluated_subframe_count"] == 1
    assert report["unevaluated_frame_count"] == 1
    pair = report["checks"]["fixed_material_contacts"]["pairs"][0]
    assert pair["evaluated_active_subframe_count"] == 1
    assert pair["missing_active_frame_count"] == 0
    velocity = report["checks"]["temporal"]["after_joint_velocity"]
    assert velocity["speed_m_per_s"]["max"] == pytest.approx(0.6)
    assert report["checks"]["original_mesh_intersections"]["queried_frames"] == [1.5]
    json.dumps(report, allow_nan=False)


def test_fractional_frame_requires_native_object_sample(case):
    for name in ("before", "after"):
        case[name]["frames"] = np.array([0.0, 1.0, 1.5, 2.0])
    with pytest.raises(ValueError, match="Subframe review requires"):
        review_candidate(**case)


def test_native_object_scale_cannot_silently_differ_from_bound_scene(case):
    case["candidate"]["obj_scale"] = np.asarray(2.0)
    with pytest.raises(ValueError, match="object scale disagrees"):
        review_candidate(**case)


def test_frozen_ray_uses_nearest_of_original_human_and_object(case, tmp_path):
    _depth_case(case, tmp_path)
    # Move the object's original triangle in front of the human on the same ray.
    case["scene"]["object_vertices"] = np.array(
        [[-1.0, -1.0, 1.0], [1.0, -1.0, 1.0], [0.0, 1.0, 1.0]]
    )
    depth = review_candidate(**case)["checks"]["fixed_ray_depth"]
    assert depth["rows"][0]["before"] == {
        "surface": "object",
        "original_face_id": 0,
        "distance_m": 1.0,
    }
    assert depth["rows"][0]["after"]["distance_m"] == 1.0


def test_frozen_ray_rejects_camera_origin_mismatch_even_with_valid_cache_hash(case, tmp_path):
    _depth_case(case, tmp_path)
    pack = case["evidence"]["fixed_depth"]
    with np.load(pack["path"]) as archive:
        values = {name: archive[name].copy() for name in archive.files}
    values["origins"] += [0.1, 0, 0]
    np.savez(pack["path"], **values)
    from pathlib import Path

    pack["sha256"] = hashlib.sha256(Path(pack["path"]).read_bytes()).hexdigest()
    with pytest.raises(ValueError, match="origins differ"):
        review_candidate(**case)


def test_bulk_topology_matches_exact_weld_reference():
    from grail.agentic_kinematic.hoi_refinement.review import _topology_counts
    from grail.core.contact import _components

    # Closed tetrahedron, disconnected open shell, exact-coordinate aliases,
    # nonmanifold shared edge, reversed winding and a degenerate face.
    vertices = np.array(
        [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [3, 0, 0], [4, 0, 0], [3, 1, 0], [0, 0, 0]],
        float,
    )
    shell = [[0, 2, 1], [7, 1, 3], [1, 2, 3], [2, 0, 3]]
    for extra in ([], [[4, 5, 6]], [[0, 1, 2]], [[0, 0, 1]], [[0, 2, 1]]):
        faces = np.array(shell + extra)
        groups = _components(vertices, faces)
        assert _topology_counts(vertices, faces) == (
            len(groups),
            sum(not closed for _, closed in groups),
        )
