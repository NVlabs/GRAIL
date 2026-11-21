"""Original-pixel evidence must not reward missing coverage or wrong-body hits."""

import json

import numpy as np
import pytest

from grail.visualization.hoi_diagnostics import (
    compare_geometry,
    depth_alignment,
    frame_alignment,
    geometry_metrics,
    load_depth_and_masks,
    temporal_metrics,
)


def geometry(frames=4):
    points = np.array([[0.0, 0.0, 2.0], [0.2, 0.0, 2.0], [0.0, 0.2, 2.0]])
    points = np.repeat(points[None], frames, axis=0)
    return {
        "human_vertices": points.copy(), "object_vertices": points.copy(),
        "human_faces": np.array([[0, 1, 2]]), "object_faces": np.array([[0, 1, 2]]),
        "camera_R": np.eye(3), "camera_T": np.zeros(3), "focal": np.array([20.0, 20.0]),
        "image_size": np.array([2, 3]), "joints": points.copy(),
    }


def test_world_velocity_uses_source_rate_and_object_rotation_is_not_hidden_by_centroid():
    points = np.zeros((5, 2, 3))
    points[:, :, 0] = np.arange(5)[:, None] / 24
    report = temporal_metrics(points, 24)
    assert report["speed_m_s"]["mean"] == pytest.approx(1.0)
    assert report["acceleration_m_s2"]["max"] == pytest.approx(0.0)
    g = geometry()
    g["object_vertices"][1] = g["object_vertices"][1, ::-1]
    metrics = geometry_metrics(g, 24, expected_frames=4)
    assert metrics["object_centroid"]["speed_m_s"]["max"] == pytest.approx(0.0, abs=1e-6)
    assert metrics["object_material_vertices"]["speed_m_s"]["max"] > 0
    assert len(metrics["per_joint"]) == 3


def test_invalid_native_faces_timeline_and_nan_fail():
    g = geometry()
    with pytest.raises(ValueError, match="expected 121"):
        geometry_metrics(g, 24, expected_frames=121)
    g["object_faces"][0, 0] = 4
    with pytest.raises(ValueError, match="original vertices"):
        geometry_metrics(g, 24)
    g = geometry()
    g["human_vertices"][0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="nonfinite"):
        geometry_metrics(g, 24)


def test_empty_derivatives_and_explicit_geometry_changes():
    g = geometry(1)
    assert temporal_metrics(g["joints"], 24)["speed_m_s"]["count"] == 0
    changed = {key: value.copy() for key, value in g.items()}
    changed["object_vertices"][:, :, 0] += 0.01
    comparison = compare_geometry(g, changed)
    assert all(comparison["exact_equal_fields"].values())
    assert comparison["geometry_change"]["human_vertices"]["exactly_equal"]
    assert comparison["geometry_change"]["object_vertices"]["displacement_m"]["max"] == pytest.approx(0.01)


def test_common_rays_do_not_hide_coverage_and_wrong_surface_hits():
    target = np.array([[2.0, 2.0, 2.0, np.nan]])
    masks = {"human": np.zeros_like(target), "object": np.ones_like(target)}
    inf = np.full_like(target, np.inf)
    predictions = {
        "before": {"human": inf.copy(), "object": np.array([[2.2, 2.2, 2.2, 2.2]])},
        "after": {"human": np.array([[np.inf, np.inf, 2.0, np.inf]]),
                  "object": np.array([[2.1, np.inf, 2.2, 2.0]])},
    }
    rows = frame_alignment(target, masks, predictions)
    after = rows["after"]["object"]
    assert after["target_pixels"] == 3
    assert after["invalid_target_depth_pixels"] == 1
    assert after["covered_pixels"] == 1
    assert after["missing_pixels"] == 2
    assert after["wrong_surface_hit_pixels"] == 1
    assert after["common_pixels"] == 1
    assert after["common_mae_m"] == pytest.approx(0.1)
    assert after["frontmost_any_surface_mae_m"] == pytest.approx(0.05)
    assert rows["before"]["object"]["common_mae_m"] == pytest.approx(0.2)


def test_empty_source_is_not_perfect_alignment_and_resolution_is_never_resized():
    target = np.ones((2, 3))
    masks = {"human": np.zeros_like(target), "object": np.zeros_like(target)}
    layers = {"a": {"human": np.full_like(target, np.inf), "object": np.full_like(target, np.inf)}}
    result = frame_alignment(target, masks, layers)["a"]["object"]
    assert result["target_pixels"] == 0
    assert result["common_mae_m"] is None
    assert result["silhouette_iou"] is None
    layers["a"]["object"] = np.ones((1, 3))
    with pytest.raises(ValueError, match="original resolution"):
        frame_alignment(target, masks, layers)


class StubRaster:
    def raster(self, vertices, faces, focal, height, width, slot):
        return np.full((height, width), 2 if "object" in slot else np.inf), None


def test_all_frames_including_zero_and_pixel_weighted_summary():
    g = geometry(2)
    depth = [np.full((2, 3), 2.0), np.full((2, 3), 3.0)]
    masks = {0: {0: np.ones((2, 3)), 1: np.zeros((2, 3))},
             1: {0: np.array([[1, 0, 0], [0, 0, 0]]), 1: np.zeros((2, 3))}}
    report = depth_alignment({"a": g}, depth, masks, StubRaster())
    assert [row["frame"] for row in report["frames"]["a"]] == [0, 1]
    assert report["summary"]["a"]["object"]["target_pixels"] == 7
    assert report["summary"]["a"]["object"]["pixel_weighted_common_mae_m"] == pytest.approx(1 / 7)
    with pytest.raises(ValueError, match="including frame 0"):
        depth_alignment({"a": g}, depth, {1: masks[1]}, StubRaster())


def test_candidates_cannot_change_camera_for_common_pixel_evaluation():
    g, changed = geometry(1), geometry(1)
    changed["camera_T"][0] = 0.1
    with pytest.raises(ValueError, match="different calibrated camera_T"):
        depth_alignment({"a": g, "b": changed}, [np.ones((2, 3))],
                        {0: {0: np.ones((2, 3)), 1: np.zeros((2, 3))}}, StubRaster())


def test_existing_depth_cache_and_mask_format_require_background_calibration(tmp_path):
    torch = pytest.importorskip("torch")
    path = tmp_path / "moge3.pt"
    mask_path = tmp_path / "masks.npz"
    torch.save([torch.ones((2, 3))], path)
    np.savez(mask_path, masks={0: {0: np.ones((2, 3)), 1: np.zeros((2, 3))}})
    metadata = {"frame_count": 1, "model": {"depth_unit": "meter"}, "gt_calibration": {
        "status": "aligned", "method": "per_frame_background_linear_depth_scale_shift",
        "first_frame_obj_mask_supplied": True, "first_frame_human_mask_supplied": True,
        "video_masks_supplied": True, "frames": [{"frame_idx": 0, "status": "aligned",
             "foreground_excluded": True, "scale": 1.0, "shift": 0.1}]}}
    metadata_path = tmp_path / "moge3.pt.json"
    metadata_path.write_text(json.dumps(metadata))
    depth, masks, records = load_depth_and_masks(path, mask_path, 1)
    assert len(depth) == len(masks) == 1
    assert len(records["depth_metadata"]["sha256"]) == 64
    metadata["gt_calibration"]["frames"][0]["foreground_excluded"] = False
    metadata_path.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="background calibration"):
        load_depth_and_masks(path, mask_path, 1)
