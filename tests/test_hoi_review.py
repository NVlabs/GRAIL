"""Native CPU visualization preserves camera convention, occlusion, and timeline."""

import json
import shutil

import numpy as np
import pytest

from grail.visualization.hoi_review import (
    CpuRasterizer,
    build_raster,
    fixed_oblique_camera,
    load_geometry,
    make_review,
    render_frame,
    select_frames,
    source_camera,
)


def geometry(count=3):
    triangle = np.array([[-.3, -.3, 2.], [.3, -.3, 2.], [0., .3, 2.]])
    return {
        "human_vertices": np.tile(triangle, (count, 1, 1)),
        "object_vertices": np.tile(triangle * [1, 1, 1.5], (count, 1, 1)),
        "human_faces": np.array([[0, 1, 2]], dtype=np.int64),
        "object_faces": np.array([[0, 1, 2]], dtype=np.int64),
        "camera_R": np.eye(3)[None], "camera_T": np.zeros((1, 3)),
        "focal": np.array(120.), "image_size": np.array([128, 128]),
    }


@pytest.fixture(scope="module")
def raster(tmp_path_factory):
    if not shutil.which("g++"):
        pytest.skip("CPU C++ compiler unavailable")
    path, _ = build_raster(tmp_path_factory.mktemp("raster"))
    return CpuRasterizer(path)


def test_source_camera_matches_pytorch3d_row_vector_projection():
    g = geometry()
    g["camera_R"] = np.array([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])[None]
    g["camera_T"] = np.array([[.2, .4, 5.]])
    point = np.array([[.7, -.6, 2.]])
    original = point @ g["camera_R"][0] + g["camera_T"][0]
    rotation, translation = source_camera(g)
    actual = point @ rotation + translation
    np.testing.assert_allclose(actual, original * [-1, -1, 1])
    np.testing.assert_allclose(64 + 120*actual[:, :2]/actual[:, 2:],
                               64 - 120*original[:, :2]/original[:, 2:])


def test_raster_perspective_depth_and_near_clip(raster):
    # These vertices project to (1,1), (7,1), (1,7); center (2.5,2.5)
    # has barycentrics .5,.25,.25, giving perspective depth 1/.6875.
    points = np.array([[-.375, -.375, 1.], [.75, -.75, 2.], [-1.5, 1.5, 4.]])
    faces = np.array([[0, 1, 2]])
    depth, ids = raster.raster(points, faces, [8., 8.], 8, 8)
    assert ids[2, 2] == 0
    assert depth[2, 2] == pytest.approx(1/.6875)
    assert ids[7, 7] == -1 and np.isinf(depth[7, 7])
    points[0, 2] = -1
    depth, ids = raster.raster(points, faces, [8., 8.], 8, 8)
    assert (ids >= 0).any(), "A crossing triangle must be clipped, not completely omitted"
    assert (depth[np.isfinite(depth)] > 0).all()


def test_human_object_occlusion_and_buffer_reuse(raster):
    g = geometry(1)
    image = render_frame(raster, g, 0, np.eye(3), np.zeros(3), [120, 120], 128, 128)
    assert image[64, 64, 0] > image[64, 64, 2], "Closer human must be blue"
    g["object_vertices"][:, :, 2] = 1.5
    image = render_frame(raster, g, 0, np.eye(3), np.zeros(3), [120, 120], 128, 128)
    assert image[64, 64, 2] > image[64, 64, 0], "Closer object must occlude human"
    _, ids = raster.raster(g["human_vertices"][0], g["human_faces"], [120]*2, 128, 128)
    _, reused = raster.raster(g["human_vertices"][0], g["human_faces"], [120]*2, 128, 128)
    assert ids is reused


def test_loader_rejects_corrupt_geometry_and_frame_selection(tmp_path):
    path = tmp_path / "geometry.npz"
    g = geometry()
    np.savez(path, **g)
    loaded = load_geometry(path)
    assert loaded["focal"].tolist() == [120., 120.]
    assert select_frames("2,0,2", 3) == [0, 2]
    for frames in ["-1", "3", "1.5", ""]:
        with pytest.raises(ValueError):
            select_frames(frames, 3)
    g["human_faces"] = np.array([[0, 1, 3]])
    np.savez(path, **g)
    with pytest.raises(ValueError, match="face indices"):
        load_geometry(path)


def test_fixed_oblique_contains_sequence_not_only_first_frame():
    g = geometry()
    g["human_vertices"][2] += [2., 0., 1.]
    rotation, translation = fixed_oblique_camera(g)
    np.testing.assert_allclose(rotation.T @ rotation, np.eye(3), atol=1e-12)
    for key in ("human_vertices", "object_vertices"):
        points = g[key] @ rotation + translation
        assert (points[..., 2] > 0).all()
        projection = 1.1 * points[..., :2] / points[..., 2:] + .5
        assert ((projection > 0) & (projection < 1)).all()


def test_cached_compile_and_complete_video_receipt(tmp_path, monkeypatch):
    cv2 = pytest.importorskip("cv2")
    if not all(shutil.which(command) for command in ("g++", "ffmpeg", "ffprobe")):
        pytest.skip("Video review tools unavailable")
    cache = tmp_path / "cache"
    first, cold = build_raster(cache)
    second, warm = build_raster(cache)
    assert first == second and not cold["cache_hit"] and warm["cache_hit"]
    g = geometry()
    path, video = tmp_path / "geometry.npz", tmp_path / "source.mp4"
    np.savez(path, **g)
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), 24, (128, 128))
    assert writer.isOpened()
    for index in range(3):
        writer.write(np.full((128, 128, 3), 30+index*30, np.uint8))
    writer.release()
    receipt = make_review(path, video, tmp_path / "full", frames="0,2", cache_dir=cache,
                          panel_size=128)
    assert receipt["status"] == "RENDER_COMPLETE_UNREVIEWED"
    assert receipt["rendered_frames"] == [0, 1, 2] and receipt["still_frames"] == [0, 2]
    assert receipt["encoded_stream"]["nb_read_frames"] == "3"
    assert receipt["raster_build"]["cache_hit"]
    assert json.loads((tmp_path / "full/review_receipt.json").read_text())["fps"] == 24
    original_capture = cv2.VideoCapture

    class NoSeekCapture:
        def __init__(self, *args, **kwargs):
            self.capture = original_capture(*args, **kwargs)

        def __getattr__(self, name):
            return getattr(self.capture, name)

        def set(self, prop, value):
            if prop == cv2.CAP_PROP_POS_FRAMES:
                raise AssertionError("Keyframe review must decode source sequentially")
            return self.capture.set(prop, value)

    monkeypatch.setattr(cv2, "VideoCapture", NoSeekCapture)
    still = make_review(path, video, tmp_path / "stills", frames="0,2", keyframes_only=True,
                        cache_dir=cache, panel_size=128, camera_reference=path)
    assert still["rendered_frames"] == [0, 2] and "video" not in still["outputs"]
    assert receipt["fixed_oblique_camera"] == still["fixed_oblique_camera"]
    assert receipt["oblique_camera_reference"]["sha256"] == still["geometry"]["sha256"]
    # Still-only and full-video reviews must align identical source backgrounds.
    for index in (0, 2):
        name = f"frame_{index:05d}.jpg"
        np.testing.assert_array_equal(cv2.imread(str(tmp_path / "full" / name)),
                                      cv2.imread(str(tmp_path / "stills" / name)))
    first_background = cv2.imread(str(tmp_path / "stills/frame_00000.jpg"))[110:120, 5:15]
    last_background = cv2.imread(str(tmp_path / "stills/frame_00002.jpg"))[110:120, 5:15]
    assert last_background.mean() > first_background.mean() + 40
    with pytest.raises(FileExistsError):
        make_review(path, video, tmp_path / "full", cache_dir=cache)
    g["human_vertices"] = g["human_vertices"][:2]
    g["object_vertices"] = g["object_vertices"][:2]
    np.savez(path, **g)
    with pytest.raises(ValueError, match="timeline or size mismatch"):
        make_review(path, video, tmp_path / "mismatch", cache_dir=cache)
