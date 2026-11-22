"""Check a fixed-camera capture in GRAIL's generation layout and draw frame zero."""

import argparse
import pickle
from pathlib import Path

import cv2
import numpy as np
from grail.retargeting.manipulation.handoff import ordered_obj


def contours(mask):
    return cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[0]


def check(root, sequence, output, *, scene_id=None, conditioning_only=False, static_terrain=False):
    dataset, category, clip = sequence.split("/")
    group = Path(dataset, category)
    scene_id = scene_id or clip
    if Path(scene_id).name != scene_id or scene_id in (".", ".."):
        raise ValueError("Use the basename of the conditioning render for --scene-id")
    case = group / scene_id
    video = root / "videos_kling" / group / f"{clip}.mp4"
    mesh = root / "mesh" / group / "model.obj"
    scene = root / "foundation_pose" / case

    with (root / "cameras" / group / f"{scene_id}.pickle").open("rb") as stream:
        camera = pickle.load(stream)
    with (scene / "first_frame_pose.pickle").open("rb") as stream:
        pose = pickle.load(stream)
    render = root / "asset_renders" / group / f"{scene_id}.png"
    if conditioning_only:
        frame = cv2.imread(str(render))
        if frame is None:
            raise ValueError(f"Cannot read conditioning frame {render}")
    else:
        cap = cv2.VideoCapture(str(video))
        ok, frame = cap.read()
        cap.release()
        if not ok:
            raise ValueError(f"Cannot read {video}")
    height, width = frame.shape[:2]
    if (width, height) != tuple(camera["resolution"]) or (
        width, height
    ) != (pose["frame_width"], pose["frame_height"]):
        raise ValueError("Frame zero and calibration have different dimensions")
    for field, other in (("R", "cam_R"), ("t", "cam_t"), ("obj_R", "obj_R"),
                         ("obj_t", "obj_t"), ("obj_scale", "obj_scale")):
        if not np.allclose(np.ravel(camera[field]), np.ravel(pose[other])):
            raise ValueError(f"Camera and first-frame pose disagree on {field}")

    mask = cv2.imread(str(scene / "masks/000000.png"), cv2.IMREAD_GRAYSCALE)
    human = cv2.imread(str(scene / "human_masks/000000.png"), cv2.IMREAD_GRAYSCALE)
    depth = cv2.imread(str(root / "depth_maps" / case / "000000.png"), cv2.IMREAD_UNCHANGED)
    for name, image in (("object mask", mask), ("human mask", human), ("depth", depth)):
        if image is None or image.shape != (height, width) or not np.any(image):
            raise ValueError(f"Missing, empty or incorrectly sized {name}")
    if depth.dtype != np.uint16:
        raise ValueError("Provide registered frame-zero depth as uint16 millimetres")

    # Frame-zero input check. The model adds its own validity mask during alignment.
    background = (human == 0) & (mask == 0) & (depth > 100) & (depth < 65000)
    background_count = int(np.count_nonzero(background))
    if background_count < 100:
        raise ValueError(f"Only {background_count} usable background depth pixels; need at least 100")

    intrinsics = np.loadtxt(scene / "cam_K.txt").reshape(3, 3)
    focal = float(pose["focal_length"])
    fitted = np.array([[focal, 0, width / 2], [0, focal, height / 2], [0, 0, 1]])
    if not (np.isfinite(focal) and focal > 0 and
            np.isclose(camera["focal_length"], focal, rtol=0, atol=0.01) and
            np.allclose(intrinsics, fitted, rtol=0, atol=0.01)):
        raise ValueError("Camera files must agree with the fitter's centered, single-focal camera in pixels")
    if static_terrain:
        from grail.retargeting.terrain import read_obj_surface
        vertices, faces = read_obj_surface(mesh)
    else:
        vertices, faces = ordered_obj(mesh)
    if not len(faces):
        raise ValueError("Object OBJ must contain faces")
    world = (vertices * np.ravel(pose["obj_scale"])) @ pose["obj_R"].T
    world += np.ravel(pose["obj_t"])
    camera_xyz = (world - np.ravel(pose["cam_t"])) @ pose["cam_R"] * [1, -1, -1]
    if not np.all(np.isfinite(camera_xyz)) or np.any(camera_xyz[:, 2] <= 0):
        raise ValueError("Expected the entire object in front of the camera at frame zero")
    image_xyz = camera_xyz @ intrinsics.T
    pixels = np.rint(image_xyz[:, :2] / image_xyz[:, 2:]).astype(np.int32)
    silhouette = np.zeros((height, width), dtype=np.uint8)
    for indices in faces:
        cv2.fillPoly(silhouette, [pixels[indices]], 255)
    if not np.any(silhouette):
        raise ValueError("Object projects outside the video")
    cv2.drawContours(frame, contours(mask), -1, (0, 200, 0), 3)
    cv2.drawContours(frame, contours(silhouette), -1, (255, 0, 255), 1)
    if not cv2.imwrite(str(output), frame):
        raise OSError(f"Could not write {output}")
    print(f"{sequence}  {width} x {height}\nInput: {(render if conditioning_only else video).resolve()}\nScene: {scene_id}\nMesh: {mesh.resolve()}")
    face_kind = "polygons" if static_terrain else "triangles"
    print(f"{len(faces)} {face_kind}, focal {focal:.2f} px, {background_count} background depth pixels")
    print(f"Overlay: {output.resolve()} (green mask, magenta projected mesh)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("generation", type=Path)
    parser.add_argument("sequence", help="dataset/category/clip")
    parser.add_argument("overlay", type=Path, help="output image; its parent must exist")
    parser.add_argument("--scene-id", help="basename of the conditioning frame and calibration; defaults to clip")
    parser.add_argument("--conditioning-only", action="store_true", help="check the rendered PNG before video generation")
    parser.add_argument("--static-terrain", action="store_true", help="accept the original terrain collider polygons")
    args = parser.parse_args()
    check(args.generation, args.sequence, args.overlay, scene_id=args.scene_id,
          conditioning_only=args.conditioning_only, static_terrain=args.static_terrain)
