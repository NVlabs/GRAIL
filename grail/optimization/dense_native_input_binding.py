"""Optional original-input identity for dense contact; no attraction targets."""

import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import torch


def array(value):
    return value.detach().cpu().numpy() if torch.is_tensor(value) else np.asarray(value)


def exact(actual, expected, label, *, flatten=False, dtype=True):
    a, b = array(actual), np.asarray(expected)
    if flatten:
        a, b = a.reshape(-1), b.reshape(-1)
    if a.shape != b.shape or (dtype and a.dtype != b.dtype) or not np.array_equal(a, b):
        raise ValueError("Dense native binding mismatch: " + label)


def fingerprint(value):
    a = np.ascontiguousarray(array(value))
    return dict(
        shape=list(a.shape), dtype=str(a.dtype), sha256=hashlib.sha256(a.tobytes()).hexdigest()
    )


def bound_path(item):
    if not isinstance(item, dict) or set(item) != {"path", "sha256"}:
        raise ValueError("Dense binding assets require explicit path and SHA256")
    p = Path(item["path"])
    if hashlib.sha256(p.read_bytes()).hexdigest() != item["sha256"]:
        raise ValueError("Dense bound file changed: " + str(p))
    return p


class NativeDenseInputBinding:
    """Checks immutable input files once, live original arrays/cameras every call."""

    def __init__(self, settings):
        option = settings["native_input_binding"]
        if not isinstance(option, dict) or set(option) != {
            "path",
            "sha256",
            "sequence",
            "prediction_id",
        }:
            raise ValueError("Explicit dense native binding path/hash/sequence/prediction required")
        binding_path = bound_path({k: option[k] for k in ["path", "sha256"]})
        spec = json.loads(binding_path.read_text())
        if (
            spec.get("schema") != "dense_native_input_binding_v001"
            or spec.get("sequence") != option["sequence"]
            or spec.get("prediction_id") != option["prediction_id"]
            or type(spec.get("frame_count")) is not int
            or spec["frame_count"] < 1
            or not isinstance(spec.get("source_fps"), (int, float))
            or not 0 < spec["source_fps"] < float("inf")
            or not isinstance(spec.get("saved_motion_fps"), (int, float))
            or not 0 < spec["saved_motion_fps"] < float("inf")
        ):
            raise ValueError("Dense explicit sequence/prediction/timeline differs")
        prediction = Path(settings["prediction_dir"])
        if prediction.name != spec["prediction_id"]:
            raise ValueError("Dense prediction directory identity differs")
        paths = {
            "canonical": Path(
                settings.get(
                    "canonical_geometry", prediction / "geometry_audit/fixed014_dense_geometry.npz"
                )
            ),
            "anatomy": Path(
                settings.get(
                    "anatomy_path", prediction / "geometry_audit/dense_anatomical_candidates.npz"
                )
            ),
            "semantic": prediction / "prediction/semantic_patch_fields.npz",
            "surface": prediction / "prediction/dense_object_surface.npz",
            "metadata": prediction / "prediction/prediction_metadata.json",
        }
        metadata = json.loads(paths["metadata"].read_text())
        for row in metadata.get("optimization_interfaces", []):
            dataset = row.get("surface_dataset")
            if dataset is not None:
                path = prediction / dataset["path"]
                if not path.resolve().is_relative_to(prediction.resolve()):
                    raise ValueError("Dense dataset escapes prediction directory")
                paths["dataset:" + row["name"]] = path
                if (
                    spec["assets"].get("dataset:" + row["name"], {}).get("sha256")
                    != dataset["sha256"]
                ):
                    raise ValueError("Dense dedicated dataset binding differs")
        required = set(paths) | {"original_motion", "model", "scene", "source_video"}
        if set(spec["assets"]) != required:
            raise ValueError(
                "Dense binding must cover exactly consumed files and original references"
            )
        for name, item in spec["assets"].items():
            p = bound_path(item)
            if name in paths and p.resolve() != paths[name].resolve():
                raise ValueError("Dense bound consumed path differs: " + name)
            paths[name] = p
        if metadata["frame_count"] != spec["frame_count"] or metadata["fps"] != spec["source_fps"]:
            raise ValueError("Dense metadata source timing differs")
        if (
            "metadata_sequence_alias" not in spec
            or metadata.get("sequence") != spec["metadata_sequence_alias"]
        ):
            raise ValueError("Dense metadata sequence contradicts explicit legacy alias")
        sequence = Path(spec["sequence"])
        video = paths["source_video"]
        if (
            not sequence.parts
            or sequence.is_absolute()
            or ".." in sequence.parts
            or video.suffix != ".mp4"
            or video.with_suffix("").parts[-len(sequence.parts) :] != sequence.parts
        ):
            raise ValueError("Dense full sequence differs from exact source-video path")
        if (
            metadata.get("source_video") is not None
            and Path(metadata["source_video"]).resolve() != paths["source_video"].resolve()
        ):
            raise ValueError("Dense metadata source video contradicts binding")
        self.spec, self.paths = copy.deepcopy(spec), paths
        self.provenance = dict(
            binding_path=str(binding_path),
            binding_sha256=option["sha256"],
            sequence=spec["sequence"],
            metadata_sequence_alias=spec["metadata_sequence_alias"],
            prediction_id=spec["prediction_id"],
            assets=copy.deepcopy(spec["assets"]),
            historical_fixed_motion_metadata=metadata.get("fixed_motion"),
            original_motion_rule="Explicit bound original motion; historical fixed_motion metadata never selects native references",
            current_targets="Only canonical local geometry is passed to the unchanged dense loss; no saved world target",
            tracking_mesh_rule="Simplified tracking mesh is not indexed by canonical original object IDs",
            live_checks=0,
            calibration_probe_count_per_check=9,
        )

    def initialize(
        self,
        data,
        human_model,
        cameras,
        canonical_vertices,
        canonical_faces,
        anatomy_faces,
        anatomy_rest,
    ):
        if human_model is None or cameras is None:
            raise ValueError(
                "Bound dense input requires actual native human_model and LossComputer cameras"
            )
        with np.load(self.paths["original_motion"], allow_pickle=False) as z:
            self.original = {
                k: z[k].copy()
                for k in (
                    "poses",
                    "trans",
                    "left_hand_pose",
                    "right_hand_pose",
                    "betas",
                    "obj_R",
                    "obj_t",
                    "human_scale",
                    "obj_scale",
                    "fps",
                )
            }
        with np.load(self.paths["model"], allow_pickle=False) as z:
            self.model = {
                k: z[k].copy()
                for k in ("v_template", "faces", "parents", "lbs_weights", "posedirs")
            }
        with np.load(self.paths["scene"], allow_pickle=False) as z:
            self.scene = {
                k: z[k].copy()
                for k in (
                    "object_vertices",
                    "object_faces",
                    "object_scale",
                    "camera_R",
                    "camera_t",
                    "focal_length_px",
                    "frame_width",
                    "frame_height",
                    "source_video_fps",
                )
            }
        scaled = self.scene["object_vertices"] * self.scene["object_scale"].reshape(3)
        canonical = np.asarray(canonical_vertices)
        if canonical.dtype.kind != "f" or not np.isfinite(canonical).all():
            raise ValueError("Dense canonical vertices must be finite floating point")
        exact(
            canonical,
            scaled.astype(canonical.dtype),
            "scene-scaled canonical vertices at original storage dtype",
        )
        exact(
            canonical_faces,
            self.scene["object_faces"],
            "canonical original object faces",
            dtype=False,
        )
        exact(anatomy_faces, self.model["faces"], "anatomy original human faces", dtype=False)
        exact(anatomy_rest, self.model["v_template"], "anatomy baked human rest")
        if (
            len(self.original["poses"]) != self.spec["frame_count"]
            or float(self.original["fps"]) != self.spec["saved_motion_fps"]
            or float(self.scene["source_video_fps"]) != self.spec["source_fps"]
        ):
            raise ValueError("Bound original timeline differs")
        exact(
            self.original["obj_scale"],
            self.scene["object_scale"],
            "original scene object scale",
            flatten=True,
            dtype=False,
        )
        self.provenance["canonical_storage_dtype"] = str(canonical.dtype)
        self.provenance["scene_scaled_cast_exact"] = True
        self.provenance["canonical_full_object_vertices"] = len(canonical)
        self.provenance["canonical_full_object_faces"] = len(canonical_faces)
        # Model NPZ has no J_regressor. This guards subsequent mutation only;
        # it does not establish original-file provenance for the first-seen regressor.
        self.regressor_fingerprint = fingerprint(human_model.model.J_regressor)
        self.provenance["J_regressor_first_seen_fingerprint"] = copy.deepcopy(
            self.regressor_fingerprint
        )
        self.provenance["J_regressor_scope"] = (
            "First-seen mutation guard only; no original-file regressor proof or rest-product exactness claim"
        )
        self.validate_live(data, human_model, cameras)

    def validate_live(self, data, human_model, cameras):
        if human_model is None or cameras is None or data.frame_num != self.spec["frame_count"]:
            raise ValueError("Current dense native references/timeline unavailable")
        for key in ("poses", "trans", "left_hand_pose", "right_hand_pose", "betas"):
            exact(
                data.human.motion_data[key],
                self.original[key],
                "live original human " + key,
                flatten=(key == "betas"),
            )
        exact(array(data.obj.poses)[:, :3, :3], self.original["obj_R"], "live original object R")
        exact(array(data.obj.poses)[:, :3, 3], self.original["obj_t"], "live original object t")
        for key, current, expected in (
            ("human scale", data.human.motion_data["scale"], self.original["human_scale"]),
            ("object scale", data.obj.scale, self.original["obj_scale"]),
            ("saved motion FPS", data.human.motion_data["mocap_frame_rate"], self.original["fps"]),
        ):
            exact(current, expected, "live original " + key, flatten=True, dtype=False)
        native = human_model.model
        exact(data.human.faces, self.model["faces"], "live original human topology", dtype=False)
        exact(native.faces, self.model["faces"], "live model topology", dtype=False)
        for key in ("v_template", "parents", "lbs_weights", "posedirs"):
            exact(getattr(native, key), self.model[key], "live baked model " + key)
        if fingerprint(native.J_regressor) != self.regressor_fingerprint:
            raise ValueError("Dense first-seen native J_regressor changed")
        size = [int(self.scene["frame_width"]), int(self.scene["frame_height"])]
        if [data.camera.frame_width, data.camera.frame_height] != size or float(
            data.camera.focal_length
        ) != float(self.scene["focal_length_px"]):
            raise ValueError("Native camera scalar metadata differs")
        from grail.rendering.camera import project_world_to_screen

        optical = np.asarray(
            [
                [0, 0, -1],
                [0.25, 0, -1],
                [-0.25, 0, -1],
                [0, 0.2, -1],
                [0, -0.2, -1],
                [0.25, 0.2, -2],
                [-0.25, 0.2, -2],
                [0.25, -0.2, -3],
                [-0.25, -0.2, -3],
            ],
            float,
        )
        world = optical @ np.linalg.inv(self.scene["camera_R"]) + self.scene["camera_t"]
        points = torch.as_tensor(world, device=cameras.R.device, dtype=cameras.R.dtype)
        with torch.no_grad():
            cam = (points - points.new_tensor(self.scene["camera_t"])) @ points.new_tensor(
                self.scene["camera_R"]
            )
            if bool((cam[:, 2] >= -1e-6).any()):
                raise ValueError("Dense calibration probe crossed source camera plane")
            focal = float(self.scene["focal_length_px"])
            expected = (
                torch.stack(
                    [focal * cam[:, 0] / (-cam[:, 2]), -focal * cam[:, 1] / (-cam[:, 2])], -1
                )
                + points.new_tensor(size) / 2
            )
            actual = project_world_to_screen(points, cameras).reshape(-1, 3)[:, :2]
            if actual.shape != expected.shape:
                raise ValueError("Dense native camera batch changed")
            error = float((actual - expected).abs().max())
            if not np.isfinite(error) or error > 0.01:
                raise ValueError("Dense source/native camera projection differs")
        self.provenance["live_checks"] += 1
        self.provenance["native_camera_projection_max_error_px"] = error
        self.provenance["native_camera_projection_tolerance_px"] = 0.01
