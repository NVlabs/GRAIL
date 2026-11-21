"""Explicit optional original-human support schema adapter; optional installed component."""

import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from grail.optimization.human_region_support import HumanRegionSupportLoss, SupportOptions

_INPUT_ROLES = (
    "schema_path",
    "model_data_path",
    "scene_data_path",
    "original_motion_path",
    "evidence_schema_path",
    "source_audit_path",
    "inventory_path",
    "dense_prediction_path",
)


def read_bound_support_schema(cfg, frame_count):
    """Validate sequence/evidence identity before decoding any geometry archive."""
    paths = {}
    for key in _INPUT_ROLES:
        p = Path(cfg[key])
        if hashlib.sha256(p.read_bytes()).hexdigest() != cfg["input_sha256"][key]:
            raise ValueError("Hand-hand input hash mismatch:" + key)
        paths[key] = p
    binding = json.loads(paths["schema_path"].read_text())
    if (
        binding["schema"] != "native_hand_hand_support_binding_v001"
        or binding["helper_schema"] != "current_human_region_support_v001"
        or binding["sequence"] != cfg["sequence"]
        or binding["frame_count"] != frame_count
        or binding["prediction_id"] != cfg["prediction_id"]
    ):
        raise ValueError("Explicit reviewed hand-hand sequence binding differs")
    roles = {
        "model": "model_data_path",
        "scene": "scene_data_path",
        "original_motion": "original_motion_path",
        "evidence_schema": "evidence_schema_path",
        "source_audit": "source_audit_path",
        "inventory": "inventory_path",
        "dense_prediction": "dense_prediction_path",
    }
    for name, key in roles.items():
        item = binding[name]
        if (
            Path(item["path"]).resolve() != paths[key].resolve()
            or item["sha256"] != cfg["input_sha256"][key]
        ):
            raise ValueError("Hand-hand bound asset differs:" + name)
    spec = json.loads(paths["evidence_schema_path"].read_text())
    if (
        spec["schema"] != "video01_existing_support_regions_v001"
        or spec["helper_schema"] != binding["helper_schema"]
        or spec["frame_count"] != frame_count
        or spec["source"] != binding["source_audit"]
        or spec["inventory"] != binding["inventory"]
        or spec["prediction010_dense_byte_reuse"] != binding["dense_prediction"]
        or binding["source_record_key"] != "01"
    ):
        raise ValueError("Reviewed Video01 hand-hand evidence differs")
    record = json.loads(paths["source_audit_path"].read_text())["records"][
        binding["source_record_key"]
    ]
    expected_video = "results/generation/videos_kling/" + binding["sequence"] + ".mp4"
    if (
        record["source_video"] != expected_video
        or record["frame_count"] != frame_count
        or record["fixed_motion"] != binding["original_motion"]["path"]
        or record["fixed_motion_sha256"] != binding["original_motion"]["sha256"]
    ):
        raise ValueError("Hand-hand source-derived sequence/original-motion mismatch")
    return spec, binding, paths


def spatial_prior(ids, values, vertex_count):
    ids, values = np.asarray(ids), np.asarray(values)
    if (
        ids.dtype.kind not in "iu"
        or ids.ndim != 1
        or ids.size == 0
        or len(np.unique(ids)) != len(ids)
        or ids.min() < 0
        or ids.max() >= vertex_count
        or values.shape != ids.shape
        or not np.isfinite(values).all()
        or np.any(values < 0)
    ):
        raise ValueError("Invalid nonempty original hand-hand spatial field")
    result = np.zeros(vertex_count)
    result[ids] = values
    return result


def validate_original_motion_identity(data, original):
    """Source-initialization reference proof; never compares optimized prediction."""

    def array(value):
        return value.detach().cpu().numpy() if torch.is_tensor(value) else np.asarray(value)

    record = {}
    # data.human.motion_data remains the ORIGINAL reference for native residuals.
    # Current body/object optimization parameters and pred meshes are separate.
    for key in ("poses", "trans", "left_hand_pose", "right_hand_pose", "betas"):
        actual, expected = array(data.human.motion_data[key]), original[key]
        if key == "betas":
            actual, expected = actual.reshape(-1), expected.reshape(-1)
        if (
            actual.shape != expected.shape
            or actual.dtype != expected.dtype
            or not np.array_equal(actual, expected)
        ):
            raise ValueError("Live original human initialization differs: " + key)
        record["human_" + key] = {
            "shape": list(actual.shape),
            "dtype": str(actual.dtype),
            "sha256": hashlib.sha256(np.ascontiguousarray(actual).tobytes()).hexdigest(),
        }
    for key, current, expected in (
        ("object_R", array(data.obj.poses)[:, :3, :3], original["obj_R"]),
        ("object_t", array(data.obj.poses)[:, :3, 3], original["obj_t"]),
    ):
        if (
            current.shape != expected.shape
            or current.dtype != expected.dtype
            or not np.array_equal(current, expected)
        ):
            raise ValueError("Live original object initialization differs: " + key)
        record[key] = {
            "shape": list(current.shape),
            "dtype": str(current.dtype),
            "sha256": hashlib.sha256(np.ascontiguousarray(current).tobytes()).hexdigest(),
        }
    for key, current, expected in (
        ("human_scale", array(data.human.motion_data["scale"]), original["human_scale"]),
        ("object_scale", array(data.obj.scale), original["obj_scale"]),
        ("saved_motion_fps", array(data.human.motion_data["mocap_frame_rate"]), original["fps"]),
    ):
        if not np.array_equal(np.asarray(current).reshape(-1), np.asarray(expected).reshape(-1)):
            raise ValueError("Live original scale/timing differs: " + key)
        record[key] = np.asarray(current).reshape(-1).tolist()
    return record


class HandHandSupportAdapter:
    def __init__(self, data, human_model, cameras, cfg, device, dtype):
        spec, binding, paths = read_bound_support_schema(cfg, data.frame_num)
        model_path, scene_path = paths["model_data_path"], paths["scene_data_path"]
        with np.load(paths["original_motion_path"], allow_pickle=False) as z:
            self.original_motion = {
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
        source_identity = validate_original_motion_identity(data, self.original_motion)
        with np.load(model_path, allow_pickle=False) as z:
            faces, rest = z["faces"].copy(), z["v_template"].copy()
        if not np.array_equal(faces, data.human.faces.detach().cpu().numpy()) or not np.array_equal(
            rest, human_model.model.v_template.detach().cpu().numpy()
        ):
            raise ValueError("Original hand-hand topology/baked rest differs")
        # Scene is an identity/calibration check, never an HH attraction target.
        with np.load(scene_path, allow_pickle=False) as z:
            self.camera_R = z["camera_R"].copy()
            self.camera_t = z["camera_t"].copy()
            self.focal = float(z["focal_length_px"])
            self.size = [int(z["frame_width"]), int(z["frame_height"])]
            if self.size != [720, 1280] or not np.array_equal(
                data.obj.scale.detach().cpu().numpy().reshape(3), z["object_scale"].reshape(3)
            ):
                raise ValueError("Original sequence calibration/scale differs")
        self.cameras = cameras
        self.calibration_checked = False
        options = dict(cfg["options"])
        if options.get("orientation_distance_scale_m") is None:
            raise ValueError("Fitting requires an explicitly selected coupled orientation length")
        interfaces = []
        for row in spec["interfaces"]:
            priors = {}
            for side in ["left", "right"]:
                priors[side] = spatial_prior(
                    row[side + "_original_vertex_ids"], row[side + "_spatial_prior"], len(rest)
                )
            interfaces.append(
                dict(
                    name=row["name"],
                    left_face_ids=row["left_face_ids"],
                    right_face_ids=row["right_face_ids"],
                    left_vertex_prior=priors["left"],
                    right_vertex_prior=priors["right"],
                )
            )
        self.helper = HumanRegionSupportLoss(
            human_rest_vertices=rest,
            human_faces=faces,
            interfaces=interfaces,
            interface_strengths=spec["interface_strengths"],
            enabled=True,
            options=SupportOptions(**options),
        ).to(device=device, dtype=dtype)
        supplied = cfg.get("frame_indices", list(range(data.frame_num)))
        if (
            not isinstance(supplied, list)
            or not supplied
            or any(type(x) is not int or not 0 <= x < data.frame_num for x in supplied)
            or len(set(supplied)) != len(supplied)
        ):
            raise ValueError("Explicit integral original hand-hand frame IDs required")
        self.frames = supplied
        self.frame_num = data.frame_num
        self.last_report = None
        self.provenance = {
            "schema": "native_hand_hand_support_adapter_v001",
            "binding": binding,
            "live_original_motion_identity": source_identity,
            "source_reference_rule": "Exact native source-initialization arrays; optimized pred/runtime arrays are not this identity proof",
            "inputs": {
                k: {"path": str(cfg[k]), "sha256": cfg["input_sha256"][k]} for k in _INPUT_ROLES
            },
            "original_frames": supplied,
            "frame_denominator": len(supplied),
            "source_evidence_rule": self.helper.provenance["strength_rule"],
            "helper": self.helper.provenance,
            "current_geometry": "Both surfaces from current full native human mesh; canonical IDs/bary/area only cached",
            "object_targets": False,
            "localization": "No HH pixel localization term; source008 visible-skin keeps its own original denominator/uncertainty",
            "fitting_value": "coupled joint_m2 only; gap/orientation remain diagnostic",
        }

    def __call__(self, data, pred):
        if data.frame_num != self.frame_num or len(pred.human.verts_seq) != self.frame_num:
            raise ValueError("Current native full timeline changed")
        current_identity = validate_original_motion_identity(data, self.original_motion)
        if current_identity != self.provenance["live_original_motion_identity"]:
            raise ValueError("Cached original motion identity changed")
        if not self.calibration_checked:
            from grail.rendering.camera import project_world_to_screen

            points = pred.human.verts_seq[self.frames[0], ::101]
            with torch.no_grad():
                cam = (points - points.new_tensor(self.camera_t)) @ points.new_tensor(self.camera_R)
                if bool((cam[:, 2] >= -1e-6).any()):
                    raise ValueError("HH calibrated probe crosses camera plane")
                xy = (
                    torch.stack(
                        [
                            self.focal * cam[:, 0] / (-cam[:, 2]),
                            -self.focal * cam[:, 1] / (-cam[:, 2]),
                        ],
                        -1,
                    )
                    + points.new_tensor(self.size) / 2
                )
                reference = project_world_to_screen(points, self.cameras).reshape(-1, 3)[:, :2]
                error = float((xy - reference).abs().max())
                if not np.isfinite(error) or error > 0.01:
                    raise ValueError("HH source/native camera differs")
            self.provenance["native_calibration_max_error_px"] = error
            self.calibration_checked = True
        self.last_report = self.helper(pred.human.verts_seq[self.frames], frame_indices=self.frames)
        if self.last_report["joint_m2"] is None:
            raise RuntimeError("Coupled fitting scalar unavailable")
        return self.last_report["joint_m2"]
