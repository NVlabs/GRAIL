"""Hash-bound native SOMA03 fitting with immutable static terrain.

This is the locomotion numerical path, not manipulation contact/placement.
Run an initial-geometry probe with --iterations 0, author native support from
that immutable probe, then run the single native fit into a new directory.
Positive-iteration fitting requires the reviewed finite terrain support manifest.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np

from grail.agentic_kinematic.hoi_stages import bind, dump
from grail.optimization.ground_config import (
    bind_scene_support_losses,
    validate_foot_support_config,
)


def load_case(path):
    path = Path(path).resolve(strict=True)
    case = json.loads(path.read_text())
    required = {"schema_version", "inventory", "backend", "human_model", "optimization_config"}
    if set(case) != required or case["schema_version"] != 1:
        raise ValueError("Expected explicit static locomotion case fields")
    if case["backend"] != "soma03":
        raise ValueError("Native locomotion requires soma03")
    if case["human_model"].get("body_model") != case["backend"]:
        raise ValueError("Human model and backend must agree")
    inventory_path = (path.parent / case["inventory"]).resolve(strict=True)
    inventory = json.loads(inventory_path.read_text())
    if (
        inventory.get("schema") != "grail.static_locomotion_inputs.v1"
        or inventory.get("is_static_obj") is not True
    ):
        raise ValueError("Expected explicitly static locomotion inputs")
    n, fps = inventory["frame_count"], inventory["fps"]
    if (
        type(n) is not int
        or n < 2
        or isinstance(fps, bool)
        or not isinstance(fps, (int, float))
        or not math.isfinite(fps)
        or fps <= 0
    ):
        raise ValueError("Invalid source timing")
    if len(inventory.get("sequence", "").split("/")) != 3:
        raise ValueError("sequence must be dataset/category/video")
    scale = np.asarray(inventory.get("original_object_scale"), dtype=float)
    if scale.size not in (1, 3) or not np.isfinite(scale).all() or (scale <= 0).any():
        raise ValueError("Original object scale must contain one or three positive values")
    files = {}
    for name, record in inventory["files"].items():
        target = (inventory_path.parent / record["path"]).resolve(strict=True)
        if not target.is_relative_to(inventory_path.parent):
            raise ValueError(f"Input record escapes inventory directory: {name}")
        actual = bind(target)
        if actual["sha256"] != record["sha256"] or actual["bytes"] != record["bytes"]:
            raise ValueError(f"Changed input: {name}")
        files[name] = actual
    required_inputs = (
        "source_video",
        "object_mesh",
        "object_poses",
        "render_config",
        "mask",
        "depth",
        "depth_metadata",
        f"{case['backend']}/human_motion",
    )
    for name in required_inputs:
        if name not in files:
            raise ValueError(f"Missing bound input {name}")
    if "placement" in inventory or any("contact_map" in name for name in files):
        raise ValueError("Static locomotion has no manipulation placement/contact map")
    case["optimization_config"] = str(
        (path.parent / case["optimization_config"]).resolve(strict=True)
    )
    return case, inventory, files, bind(inventory_path)


def build_config(config, human_model, iterations=None, support_manifest=None, *, source_fps=None):
    """Resolve the single native terrain fit from the config-family losses."""
    cfg = copy.deepcopy(config["optimization"])
    if validate_foot_support_config(cfg) != "terrain":
        raise ValueError("Native locomotion requires the terrain config family")
    if set(cfg["opt_stage_specs"]) != {"init_opt"}:
        raise ValueError("Static locomotion requires the native init_opt configuration")
    stage = cfg["opt_stage_specs"]["init_opt"]
    if iterations is None:
        iterations = stage["niter"]
    if type(iterations) is not int or iterations < 0:
        raise ValueError("iterations/niter must be a nonnegative integer")
    if any(
        name
        not in {"human_trans_global", "human_trans_res", "human_pose_res", "obj_t_res", "obj_R_res"}
        for name in stage["opt_vars"]
    ):
        raise ValueError("Unsupported editable controls for static locomotion")
    stage["opt_vars"] = {k: v for k, v in stage["opt_vars"].items() if k.startswith("human_")}
    if not stage["opt_vars"]:
        raise ValueError("No editable human controls")
    for key in ("contact", "phase_contact_temporal", "human_foot_ground"):
        if key in stage["loss_cfg"]:
            raise ValueError(f"Base locomotion configuration must not contain {key}")
    if (
        cfg.get("reviewed_contacts")
        or cfg.get("hoi_mode")
        or cfg.get("object_stationary_intervals")
    ):
        raise ValueError("Manipulation reviewed-contact mode is unsupported")
    stage["niter"] = iterations
    stage["loss_cfg"] = bind_scene_support_losses(
        stage["loss_cfg"], support_manifest, source_fps=source_fps, probe=iterations == 0
    )
    cfg.update(
        human_model=copy.deepcopy(human_model),
        is_static_obj=True,
        vis_cfg={"enable": False},
        eval={},
        output_smoothing=False,
    )
    return cfg


def validate_static_poses(poses, frame_count):
    poses = np.asarray(poses)
    if poses.shape != (frame_count, 4, 4) or not np.isfinite(poses).all():
        raise ValueError("Static poses must be finite [T,4,4]")
    # Historical static tracks were formed through a float32 4x4 inverse;
    # one-ULP homogeneous-row residue is expected. Validate without rewriting
    # the input matrices, and retain the separate exact stationarity check.
    if not np.allclose(poses[:, 3], [0, 0, 0, 1], atol=1e-6, rtol=0):
        raise ValueError("Invalid homogeneous object pose")
    r = poses[:, :3, :3]
    if not np.allclose(r @ r.transpose(0, 2, 1), np.eye(3), atol=1e-5, rtol=0) or not np.allclose(
        np.linalg.det(r), 1, atol=1e-5, rtol=0
    ):
        raise ValueError("Invalid object rotation")
    if not np.array_equal(poses, np.broadcast_to(poses[0], poses.shape)):
        raise ValueError("Terrain must be exactly static across the source timeline")


def soma_editable_mask(model, device):
    import torch

    keep = torch.zeros(77, dtype=torch.bool, device=device)
    keep[[i + 1 for i in model.BODY_POSE_INDICES]] = True
    keep[0] = False
    return keep


def freeze_soma_pose_channels(model, device):
    """Restore root/fingers/wrists before FK, even at zero rotation residual."""
    import torch

    editable = soma_editable_mask(model, device)
    original_apply = model.apply_pose_residuals

    def apply(motion, body, hands, count):
        original = motion["poses"].reshape(count, 77, 3).clone()
        result = original_apply(motion, body, hands, count)
        result["poses"] = torch.where(
            editable[None, :, None], result["poses"].reshape(count, 77, 3), original
        )
        return result

    model.apply_pose_residuals = apply
    return editable


def freeze_native_pose_channels(model, device):
    """Freeze native root/finger/face fields without changing the body layout."""
    if model.model_type != "soma03":
        raise ValueError("Locomotion requires a native soma03 model")
    return freeze_soma_pose_channels(model, device)


def native_model_assets(model, human_model, character_name):
    """Bind the native runtime, geometry and selected identity."""
    from grail.models.soma_x_runtime import official_runtime_inputs

    if model.model_type != "soma03":
        raise ValueError("Locomotion requires a native soma03 model")
    paths, _ = official_runtime_inputs(model.model.asset_root)
    identity = human_model.get("soma_identity_params_path")
    if identity:
        paths += [Path(identity), Path(identity).with_suffix(".json")]
    return paths


def support_binding(
    inventory_binding, source_binding, faces, vertex_count, regions, *, model_type="soma03"
):
    if model_type != "soma03":
        raise ValueError("Unsupported support model type")
    return {
        "schema": "grail.static_locomotion_support_binding.v1",
        "inventory_sha256": inventory_binding["sha256"],
        "source_video_sha256": source_binding["sha256"],
        "model_type": model_type,
        "human_vertex_count": int(vertex_count),
        "human_faces_sha256": hashlib.sha256(np.asarray(faces, dtype="<i8").tobytes()).hexdigest(),
        "foot_regions_sha256": hashlib.sha256(
            json.dumps(regions, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
    }


def validate_support_reference(manifest, vertices):
    """Bind sole selection to the same initial native shape, pose and placement."""
    expected = (
        manifest.get("source", {})
        .get("material_remapping", {})
        .get("reference_vertices_sha256_float32_c")
    )
    if not isinstance(expected, str) or len(expected) != 64:
        raise ValueError("Support manifest requires the initial-reference vertex hash")
    vertices = np.asarray(vertices, dtype="<f4")
    if vertices.ndim != 3 or vertices.shape[-1] != 3 or not np.isfinite(vertices).all():
        raise ValueError("Initial support-reference vertices must be finite [T,V,3]")
    actual = hashlib.sha256(vertices.tobytes(order="C")).hexdigest()
    if actual != expected:
        raise ValueError("Support material selection belongs to different initial native geometry")
    return dict(reference_vertices_sha256_float32_c=actual, matches_authored_reference=True)


def validate_support_reference_replay(
    manifest_path, reference_path, current_path, *, model_type="soma03"
):
    """Verify immutable authored reference bytes, then bound repeat-FK roundoff."""
    manifest = json.loads(Path(manifest_path).read_text())
    receipt_path = Path(manifest_path).parent / "receipt.json"
    receipt = json.loads(receipt_path.read_text())
    manifest_record, reference_record = bind(manifest_path), bind(reference_path)
    expected = [
        value
        for key, value in receipt.get("inputs", {}).items()
        if Path(key).name == "initial_geometry.npz"
    ]
    receipt_matches_model = (
        model_type == "soma03"
        and receipt.get("schema") == "grail.soma_support_remapping_receipt.v1"
    ) or (
        receipt.get("schema") == "grail.static_support_remapping_receipt.v1"
        and receipt.get("model_type") == model_type
    )
    if (
        not receipt_matches_model
        or receipt.get("output", {}).get("sha256") != manifest_record["sha256"]
        or len(expected) != 1
        or expected[0] != reference_record["sha256"]
    ):
        raise ValueError("Saved support reference or manifest differs from its authored receipt")
    tolerances = {"human_vertices": 1e-5, "joints": 1e-5, "body_2d": 0.002, "hand_2d": 0.002}
    deltas = {}
    with (
        np.load(reference_path, allow_pickle=False) as reference,
        np.load(current_path, allow_pickle=False) as current,
    ):
        if set(reference.files) != set(current.files):
            raise ValueError("Initial support-reference geometry fields changed")
        authored = validate_support_reference(manifest, reference["human_vertices"])
        for key in reference.files:
            original, repeated = reference[key], current[key]
            if (
                original.shape != repeated.shape
                or not np.isfinite(original).all()
                or not np.isfinite(repeated).all()
            ):
                raise ValueError(f"Invalid repeated initial geometry: {key}")
            delta = float(
                np.abs(original.astype(np.float64) - repeated.astype(np.float64)).max(initial=0)
            )
            deltas[key] = delta
            if key in tolerances:
                if delta > tolerances[key]:
                    raise ValueError(
                        f"Repeated initial geometry exceeds roundoff tolerance: {key} ({delta})"
                    )
            elif not np.array_equal(original, repeated):
                raise ValueError(f"Frozen initial reference field changed: {key}")
    return dict(
        **authored,
        reference_geometry=reference_record,
        remapping_receipt=bind(receipt_path),
        comparison_geometry=bind(current_path),
        maximum_absolute_differences=deltas,
        tolerated_repeat_fk_fields=tolerances,
        other_fields_exact=True,
    )


def run(
    case_path,
    output,
    *,
    iterations=None,
    support_manifest=None,
    support_reference_geometry=None,
    depth_reference=None,
    device="cuda",
):
    import cv2
    import joblib
    import torch
    import yaml

    from grail.core.io import (
        load_character_data,
        load_human_motion_data,
        load_mesh,
        load_object_pose_data,
    )
    from grail.core.torch_utils import tensor_to_numpy
    from grail.optimization.foot_ground import FOOT_PARTS, foot_region_indices
    from grail.optimization.hoi_optimizer import HOIOptimizer
    from grail.optimization.loss_computer import LossComputer
    from grail.optimization.scene_support import SceneSupportConstraint
    from grail.preprocessing.preprocess import load_depth_from_cache, load_masks_from_cache
    from grail.rendering.camera import transform_pose_c2w

    case, inventory, inputs, inventory_record = load_case(case_path)
    backend = case["backend"]
    if (support_manifest is None) != (support_reference_geometry is None):
        raise ValueError(
            "Support fitting requires both manifest and original probe reference geometry"
        )
    if depth_reference is not None and support_reference_geometry is None:
        raise ValueError("Shared depth reference requires a bound native probe parity check")
    manifest = json.loads(Path(support_manifest).read_text()) if support_manifest else None
    cfg = build_config(
        yaml.safe_load(Path(case["optimization_config"]).read_text()),
        case["human_model"],
        iterations,
        manifest,
        source_fps=inventory["fps"],
    )
    stage = cfg["opt_stage_specs"]["init_opt"]
    iterations = stage["niter"]
    support = stage["loss_cfg"].get("human_scene_support")
    transition = stage["loss_cfg"].get("human_scene_support_temporal")
    transition_origin = "yaml" if transition is not None else "disabled_in_effective_config"
    output = Path(output).resolve()
    protected = [
        Path(case_path).resolve(),
        Path(case["optimization_config"]),
        Path(inventory_record["path"]),
        *[Path(r["path"]) for r in inputs.values()],
    ]
    if support_manifest:
        protected += [
            Path(support_manifest).resolve(),
            Path(support_reference_geometry).resolve(),
            Path(support_manifest).resolve().parent / "receipt.json",
        ]
    for extra in (depth_reference,):
        if extra is not None:
            protected.append(Path(extra).resolve(strict=True))
    if output.exists() or any(output == p or output in p.parents for p in protected):
        raise ValueError("Use a new output directory outside protected inputs")
    output.mkdir(parents=True)
    started = time.monotonic()
    dump(output / "completion.json", {"status": "RUNNING", "motion_accepted": False})
    dump(
        output / "inputs.json",
        {
            "case": bind(case_path),
            "inventory": inventory_record,
            "files": inputs,
            "optimization_config": bind(case["optimization_config"]),
            "support_manifest": bind(support_manifest) if support_manifest else None,
            "support_reference_geometry": (
                bind(support_reference_geometry) if support_reference_geometry else None
            ),
            "support_transition": {"origin": transition_origin, "settings": transition},
            "support_binding_mode": (
                "geometry_probe_without_support" if iterations == 0 and manifest is None else "yaml"
            ),
            "depth_reference": bind(depth_reference) if depth_reference else None,
            "predictors": inventory.get("predictors", "HASH_BOUND_PREDICTIONS"),
            "runner": bind(__file__),
        },
    )
    torch.set_num_threads(4)
    torch.manual_seed(20261014)
    if device.startswith("cuda"):
        torch.cuda.manual_seed_all(20261014)
    np.random.seed(20261014)
    n, fps = inventory["frame_count"], float(inventory["fps"])

    def path(role):
        return inputs[role]["path"]

    class StaticOptimizer(HOIOptimizer):
        def _load_video_frames(self, video):
            directory = output / "source_frames"
            directory.mkdir()
            capture = cv2.VideoCapture(video)
            actual_fps = capture.get(cv2.CAP_PROP_FPS)
            frames = []
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                target = directory / f"{len(frames):06d}.jpg"
                if not cv2.imwrite(str(target), frame):
                    raise RuntimeError("Source frame export failed")
                frames.append(str(target))
            capture.release()
            if len(frames) != n or abs(actual_fps - fps) > 1e-3:
                raise ValueError("Source video timing changed")
            return frames, fps, n

        def _load_masks(self, video_id):
            masks = load_masks_from_cache(path("mask"))
            if set(masks) != set(range(n)):
                raise ValueError("Masks must cover the unchanged source timeline")
            return {i: masks[i][1] for i in range(n)}, {i: masks[i][0] for i in range(n)}

        def _load_object(self, mesh, scale, poses, cam_r, cam_t):
            if not np.allclose(
                np.asarray(tensor_to_numpy(scale)).reshape(-1),
                np.asarray(inventory["original_object_scale"]).reshape(-1),
                atol=1e-7,
                rtol=0,
            ):
                raise ValueError("Original terrain scale changed")
            vertices, faces, _ = load_mesh(
                mesh, mesh_scale=scale, target_num_verts=None, device=device
            )
            incam = load_object_pose_data(poses, to_tensor=True, device=device)
            validate_static_poses(tensor_to_numpy(incam), n)
            world = transform_pose_c2w(incam, cam_r, cam_t)
            validate_static_poses(tensor_to_numpy(world), n)
            # Bind the known static track to its original render, not a fitted CAD pose.
            import pickle

            with open(path("render_config"), "rb") as stream:
                render = pickle.load(stream)
            if not np.allclose(
                tensor_to_numpy(world[0, :3, :3]), render["obj_R"], atol=2e-5, rtol=0
            ) or not np.allclose(
                tensor_to_numpy(world[0, :3, 3]),
                np.asarray(render["obj_t"]).reshape(3),
                atol=2e-5,
                rtol=0,
            ):
                raise ValueError("Static terrain pose differs from original rendering")
            return vertices, faces, incam, world

        def _detect_interaction(self, *args):
            return n, n, True

        def _load_depth(self, video_id):
            from grail.core.depth_config import depth_cache_metadata_path

            if Path(depth_cache_metadata_path(path("depth"))).resolve() != Path(
                path("depth_metadata")
            ):
                raise ValueError("Bound depth metadata is not the cache sidecar")
            values = load_depth_from_cache(path("depth"), device=device)
            if len(values) != n:
                raise ValueError("Depth does not cover the source timeline")
            return values

        def _load_motion(self, hmr_file, render_config_file, cam_r, cam_t, align_frame):
            global_motion = load_human_motion_data(
                hmr_file, is_global=True, to_tensor=True, device=device
            )
            incam_motion = load_human_motion_data(
                hmr_file, is_global=False, to_tensor=True, device=device
            )
            if incam_motion.get("predicted_body_height") is None:
                raise ValueError(
                    f"Native {backend} prediction must record its predicted body height"
                )
            character = (
                load_character_data(path("character_data")) if "character_data" in inputs else None
            )
            shape = self.human_model.load_shape_params(self.cfg, self.character_name, device)
            kwargs = dict(
                cam_R=cam_r,
                cam_t=cam_t,
                align_frame=0,
                gt_shape_params=shape,
                gt_height=character.get("character_height") if character else None,
                device=device,
            )
            current = self.human_model.transform_global_motion(
                global_motion,
                incam_motion,
                use_global=bool(self.cfg.get("use_global", False)),
                **kwargs,
            )
            reference = self.human_model.transform_global_motion(
                global_motion, incam_motion, use_global=True, **kwargs
            )
            return current, reference, incam_motion["foot_contact_probs"]

        def forward(self, data, params, **kwargs):
            prediction = super().forward(data, params, **kwargs)
            prediction.obj.R = data.obj.poses[:, :3, :3].clone()
            prediction.obj.trans = data.obj.poses[:, :3, 3].clone()
            prediction.obj.verts_seq = data.obj.verts_seq.clone()
            return prediction

    opt = StaticOptimizer(
        inventory["sequence"], cfg, str(output / "cache"), str(output / "logs"), device=device
    )
    editable = freeze_native_pose_channels(opt.human_model, device)
    data = opt.init_data(
        path("source_video"),
        path(f"{backend}/human_motion"),
        path("object_mesh"),
        path("object_poses"),
        path("render_config"),
    )
    if data.frame_num != n or not data.is_static_obj:
        raise ValueError("Static source contract changed")
    code_root = Path(__file__).resolve().parents[1]
    code_paths = [
        "agentic_kinematic/hoi_locomotion.py",
        "optimization/hoi_optimizer.py",
        "optimization/loss_computer.py",
        "optimization/scene_support.py",
        "optimization/foot_ground.py",
        "optimization/ground_config.py",
        "models/human_model.py",
        "rendering/camera.py",
    ]
    code_paths += ["agentic_kinematic/depth_reference.py", "optimization/scene_support_temporal.py"]
    code_paths += ["models/soma03_model.py", "models/soma_model.py"]
    model_paths = native_model_assets(opt.human_model, case["human_model"], opt.character_name)
    model_records = [bind(p) for p in model_paths]
    dump(
        output / "model_code_binding.json",
        {
            "code": {p: bind(code_root / p) for p in code_paths},
            "model": model_records,
            "identity_provenance": getattr(opt.human_model, "identity_provenance", None),
        },
    )
    native_regions = foot_region_indices(opt.human_model)
    regions = {name: np.asarray(ids).tolist() for name, ids in zip(FOOT_PARTS, native_regions)}
    dump(output / "foot_regions.json", regions)
    with torch.no_grad():
        vertex_count = opt.human_model.generate_mesh(data.human.motion_data, require_grad=False)[
            0
        ].shape[1]
    binding = support_binding(
        inventory_record,
        inputs["source_video"],
        tensor_to_numpy(data.human.faces),
        vertex_count,
        regions,
        model_type=backend,
    )
    dump(output / "support_binding.json", binding)
    if manifest is not None and manifest.get("binding") != binding:
        raise ValueError("Support manifest is not bound to these native model observations/regions")
    joblib.dump(
        tensor_to_numpy(data.human.motion_data_global_init), output / "global_reference.pkl"
    )
    observations = dict(
        gt_body_2d=tensor_to_numpy(data.human.body_keypoints_seq),
        gt_hand_2d=tensor_to_numpy(data.human.hand_keypoints_seq),
        foot_contact_probs=tensor_to_numpy(data.human.foot_contact_probs),
    )
    np.savez_compressed(output / "source_observations.npz", **observations)
    camera_r, camera_t = opt.cameras.R.clone(), opt.cameras.T.clone()
    opt.init_params(data)
    opt.loss_computer = LossComputer(
        opt.cameras, opt.human_model, device, opt.num_body_joints, opt.logger
    )
    stage = opt.opt_stage_specs["init_opt"]
    dump(output / "effective_config.json", opt.cfg)

    def save(label):
        with torch.no_grad():
            pred = opt.forward(data, opt.params)
            before, after = data.human.motion_data, pred.human.motion_data
            frozen_fields = (
                "identity_coeffs",
                "scale_params",
                "scale",
                "left_hand_pose",
                "right_hand_pose",
            )
            checks = {
                key: np.array_equal(tensor_to_numpy(before[key]), tensor_to_numpy(after[key]))
                for key in frozen_fields
            }
            checks.update(
                frozen_native_joints_exact=bool(
                    torch.equal(
                        before["poses"].reshape(n, -1, 3)[:, ~editable],
                        after["poses"].reshape(n, -1, 3)[:, ~editable],
                    )
                ),
                terrain_rotation_exact=bool(torch.equal(pred.obj.R, data.obj.poses[:, :3, :3])),
                terrain_translation_exact=bool(
                    torch.equal(pred.obj.trans, data.obj.poses[:, :3, 3])
                ),
                terrain_geometry_exact=bool(torch.equal(pred.obj.verts_seq, data.obj.verts_seq)),
                camera_exact=bool(
                    torch.equal(camera_r, opt.cameras.R) and torch.equal(camera_t, opt.cameras.T)
                ),
                source_timing=len(after["poses"]) == n and float(after["mocap_frame_rate"]) == fps,
                finite_geometry=bool(torch.isfinite(pred.human.verts_seq).all()),
            )
            checks = {key: bool(value) for key, value in checks.items()}
            if not all(checks.values()):
                raise ValueError(f"Native invariant failure: {checks}")
            native = opt.get_optimized_data(data, pred, smooth=False)
            checks["exported_native_exact"] = all(
                np.array_equal(native["human_data"][key], tensor_to_numpy(after[key]))
                for key in ("poses", "trans", *frozen_fields)
            )
            checks["exported_terrain_exact"] = (
                np.array_equal(native["obj_data"]["obj_R"], tensor_to_numpy(pred.obj.R))
                and np.array_equal(native["obj_data"]["obj_t"], tensor_to_numpy(pred.obj.trans))
                and np.array_equal(native["obj_data"]["obj_scale"], tensor_to_numpy(data.obj.scale))
            )
            if not all(checks.values()):
                raise ValueError(f"Native export changed evaluated geometry: {checks}")
            native["meta"].update(
                model=backend,
                stage="static_locomotion",
                status="UNREVIEWED",
                source_fps=fps,
                output_smoothing=False,
            )
            joblib.dump(native, output / f"{label}.pkl")
            np.savez_compressed(
                output / f"{label}_geometry.npz",
                human_vertices=tensor_to_numpy(pred.human.verts_seq),
                human_faces=tensor_to_numpy(data.human.faces),
                object_vertices=tensor_to_numpy(pred.obj.verts_seq),
                object_faces=tensor_to_numpy(data.obj.faces),
                joints=tensor_to_numpy(opt.human_model.get_native_joints(after)),
                body_2d=tensor_to_numpy(pred.human.body_keypoints_seq),
                hand_2d=tensor_to_numpy(pred.human.hand_keypoints_seq),
                gt_body_keypoints=observations["gt_body_2d"],
                gt_hand_keypoints=observations["gt_hand_2d"],
                foot_contact_probs=observations["foot_contact_probs"],
                camera_R=tensor_to_numpy(camera_r),
                camera_T=tensor_to_numpy(camera_t),
                focal=float(data.camera.focal_length),
                image_size=[data.camera.frame_height, data.camera.frame_width],
            )
            if manifest is not None:
                if label == "initial":
                    reference_binding = validate_support_reference_replay(
                        support_manifest,
                        support_reference_geometry,
                        output / "initial_geometry.npz",
                        model_type=backend,
                    )
                    dump(output / "support_reference_binding.json", reference_binding)
                SceneSupportConstraint(
                    pred.human.verts_seq, native_regions, manifest, robust_delta_m=0.005
                )
            return checks

    checks = save("initial")
    if iterations and stage["loss_cfg"].get("depth_pointcloud", {}).get("weight", 0) > 0:
        from grail.agentic_kinematic.depth_reference import (
            load_depth_reference,
            save_depth_reference,
        )

        # Bind fixed initial rays/visibility once, before either fitting branch.
        # The shared native probe guard above independently checks the geometry.
        context = {
            "frame_count": n,
            "human_vertex_count": vertex_count,
            "object_vertex_count": int(data.obj.verts_seq.shape[1]),
            "inputs": {name: {k: r[k] for k in ("sha256", "bytes")} for name, r in inputs.items()},
            "model": [{k: record[k] for k in ("sha256", "bytes")} for record in model_records],
            "depth_config": stage["loss_cfg"]["depth_pointcloud"],
            "reference_probe_sha256": (
                bind(support_reference_geometry)["sha256"] if support_reference_geometry else None
            ),
            "camera_R": tensor_to_numpy(camera_r).tolist(),
            "camera_T": tensor_to_numpy(camera_t).tolist(),
            "focal": float(data.camera.focal_length),
        }
        if depth_reference:
            cache, receipt = load_depth_reference(depth_reference, context, device=device)
            opt.loss_computer._depth_loss_cache = cache
            dump(
                output / "depth_reference_receipt.json",
                dict(mode="REUSED_FIXED_INITIAL", **receipt),
            )
        else:
            with torch.no_grad():
                prediction = opt.forward(data, opt.params)
                opt.loss_computer._build_depth_loss_cache(
                    data,
                    prediction,
                    num_gt_samples=stage["loss_cfg"]["depth_pointcloud"].get(
                        "num_gt_samples", 3000
                    ),
                )
            receipt = save_depth_reference(
                output / "depth_reference.npz", opt.loss_computer._depth_loss_cache, context
            )
            dump(
                output / "depth_reference_receipt.json", dict(mode="BUILT_FROM_INITIAL", **receipt)
            )
    optimizer, active = opt.init_opt(data, opt.params, stage)
    if not active or any(not key.startswith("human_") for key in active):
        raise ValueError("Only native human body/translation controls may be active")
    history = []
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    solve_started = time.monotonic()
    for iteration in range(iterations):
        optimizer.zero_grad(set_to_none=True)
        pred = opt.forward(data, opt.params)
        total, parts = opt.loss_computer.compute_loss(data, pred, stage["loss_cfg"])
        if not torch.isfinite(total):
            raise FloatingPointError("Nonfinite loss")
        total.backward()
        if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in active.values()):
            raise FloatingPointError("Nonfinite gradient")
        optimizer.step()
        history.append(
            dict(
                iteration=iteration + 1,
                total=float(total.detach()),
                losses=parts,
                seconds=time.monotonic() - solve_started,
            )
        )
        if iteration == 0 or (iteration + 1) % 50 == 0:
            dump(output / "progress.json", history[-1])
            print(json.dumps(history[-1]), flush=True)
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    solve_seconds = time.monotonic() - solve_started
    checks = save("result")
    dump(output / "guards.json", checks)
    dump(output / "history.json", history)
    cache = opt.loss_computer._depth_loss_cache
    if cache is not None:
        depth_cache_binding = {}
        for group, values in cache.items():
            depth_cache_binding[group] = {}
            for frame, value in sorted(values.items()):
                array = np.ascontiguousarray(tensor_to_numpy(value))
                depth_cache_binding[group][str(frame)] = dict(
                    shape=list(array.shape),
                    dtype=str(array.dtype),
                    sha256=hashlib.sha256(array.tobytes()).hexdigest(),
                )
        dump(output / "depth_sampling_binding.json", depth_cache_binding)
    completion = dict(
        status="OPTIMIZATION_COMPLETE_REVIEW_PENDING" if iterations else "PREFLIGHT_ONLY",
        stage="static_locomotion",
        variant=(
            "transition"
            if transition is not None
            else ("support" if support is not None else "control")
        ),
        support_transition_origin=transition_origin,
        backend=backend,
        iterations=iterations,
        checks=checks,
        solve_seconds=solve_seconds,
        elapsed_seconds=time.monotonic() - started,
        motion_accepted=False,
        output=bind(output / "result.pkl"),
    )
    dump(output / "completion.json", completion)
    return completion


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--iterations", type=int, help="Override YAML niter; use 0 for a geometry probe"
    )
    parser.add_argument("--support-manifest")
    parser.add_argument("--support-reference-geometry")
    parser.add_argument(
        "--depth-reference",
        help="Reuse bound baseline initial rays/visibility NPZ in a paired experiment",
    )
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    print(
        json.dumps(
            run(
                args.case,
                args.output,
                iterations=args.iterations,
                support_manifest=args.support_manifest,
                support_reference_geometry=args.support_reference_geometry,
                depth_reference=args.depth_reference,
                device=args.device,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
