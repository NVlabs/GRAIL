"""Loss computation for HOI optimization, extracted from HOIOptimizer."""

import math

import torch
from pytorch3d.structures import Meshes
from pytorch3d.transforms import axis_angle_to_matrix

from grail.optimization.loss_terms import (
    bidirectional_chamfer_loss,
    ground_loss,
    huber_loss,
    keypoint_loss,
    l1_loss,
    penetration_loss,
    reg_loss,
    smoothness_loss,
)
from grail.rendering.camera import project_world_to_screen, unproject_depth_map_to_world


class LossComputer:
    """Computes all loss terms for HOI optimization."""

    def __init__(
        self,
        cameras,
        human_model,
        device,
        num_body_joints,
        logger,
    ):
        self.cameras = cameras
        self.human_model = human_model
        self.device = device
        self.num_body_joints = num_body_joints
        self.logger = logger
        self._depth_loss_cache = None
        self._dense_contact_cache = {}
        self._fixed_contact_cache = {}
        self._phase_contact_map_cache = {}
        self._phase_contact_temporal_cache = {}
        self._phase_contact_temporal_provenance = {}
        self._visible_skin_cache = {}
        self._visible_object_boundary_cache = {}
        self._contact_temporal_cache = {}
        self._surface_collision_cache = {}
        self._global_motion_reference_cache = None
        self._foot_ground_constraints = {}
        self._foot_ground_temporal_constraints = {}
        self._scene_support_constraints = {}
        self._scene_support_temporal_constraints = {}

    # ── Dispatch ─────────────────────────────────────────────────────────────

    def reset_foot_ground_temporal_reference(self):
        """Start a new numerical stage without changing labels or other loss caches.

        Regular HOI fitting reuses this object across stages. Call once before
        the next stage's first objective evaluation, never inside its iterations.
        The next foot-transition evaluation captures that stage's native seed.
        """
        self._foot_ground_temporal_constraints.clear()

    _LOSS_FN = {
        "contact": "_contact_loss",
        "hand_hand_support": "_hand_hand_support_loss",
        "visible_skin": "_visible_skin_loss",
        "visible_object_boundary": "_visible_object_boundary_loss",
        "contact_temporal": "_contact_temporal_loss",
        "phase_contact_temporal": "_phase_contact_temporal_loss",
        "surface_collision": "_surface_collision_loss",
        "keypoint_tracking": "_keypoint_tracking_loss",
        "ground": "_ground_loss",
        "human_global_init_reg": "_human_global_init_reg_loss",
        "human_smoothness": "_human_smoothness_loss",
        "human_traj_reg": "_human_traj_reg_loss",
        "human_pose_reg": "_human_pose_reg_loss",
        "human_foot_contact": "_human_foot_contact_loss",
        "human_foot_ground": "_human_foot_ground_loss",
        "human_foot_ground_temporal": "_human_foot_ground_temporal_loss",
        "human_scene_support": "_human_scene_support_loss",
        "human_scene_support_temporal": "_human_scene_support_temporal_loss",
        "verts_tracking": "_verts_tracking_loss",
        "obj_smoothness": "_obj_smoothness_loss",
        "obj_traj_reg": "_obj_traj_reg_loss",
        "obj_rot_reg": "_obj_rot_reg_loss",
        "depth_pointcloud": "_depth_pointcloud_loss",
        "obj_precontact_reg": "_obj_precontact_reg_loss",
        "penetration": "_penetration_loss",
    }

    def compute_loss(self, data, pred, loss_cfg):
        temporal_support = loss_cfg.get("human_scene_support_temporal", {})
        contact = loss_cfg.get("human_foot_contact", {})
        if contact.get("weight", 0) > 0 and contact.get("label_source") == "scene_support":
            support = loss_cfg.get("human_scene_support", {})
            if (
                support.get("weight", 0) <= 0
                or contact.get("manifest") != support.get("manifest")
                or contact.get("options", {}).get("tangent_blend_frames")
                != support.get("tangent_blend_frames")
            ):
                raise ValueError(
                    "Reviewed foot contact requires the same enabled scene-support manifest and blend"
                )
        if (
            temporal_support.get("weight", 0) > 0
            and loss_cfg.get("human_foot_ground_temporal", {}).get("weight", 0) > 0
        ):
            raise ValueError("Do not combine predicted foot-label and reviewed scene transitions")
        if temporal_support.get("weight", 0) > 0:
            support = loss_cfg.get("human_scene_support", {})
            if support.get("weight", 0) <= 0 or temporal_support.get("manifest") != support.get(
                "manifest"
            ):
                raise ValueError(
                    "Temporal foot support requires the same enabled scene-support manifest"
                )
            if (
                loss_cfg.get("human_foot_contact", {}).get("weight", 0) != 0
                and temporal_support.get("component_weights", {}).get("no_slip", 0) != 0
            ):
                raise ValueError(
                    "Do not combine reviewed scene no-slip with a separate GEM contact schedule"
                )
            if temporal_support.get("options", {}).get("tangent_blend_frames") != support.get(
                "tangent_blend_frames"
            ):
                raise ValueError(
                    "Scene support and temporal no-slip must share tangential blending"
                )
        total_loss = 0.0
        loss_dict = {}
        for loss_name, cfg in loss_cfg.items():
            weight = cfg["weight"]
            fn_name = self._LOSS_FN.get(loss_name)
            if fn_name is None:
                raise ValueError(f"Invalid loss name: {loss_name}")
            loss = getattr(self, fn_name)(data, pred, cfg, weight)
            total_loss += loss
            loss_dict[loss_name] = loss.item()
        return total_loss, loss_dict

    # ── Individual loss methods ──────────────────────────────────────────────


    def _hand_hand_support_loss(self, data, pred, cfg, weight):
        """Optional current HUMAN/HUMAN coupled support, never object contact."""
        if weight == 0 or not cfg.get("enabled", False):
            return pred.human.verts_seq.new_zeros(())
        import json
        import math

        if (
            type(cfg.get("enabled")) is not bool
            or isinstance(weight, bool)
            or not isinstance(weight, (float, int))
            or not math.isfinite(weight)
            or weight <= 0
        ):
            raise ValueError("Explicit enabled support and positive finite weight required")
        from grail.optimization.hand_hand_support_adapter import HandHandSupportAdapter

        if not hasattr(self, "_hand_hand_support_cache"):
            self._hand_hand_support_cache = {}
        key = (id(data), json.dumps(cfg, sort_keys=True))
        if key not in self._hand_hand_support_cache:
            self._hand_hand_support_cache[key] = HandHandSupportAdapter(
                data,
                self.human_model,
                self.cameras,
                cfg,
                pred.human.verts_seq.device,
                pred.human.verts_seq.dtype,
            )
        return weight * self._hand_hand_support_cache[key](data, pred)

    def _surface_collision_loss(self, data, pred, cfg, weight):
        """Optional original-hand area collision; no assets loaded when inactive."""
        if weight == 0 or not cfg.get("enabled", False):
            return pred.human.verts_seq.new_zeros(())
        if (
            not isinstance(cfg["enabled"], bool)
            or not isinstance(weight, (float, int))
            or not 0 < weight < float("inf")
        ):
            raise ValueError(
                "Surface collision requires explicit boolean enabled and a positive finite weight"
            )
        if set(cfg.get("options", {})) - {"tolerance_m", "max_edge_m"}:
            raise ValueError("Surface collision options are tolerance_m and max_edge_m only")
        import hashlib
        import json
        from pathlib import Path

        import numpy as np

        from grail.optimization.surface_collision import SurfaceCollisionLoss, canonical_hash

        key = (id(data), json.dumps(cfg, sort_keys=True))
        if key not in self._surface_collision_cache:

            def array(value):
                return value.detach().cpu().numpy() if torch.is_tensor(value) else np.asarray(value)

            def record(value):
                value = np.ascontiguousarray(array(value))
                return {
                    "shape": list(value.shape),
                    "dtype": str(value.dtype),
                    "sha256": hashlib.sha256(value.tobytes()).hexdigest(),
                }

            inputs = {}
            for name in ("scene_data_path", "model_data_path", "anatomy_path", "grid_path"):
                path = Path(cfg[name])
                digest = hashlib.sha256(path.read_bytes()).hexdigest()
                if digest != cfg["input_sha256"][name]:
                    raise ValueError(f"Surface collision input hash mismatch: {name}")
                inputs[name] = {"path": str(path.resolve()), "sha256": digest}
            with np.load(cfg["model_data_path"], allow_pickle=False) as model:
                faces, rest = model["faces"].copy(), model["v_template"].copy()
            with np.load(cfg["anatomy_path"], allow_pickle=False) as archive:
                anatomy = {
                    k: archive[k].copy()
                    for k in (
                        "human_faces",
                        "human_rest_vertices_m",
                        "left_complete_hand_original_face_ids",
                        "right_complete_hand_original_face_ids",
                    )
                }
            if not np.array_equal(array(data.human.faces), faces) or not np.array_equal(
                faces, anatomy["human_faces"]
            ):
                raise ValueError(
                    "Surface collision actual native/anatomy original human topology differs"
                )
            native_rest = array(self.human_model.model.v_template)
            if not np.array_equal(native_rest, rest) or not np.array_equal(
                rest, anatomy["human_rest_vertices_m"]
            ):
                raise ValueError(
                    "Surface collision actual baked native template differs; do not apply human scale twice"
                )
            tracking = {name: record(getattr(data.obj, name)) for name in ("verts", "faces")}
            if tracking != cfg["native_tracking_mesh"]:
                raise ValueError(
                    "Surface collision actual simplified tracking mesh identity differs"
                )
            with np.load(cfg["scene_data_path"], allow_pickle=False) as scene:
                if not np.array_equal(
                    array(data.obj.scale).reshape(3), scene["object_scale"].reshape(3)
                ):
                    raise ValueError(
                        "Surface collision exact native object scale differs from canonical scene"
                    )
                helper = SurfaceCollisionLoss(
                    human_faces=faces,
                    rest_vertices=rest,
                    expected_human_faces_sha256=canonical_hash(anatomy["human_faces"], "faces"),
                    expected_human_rest_sha256=canonical_hash(
                        anatomy["human_rest_vertices_m"], "vertices"
                    ),
                    hand_face_ids={
                        s: anatomy[s + "_complete_hand_original_face_ids"]
                        for s in ("left", "right")
                    },
                    grid_path=cfg["grid_path"],
                    expected_grid_sha256=cfg["input_sha256"]["grid_path"],
                    object_vertices_scaled=scene["object_vertices"]
                    * scene["object_scale"].reshape(3),
                    object_faces=scene["object_faces"],
                    object_scale=scene["object_scale"],
                    enabled=True,
                    **cfg.get("options", {}),
                ).to(device=pred.human.verts_seq.device, dtype=pred.human.verts_seq.dtype)
            helper.provenance["native_integration"] = {
                "inputs": inputs,
                "actual_native_tracking_mesh": tracking,
                "tracking_mesh_use": "Identity only; original IDs never index this simplified mesh.",
                "actual_human_faces": record(data.human.faces),
                "actual_baked_rest": record(native_rest),
                "camera_dependency": "None for object-local collision; native scene/camera remain unchanged.",
            }
            self._surface_collision_cache[key] = helper
        result = self._surface_collision_cache[key](
            pred.human.verts_seq, pred.obj.R, pred.obj.trans
        )
        return weight * result["total"]

    def _visible_skin_loss(self, data, pred, cfg, weight):
        """Explicit original-topology source skin observations; optional/off by omission."""
        return self._source_surface_loss(data, pred, cfg, weight, object_boundary=False)

    def _visible_object_boundary_loss(self, data, pred, cfg, weight):
        """Original-object live silhouette curves; distinct from contact labels."""
        return self._source_surface_loss(data, pred, cfg, weight, object_boundary=True)

    def _source_surface_loss(self, data, pred, cfg, weight, *, object_boundary):
        if weight == 0:
            return pred.human.verts_seq.sum() * 0
        helper = self._source_surface_helper(data, pred, cfg, object_boundary=object_boundary)
        return weight * helper(pred.human.verts_seq, pred.obj.R, pred.obj.trans)

    def _source_surface_helper(self, data, pred, cfg, *, object_boundary=False, observer=False):
        """Build a calibrated helper; support observers never enter an objective cache."""
        if observer and object_boundary:
            raise ValueError("Object-boundary helpers cannot be skin support observers")
        import hashlib
        import json
        from pathlib import Path

        import numpy as np

        from grail.optimization.visible_skin import VisibleSkinLoss
        from grail.rendering.camera import project_world_to_screen

        helper_class = VisibleSkinLoss
        cache = self._visible_skin_cache
        if observer:
            if not hasattr(self, "_visible_skin_observer_cache"):
                self._visible_skin_observer_cache = {}
            cache = self._visible_skin_observer_cache
        if object_boundary:
            from grail.optimization.visible_object_boundary import VisibleObjectBoundaryLoss

            helper_class = VisibleObjectBoundaryLoss
            cache = self._visible_object_boundary_cache

        key = (id(data), json.dumps(cfg, sort_keys=True))
        if key not in cache:
            scene_path = Path(cfg["scene_data_path"])
            record = json.loads(Path(cfg["observations_path"]).read_text())
            if record["scene_sha256"] != hashlib.sha256(scene_path.read_bytes()).hexdigest():
                raise ValueError("Visible skin observations use a different calibrated scene")
            if record["frame_count"] != data.frame_num:
                raise ValueError("Visible skin source timeline differs from native data")
            with np.load(cfg["model_data_path"], allow_pickle=False) as model:
                human_faces, rest = model["faces"].copy(), model["v_template"].copy()
            if not np.array_equal(human_faces, data.human.faces.detach().cpu().numpy()):
                raise ValueError("Visible skin original human topology differs")
            with np.load(scene_path, allow_pickle=False) as scene:
                if not np.allclose(
                    torch.as_tensor(data.obj.scale).detach().cpu().numpy().reshape(3),
                    scene["object_scale"].reshape(3),
                    atol=1e-7,
                    rtol=0,
                ):
                    raise ValueError("Source surface scene and native fixed object scale differ")
                for name, array in [
                    (
                        "original_object_faces_int64_sha256",
                        np.asarray(scene["object_faces"], dtype="<i8"),
                    ),
                    (
                        "original_object_vertices_scaled_float64_sha256",
                        np.asarray(
                            scene["object_vertices"] * scene["object_scale"].reshape(3), dtype="<f8"
                        ),
                    ),
                ]:
                    if (
                        name in record
                        and hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()
                        != record[name]
                    ):
                        raise ValueError(
                            "Source surface observation original-object identity differs"
                        )
                helper = helper_class(
                    human_faces=human_faces,
                    rest_vertices=rest,
                    object_vertices=scene["object_vertices"] * scene["object_scale"].reshape(3),
                    object_faces=scene["object_faces"],
                    camera_R=scene["camera_R"],
                    camera_t=scene["camera_t"],
                    focal_length_px=float(scene["focal_length_px"]),
                    image_size=[float(scene["frame_width"]), float(scene["frame_height"])],
                    observations=record["observations"],
                    regions=record["regions"],
                    frame_count=data.frame_num,
                    device=self.device,
                    **cfg.get("options", {}),
                )
            with torch.no_grad():
                probes = pred.human.verts_seq[0, ::101]
                projected, _ = helper.project(probes)
                native = project_world_to_screen(probes, self.cameras).reshape(-1, 3)[:, :2]
                error = (projected - native).abs().max()
                if not torch.isfinite(error) or error > 0.01:
                    raise ValueError("Source surface scene does not match native calibrated camera")
                helper.native_calibration_max_error_px = float(error)
            cache[key] = helper
        return cache[key]


    def _contact_temporal_loss(self, data, pred, cfg, weight):
        """Explicit-fps current arm/object acceleration; optional/off by omission."""
        if weight == 0:
            return pred.human.verts_seq.sum() * 0
        import json

        import numpy as np

        from grail.optimization.contact_temporal import ContactTemporalLoss

        key = (id(data), json.dumps(cfg, sort_keys=True))
        if key not in self._contact_temporal_cache:
            with np.load(cfg["canonical_geometry"], allow_pickle=False) as geometry:
                self._contact_temporal_cache[key] = ContactTemporalLoss(
                    geometry["object_vertices_scaled_m"],
                    geometry["object_faces"],
                    cfg["options"],
                ).to(self.device)
        # Obtain body indices from the bound native model.
        # current native joints explicitly; this extra forward is optional.
        _, _, native_joints = self.human_model.generate_mesh(
            pred.human.motion_data, output_joints=True, require_grad=True
        )
        result = self._contact_temporal_cache[key](native_joints, pred.obj.R, pred.obj.trans)
        return weight * result["total"]

    def prepare_fixed_contact(self, data, cfg):
        """Load immutable predicted pairs before Stage 2; never infer near pairs."""
        from grail.optimization.fixed_contact import FixedContactLoss

        allowed = {"mode", "prediction_path", "robust_delta_m", "weight", "threshold"}
        if set(cfg) - allowed:
            raise ValueError(
                "Fixed contact uses predicted phase windows and full 3D pairs; "
                f"remove legacy/dense options: {sorted(set(cfg) - allowed)}"
            )
        path = cfg.get("prediction_path")
        if not isinstance(path, str) or not path or "{" in path:
            raise ValueError("Fixed contact requires a resolved prediction_path before Stage 2")
        context = getattr(data, "fixed_contact_context", None)
        if context is None:
            raise ValueError(
                "Fixed contact requires original-geometry/source binding from init_data"
            )
        if context["binding"]["frame_count"] != data.frame_num:
            raise ValueError(
                "Fixed contact source timeline changed; regenerate predictions for this timeline"
            )
        delta = cfg.get("robust_delta_m", 0.01)
        if type(delta) not in (int, float) or not math.isfinite(delta) or delta <= 0:
            raise ValueError("Fixed contact robust_delta_m must be positive and finite")
        key = (id(data), path, delta)
        if key not in self._fixed_contact_cache:
            try:
                self._fixed_contact_cache[key] = FixedContactLoss.from_file(
                    path,
                    binding=context["binding"],
                    object_vertices=context["object_vertices"],
                    object_faces=context["object_faces"],
                    human_faces=data.human.faces,
                    device=self.device,
                    dtype=context["object_vertices"].dtype,
                    robust_delta_m=delta,
                    require_video_review=True,
                )
            except FileNotFoundError as exc:
                raise FileNotFoundError(
                    f"Fixed contact input is missing: {path}. Before Stage 2, review "
                    "the source video and compile per-hand contact intervals and "
                    "material pairs; see docs/fixed_contact_points.md."
                ) from exc
        return self._fixed_contact_cache[key]

    def fixed_contact_export(self, data):
        """Preserve the exact consumed pairs/phases in native output metadata."""
        helpers = [
            helper for key, helper in self._fixed_contact_cache.items() if key[0] == id(data)
        ]
        if not helpers:
            return None
        hashes = {helper.provenance["sha256"] for helper in helpers}
        if len(hashes) != 1:
            raise ValueError("A reconstruction cannot export conflicting fixed-contact predictions")
        helper = helpers[0]
        return {
            "provenance": helper.provenance,
            "contacts": helper.export_contacts(),
            "obj_contact_points": helper.contact_points_by_hand(),
            "semantics": "predicted material targets; not measured contact or physical validation",
        }

    def prepare_phase_contact_map(self, data, cfg):
        """Bind source-authored independent region fields before any update."""
        import json

        from grail.optimization.phase_contact_map_loss import PhaseContactMapLoss

        settings = {
            "partial_mass_fraction",
            "pair_temperature_m2",
            "coverage_temperature_m2",
            "robust_delta_m",
            "gap_tolerance_m",
            "normal_distance_scale_m",
            "min_opposing_cosine",
            "activation_ramp_frames",
            "phase_blend_frames",
        }
        allowed = settings | {"mode", "prediction_path", "weight", "threshold"}
        if set(cfg) - allowed:
            raise ValueError(
                f"Unsupported phase contact-map settings: {sorted(set(cfg) - allowed)}"
            )
        path = cfg.get("prediction_path")
        if not isinstance(path, str) or not path or "{" in path:
            raise ValueError("Phase contact maps require a resolved prediction_path before Stage 2")
        context = getattr(data, "fixed_contact_context", None)
        if context is None or context["binding"]["frame_count"] != data.frame_num:
            raise ValueError(
                "Phase contact maps require matching original geometry/source timeline"
            )
        options = {key: cfg[key] for key in settings if key in cfg}
        key = (id(data), path, json.dumps(options, sort_keys=True))
        if key not in self._phase_contact_map_cache:
            self._phase_contact_map_cache[key] = PhaseContactMapLoss.from_file(
                path,
                binding=context["binding"],
                object_vertices=context["object_vertices"],
                object_faces=context["object_faces"],
                human_faces=data.human.faces,
                device=self.device,
                dtype=context["object_vertices"].dtype,
                **options,
            )
        return self._phase_contact_map_cache[key]

    def phase_contact_map_export(self, data):
        """Save the consumed map and objective; never export inferred point pairs."""
        helpers = [
            value for key, value in self._phase_contact_map_cache.items() if key[0] == id(data)
        ]
        if not helpers:
            return None
        first = helpers[0].export_spec()
        if any(helper.export_spec() != first for helper in helpers[1:]):
            raise ValueError("A reconstruction cannot export conflicting phase contact maps")
        temporal = [
            dict(helper.provenance, **self._phase_contact_temporal_provenance[key])
            for key, helper in self._phase_contact_temporal_cache.items()
            if key[0] == id(data)
        ]
        return {
            "map": first,
            "objectives": [helper.provenance for helper in helpers],
            "temporal_objectives": temporal,
            "semantics": "source-authored region hypotheses with partial soft coverage; not measured contact",
        }

    def prepare_phase_contact_temporal(self, data, cfg):
        """Bind explicitly selected material holds and local transition windows."""
        import json

        from grail.optimization.phase_contact_temporal import PhaseContactTemporalLoss

        if set(cfg) - {"weight", "contact", "options", "component_weights", "scope_path"}:
            raise ValueError("Unknown phase contact temporal configuration field")
        weights = cfg.get("component_weights", {})
        components = {"hold_relative_velocity", "window_velocity", "window_acceleration"}
        if set(weights) != components or any(
            type(value) not in (int, float) or not math.isfinite(value) or value < 0
            for value in weights.values()
        ):
            raise ValueError("Declare all three finite nonnegative temporal component weights")
        contact = cfg["contact"]
        if contact.get("mode") != "phase_contact_map":
            raise ValueError("Temporal regions require a bound phase contact map")
        helper = self.prepare_phase_contact_map(data, contact)
        key = (id(data), json.dumps(cfg, sort_keys=True))
        if key not in self._phase_contact_temporal_cache:
            import copy

            options = copy.deepcopy(cfg.get("options", {}))
            scope = None
            if "scope_path" in cfg:
                from grail.optimization.contact_temporal_scope import load_temporal_scope

                options, scope = load_temporal_scope(
                    cfg["scope_path"],
                    defaults=options,
                    contact_map_sha256=helper.provenance["contact_map"]["manifest"]["sha256"],
                )
            self._phase_contact_temporal_cache[key] = PhaseContactTemporalLoss(
                helper, **options
            ).to(device=self.device, dtype=helper.object_local_vertices.dtype)
            self._phase_contact_temporal_provenance[key] = dict(
                scope=scope,
                resolved_options=copy.deepcopy(options),
                component_weights=copy.deepcopy(weights),
                weight=cfg.get("weight", 1.0),
            )
        return self._phase_contact_temporal_cache[key]

    def _phase_contact_temporal_loss(self, data, pred, cfg, weight):
        if weight == 0:
            return (pred.human.verts_seq.sum() + pred.obj.R.sum() + pred.obj.trans.sum()) * 0
        helper = self.prepare_phase_contact_temporal(data, cfg)
        components = helper(pred.human.verts_seq, pred.obj.R, pred.obj.trans)
        return weight * sum(
            cfg["component_weights"][key] * components[key] for key in cfg["component_weights"]
        )

    def _contact_loss(self, data, pred, cfg, weight):
        if weight == 0:
            return (pred.human.verts_seq.sum() + pred.obj.R.sum() + pred.obj.trans.sum()) * 0
        # Existing explicit dense experiments remain opt-in. An explicitly fixed
        # mode always takes the fixed loader, which rejects mixed dense options.
        mode = cfg.get(
            "mode", "legacy_nearest" if cfg.get("dense_contact_prediction") else "fixed_prediction"
        )
        if mode == "phase_contact_map":
            helper = self.prepare_phase_contact_map(data, cfg)
            return weight * helper(pred.human.verts_seq, pred.obj.R, pred.obj.trans)
        if mode == "fixed_prediction":
            # A stationary object can still be touched; reviewed phases govern
            # activation, not the motion heuristic used to freeze object pose.
            helper = self.prepare_fixed_contact(data, cfg)
            return weight * helper(pred.human.verts_seq, pred.obj.R, pred.obj.trans)
        if cfg.get("dense_contact_prediction") and mode == "legacy_nearest":
            if cfg.get("depth_only", False) or cfg.get("use_center_loss", False):
                raise ValueError(
                    "Dense contact regions use their own 3D objective; disable legacy contact modes"
                )
            import json

            from grail.optimization.dense_contact_adapter import DenseContactAdapter

            settings = cfg["dense_contact_prediction"]
            cache_key = (id(data), json.dumps(settings, sort_keys=True))
            bound = (
                isinstance(settings, dict)
                and settings.get("native_input_binding") is not None
                and settings["native_input_binding"] is not False
            )
            if cache_key not in self._dense_contact_cache:
                if bound:
                    self._dense_contact_cache[cache_key] = DenseContactAdapter(
                        data,
                        settings,
                        self.device,
                        human_model=self.human_model,
                        cameras=self.cameras,
                    )
                else:
                    self._dense_contact_cache[cache_key] = DenseContactAdapter(
                        data, settings, self.device
                    )
            if bound:
                loss = self._dense_contact_cache[cache_key](
                    data, pred, human_model=self.human_model, cameras=self.cameras
                )
            else:
                loss = self._dense_contact_cache[cache_key](data, pred)
            return loss * weight * (0.0 if data.is_static_obj else 1.0)

        raise ValueError(
            f"Unsupported contact mode: {mode}. VLM-label/nearest contact losses "
            "have been removed. Use phase_contact_map for reviewed independent regions "
            "or fixed_prediction for explicit material pairs; see docs/fixed_contact_points.md."
        )


    def _keypoint_tracking_loss(self, data, pred, cfg, weight):
        # Confidence scales belong to the estimator: SOMA uses one threshold;
        common = cfg.get("confidence_threshold")
        group_keys = ("body_confidence_threshold", "hand_confidence_threshold")
        if common is not None and any(cfg.get(key) is not None for key in group_keys):
            raise ValueError(
                "keypoint confidence_threshold cannot be combined with "
                "body_confidence_threshold or hand_confidence_threshold"
            )

        def threshold(key, default):
            value = cfg.get(key)
            if value is None:
                return default
            if isinstance(value, bool):
                raise ValueError(f"keypoint {key} must be finite and in [0, 1]")
            try:
                value = float(value)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"keypoint {key} must be finite and in [0, 1]") from exc
            if not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"keypoint {key} must be finite and in [0, 1]")
            return value

        if common is not None:
            body_threshold = hand_threshold = threshold("confidence_threshold", None)
        else:
            body_threshold = threshold("body_confidence_threshold", 0.6)
            hand_threshold = threshold("hand_confidence_threshold", 0.2)
        pred_body_kp = pred.human.body_keypoints_seq
        gt_body_kp = data.human.body_keypoints_seq
        gt_body_conf = gt_body_kp[:, :, 2]
        if cfg.get("excluded_body_intervals"):
            from grail.optimization.hoi_depth_alignment import (
                exclude_body_observation_confidence,
            )

            gt_body_conf = exclude_body_observation_confidence(
                gt_body_conf, cfg["excluded_body_intervals"]
            )
        gt_body_kp = gt_body_kp[:, :, :2]

        pred_hand_kp = pred.human.hand_keypoints_seq
        gt_hand_kp = data.human.hand_keypoints_seq
        gt_hand_conf = gt_hand_kp[:, :, 2]
        gt_hand_kp = gt_hand_kp[:, :, :2]

        duration = cfg.get("duration", "all")
        if duration == "start":
            inter_start_idx = data.inter_start_idx
            window_size = 8
            ws = max(0, inter_start_idx - window_size // 2)
            we = min(len(pred_body_kp), inter_start_idx + window_size // 2)
            pred_body_kp = pred_body_kp[ws:we]
            gt_body_kp = gt_body_kp[ws:we]
            gt_body_conf = gt_body_conf[ws:we]
            pred_hand_kp = pred_hand_kp[ws:we]
            gt_hand_kp = gt_hand_kp[ws:we]
            gt_hand_conf = gt_hand_conf[ws:we]
        elif duration != "all":
            raise ValueError(f"Invalid duration: {duration}")

        loss_body = keypoint_loss(
            pred_body_kp.reshape(-1, 2),
            gt_body_kp.reshape(-1, 2),
            gt_body_conf.reshape(-1),
            conf_thres=body_threshold,
        )
        loss_hand = keypoint_loss(
            pred_hand_kp.reshape(-1, 2),
            gt_hand_kp.reshape(-1, 2),
            gt_hand_conf.reshape(-1),
            conf_thres=hand_threshold,
        )
        beta = cfg.get("beta", 0.3)
        return weight * (loss_body + beta * loss_hand)

    def _ground_loss(self, data, pred, cfg, weight):
        height = cfg.get("height", 0.14)
        gravity_axis = cfg.get("gravity_axis", "z")
        return weight * ground_loss(pred.human.verts_seq, gravity_axis=gravity_axis, height=height)

    def _human_global_init_reg_loss(self, data, pred, cfg, weight):
        mode = cfg.get("mode", "speed")
        if mode == "root_relative_velocity":
            return self._root_relative_velocity_loss(data, pred, cfg, weight)
        pred_trans = pred.human.trans
        ref_trans = data.human.motion_data_global_init["trans"]

        pred_vel = pred_trans[1:] - pred_trans[:-1]
        ref_vel = ref_trans[1:] - ref_trans[:-1]

        if mode == "speed":
            return weight * reg_loss(pred_vel.norm(dim=-1), ref_vel.norm(dim=-1), use_l2=True)
        if mode != "camera_depth":
            raise ValueError(f"Invalid human_global_init_reg mode: {mode}")

        camera_depth_axis = self._camera_depth_axes(pred_trans.shape[0], cfg)
        pred_camera_depth = (pred_vel * camera_depth_axis).sum(dim=-1)
        ref_camera_depth = (ref_vel * camera_depth_axis).sum(dim=-1)

        delta = cfg.get("huber_delta", 0.01)

        def component_loss(pred_component, ref_component):
            error = pred_component - ref_component
            if delta is None:
                return (error**2).mean()
            return huber_loss(error, delta=delta)

        velocity_weight = float(cfg.get("velocity_weight", 1.0))
        acceleration_weight = float(cfg.get("acceleration_weight", 0.0))
        if velocity_weight < 0 or acceleration_weight < 0:
            raise ValueError("camera_depth velocity and acceleration weights must be non-negative")
        if velocity_weight + acceleration_weight <= 0:
            return pred_vel.sum() * 0.0

        loss = velocity_weight * component_loss(pred_camera_depth, ref_camera_depth)
        if acceleration_weight > 0 and pred_camera_depth.shape[0] > 1:
            loss = loss + acceleration_weight * component_loss(
                pred_camera_depth[1:] - pred_camera_depth[:-1],
                ref_camera_depth[1:] - ref_camera_depth[:-1],
            )
        return weight * loss

    def _root_relative_velocity_loss(self, data, pred, cfg, weight):
        """Root speed + nonroot velocity vectors in each motion's root frame.

        Use native FK roots rather than model translation parameters. Differences
        are metres per source-frame interval (as in the legacy speed loss); both
        motions already share the source timeline, world placement and identity.
        Canonicalize positions with each frame's full inverse root rotation before
        differencing, so root turning/tilting does not appear as articulation.
        """
        if cfg.get("relative_frame", "root") != "root":
            raise ValueError("root_relative_velocity requires relative_frame: root")
        root_weight = float(cfg.get("root_speed_weight", 1.0))
        joint_weight = float(cfg.get("joint_velocity_weight", 1.0))
        if (
            not all(math.isfinite(w) and w >= 0 for w in (root_weight, joint_weight))
            or root_weight + joint_weight <= 0
        ):
            raise ValueError("Velocity weights must be finite, nonnegative and not both zero")
        joints = self.human_model.get_native_joints(pred.human.motion_data, require_grad=True)
        root_idx = cfg.get("root_joint_index", 0)
        if (
            joints.ndim != 3
            or joints.shape[-1] != 3
            or joints.shape[1] < 2
            or type(root_idx) is not int
            or not 0 <= root_idx < joints.shape[1]
        ):
            raise ValueError("Native joints must be [T, J>=2, 3] with a valid root_joint_index")

        reference_motion = data.human.motion_data_global_init
        cache = getattr(self, "_global_motion_reference_cache", None)
        if cache is None or cache[0] is not reference_motion or cache[1] is not self.human_model:
            # The global initialization is an immutable target across stages,
            # not the stage's evolving candidate. A new dataset replaces it.
            with torch.no_grad():
                reference = (
                    self.human_model.get_native_joints(reference_motion, require_grad=False)
                    .detach()
                    .clone()
                )
                reference_pose, _ = self.human_model.extract_root_body_pose(reference_motion)
                reference_rotation = axis_angle_to_matrix(reference_pose).detach().clone()
            self._global_motion_reference_cache = (
                reference_motion,
                self.human_model,
                reference,
                reference_rotation,
            )
        reference = self._global_motion_reference_cache[2]
        if reference.shape != joints.shape:
            raise ValueError("Predicted and reference native joints must share shape and timeline")
        root_pose, _ = self.human_model.extract_root_body_pose(pred.human.motion_data)
        root_rotation = axis_angle_to_matrix(root_pose)
        reference_rotation = self._global_motion_reference_cache[3]
        if (
            root_rotation.shape != (len(joints), 3, 3)
            or reference_rotation.shape != root_rotation.shape
        ):
            raise ValueError("Native root rotations must share the joint timeline")
        if joints.shape[0] < 2:
            return joints.sum() * 0.0

        root = joints[:, root_idx]
        reference_root = reference[:, root_idx]
        root_speed = (root[1:] - root[:-1]).norm(dim=-1)
        reference_speed = (reference_root[1:] - reference_root[:-1]).norm(dim=-1)
        root_loss = (root_speed - reference_speed).square().mean()

        nonroot = torch.arange(joints.shape[1], device=joints.device) != root_idx
        # Row vectors: multiplying by R applies R^-1 to the world-space vector.
        # Rotating velocities after differencing would retain root angular motion.
        relative = (joints[:, nonroot] - root[:, None]) @ root_rotation
        reference_relative = (reference[:, nonroot] - reference_root[:, None]) @ reference_rotation
        velocity_error = (relative[1:] - relative[:-1]) - (
            reference_relative[1:] - reference_relative[:-1]
        )
        joint_loss = velocity_error.square().sum(dim=-1).mean()
        return weight * (root_weight * root_loss + joint_weight * joint_loss)

    def _camera_depth_axes(self, frame_num, cfg):
        """Return horizontal world-space camera optical axes per frame interval.

        PyTorch3D camera-space +Z is the depth direction used by unprojection.
        Transforming the origin and a +Z point through camera-to-world avoids
        relying on storage conventions for the camera rotation.  The gravity
        component is removed so this term does not oppose stairs motion.
        """
        if self.cameras is None:
            raise ValueError("Camera-depth regularization requires cameras")

        camera_to_world = self.cameras.get_world_to_view_transform().inverse()
        batch_size = int(self.cameras.R.shape[0])
        dtype = self.cameras.R.dtype
        device = self.cameras.R.device
        camera_origin = torch.zeros(batch_size, 1, 3, dtype=dtype, device=device)
        camera_depth_point = camera_origin.clone()
        camera_depth_point[:, :, 2] = 1.0
        depth_axis = (
            camera_to_world.transform_points(camera_depth_point)
            - camera_to_world.transform_points(camera_origin)
        ).squeeze(1)

        gravity_axis = cfg.get("gravity_axis", "z")
        if isinstance(gravity_axis, str):
            try:
                gravity_axis = {"x": 0, "y": 1, "z": 2}[gravity_axis.lower()]
            except KeyError as exc:
                raise ValueError(f"Invalid gravity axis: {gravity_axis}") from exc
        depth_axis[:, gravity_axis] = 0.0
        axis_norm = depth_axis.norm(dim=-1, keepdim=True)
        if torch.any(axis_norm <= 1e-6):
            raise ValueError("Camera depth axis is parallel to gravity")
        depth_axis = depth_axis / axis_norm

        interval_num = frame_num - 1
        if batch_size == 1:
            interval_depth = depth_axis.expand(interval_num, -1)
        elif batch_size == frame_num:
            interval_depth = depth_axis[:-1] + depth_axis[1:]
            interval_depth = torch.nn.functional.normalize(interval_depth, dim=-1, eps=1e-6)
        elif batch_size == interval_num:
            interval_depth = depth_axis
        else:
            raise ValueError(
                f"Expected 1, {frame_num}, or {interval_num} cameras, got {batch_size}"
            )
        return interval_depth.detach()

    def _human_smoothness_loss(self, data, pred, cfg, weight):
        beta = cfg.get("beta", 1.0)
        return weight * smoothness_loss(pred.human.verts_seq, beta=beta)

    def _human_traj_reg_loss(self, data, pred, cfg, weight):
        res = pred.human.trans_res
        res = res.reshape(res.shape[0], 3)
        return weight * reg_loss(res, torch.zeros_like(res))

    def _human_pose_reg_loss(self, data, pred, cfg, weight):
        res = pred.human.pose_res
        frame_num = res.shape[0]
        reg_target = (
            torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0])
            .reshape(1, 1, 6)
            .repeat(frame_num, self.num_body_joints, 1)
            .to(self.device)
        )
        return weight * reg_loss(res, reg_target)

    def _human_foot_contact_loss(self, data, pred, cfg, weight):
        label_source = cfg.get("label_source", "gem")
        if label_source == "scene_support":
            if cfg.get("velocity_space") != "planar" or cfg.get("point_mode") != "foot_parts":
                raise ValueError("Reviewed foot contact requires planar foot_parts")
            if cfg.get("options", {}).get("normal_velocity_weight") != 0:
                raise ValueError("Planar reviewed foot contact requires normal_velocity_weight: 0")
            return self._human_scene_support_temporal_loss(
                data,
                pred,
                {
                    "manifest": cfg.get("manifest"),
                    "options": cfg.get("options", {}),
                    "component_weights": {
                        "no_slip": 1.0,
                        "window_velocity": 0.0,
                        "window_acceleration": 0.0,
                    },
                },
                weight,
            )
        if label_source != "gem":
            raise ValueError(f"Invalid human_foot_contact label_source: {label_source}")
        body_joints_seq = pred.human.body_joints_seq
        foot_contact_probs = data.human.foot_contact_probs

        if foot_contact_probs is None:
            return body_joints_seq.new_zeros(())

        contact_threshold = cfg.get("threshold", 0.5)
        point_mode = cfg.get("point_mode", "ankle")
        if point_mode == "ankle":
            left_idx, right_idx = self.human_model.get_foot_joint_indices()
            contact_points = torch.stack(
                [body_joints_seq[:, left_idx, :], body_joints_seq[:, right_idx, :]], dim=1
            )
            contact_probs = torch.stack(
                [
                    torch.max(foot_contact_probs[:, 0], foot_contact_probs[:, 1]),
                    torch.max(foot_contact_probs[:, 2], foot_contact_probs[:, 3]),
                ],
                dim=1,
            )
        elif point_mode == "foot_parts":
            # GEM predicts contact for [L ankle, L toe, R ankle, R toe].
            # Use the corresponding mesh-region centroids so a planted toe can
            # remain fixed while the heel legitimately pivots (and vice versa).
            labels = ("L_Ankle", "L_Toe", "R_Ankle", "R_Toe")
            contact_points = torch.stack(
                [
                    self.human_model.get_verts_segment(pred.human.verts_seq, [label]).mean(dim=1)
                    for label in labels
                ],
                dim=1,
            )
            contact_probs = foot_contact_probs[:, :4]
        else:
            raise ValueError(f"Invalid human_foot_contact point_mode: {point_mode}")

        if len(contact_points) < 2:
            return contact_points.sum() * 0.0

        velocity_space = cfg.get("velocity_space", "3d")
        if velocity_space not in ("3d", "planar"):
            raise ValueError(f"Invalid human_foot_contact velocity_space: {velocity_space}")

        gravity_axis = cfg.get("gravity_axis", "z")
        if isinstance(gravity_axis, str):
            try:
                gravity_axis = {"x": 0, "y": 1, "z": 2}[gravity_axis.lower()]
            except KeyError as exc:
                raise ValueError(f"Invalid gravity axis: {gravity_axis}") from exc

        weight_mode = cfg.get("weight_mode", "hard")
        if weight_mode not in ("hard", "soft"):
            raise ValueError(f"Invalid human_foot_contact weight_mode: {weight_mode}")

        def interval_weights(lag):
            # Requiring contact throughout the interval prevents a swing/contact
            # transition from being mistaken for foot sliding.
            interval_conf = contact_probs.unfold(0, lag + 1, 1).amin(dim=-1)
            if weight_mode == "hard":
                return (interval_conf > contact_threshold).float()
            soft_floor = cfg.get("soft_floor", 0.3)
            soft_gamma = cfg.get("soft_gamma", 1.0)
            weights = ((interval_conf - soft_floor) / max(1.0 - soft_floor, 1e-6)).clamp(0, 1)
            return weights.pow(soft_gamma)

        def displacement_loss(lag):
            displacement = contact_points[lag:] - contact_points[:-lag]
            if velocity_space == "planar":
                displacement = displacement.clone()
                displacement[..., gravity_axis] = 0.0
            else:
                vertical_weight = cfg.get("vertical_weight", 1.0)
                if vertical_weight != 1.0:
                    displacement = displacement.clone()
                    displacement[..., gravity_axis] *= vertical_weight
            interval_weight = interval_weights(lag)
            weighted_sum = (displacement.norm(dim=-1) * interval_weight).sum()
            return weighted_sum / interval_weight.sum().clamp_min(1e-6)

        loss = displacement_loss(1)
        long_horizon_weight = cfg.get("long_horizon_weight", 0.0)
        if long_horizon_weight > 0:
            long_horizon_lag = cfg.get("long_horizon_lag", 4)
            if not 1 < long_horizon_lag < len(contact_points):
                raise ValueError(
                    "human_foot_contact long_horizon_lag must be between 2 and sequence length - 1"
                )
            loss = loss + long_horizon_weight * displacement_loss(long_horizon_lag)
        stance_weight = cfg.get("stance_weight", 0.0)
        if not math.isfinite(stance_weight) or stance_weight < 0:
            raise ValueError("human_foot_contact stance_weight must be finite and nonnegative")
        if stance_weight > 0:
            from grail.optimization.foot_support import stance_drift_loss

            loss = loss + stance_weight * stance_drift_loss(
                contact_points,
                contact_probs,
                threshold=cfg.get("stance_threshold", contact_threshold),
                min_frames=cfg.get("stance_min_frames", 3),
                gravity_axis=gravity_axis,
                robust_delta=cfg.get("stance_robust_delta_m", 0.005),
            )
        return weight * loss

    def _human_foot_ground_loss(self, data, pred, cfg, weight):
        from grail.optimization.foot_ground import FootGroundConstraint, foot_region_indices

        if weight == 0:
            return pred.human.verts_seq.sum() * 0.0
        penetration_weight = cfg.get("penetration_weight", 1.0)
        support_weight = cfg.get("support_weight", 1.0)
        if any(
            not math.isfinite(value) or value < 0
            for value in (weight, penetration_weight, support_weight)
        ):
            raise ValueError("human_foot_ground weights must be finite and nonnegative")
        # Stage 1 and Stage 2 may differ in weights; cache geometry/labels only
        # for identical settings and fixed observation identities.
        import json

        key = (
            json.dumps({k: v for k, v in cfg.items() if k != "weight"}, sort_keys=True),
            id(data.human.foot_contact_probs),
        )
        if key not in self._foot_ground_constraints:
            self._foot_ground_constraints[key] = FootGroundConstraint(
                pred.human.verts_seq,
                foot_region_indices(self.human_model),
                data.human.foot_contact_probs,
                plane=cfg.get("plane"),
                support_threshold=cfg.get("support_threshold", 0.5),
                robust_delta_m=cfg.get("robust_delta_m", 0.005),
                support_height_m=cfg.get("support_height_m", 0.0),
                penetration_margin_m=cfg.get("penetration_margin_m", 0.0),
            )
        terms = self._foot_ground_constraints[key].terms(pred.human.verts_seq)
        return weight * (
            penetration_weight * terms["penetration"] + support_weight * terms["support_height"]
        )

    def _human_foot_ground_temporal_loss(self, data, pred, cfg, weight):
        """Label-bound transition smoothing; retains the separate stance objective."""
        import json

        from grail.optimization.foot_ground import foot_region_indices
        from grail.optimization.foot_ground_temporal import FootGroundTemporal

        if set(cfg) - {"weight", "component_weights", "options"}:
            raise ValueError("Unknown foot-ground temporal configuration field")
        component_weights = cfg.get("component_weights", {})
        if not isinstance(component_weights, dict) or set(component_weights) != {
            "window_velocity",
            "window_acceleration",
        }:
            raise ValueError("Foot-ground temporal requires two explicit component weights")
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value < 0
            for value in [weight, *component_weights.values()]
        ):
            raise ValueError("Foot-ground temporal weights must be finite and nonnegative")
        if weight == 0:
            return pred.human.verts_seq.sum() * 0.0
        options = cfg.get("options")
        allowed = {
            "fps",
            "threshold",
            "window_frames",
            "velocity_scale_mps",
            "acceleration_scale_mps2",
            "correction_speed_tolerance_mps",
            "acceleration_tolerance_mps2",
        }
        if not isinstance(options, dict) or "fps" not in options or set(options) - allowed:
            raise ValueError("Foot-ground temporal requires explicit FPS and known options")
        key = (
            id(data),
            id(data.human.foot_contact_probs),
            json.dumps(options, sort_keys=True, allow_nan=False),
        )
        if key not in self._foot_ground_temporal_constraints:
            # The first evaluation binds this stage's starting native geometry.
            # Repeated optimizer evaluations must never reset the reference.
            self._foot_ground_temporal_constraints[key] = FootGroundTemporal(
                pred.human.verts_seq.detach(),
                foot_region_indices(self.human_model),
                data.human.foot_contact_probs,
                **options,
            )
        terms = self._foot_ground_temporal_constraints[key].terms(pred.human.verts_seq)
        return weight * sum(terms[name] * value for name, value in component_weights.items())

    def _human_scene_support_loss(self, data, pred, cfg, weight):
        """Fixed phase/surface assignments, shared by numerical Stage 1 and 2."""
        import json

        from grail.optimization.foot_ground import foot_region_indices
        from grail.optimization.scene_support import SceneSupportConstraint

        terms_weights = {
            "penetration": cfg.get("penetration_weight", 2.0),
            "support_height": cfg.get("support_weight", 1.0),
            "tangent_anchor": cfg.get("tangent_weight", 2.0),
            "boundary": cfg.get("boundary_weight", 1.0),
        }
        values = [weight, *terms_weights.values()]
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value < 0
            for value in values
        ):
            raise ValueError("human_scene_support weights must be finite and nonnegative")
        if weight == 0:
            return pred.human.verts_seq.sum() * 0.0
        manifest = cfg.get("manifest")
        blend_options = {
            name: cfg.get(name) for name in ("height_blend_frames", "tangent_blend_frames")
        }
        key = json.dumps(
            {
                "manifest": manifest,
                "robust_delta_m": cfg.get("robust_delta_m", 0.005),
                **blend_options,
            },
            sort_keys=True,
            allow_nan=False,
        )
        if key not in self._scene_support_constraints:
            self._scene_support_constraints[key] = SceneSupportConstraint(
                pred.human.verts_seq,
                foot_region_indices(self.human_model),
                manifest,
                robust_delta_m=cfg.get("robust_delta_m", 0.005),
                **blend_options,
            )
        terms = self._scene_support_constraints[key].terms(pred.human.verts_seq)
        return weight * sum(
            terms[name] * term_weight for name, term_weight in terms_weights.items()
        )

    def _human_scene_support_temporal_loss(self, data, pred, cfg, weight):
        """Foot-material continuity around reviewed touchdown and release events."""
        import json

        from grail.optimization.foot_ground import foot_region_indices
        from grail.optimization.scene_support_temporal import SceneSupportTemporal

        component_weights = cfg.get("component_weights", {})
        expected = {"no_slip", "window_velocity", "window_acceleration"}
        if set(component_weights) != expected:
            raise ValueError("Scene-support temporal requires three explicit component weights")
        if any(
            isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v < 0
            for v in [weight, *component_weights.values()]
        ):
            raise ValueError("Scene-support temporal weights must be finite and nonnegative")
        if weight == 0:
            return pred.human.verts_seq.sum() * 0.0
        options = cfg.get("options", {})
        key = json.dumps(
            {"manifest": cfg.get("manifest"), "options": options}, sort_keys=True, allow_nan=False
        )
        if key not in self._scene_support_temporal_constraints:
            self._scene_support_temporal_constraints[key] = SceneSupportTemporal(
                pred.human.verts_seq.detach(),
                foot_region_indices(self.human_model),
                cfg.get("manifest"),
                **options,
            )
        terms = self._scene_support_temporal_constraints[key].terms(pred.human.verts_seq)
        return weight * sum(terms[name] * value for name, value in component_weights.items())

    def _verts_tracking_loss(self, data, pred, cfg, weight):
        pred_verts = pred.obj.verts_seq
        gt_verts = data.obj.verts_tracking_seq
        frame_num = pred_verts.shape[0]
        pred_2d = project_world_to_screen(pred_verts.reshape(-1, 3), self.cameras).reshape(
            frame_num, -1, 3
        )[:, :, :2]
        return weight * l1_loss(pred_2d, gt_verts)

    def _obj_smoothness_loss(self, data, pred, cfg, weight):
        verts = pred.obj.verts_seq
        beta = cfg.get("beta", 1.0)
        return weight * smoothness_loss(verts.reshape(verts.shape[0], -1, 3), beta=beta)

    def _obj_traj_reg_loss(self, data, pred, cfg, weight):
        pred_trans = pred.obj.trans.reshape(-1, 3)
        orig_trans = data.obj.poses[:, :3, 3].reshape(-1, 3)
        return weight * reg_loss(pred_trans, orig_trans)

    def _obj_rot_reg_loss(self, data, pred, cfg, weight):
        return weight * reg_loss(pred.obj.R, data.obj.poses[:, :3, :3])


    def _depth_pointcloud_loss(self, data, pred, cfg, weight):
        if self._depth_loss_cache is None:
            num_gt_samples = cfg.get("num_gt_samples", 3000)
            self._build_depth_loss_cache(data, pred, num_gt_samples=num_gt_samples)
        cache = self._depth_loss_cache

        human_verts_seq = pred.human.verts_seq
        obj_verts_seq = pred.obj.verts_seq
        trim_pct = cfg.get("trim_pct", 0.2)

        frame_num = human_verts_seq.shape[0]
        interval = cfg.get("interval", 1)
        all_frames = list(range(frame_num))
        if interval > 1:
            start_offset = torch.randint(0, interval, (1,)).item()
            frame_indices = all_frames[start_offset::interval]
        else:
            frame_indices = all_frames

        loss = 0.0
        count = 0
        for i in frame_indices:
            h_visible = cache["human_vis_masks"][i]
            o_visible = cache["obj_vis_masks"][i]
            pred_human_visible = human_verts_seq[i][h_visible]
            pred_obj_visible = obj_verts_seq[i][o_visible]
            gt_human_pc = cache["gt_human_pcs"][i]
            gt_obj_pc = cache["gt_obj_pcs"][i]

            frame_loss = 0.0
            for pred_vis, gt_pc in [
                (pred_human_visible, gt_human_pc),
                (pred_obj_visible, gt_obj_pc),
            ]:
                if pred_vis.shape[0] > 0 and gt_pc.shape[0] > 0:
                    frame_loss = frame_loss + bidirectional_chamfer_loss(
                        pred_vis, gt_pc, trim_pct=trim_pct
                    )
            loss += weight * frame_loss
            count += 1

        if count > 0:
            loss /= count
        else:
            loss = torch.tensor(0.0, device=self.device, requires_grad=True)
        return loss

    def _obj_precontact_reg_loss(self, data, pred, cfg, weight):
        pred_verts = pred.obj.verts_seq
        inter_start_idx = data.inter_start_idx
        if inter_start_idx <= 0:
            return pred_verts[:0].sum()
        # Preserve the prefix pose relative to its current first frame, leaving
        # shared placement free and gradients active on the first-frame anchor.
        return weight * l1_loss(
            pred_verts[:inter_start_idx], pred_verts[0:1].repeat(inter_start_idx, 1, 1)
        )

    def _penetration_loss(self, data, pred, cfg, weight):
        human_verts_seq = pred.human.verts_seq
        obj_t = pred.obj.trans
        obj_R = pred.obj.R
        obj_sdf = data.obj_sdf
        frame_num = human_verts_seq.shape[0]

        loss_accum = 0.0
        count = 0
        for i in range(frame_num):
            human_verts_centered = human_verts_seq[i] - obj_t[i].unsqueeze(0)
            human_verts_ocs = torch.matmul(human_verts_centered, obj_R[i])
            loss_accum += penetration_loss(verts_A=human_verts_ocs, sdf_B=obj_sdf)
            count += 1

        return weight * (loss_accum / count) if count > 0 else human_verts_seq.new_zeros(())

    # ── Helpers ──────────────────────────────────────────────────────────────

    def _build_depth_loss_cache(self, data, pred, num_gt_samples=3000):
        """Pre-compute GT point clouds and vertex visibility masks for depth_pointcloud loss."""
        from pytorch3d.renderer.mesh.rasterizer import MeshRasterizer, RasterizationSettings

        full_h = int(data.camera.frame_height)
        full_w = int(data.camera.frame_width)
        half_h = full_h // 2
        half_w = full_w // 2
        frame_num = data.frame_num

        human_faces = data.human.faces
        obj_faces = data.obj.faces
        human_verts_seq = pred.human.verts_seq
        obj_verts_seq = pred.obj.verts_seq

        raster_settings = RasterizationSettings(
            image_size=(half_h, half_w),
            blur_radius=0.0,
            faces_per_pixel=1,
            bin_size=None,
            max_faces_per_bin=50000,
            cull_backfaces=True,
        )
        rasterizer = MeshRasterizer(cameras=self.cameras, raster_settings=raster_settings)
        depth_tol = 0.02

        cache = {"gt_human_pcs": {}, "gt_obj_pcs": {}, "human_vis_masks": {}, "obj_vis_masks": {}}
        self.logger.info(f"Building depth loss cache for {frame_num} frames...")

        for i in range(frame_num):
            human_mask = torch.from_numpy(data.human.masks[i]).squeeze().bool().to(self.device)
            obj_mask = torch.from_numpy(data.obj.masks[i]).squeeze().bool().to(self.device)

            depth_map = data.depth_maps[i]
            if not isinstance(depth_map, torch.Tensor):
                depth_map = torch.tensor(depth_map, dtype=torch.float32)
            depth_map = depth_map.to(self.device)

            # GT point clouds via unprojection (full resolution)
            for mask, pc_key in [(human_mask, "gt_human_pcs"), (obj_mask, "gt_obj_pcs")]:
                valid = mask & (depth_map > 0)
                ys, xs = torch.where(valid)
                if len(xs) > 0:
                    pts = torch.stack([xs.float(), ys.float(), depth_map[ys, xs]], dim=1)
                    pc = unproject_depth_map_to_world(pts, self.cameras).detach()
                    if pc.shape[0] > num_gt_samples:
                        pc = pc[torch.randperm(pc.shape[0], device=self.device)[:num_gt_samples]]
                    cache[pc_key][i] = pc
                else:
                    cache[pc_key][i] = torch.zeros((0, 3), device=self.device)

            # Vertex visibility masks via rasterization (half resolution)
            with torch.no_grad():
                for verts, faces, vis_key in [
                    (human_verts_seq[i], human_faces, "human_vis_masks"),
                    (obj_verts_seq[i], obj_faces, "obj_vis_masks"),
                ]:
                    mesh = Meshes(verts=[verts], faces=[faces])
                    zbuf = rasterizer(mesh).zbuf[..., 0].squeeze(0)

                    screen = self.cameras.transform_points_screen(verts.unsqueeze(0)).squeeze(0)
                    cam_pts = (
                        self.cameras.get_world_to_view_transform()
                        .transform_points(verts.unsqueeze(0))
                        .squeeze(0)
                    )
                    px = (screen[:, 0] * half_w / full_w).long().clamp(0, half_w - 1)
                    py = (screen[:, 1] * half_h / full_h).long().clamp(0, half_h - 1)
                    surface_depth = zbuf[py, px]
                    cache[vis_key][i] = (surface_depth > 0) & (
                        torch.abs(cam_pts[:, 2] - surface_depth) < depth_tol
                    )

        self._depth_loss_cache = cache
        self.logger.info("Depth loss cache built successfully.")

    def invalidate_cache(self):
        """Invalidate the depth loss cache (e.g. after data truncation)."""
        self._depth_loss_cache = None
