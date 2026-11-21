"""Bound, incremental native-motion audit for explicit local repair scopes.

This is finite-sample geometric evidence. A passing declared scope is never a
motion selection, continuous collision certificate or dynamics result.
"""

from __future__ import annotations

import copy
import hashlib
import json
import re
import time
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from .contact_edit import native_io
from .contact_edit.pose import (
    _finite,
    _integer,
    _keys,
    _model_dependencies,
    _sha,
    _unit,
    _weight,
)
from .repair_geometry import TriangleMesh, mesh_intersections
from .repair_scope import geometry_coverage
from .contact_tolerance import evaluate_contact_tolerance, validate_policy
from .static_scene import static_scene_payload, assert_static_scene_equal
from .joint_velocity import JointVelocityGate


def digest(value):
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, allow_nan=False, separators=(",", ":")
        ).encode()
    ).hexdigest()


def skew(v):
    x, y, z = v
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])


class AuditContext:
    """Compile geometry/model once, reuse exact unchanged sampled states."""

    def __init__(self, config, config_dir=".", *, cache_dir=None):
        _keys(
            config,
            {
                "schema_version",
                "model_xml",
                "trajectory_npz",
                "object_mesh_npz",
                "window",
                "editable_joints",
                "contacts",
            },
            {
                "protected_frames",
                "robot_geoms",
                "collision_pairs",
                "audit",
                "solver",
                "temporal",
                "source",
                "environment",
                "static_scene",
                "derived_fields",
                "expected_input_sha256",
                "backend",
                "coverage",
                "iteration",
                "contact_gap_policy",
                "contact_normal_policy",
                "diagnostics",
            },
            "repair config",
        )
        if type(config["schema_version"]) is not int or config["schema_version"] != 1:
            raise ValueError("Unsupported repair schema_version")
        self.config = copy.deepcopy(config)
        # Parse only the bound snapshot. Caller-side policy mutations must not
        # alter later audits while leaving their original digest unchanged.
        config = self.config
        self.config_digest = digest(config)
        self.contact_gap_policy = (
            validate_policy(config["contact_gap_policy"])
            if "contact_gap_policy" in config else None
        )
        self.contact_normal_policy = (
            validate_policy(config["contact_normal_policy"], "normal")
            if "contact_normal_policy" in config else None
        )
        self.repair_backend = self.config.get("backend", {"name": "native"})
        if not isinstance(self.repair_backend, dict):
            raise ValueError("Repair backend must be an object with a name")
        backend_name = self.repair_backend.get("name")
        if backend_name not in ("native", "material"):
            raise ValueError("Unknown repair backend; choose native or material")
        if backend_name == "native" and set(self.repair_backend) != {"name"}:
            raise ValueError("Native repair backend has no additional backend options")
        self.code_bindings = {
            str(path): _sha(path)
            for path in (
                Path(__file__),
                Path(__file__).with_name("local_repair.py"),
                Path(__file__).with_name("repair_geometry.py"),
                Path(__file__).with_name("repair_scope.py"),
                Path(__file__).with_name("repair_iteration.py"),
                Path(__file__).with_name("joint_velocity.py"),
                Path(__file__).with_name("audit_cache.py"),
                Path(__file__).with_name("contact_tolerance.py"),
                Path(__file__).with_name("static_scene.py"),
                Path(native_io.__file__),
                Path(native_io.__file__).with_name("pose.py"),
            )
        }
        if backend_name == "material":
            # Bind optional proposal code even during replay-only audits. Heavy
            # Torch/Warp runtimes are imported only by the selected solver.
            for filename in ("repair_material.py", "repair_tensor.py"):
                path = Path(__file__).with_name(filename)
                self.code_bindings[str(path)] = _sha(path)
        self.paths = {
            key: (Path(config_dir) / config[key]).resolve(strict=True)
            for key in ("model_xml", "trajectory_npz", "object_mesh_npz")
        }
        self.bindings = {
            key: {"path": str(path), "sha256": _sha(path)}
            for key, path in self.paths.items()
        }
        expected = config.get("expected_input_sha256", {})
        if not isinstance(expected, dict) or set(expected) - set(self.paths):
            raise ValueError("Unknown expected input hash key")
        for key, value in expected.items():
            if self.bindings[key]["sha256"] != value:
                raise ValueError("Input hash mismatch: " + key)
        self.dependencies = {
            str(p): _sha(p) for p in _model_dependencies(self.paths["model_xml"])
        }
        self.model = mujoco.MjModel.from_xml_path(str(self.paths["model_xml"]))
        if self.model.neq:
            raise ValueError("Equality constraints require a constraint-aware solver")
        self.data = mujoco.MjData(self.model)
        self.parent = native_io.load_npz(self.paths["trajectory_npz"])
        self.static_scene, scene_arrays = static_scene_payload(self.parent)
        scene_policy = config.get("static_scene", {})
        _keys(scene_policy, set(), {"clearance_m", "reference_frame"}, "static_scene")
        q = native_io._native_qpos(self.parent)
        self.T = len(q)
        self.fps = float(self.parent["fps"])
        if q.shape[1] != self.model.nq:
            raise ValueError("Native qpos dimension differs from compiled model")
        self.model_snapshot = dict(
            joint_names=np.array(
                [self.model.joint(i).name for i in range(self.model.njnt)]
            ),
            jnt_type=self.model.jnt_type.copy(),
            jnt_limited=self.model.jnt_limited.copy(),
            jnt_range=self.model.jnt_range.copy(),
            jnt_qposadr=self.model.jnt_qposadr.copy(),
        )
        native_io._validate_native_binding(
            self.parent, native_io._model_joints(self.model_snapshot)
        )
        compiled = np.empty(mujoco.mj_sizeModel(self.model), dtype=np.uint8)
        mujoco.mj_saveModel(self.model, None, compiled)
        self.compiled_digest = hashlib.sha256(compiled.tobytes()).hexdigest()
        if (
            "compiled_model_sha256" in self.parent
            and str(self.parent["compiled_model_sha256"].item()) != self.compiled_digest
        ):
            raise ValueError("Compiled model hash differs from native binding")
        for name, neutral in [
            ("scene_scale", 1.0),
            ("height_scale", 1.0),
            ("height_offset", 0.0),
            ("pelvis_rotation", 0.0),
        ]:
            if name in self.parent and (
                self.parent[name].shape != () or float(self.parent[name]) != neutral
            ):
                raise ValueError(
                    "Display placement requires an explicit native bake: " + name
                )
        self.quaternion_starts = []
        for j, kind in enumerate(self.model.jnt_type):
            a = int(self.model.jnt_qposadr[j])
            if kind in (0, 1):
                a += 3 if kind == 0 else 0
                qs = q[:, a : a + 4]
                if not np.allclose(
                    np.linalg.norm(qs, axis=1), 1.0, atol=1e-6, rtol=0
                ) or np.any(np.sum(qs[:-1] * qs[1:], axis=1) < 0):
                    raise ValueError(
                        "Frozen native quaternions require unit values and hemisphere continuity"
                    )
                self.quaternion_starts.append(a)
            elif self.model.jnt_limited[j] and (
                np.any(q[:, a] < self.model.jnt_range[j, 0] - 1e-7)
                or np.any(q[:, a] > self.model.jnt_range[j, 1] + 1e-7)
            ):
                raise ValueError(
                    "Parent exceeds native joint limits; no clipping is performed"
                )
        for name, shape in [("obj_R", (self.T, 3, 3)), ("obj_t", (self.T, 3))]:
            if name not in self.parent:
                raise ValueError("Native motion requires " + name)
            _finite(self.parent[name], shape, name)
        rr = self.parent["obj_R"]
        if not np.allclose(
            rr @ rr.swapaxes(-1, -2), np.eye(3), atol=1e-6, rtol=0
        ) or not np.allclose(np.linalg.det(rr), 1.0, atol=1e-6):
            raise ValueError("Object rotations must be proper orthonormal matrices")
        scale = np.asarray(self.parent["object_scale"])
        if (
            scale.shape not in ((), (3,))
            or not np.isfinite(scale).all()
            or np.any(scale <= 0)
        ):
            raise ValueError("Object scale must be positive scalar or XYZ")
        self.object_slerp = (
            Slerp(np.arange(self.T), Rotation.from_matrix(rr)) if self.T > 1 else None
        )
        if not isinstance(config["window"], list) or len(config["window"]) != 2:
            raise ValueError("window requires [start,end] integers")
        self.start, self.end = [_integer(x, "window") for x in config["window"]]
        if not 0 <= self.start < self.end < self.T:
            raise ValueError("Invalid source-frame window")
        protected = config.get("protected_frames", [])
        if not isinstance(protected, list):
            raise ValueError("protected_frames must be a list")
        protected = [_integer(x, "protected frame") for x in protected]
        if len(set(protected)) != len(protected) or any(
            not self.start <= x <= self.end for x in protected
        ):
            raise ValueError("Protected frames must be unique and within the window")
        self.protected = sorted(set(protected + [self.start, self.end]))
        self.frames = np.array(
            [f for f in range(self.start + 1, self.end) if f not in self.protected],
            dtype=int,
        )
        if not len(self.frames):
            raise ValueError("Window has no unprotected interior frames")
        edits = config["editable_joints"]
        if not isinstance(edits, dict) or not edits:
            raise ValueError("editable_joints requires named native limited hinges")
        ids, qids, dids, deltas = [], [], [], []
        for name, values in edits.items():
            j = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if j < 0 or self.model.jnt_type[j] != 3 or not self.model.jnt_limited[j]:
                raise ValueError("Editable joint must be a limited hinge: " + name)
            _keys(values, {"delta_bounds_rad"}, set(), name)
            delta = _finite(values["delta_bounds_rad"], (2,), "delta bounds")
            if not delta[0] <= 0 <= delta[1] or delta[0] >= delta[1]:
                raise ValueError(
                    "Correction corridor must include zero and have positive width"
                )
            ids.append(j)
            qids.append(self.model.jnt_qposadr[j])
            dids.append(self.model.jnt_dofadr[j])
            deltas.append(delta)
        self.jids = np.array(ids)
        self.qids = np.array(qids)
        self.dids = np.array(dids)
        self.names = list(edits)
        initial = q[np.ix_(self.frames, self.qids)]
        self.lower = np.maximum(
            initial + np.array(deltas)[:, 0], self.model.jnt_range[self.jids, 0]
        )
        self.upper = np.minimum(
            initial + np.array(deltas)[:, 1], self.model.jnt_range[self.jids, 1]
        )
        if np.any(self.lower >= self.upper):
            raise ValueError("Empty native correction corridor")
        self.mask = np.zeros(q.shape, dtype=bool)
        self.mask[np.ix_(self.frames, self.qids)] = True
        native_io._declared_derived_fields(config)
        mesh = native_io.load_npz(self.paths["object_mesh_npz"])
        self.meshes = {
            "object": dict(
                mesh=TriangleMesh(mesh["vertices"] * scale, mesh["faces"]),
                body=None,
                geom=None,
            )
        }
        geoms = config.get("robot_geoms", [])
        if not isinstance(geoms, list) or len(set(geoms)) != len(geoms):
            raise ValueError("robot_geoms must be a list of unique original geom IDs")
        # A carried table is part of the native scene, not optional decoration.
        # Include all original robot meshes against it even for a hand-only edit;
        # inherited collisions remain failures outside the editable support.
        if self.static_scene["meshes"]:
            geoms = sorted(set(geoms) | {
                g for g in range(self.model.ngeom)
                if self.model.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH
                and self.model.geom_bodyid[g] > 0
            })
        for g in geoms:
            g = _integer(g, "robot geom")
            if (
                g >= self.model.ngeom
                or self.model.geom_type[g] != mujoco.mjtGeom.mjGEOM_MESH
            ):
                raise ValueError("Robot original geom must be a native mesh")
            m = int(self.model.geom_dataid[g])
            va = int(self.model.mesh_vertadr[m])
            vn = int(self.model.mesh_vertnum[m])
            fa = int(self.model.mesh_faceadr[m])
            fn = int(self.model.mesh_facenum[m])
            self.meshes[f"geom:{g}"] = dict(
                mesh=TriangleMesh(
                    self.model.mesh_vert[va : va + vn],
                    self.model.mesh_face[fa : fa + fn],
                ),
                body=int(self.model.geom_bodyid[g]),
                geom=g,
            )
        environment = config.get("environment", [])
        for item in environment:
            _keys(item, {"name", "mesh_npz", "matrix_world"}, set(), "environment")
            if (
                not isinstance(item["name"], str)
                or not item["name"]
                or item["name"] in self.meshes
            ):
                raise ValueError("Environment name must be unique")
            path = (Path(config_dir) / item["mesh_npz"]).resolve(strict=True)
            mesh = native_io.load_npz(path)
            matrix = _finite(item["matrix_world"], (4, 4), "environment transform")
            if (
                not np.array_equal(matrix[3], np.array([0, 0, 0, 1]))
                or abs(np.linalg.det(matrix[:3, :3])) < 1e-12
            ):
                raise ValueError("Environment transform must be affine and nonsingular")
            self.meshes[item["name"]] = dict(
                mesh=TriangleMesh(mesh["vertices"], mesh["faces"]).transformed(matrix),
                body=None,
                geom=None,
                static=True,
            )
            self.bindings["environment:" + item["name"]] = {
                "path": str(path),
                "sha256": _sha(path),
            }
        scene_labels = []
        for index, item in enumerate(self.static_scene["meshes"]):
            label = f"scene:{index:03d}"
            if label in self.meshes:
                raise ValueError("Environment uses a reserved native scene label: " + label)
            scene_labels.append(label)
            self.meshes[label] = dict(
                mesh=TriangleMesh(scene_arrays[item["vertices_key"]],
                                  scene_arrays[item["faces_key"]]),
                body=None, geom=None, static=True,
            )
        self.pairs = []
        seen = set()
        declared_pairs = config.get("collision_pairs", [])
        # Explicit pairs may override the proposal clearance/reference, but may
        # not silently omit a body part from robot/static-scene screening.
        declared_keys = {tuple(sorted((p["a"], p["b"]))) for p in declared_pairs}
        automatic_pairs = [dict(a=f"geom:{g}", b=label,
                                clearance_m=scene_policy.get("clearance_m", .0003),
                                reference_frame=scene_policy.get("reference_frame", self.start))
                           for g in geoms for label in scene_labels
                           if tuple(sorted((f"geom:{g}", label))) not in declared_keys]
        for item in [*declared_pairs, *automatic_pairs]:
            _keys(
                item, {"a", "b"}, {"clearance_m", "reference_frame"}, "collision pair"
            )
            a, b = item["a"], item["b"]
            if (
                a not in self.meshes
                or b not in self.meshes
                or a == b
                or tuple(sorted((a, b))) in seen
            ):
                raise ValueError("Unknown or duplicate collision mesh pair")
            seen.add(tuple(sorted((a, b))))
            reference = _weight(
                item.get("reference_frame", self.start), "reference frame"
            )
            if not 0 <= reference < self.T:
                raise ValueError("Collision reference_frame outside timeline")
            self.pairs.append(
                dict(
                    a=a,
                    b=b,
                    clearance_m=_weight(item.get("clearance_m", 0.0003), "clearance"),
                    reference_frame=reference,
                )
            )
        coverage = config.get("coverage", {})
        _keys(coverage, set(), {"omissions"}, "coverage")
        # Enumeration only: it exposes gaps in the declared pair policy, without
        # changing the configured geometry tests or their numeric pass/fail.
        self.geometry_scope = geometry_coverage(
            self.model, self.meshes, self.pairs, self.names,
            omission_reasons=coverage.get("omissions", ()),
        )
        # Store original face IDs once, not once per pair and sampled pose.
        # World-space degeneracy can differ due to floating point precision;
        # each row retains exact additions/removals against this catalog.
        self.geometry_issues = {
            name: dict(degenerate_faces=np.flatnonzero(~item["mesh"].normal_valid).tolist())
            for name, item in self.meshes.items()
        }
        self.contacts = []
        names = set()
        if not isinstance(config["contacts"], list):
            raise ValueError("contacts must be a list")
        for item in config["contacts"]:
            _keys(
                item,
                {
                    "name",
                    "body_name",
                    "point_local",
                    "normal_local",
                    "object_face_id",
                    "object_barycentric",
                },
                {
                    "object_normal_sign",
                    "position_weight",
                    "normal_weight",
                    "active_frames",
                    "max_gap_m",
                    "max_normal_deg",
                    "target_gap_m",
                },
                "contact",
            )
            name = item["name"]
            body = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_BODY, item["body_name"]
            )
            if not isinstance(name, str) or not name or name in names or body <= 0:
                raise ValueError("Contacts need unique names and known body names")
            names.add(name)
            face = _integer(item["object_face_id"], "object face")
            if face >= len(self.meshes["object"]["mesh"].faces):
                raise ValueError("Contact face outside original mesh")
            bary = _finite(item["object_barycentric"], (3,), "barycentric")
            if np.any(bary < -1e-12) or not np.isclose(
                bary.sum(), 1.0, atol=1e-10, rtol=0
            ):
                raise ValueError(
                    "Barycentric weights must be nonnegative and sum to one"
                )
            # Preserve edge anchors returned with machine-roundoff negatives.
            bary = np.maximum(bary, 0.0)
            bary /= bary.sum()
            tri = self.meshes["object"]["mesh"].vertices[
                self.meshes["object"]["mesh"].faces[face]
            ]
            normal = np.cross(tri[1] - tri[0], tri[2] - tri[0])
            length = np.linalg.norm(normal)
            if length < 1e-14:
                raise ValueError("Authored contact triangle is degenerate")
            sign = item.get("object_normal_sign", 1)
            if type(sign) not in (int, float) or sign not in (-1, 1):
                raise ValueError("Contact normal sign must be +1/-1")
            active = _finite(
                item.get("active_frames", [0, self.T - 1]), (2,), "active frames"
            )
            if not 0 <= active[0] <= active[1] < self.T:
                raise ValueError("Invalid active contact frames")
            # Alignment is diagnostic by default; a stricter angle is opt-in.
            angle = _weight(item.get("max_normal_deg", 180.0), "normal gate")
            if angle > 180:
                raise ValueError("Normal gate exceeds 180 degrees")
            self.contacts.append(
                dict(
                    name=name,
                    body=body,
                    body_name=item["body_name"],
                    local=_finite(item["point_local"], (3,), "pad point"),
                    pad_normal=_unit(item["normal_local"], "pad normal"),
                    face=face,
                    bary=bary,
                    target_local=bary @ tri,
                    normal_local=sign * normal / length,
                    active=active,
                    max_gap_m=_weight(
                        item.get("max_gap_m", 0.006), "gap gate", positive=True
                    ),
                    max_normal_deg=angle,
                    position_weight=_weight(
                        item.get("position_weight", 1.0),
                        "contact weight",
                        positive=True,
                    ),
                    normal_weight=_weight(
                        item.get("normal_weight", 0.0), "normal weight"
                    ),
                )
            )
            target_gap = _weight(item.get("target_gap_m", 0.0), "target gap")
            if target_gap > self.contacts[-1]["max_gap_m"]:
                raise ValueError("Target gap exceeds the contact gate")
            self.contacts[-1]["target_gap_m"] = target_gap
        audit = config.get("audit", {})
        _keys(
            audit, set(),
            {"substeps", "halo", "witness_limit", "geometry_backend", "device"},
            "audit",
        )
        self.geometry_backend = audit.get("geometry_backend", "cpu")
        if self.geometry_backend not in ("cpu", "warp", "numba"):
            raise ValueError("audit geometry_backend must be cpu, warp or numba")
        device = audit.get(
            "device", "cuda:0" if self.geometry_backend == "warp" else "cpu"
        )
        if not isinstance(device, str) or not re.fullmatch(r"cpu|cuda:[0-9]+", device):
            raise ValueError("audit device must be cpu or an explicit cuda:N device")
        self.geometry_queries = None
        if self.geometry_backend == "warp":
            from .repair_gpu import WarpMeshQueries

            backend_path = Path(__file__).with_name("repair_gpu.py")
            self.code_bindings[str(backend_path)] = _sha(backend_path)
            self.geometry_queries = WarpMeshQueries(
                {name: item["mesh"] for name, item in self.meshes.items()},
                device=device,
            )
        elif device != "cpu":
            raise ValueError("CPU geometry backend requires device cpu")
        elif self.geometry_backend == "numba":
            from .repair_numba import NumbaMeshQueries

            backend_path = Path(__file__).with_name("repair_numba.py")
            self.code_bindings[str(backend_path)] = _sha(backend_path)
            self.geometry_queries = NumbaMeshQueries(
                {name: item["mesh"] for name, item in self.meshes.items()},
                device=device,
            )
        self.substeps = _integer(audit.get("substeps", 4), "substeps", 1)
        self.halo = _integer(audit.get("halo", 2), "halo", 1)
        if self.substeps > 32:
            raise ValueError("Audit substeps exceeds 32")
        self.witness_limit = _integer(
            audit.get("witness_limit", 16), "witness limit", 1
        )
        self.temporal = config.get("temporal", {})
        self.joint_velocity_gate = JointVelocityGate(self.temporal, self.model_snapshot)
        self.source = config.get("source")
        self.guides = []
        if self.source is not None:
            _keys(
                self.source,
                {"camera_matrix", "world_to_camera", "observations"},
                set(),
                "source",
            )
            self.K = _finite(self.source["camera_matrix"], (3, 3), "camera matrix")
            self.E = _finite(self.source["world_to_camera"], (4, 4), "world to camera")
            if (
                not np.allclose(self.E[3], [0, 0, 0, 1])
                or not np.allclose(
                    self.E[:3, :3] @ self.E[:3, :3].T, np.eye(3), atol=1e-6
                )
                or not np.isclose(np.linalg.det(self.E[:3, :3]), 1.0, atol=1e-6)
            ):
                raise ValueError("Camera extrinsic must be a proper rigid transform")
            for item in self.source["observations"]:
                _keys(
                    item,
                    {"frame", "body_name", "point_local", "uv", "max_error_px"},
                    set(),
                    "source observation",
                )
                f = _weight(item["frame"], "source frame")
                body = mujoco.mj_name2id(
                    self.model, mujoco.mjtObj.mjOBJ_BODY, item["body_name"]
                )
                if not 0 <= f < self.T or body <= 0:
                    raise ValueError("Invalid source observation binding")
                self.guides.append(
                    dict(
                        frame=f,
                        body=body,
                        body_name=item["body_name"],
                        local=_finite(item["point_local"], (3,), "source point"),
                        uv=_finite(item["uv"], (2,), "source uv"),
                        max_error_px=_weight(
                            item["max_error_px"], "source gate", positive=True
                        ),
                    )
                )
        self.cache = {}
        self.cache_hits = 0
        self.cache_misses = 0
        self.jp = np.zeros((3, self.model.nv))
        self.jr = np.zeros((3, self.model.nv))
        self.iteration = None
        self.initial_qpos = self.parent["qpos"]
        if "iteration" in config:
            from .repair_iteration import RepairIteration

            self.iteration = RepairIteration(self, config_dir)
            self.initial_qpos = self.iteration.qpos
            self.halo = max(self.halo, 3)

        self.persistent_cache = None
        self.motion_diagnostics = None
        if "diagnostics" in config:
            from .motion_diagnostics import MotionDiagnostics

            self.motion_diagnostics = MotionDiagnostics(
                self.model, self.parent, config["diagnostics"], config_dir)
            self.bindings.update(self.motion_diagnostics.bindings)
            diagnostic_code = Path(__file__).with_name("motion_diagnostics.py")
            self.code_bindings[str(diagnostic_code)] = _sha(diagnostic_code)
        if cache_dir is not None:
            from .audit_cache import AuditCache, runtime_binding

            # Bind all policy, original input bytes, compiled geometry and code.
            # Relocation alone must not invalidate a reusable baseline. Parent
            # motion is still fixed; candidate samples get separate exact keys.
            policy = copy.deepcopy(config)
            for key, record in self.bindings.items():
                if key in self.paths:
                    policy[key] = record["sha256"]
            for item in policy.get("environment", []):
                item["mesh_npz"] = self.bindings["environment:" + item["name"]]["sha256"]
            if "iteration" in policy:
                policy["iteration"].pop("seed_npz", None)
            self.persistent_cache = AuditCache(cache_dir, dict(
                schema_version=1, policy=policy, compiled=self.compiled_digest,
                inputs={k: v["sha256"] for k, v in self.bindings.items()},
                dependencies=sorted(self.dependencies.values()),
                code={Path(k).name: v for k, v in self.code_bindings.items()},
                runtime=runtime_binding()))

    def check_inputs_unchanged(self):
        for record in self.bindings.values():
            if _sha(record["path"]) != record["sha256"]:
                raise ValueError("Bound input changed during work: " + record["path"])
        for path, sha in self.dependencies.items():
            if _sha(path) != sha:
                raise ValueError("Model dependency changed during work: " + path)
        for path, sha in self.code_bindings.items():
            if _sha(path) != sha:
                raise ValueError(
                    "Audit/repair implementation changed during work: " + path
                )

    def affected_by_edit(self, frame):
        """Whether a sampled pose depends on any permitted integer-frame edit."""
        lo = int(np.floor(frame))
        alpha = float(frame) - lo
        return bool(
            np.any(self.mask[lo])
            or (alpha > 0 and lo + 1 < self.T and np.any(self.mask[lo + 1]))
        )

    def sample(self, qpos, frame):
        f = float(frame)
        if not np.isfinite(f) or not 0 <= f < self.T:
            raise ValueError("Sample outside timeline")
        lo = int(np.floor(f))
        hi = min(lo + 1, self.T - 1)
        alpha = f - lo
        q = (1 - alpha) * qpos[lo] + alpha * qpos[hi]
        if alpha == 0:
            R = self.parent["obj_R"][lo].astype(float)
        else:
            R = self.object_slerp([f]).as_matrix()[0]
        t = (1 - alpha) * self.parent["obj_t"][lo] + alpha * self.parent["obj_t"][hi]
        return q, R, t

    def forward(self, qpos):
        self.data.qpos[:] = qpos
        mujoco.mj_normalizeQuat(self.model, self.data.qpos)
        mujoco.mj_forward(self.model, self.data)

    def point(self, body, local, jac=False):
        R = self.data.xmat[body].reshape(3, 3)
        p = R @ local + self.data.xpos[body]
        if not jac:
            return p
        mujoco.mj_jac(self.model, self.data, self.jp, self.jr, p, body)
        return p, self.jp[:, self.dids].copy(), self.jr[:, self.dids].copy()

    def transform(self, name, R, t):
        item = self.meshes[name]
        if item.get("static"):
            return np.eye(4)
        M = np.eye(4)
        if name == "object":
            M[:3, :3] = R
            M[:3, 3] = t
        else:
            g = item["geom"]
            M[:3, :3] = self.data.geom_xmat[g].reshape(3, 3)
            M[:3, 3] = self.data.geom_xpos[g]
        return M

    def mesh_point(self, name, local, R, t, jac=False):
        M = self.transform(name, R, t)
        p = M[:3, :3] @ local + M[:3, 3]
        if not jac:
            return p
        body = self.meshes[name]["body"]
        if body is None:
            return p, np.zeros((3, len(self.qids))), np.zeros((3, len(self.qids)))
        mujoco.mj_jac(self.model, self.data, self.jp, self.jr, p, body)
        return p, self.jp[:, self.dids].copy(), self.jr[:, self.dids].copy()

    def times(self, substeps=None):
        substeps = self.substeps if substeps is None else substeps
        start = max(0, self.start - self.halo)
        end = min(self.T - 1, self.end + self.halo)
        return sorted(
            set(
                (np.arange(start * substeps, end * substeps + 1) / substeps).tolist()
                + [g["frame"] for g in self.guides if start <= g["frame"] <= end]
            )
        )

    def _row(self, qpos, frame):
        q, R, t = self.sample(qpos, frame)
        key = hashlib.sha256(
            np.asarray(frame, dtype="<f8").tobytes()
            + q.tobytes()
            + R.tobytes()
            + t.tobytes()
        ).hexdigest()
        if key in self.cache:
            self.cache_hits += 1
            return copy.deepcopy(self.cache[key])
        if self.persistent_cache is not None:
            row = self.persistent_cache.get(key)
            if row is not None:
                self.cache_hits += 1
                self.cache[key] = copy.deepcopy(row)
                return row
        self.cache_misses += 1
        self.forward(q)
        row = dict(
            source_frame=float(frame),
            contacts=[],
            collisions=[],
            source=[],
            failures=[],
        )
        for c in self.contacts:
            if not c["active"][0] <= frame <= c["active"][1]:
                continue
            p = self.point(c["body"], c["local"])
            n = self.data.xmat[c["body"]].reshape(3, 3) @ c["pad_normal"]
            target = R @ c["target_local"] + t
            normal = R @ c["normal_local"]
            gap = float(np.linalg.norm(p - target))
            angle = float(np.degrees(np.arccos(np.clip(-n @ normal, -1, 1))))
            item = dict(
                name=c["name"],
                gap_m=gap,
                normal_deg=angle,
                max_gap_m=c["max_gap_m"],
                max_normal_deg=c["max_normal_deg"],
                face_id=c["face"],
            )
            row["contacts"].append(item)
            if gap > c["max_gap_m"] + 1e-10 or angle > c["max_normal_deg"] + 1e-8:
                row["failures"].append(
                    dict(kind="contact", source_frame=float(frame), **item)
                )
        names = dict.fromkeys(
            name for pair in self.pairs for name in (pair["a"], pair["b"])
        )
        transforms = {name: self.transform(name, R, t) for name in names}
        surfaces = {
            name: self.meshes[name]["mesh"].transformed(matrix)
            for name, matrix in transforms.items()
        }
        issues = {}
        for name, surface in surfaces.items():
            original = self.meshes[name]["mesh"].normal_valid
            current = surface.normal_valid
            issues[name] = dict(
                catalog_mesh=name,
                count=int(np.count_nonzero(~current)),
                added_face_ids=np.flatnonzero(original & ~current).tolist(),
                removed_face_ids=np.flatnonzero(~original & current).tolist(),
            )
        candidates = None
        if self.geometry_backend == "warp":
            candidates = self.geometry_queries.candidate_pairs_batch(
                [(pair["a"], pair["b"]) for pair in self.pairs],
                transforms,
                world_bounds={
                    name: (mesh._lower, mesh._upper)
                    for name, mesh in surfaces.items()
                },
            )
        for index, pair in enumerate(self.pairs):
            stream = None if candidates is None else [candidates[index]]
            if self.geometry_backend == "numba":
                stream = self.geometry_queries.candidate_pairs(
                    pair["a"], surfaces[pair["a"]],
                    pair["b"], surfaces[pair["b"]],
                )
            collision = mesh_intersections(
                surfaces[pair["a"]],
                surfaces[pair["b"]],
                witness_limit=self.witness_limit,
                candidate_pairs=stream,
            )
            collision.pop("degenerate_faces")
            collision["degenerate_face_refs"] = {
                side: issues[pair[side]] for side in ("a", "b")
            }
            item = dict(pair=[pair["a"], pair["b"]], **collision)
            row["collisions"].append(item)
            if collision["intersecting"] or not collision["complete"]:
                row["failures"].append(
                    dict(
                        kind="collision"
                        if collision["intersecting"]
                        else "geometry_incomplete",
                        source_frame=float(frame),
                        **item,
                    )
                )
        for guide in self.guides:
            if frame != guide["frame"]:
                continue
            world = self.point(guide["body"], guide["local"])
            cam = self.E[:3, :3] @ world + self.E[:3, 3]
            if cam[2] <= 0:
                err = None
            else:
                projected = self.K @ cam
                err = float(np.linalg.norm(projected[:2] / projected[2] - guide["uv"]))
            item = dict(
                body_name=guide["body_name"],
                error_px=err,
                max_error_px=guide["max_error_px"],
            )
            row["source"].append(item)
            if err is None or err > guide["max_error_px"]:
                row["failures"].append(
                    dict(kind="source", source_frame=float(frame), **item)
                )
        self.cache[key] = copy.deepcopy(row)
        if self.persistent_cache is not None:
            self.persistent_cache.put(key, row)
        return row

    def audit(self, trajectory=None, times=None):
        self.check_inputs_unchanged()
        started = time.perf_counter()
        native = self.parent if trajectory is None else trajectory
        assert_static_scene_equal(self.parent, native)
        q = native_io._native_qpos(native)
        if q.shape != self.parent["qpos"].shape:
            raise ValueError("Candidate timeline/dimension changed")
        for key in self.parent:
            if key.startswith(
                ("obj_", "object_", "source_", "camera_", "model_", "compiled_model_")
            ) or key in (
                "fps",
                "timestamps",
                "dof_names",
                "qposadr",
                "scene_scale",
                "height_scale",
                "height_offset",
                "pelvis_rotation",
            ):
                if key not in native or not np.array_equal(
                    native[key], self.parent[key]
                ):
                    raise ValueError("Frozen input changed: " + key)
        times = self.times() if times is None else sorted(set(float(t) for t in times))
        if not times:
            raise ValueError("Audit needs explicit sample coverage")
        failures = []
        if not np.array_equal(q[~self.mask], self.parent["qpos"][~self.mask]):
            failures.append(
                dict(kind="frozen_qpos", message="Protected or undeclared qpos changed")
            )
        values = q[np.ix_(self.frames, self.qids)]
        if np.any(values < self.lower - 1e-7) or np.any(values > self.upper + 1e-7):
            failures.append(
                dict(
                    kind="limits",
                    message="Native limit or correction corridor exceeded",
                )
            )
        hits, misses = self.cache_hits, self.cache_misses
        rows = [self._row(q, t) for t in times]
        contact_reviews = {}
        complete, violations = {}, {}
        for metric, policy in [("gap", self.contact_gap_policy), ("normal", self.contact_normal_policy)]:
            if policy is None:
                continue
            review = evaluate_contact_tolerance(
                rows, policy=policy, metric=metric, fps=self.fps,
                required_frames=self.times(),
                contacts=[dict(name=c["name"], active_frames=c["active"])
                          for c in self.contacts],
            )
            contact_reviews[metric] = review
            complete[metric] = {c["name"]: c for c in review["contacts"] if c["complete"]}
            violations[metric] = {(c["name"], v["source_frame"]): v["reasons"]
                                  for c in complete[metric].values() for v in c["violations"]}
            incomplete = [c for c in review["contacts"] if not c["complete"]]
            if incomplete:
                failures.append(dict(kind="contact_coverage", metric=metric,
                                     message="Contact duration policy needs the complete declared sample grid",
                                     contacts=[c["name"] for c in incomplete],
                                     missing_frames={c["name"]:c["missing_frames"] for c in incomplete}))
        if contact_reviews:
            for row in rows:
                # _row/cache retains strict measurements; alter only this owned
                # report after the complete scope's duration has been checked.
                row["failures"] = [f for f in row["failures"] if f["kind"] != "contact"]
                for c in row["contacts"]:
                    failed = False
                    for metric, field, tol in [("gap", "gap_m", 1e-10), ("normal", "normal_deg", 1e-8)]:
                        strict = c[field] > c["max_" + field] + tol
                        c["strict_" + metric + "_exceeded"] = strict
                        if c["name"] in complete.get(metric, {}):
                            reasons = violations[metric].get((c["name"], row["source_frame"]), [])
                            c[metric + "_policy_reasons"] = reasons
                            c[metric + "_policy_passed"] = not reasons
                            failed |= bool(reasons)
                        else:
                            failed |= strict
                    if failed:
                        row["failures"].append(dict(kind="contact", source_frame=row["source_frame"], **c))
        failures.extend(f for row in rows for f in row["failures"])
        lo = max(0, self.start - self.halo)
        hi = min(self.T - 1, self.end + self.halo)
        dq = np.diff(q[lo : hi + 1, self.qids], axis=0)
        ddq = np.diff(q[lo : hi + 1, self.qids], n=2, axis=0) * self.fps**2
        temporal = dict(
            max_step_rad=float(np.max(np.abs(dq), initial=0)),
            max_acceleration_rad_s2=float(np.max(np.abs(ddq), initial=0)),
        )
        for key, samples, coefficients, scale in (
            ("max_step_rad", dq, [-1, 1], 1.0),
            ("max_acceleration_rad_s2", ddq, [1, -2, 1], self.fps**2),
        ):
            if key not in self.temporal:
                continue
            limit = self.temporal[key]
            # Preserve every violating joint/stencil. A larger frozen halo
            # violation must not hide a smaller violation we can still repair.
            for offset, column in np.argwhere(np.abs(samples) > limit + 1e-9):
                support = [int(lo + offset + k) for k in range(len(coefficients))]
                address = int(self.qids[column])
                editable = [
                    frame
                    for frame, coefficient in zip(support, coefficients, strict=True)
                    if coefficient != 0 and self.mask[frame, address]
                ]
                failures.append(
                    dict(
                        kind="temporal",
                        metric=key,
                        value=float(abs(samples[offset, column])),
                        limit=limit,
                        window=[lo, hi],
                        joint=self.names[column],
                        qpos_address=address,
                        support_frames=support,
                        coefficients=coefficients,
                        scale=float(scale),
                        editable_support_frames=editable,
                        affected_by_edit=bool(editable),
                    )
                )
        joint_velocity = self.joint_velocity_gate.evaluate(
            q, self.fps, native.get("timestamps"), self.mask,
        )
        if self.joint_velocity_gate.configured:
            # The velocity gate always covers every native hinge and interval.
            # Geometry sample selection, edit windows and frozen scope do not
            # suppress an inherited speed failure or a phase-boundary jump.
            failures.extend(joint_velocity["violations"])
            temporal["max_velocity_rad_s"] = joint_velocity["max_velocity_rad_s"]
        scoped_contacts = [
            c for c in self.contacts if c["active"][0] <= hi and c["active"][1] >= lo
        ]
        scoped_guides = [g for g in self.guides if lo <= g["frame"] <= hi]
        missing_contacts = [
            c["name"]
            for c in scoped_contacts
            if not any(c["active"][0] <= t <= c["active"][1] for t in times)
        ]
        missing_guides = [g["frame"] for g in scoped_guides if g["frame"] not in times]
        smoothness = None
        if self.iteration is not None:
            smoothness, smoothness_failures = self.iteration.compare(q)
            failures.extend(smoothness_failures)
        if missing_contacts:
            failures.append(dict(kind="contact_coverage", contacts=missing_contacts))
        if missing_guides:
            failures.append(dict(kind="source_coverage", source_frames=missing_guides))
        categories = {
            "contacts": bool(self.contacts),
            "collision": bool(self.pairs),
            "source": bool(self.guides),
            "temporal": bool(self.temporal),
            "invariants": True,
        }
        kinds = {
            "contacts": {"contact", "contact_coverage"},
            "collision": {"collision", "geometry_incomplete"},
            "source": {"source", "source_coverage"},
            "temporal": {"temporal"},
            "invariants": {"frozen_qpos", "limits"},
        }
        if self.iteration is not None:
            categories["smoothness"] = self.iteration.preserve_smoothness
            kinds["smoothness"] = {"smoothness"}
        checks = {
            name: dict(
                status=(
                    "FAIL"
                    if any(f["kind"] in kinds[name] for f in failures)
                    else "PASS"
                )
                if configured
                else "NOT_CONFIGURED",
                passed=(not any(f["kind"] in kinds[name] for f in failures))
                if configured
                else None,
            )
            for name, configured in categories.items()
        }
        for name, configured, scoped in [
            ("contacts", self.contacts, scoped_contacts),
            ("source", self.guides, scoped_guides),
        ]:
            if (
                configured
                and not scoped
                and not any(
                    row["contacts" if name == "contacts" else "source"] for row in rows
                )
            ):
                checks[name] = dict(status="NOT_IN_SCOPE", passed=None)
        configured_and_checked = any(
            row["contacts"] or row["collisions"] or row["source"] for row in rows
        ) or bool(self.temporal) or bool(self.iteration and self.iteration.preserve_smoothness)
        passed = not failures and configured_and_checked
        contacts = [c for row in rows for c in row["contacts"]]
        def failure_is_affected(failure):
            if failure["kind"] == "temporal":
                return failure["affected_by_edit"]
            return "source_frame" not in failure or self.affected_by_edit(
                failure["source_frame"]
            )

        affected = [f for f in failures if failure_is_affected(f)]
        inherited = [f for f in failures if not failure_is_affected(f)]
        return dict(
            schema_version=1,
            status="CHECKED_SCOPE_PASS" if passed else "CHECKED_SCOPE_FAIL",
            passed=passed,
            selected=False,
            config_sha256=self.config_digest,
            # Reports are caller-owned snapshots. Editing/redacting a report
            # must not change the guards which protect later cached audits.
            bindings=copy.deepcopy(self.bindings),
            model_dependencies_sha256=digest(self.dependencies),
            code_sha256=self.code_bindings.copy(),
            geometry_scope=copy.deepcopy(self.geometry_scope),
            geometry_issues=copy.deepcopy(self.geometry_issues),
            static_scene=dict(status=self.static_scene["status"],
                              mesh_names=[m["name"] for m in self.static_scene["meshes"]],
                              policy="all_original_robot_meshes_against_bound_native_scene",
                              object_scene_contact_checked=False),
            geometry_backend=(
                copy.deepcopy(self.geometry_queries.metadata())
                if self.geometry_queries is not None
                else dict(backend="cpu", device="cpu")
            ),
            coverage=dict(
                source_frames=times,
                window=[self.start, self.end],
                halo=self.halo,
                pairs=[[p["a"], p["b"]] for p in self.pairs],
                continuous_time=False,
                contacts_outside_scope=[
                    c["name"]
                    for c in self.contacts
                    if c["active"][0] > hi or c["active"][1] < lo
                ],
                source_guides_outside_scope=[
                    g["frame"] for g in self.guides if not lo <= g["frame"] <= hi
                ],
                collision_criterion="No original triangle surface intersections, including touches/coplanar overlap; clearance_m is a repair proposal offset, not a minimum-distance acceptance gate.",
            ),
            affected_scope_passed=not affected and configured_and_checked,
            affected_failures=affected,
            failures_outside_editable_support=inherited,
            checks=checks,
            joint_velocity=joint_velocity,
            **{"contact_" + metric + "_review": review for metric, review in contact_reviews.items()},
            **({"smoothness": smoothness} if smoothness is not None else {}),
            **({"motion_diagnostics": self.motion_diagnostics.compare(q)}
               if self.motion_diagnostics is not None else {}),
            failures=failures,
            rows=rows,
            summary=dict(
                failing_samples=sorted(
                    set(f["source_frame"] for f in failures if "source_frame" in f)
                ),
                max_contact_gap_m=max((c["gap_m"] for c in contacts), default=None),
                max_contact_normal_deg=max(
                    (c["normal_deg"] for c in contacts), default=None
                ),
                intersecting_pairs=sum(
                    c["intersecting"] for row in rows for c in row["collisions"]
                ),
                **temporal,
            ),
            reuse=dict(
                cache_hits=self.cache_hits - hits,
                evaluated_samples=self.cache_misses - misses,
                persistent_namespace=(self.persistent_cache.namespace
                                      if self.persistent_cache else None),
            ),
            elapsed_seconds=time.perf_counter() - started,
            limitations=[
                "Only explicitly listed mesh pairs and sampled poses are checked. Containment, continuous time and dynamics are not certified.",
                "Source checks cover only configured projected observations; missing source or environment checks are NOT_CONFIGURED, not accepted.",
                "A declared geometric pass does not select a motion; visual/source/physical review retains its own scope.",
            ],
        )
