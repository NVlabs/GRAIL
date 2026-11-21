"""Bound native inputs shared by surface-grasp proposals and mounted holds."""
from pathlib import Path
import hashlib
import json

import numpy as np
from scipy.spatial.transform import Rotation

from .contact_edit import native_io
from .contact_edit.pose import _finite, _keys, _model_dependencies, _sha
from .repair_geometry import TriangleMesh


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def code_binding(*modules):
    return {name: _sha(Path(__file__).with_name(name)) for name in modules}


def number(value, name, low=0, high=np.inf):
    if type(value) not in (int, float) or not np.isfinite(value) or not low <= value <= high:
        raise ValueError(f"Invalid {name}: expected a finite number in [{low}, {high}]")
    return float(value)


def indices(values, count, name):
    if (not isinstance(values, list) or not values or
            any(type(v) is not int or not 0 <= v < count for v in values) or
            len(set(values)) != len(values)):
        raise ValueError(f"{name} must contain unique indices in [0, {count})")
    return values


def rigid(rotation, name):
    r = _finite(rotation, (3, 3), name)
    if not np.allclose(r.T @ r, np.eye(3), atol=1e-6) or not np.isclose(np.linalg.det(r), 1, atol=1e-6):
        raise ValueError(f"{name} must be a proper rotation")
    return r


def object_query_geometry(original, policy="strict"):
    """Retain original geometry; optionally omit exact zero-area query facets.

    Material IDs always refer to ``original``. The compact face map is the
    ascending complement of the recorded exclusions, never a repaired mesh.
    Near-degenerate and float32-degenerate facets still fail strict queries.
    """
    if policy not in ("strict", "exclude_zero_area_queries"):
        raise ValueError("Unknown object_degenerate_policy")
    if not len(original.faces):
        raise ValueError("Original object must have nondegenerate triangles")
    excluded = np.flatnonzero(~original.normal_valid)
    if len(excluded):
        triangles = original.triangles[excluded]
        cross = np.cross(triangles[:, 1]-triangles[:, 0], triangles[:, 2]-triangles[:, 0])
        if policy == "strict" or not np.all(cross == 0):
            raise ValueError("Original object must have nondegenerate triangles; only exact zero-area query exclusions are supported")
    ids = np.flatnonzero(original.normal_valid)
    if not len(ids):
        raise ValueError("Object has no nondegenerate query triangles")
    ids.setflags(write=False)
    query = (TriangleMesh(original.vertices, original.faces[ids])
             if len(excluded) else original)
    metadata = dict(
        policy=policy, original_faces=len(original.faces), query_faces=len(ids),
        query_geometry_complete=not len(excluded),
        excluded_original_zero_area_face_ids=excluded.tolist(),
        query_face_to_original="ascending original face IDs excluding excluded_original_zero_area_face_ids",
        query_face_to_original_sha256=hashlib.sha256(ids.astype("<i8").tobytes()).hexdigest(),
        original_geometry_sha256=hashlib.sha256(
            original.vertices.tobytes()+original.faces.tobytes()).hexdigest(),
        query_geometry_sha256=hashlib.sha256(query.vertices.tobytes()+query.faces.tobytes()).hexdigest(),
        original_geometry_preserved=True, motion_accepted=False,
        scope="Normal-dependent proposal query only; original-mesh screening retains every face and incomplete status.")
    return query, ids, metadata


class GraspInputs:
    """Explicit model, original object triangles, timing, hand topology and lineage.

    Compiled robot triangle arrays retain the native geometry; the compiled geom
    transform restores their body frame. No hulls or remeshing are used here.
    """
    def __init__(self, config, directory):
        import mujoco
        _keys(config, {"native_npz", "model_xml", "object_mesh_npz", "hands"},
              {"object_degenerate_policy"}, "inputs")
        self.paths = {k: (Path(directory) / config[k]).resolve(strict=True)
                      for k in ("native_npz", "model_xml", "object_mesh_npz")}
        self.native = native_io.load_npz(self.paths["native_npz"])
        self.q = native_io._native_qpos(self.native)
        self.model = m = mujoco.MjModel.from_xml_path(str(self.paths["model_xml"]))
        if self.q.shape[1] != m.nq:
            raise ValueError("Native qpos does not match model nq")
        joint_records = [dict(joint=m.joint(j).name, address=int(m.jnt_qposadr[j]),
                              joint_type=int(m.jnt_type[j])) for j in range(m.njnt)]
        native_io._validate_native_binding(self.native, joint_records)
        for j in range(m.njnt):
            if m.jnt_type[j] in (0, 1):
                a = m.jnt_qposadr[j] + (3 if m.jnt_type[j] == 0 else 0)
                if not np.allclose(np.linalg.norm(self.q[:, a:a+4], axis=1), 1, atol=1e-5):
                    raise ValueError("Native root/ball quaternions must be normalized")
        self.R = _finite(self.native["obj_R"], (len(self.q), 3, 3), "obj_R")
        for r in self.R:
            rigid(r, "obj_R")
        self.t = _finite(self.native["obj_t"], (len(self.q), 3), "obj_t")
        scale = np.asarray(self.native["object_scale"], dtype=float)
        if scale.shape not in ((), (3,)) or not np.isfinite(scale).all() or np.any(scale <= 0):
            raise ValueError("object_scale must be positive scalar or xyz")
        mesh = native_io.load_npz(self.paths["object_mesh_npz"])
        self.object = TriangleMesh(mesh["vertices"] * scale, mesh["faces"])
        policy = config.get("object_degenerate_policy", "strict")
        self.object_query, self.object_query_face_ids, self.object_query_provenance = (
            object_query_geometry(self.object, policy))
        hands = config["hands"]
        if not isinstance(hands, dict) or not hands:
            raise ValueError("hands must explicitly describe at least one hand")
        self.hands, self.meshes, self.bodies = {}, {}, {}
        owned_joints = set()
        for name, hand in hands.items():
            _keys(hand, {"wrist_body", "finger_joints", "robot_geoms", "contact_groups", "region_faces"}, set(), name)
            wrist = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, hand["wrist_body"])
            if wrist <= 0:
                raise ValueError("wrist_body must name a native body")
            js = self.joints(hand["finger_joints"])
            if owned_joints.intersection(js):
                raise ValueError("Hands must not share editable finger joints")
            owned_joints.update(js)
            def descendant(b):
                while b and b != wrist:
                    b = int(m.body_parentid[b])
                return b == wrist
            if any(not descendant(int(m.jnt_bodyid[j])) or int(m.jnt_bodyid[j]) == wrist for j in js):
                raise ValueError("Finger joints must be downstream of the wrist")
            geoms = indices(hand["robot_geoms"], m.ngeom, "robot_geoms")
            for g in geoms:
                b = int(m.geom_bodyid[g])
                if not descendant(b) or m.geom_type[g] != mujoco.mjtGeom.mjGEOM_MESH:
                    raise ValueError("Hand geoms must be original meshes at/downstream of its wrist")
                mid = m.geom_dataid[g]
                va, vn = m.mesh_vertadr[mid], m.mesh_vertnum[mid]
                fa, fn = m.mesh_faceadr[mid], m.mesh_facenum[mid]
                r = Rotation.from_quat(m.geom_quat[g][[1, 2, 3, 0]]).as_matrix()
                self.meshes[g] = TriangleMesh(m.mesh_vert[va:va+vn] @ r.T + m.geom_pos[g], m.mesh_face[fa:fa+fn])
                if not self.meshes[g].normal_valid.all():
                    raise ValueError("Original hand mesh has degenerate triangles")
                self.bodies[g] = b
            groups = hand["contact_groups"]
            if not isinstance(groups, dict) or not groups:
                raise ValueError("contact_groups must explicitly name participating hand surfaces")
            for group, ids in groups.items():
                indices(ids, m.ngeom, group)
                if not set(ids).issubset(geoms):
                    raise ValueError("Every contact group geom must be in robot_geoms")
            indices(hand["region_faces"], len(self.object.faces), "region_faces")
            if not self.object.normal_valid[hand["region_faces"]].all():
                raise ValueError("Contact region must use nondegenerate original faces")
            self.hands[name] = dict(hand, wrist=wrist, joints=js, addresses=m.jnt_qposadr[js].tolist())
        dependencies = _model_dependencies(self.paths["model_xml"])
        self.binding = dict(native_sha256=_sha(self.paths["native_npz"]),
                            object_mesh_sha256=_sha(self.paths["object_mesh_npz"]),
                            model_dependencies=[dict(path=str(p.relative_to(self.paths["model_xml"].parent))
                                                     if p.is_relative_to(self.paths["model_xml"].parent) else str(p),
                                                     sha256=_sha(p)) for p in dependencies],
                            hands_sha256=digest(hands), object_scale=scale.tolist())
        if policy != "strict":
            # Opt-in banks and mounts must agree on the exact omission policy.
            # Preserve the historical strict binding for existing banks.
            self.binding["object_query"] = self.object_query_provenance

    def joints(self, names):
        import mujoco
        if not isinstance(names, list) or not names or len(set(names)) != len(names):
            raise ValueError("Expected unique named limited hinges")
        result = []
        for name in names:
            j = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if j < 0 or self.model.jnt_type[j] != 3 or not self.model.jnt_limited[j]:
                raise ValueError(f"Not a native limited hinge: {name}")
            result.append(j)
        return result


def write_report(path, report):
    Path(path).write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
