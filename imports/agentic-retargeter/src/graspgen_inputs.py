"""Original-geometry adapters for optional, unreviewed GraspGenX proposals.

Only NumPy is imported eagerly. Preparation uses native MuJoCo FK; it never
changes a trajectory or replaces authored contacts. Distances to the current
reference wrist are diagnostics, not source-video agreement or acceptance.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from collections.abc import Mapping

import numpy as np

_JOINT_SUFFIXES = ("index_0", "middle_0", "thumb_0", "index_1", "middle_1", "thumb_1", "thumb_2")
_QOPEN = np.array([.4, .4, 0, 0, 0, .7, 0], dtype=float)
_QCLOSE = np.array([1.5708, 1.5708, 0, 1.74533, 1.74533, -1.0472, -1.5], dtype=float)


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _array_sha(value):
    array = np.ascontiguousarray(value)
    return hashlib.sha256(str((array.dtype.str, array.shape)).encode() + array.tobytes()).hexdigest()


def _finite(value, shape, label):
    array = np.asarray(value)
    if array.shape != shape or array.dtype.kind not in "fiu" or not np.isfinite(array).all():
        raise ValueError(f"{label} must be a finite numeric array with shape {shape}")
    return array.astype(float)


def _integer(value, label, low=0):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < low:
        raise ValueError(f"{label} must be an integer >= {low}")
    return int(value)


def _rotation(value, label, tolerance=1e-6):
    rotation = np.asarray(value)
    if rotation.shape[-2:] != (3, 3) or rotation.dtype.kind not in "fiu" or not np.isfinite(rotation).all():
        raise ValueError(f"{label} must contain finite 3x3 rotations")
    if not np.allclose(rotation @ rotation.swapaxes(-1, -2), np.eye(3), atol=tolerance, rtol=0) or not np.allclose(np.linalg.det(rotation), 1., atol=tolerance, rtol=0):
        raise ValueError(f"{label} must contain proper orthonormal rotations; reflections are unsupported")
    return rotation.astype(float)


def _rigid(value, shape, label, tolerance=1e-6):
    transform = _finite(value, shape, label)
    if not np.allclose(transform[..., 3, :], [0, 0, 0, 1], atol=tolerance, rtol=0):
        raise ValueError(f"{label} must have rigid homogeneous last rows")
    _rotation(transform[..., :3, :3], label, tolerance)
    return transform


def hand_profile(side):
    """Return the tested G1 Dex3 recipe, using a proper rotation on each side."""
    if side not in ("left", "right"):
        raise ValueError("side must be left or right")
    transform = np.eye(4)
    if side == "right":
        transform[:3, :3] = [[-1, 0, 0], [0, 0, 1], [0, 1, 0]]
        transform[:3, 3] = [.1115, -.023, 0]
        signs = np.ones(7)
    else:
        transform[:3, :3] = [[-1, 0, 0], [0, 0, -1], [0, -1, 0]]
        transform[:3, 3] = [.1115, .023, 0]
        signs = np.array([-1, -1, 1, -1, -1, -1, -1])
    return dict(W_G=transform, qopen=_QOPEN * signs, qclose=_QCLOSE * signs,
                qmid=(_QOPEN + _QCLOSE) * .5 * signs,
                hand_joint_names=np.asarray([f"{side}_hand_{part}_joint" for part in _JOINT_SUFFIXES]))


def sample_surface(vertices, faces, count, seed):
    """Sample original triangles by physical area; retain material coordinates."""
    count = _integer(count, "point_count", 1)
    seed = _integer(seed, "seed")
    vertices = np.asarray(vertices)
    if vertices.ndim != 2 or vertices.shape[1:] != (3,) or len(vertices) < 3:
        raise ValueError("Object vertices must be a nonempty [V,3] array")
    vertices = _finite(vertices, vertices.shape, "Object vertices")
    faces = np.asarray(faces)
    if faces.ndim != 2 or faces.shape[1:] != (3,) or not len(faces) or faces.dtype.kind not in "iu" or np.any(faces < 0) or np.any(faces >= len(vertices)):
        raise ValueError("Object faces must be nonempty integer original triangle indices")
    tri = vertices[faces]
    area = np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)
    if not np.isfinite(area).all() or not np.isfinite(area.sum()) or area.sum() <= 0:
        raise ValueError("Object triangles have no finite positive surface area")
    # Zero-area original faces remain indexed but have zero sampling probability.
    rng = np.random.default_rng(seed)
    face_ids = rng.choice(len(faces), count, p=area / area.sum())
    uv = rng.random((count, 2))
    root = np.sqrt(uv[:, 0])
    bary = np.column_stack((1 - root, root * (1 - uv[:, 1]), root * uv[:, 1]))
    points = np.einsum("bi,bij->bj", bary, tri[face_ids])
    return points, face_ids, bary


def _native_model(native, model, compiled_digest):
    from .contact_edit import native_io

    qpos = native_io._native_qpos(native)
    if qpos.shape[1] != model.nq:
        raise ValueError("Native qpos dimension differs from compiled model")
    if "dof_names" not in native or "qposadr" not in native:
        raise ValueError("Native input requires explicit dof_names and qposadr model binding")
    snapshot = dict(joint_names=np.asarray([model.joint(i).name for i in range(model.njnt)]),
                    jnt_type=model.jnt_type.copy(), jnt_limited=model.jnt_limited.copy(),
                    jnt_range=model.jnt_range.copy(), jnt_qposadr=model.jnt_qposadr.copy())
    native_io._validate_native_binding(native, native_io._model_joints(snapshot))
    if "compiled_model_sha256" in native and (np.asarray(native["compiled_model_sha256"]).shape != () or str(native["compiled_model_sha256"].item()) != compiled_digest):
        raise ValueError("Compiled model hash differs from native binding")
    for name, neutral in (("scene_scale", 1), ("height_scale", 1), ("height_offset", 0), ("pelvis_rotation", 0)):
        if name in native and (np.asarray(native[name]).shape != () or float(native[name]) != neutral):
            raise ValueError("Display placement requires explicit native bake: " + name)
    if model.neq:
        raise ValueError("Equality-constrained models require a constraint-aware adapter")
    for joint, kind in enumerate(model.jnt_type):
        address = int(model.jnt_qposadr[joint])
        if kind in (0, 1):
            start = address + (3 if kind == 0 else 0)
            if not np.allclose(np.linalg.norm(qpos[:, start:start + 4], axis=1), 1, atol=1e-6, rtol=0):
                raise ValueError("Native quaternions must already be unit length")
        elif model.jnt_limited[joint] and (np.any(qpos[:, address] < model.jnt_range[joint, 0] - 1e-7) or np.any(qpos[:, address] > model.jnt_range[joint, 1] + 1e-7)):
            raise ValueError("Parent exceeds original native joint limits; no clipping is performed")
    count = len(qpos)
    rotation = _finite(native.get("obj_R"), (count, 3, 3), "Native obj_R")
    _rotation(rotation, "Native obj_R")
    translation = _finite(native.get("obj_t"), (count, 3), "Native obj_t")
    scale = np.asarray(native.get("object_scale"))
    if scale.shape not in ((), (3,)) or scale.dtype.kind not in "fiu" or not np.isfinite(scale).all() or np.any(scale <= 0):
        raise ValueError("Native object_scale must be positive scalar or XYZ; animated scale is unsupported")
    return qpos, rotation, translation, np.broadcast_to(scale, (3,)).astype(float).copy()


def _hand_mesh(model, data, qpos, side, state, profile):
    import mujoco

    data.qpos[:] = qpos
    joints = []
    for name, value in zip(profile["hand_joint_names"], state):
        joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, str(name))
        if joint < 0 or model.jnt_type[joint] != mujoco.mjtJoint.mjJNT_HINGE or not model.jnt_limited[joint]:
            raise ValueError(f"Hand profile requires named limited native hinge {name}")
        expected_body = str(name).removesuffix("_joint") + "_link"
        if model.body(int(model.jnt_bodyid[joint])).name != expected_body:
            raise ValueError(f"Hand profile joint {name} must belong to {expected_body}")
        limits = model.jnt_range[joint]
        if not np.isfinite(limits).all() or limits[0] >= limits[1] or not limits[0] <= value <= limits[1]:
            raise ValueError(f"Hand recipe violates original native limits for {name}; no clipping is performed")
        address = int(model.jnt_qposadr[joint])
        data.qpos[address] = value
        joints.append(dict(name=str(name), qposadr=address, limits=limits.tolist(), axis=model.jnt_axis[joint].tolist()))
    mujoco.mj_forward(model, data)
    wrist = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"{side}_wrist_yaw_link")
    if wrist < 0:
        raise ValueError("Hand profile requires native wrist_yaw_link")
    wrist_R, wrist_t = data.xmat[wrist].reshape(3, 3), data.xpos[wrist]
    vertices, faces, bodies, geoms = [], [], {}, []
    offset = 0
    for geom in range(model.ngeom):
        body = int(model.geom_bodyid[geom])
        ancestor = body
        while ancestor not in (0, wrist):
            ancestor = int(model.body_parentid[ancestor])
        if ancestor != wrist:
            continue
        name = model.body(body).name
        if model.geom_group[geom] != 2:
            continue
        if model.geom_type[geom] != mujoco.mjtGeom.mjGEOM_MESH:
            raise ValueError("Hand visual geometry must use original triangle meshes")
        mesh = int(model.geom_dataid[geom])
        va, vn = int(model.mesh_vertadr[mesh]), int(model.mesh_vertnum[mesh])
        fa, fn = int(model.mesh_faceadr[mesh]), int(model.mesh_facenum[mesh])
        local = model.mesh_vert[va:va + vn].astype(float)
        mesh_faces = model.mesh_face[fa:fa + fn].astype(np.int64)
        world = local @ data.geom_xmat[geom].reshape(3, 3).T + data.geom_xpos[geom]
        wrist_vertices = (world - wrist_t) @ wrist_R
        canonical = (wrist_vertices - profile["W_G"][:3, 3]) @ profile["W_G"][:3, :3]
        vertices.append(wrist_vertices)
        faces.append(mesh_faces + offset)
        bodies.setdefault(name, []).append(canonical)
        geoms.append(dict(geom_id=geom, body_name=name, mesh_name=model.mesh(mesh).name,
                          compiled_vertices_sha256=_array_sha(local), original_faces_sha256=_array_sha(mesh_faces),
                          canonical_vertices_sha256=_array_sha(canonical), vertex_count=vn, face_count=fn))
        offset += vn
    if not vertices:
        raise ValueError("No original visual group 2 hand meshes found")
    required = [f"{side}_hand_{part}_link" for part in _JOINT_SUFFIXES]
    if any(name not in bodies for name in required):
        raise ValueError("Hand profile lacks original visual meshes for each named digit link")
    return dict(vertices=np.concatenate(vertices), faces=np.concatenate(faces),
                bodies={name: np.concatenate(values) for name, values in bodies.items()}, geoms=geoms, joints=joints)


def _box(hand, side):
    thumb = hand["bodies"][f"{side}_hand_thumb_2_link"]
    fingers = np.concatenate([hand["bodies"][f"{side}_hand_index_1_link"], hand["bodies"][f"{side}_hand_middle_1_link"]])
    union = np.concatenate([thumb, fingers])
    low, high = union.min(axis=0), union.max(axis=0)
    left, right = (fingers, thumb) if fingers[:, 0].mean() < thumb[:, 0].mean() else (thumb, fingers)
    low[0], high[0] = left[:, 0].max(), right[:, 0].min()
    if np.any(high <= low):
        raise ValueError("Actual native hand sweep lacks a positive distal inner box")
    return (high - low).tolist(), ((high + low) / 2).tolist()


def prepare_inputs(config: dict, output: Path) -> dict:
    """Prepare one explicit native frame and hand; write three new bound files.

    Paths must already be absolute. ``hand_profile='g1_dex3'``, ``side`` and a
    zero-based integer ``frame`` are mandatory. The external controller owns
    configuration and output-directory creation. This function performs no IK,
    model inference, finger closure, trajectory edits or acceptance decision.
    """
    if not isinstance(config, dict) or config.get("hand_profile") != "g1_dex3":
        raise ValueError("Explicit hand_profile='g1_dex3' is required")
    side = config.get("side")
    profile = hand_profile(side)
    frame = _integer(config.get("frame"), "frame")
    count = _integer(config.get("point_count", 4096), "point_count", 1)
    seed = _integer(config.get("seed", 20260929), "seed")
    paths = {}
    for key in ("native", "model_xml", "object_mesh"):
        value = config.get(key)
        if not isinstance(value, (str, Path)) or not Path(value).is_absolute():
            raise ValueError(f"{key} must be an absolute path")
        paths[key] = Path(value).resolve(strict=True)
        if not paths[key].is_file():
            raise ValueError(f"{key} must be a file")
    output = Path(output)
    if not output.is_dir():
        raise ValueError("output must be an existing new directory")
    for name in ("input.npz", "hand_description.json", "inputs.json"):
        if (output / name).exists() or (output / name).is_symlink():
            raise FileExistsError("Refusing to overwrite " + str(output / name))
    bindings = {key: dict(path=str(path), sha256=_sha(path)) for key, path in paths.items()}
    expected = config.get("expected_input_sha256", {})
    if not isinstance(expected, dict) or set(expected) - set(paths):
        raise ValueError("expected_input_sha256 has unknown input keys")
    for key, digest in expected.items():
        if digest != bindings[key]["sha256"]:
            raise ValueError("Input hash mismatch: " + key)
    from .contact_edit import native_io
    from .contact_edit.pose import _model_dependencies
    import mujoco

    dependencies = {str(path): _sha(path) for path in _model_dependencies(paths["model_xml"])}
    model = mujoco.MjModel.from_xml_path(str(paths["model_xml"]))
    compiled = np.empty(mujoco.mj_sizeModel(model), dtype=np.uint8)
    mujoco.mj_saveModel(model, None, compiled)
    compiled_digest = hashlib.sha256(compiled.tobytes()).hexdigest()
    native = native_io.load_npz(paths["native"])
    qpos, rotations, translations, scale = _native_model(native, model, compiled_digest)
    if frame >= len(qpos):
        raise ValueError("frame is outside the native timeline")
    if "model_xml_sha256" in native and (np.asarray(native["model_xml_sha256"]).shape != () or str(native["model_xml_sha256"].item()) != bindings["model_xml"]["sha256"]):
        raise ValueError("Model XML hash differs from native binding")
    raw_mesh = native_io.load_npz(paths["object_mesh"])
    if "vertices" not in raw_mesh or "faces" not in raw_mesh:
        raise ValueError("Original object mesh NPZ requires vertices and faces")
    vertices = np.asarray(raw_mesh["vertices"])
    if vertices.ndim != 2 or vertices.shape[1:] != (3,):
        raise ValueError("Object vertices must have shape [V,3]")
    vertices = _finite(vertices, vertices.shape, "Original object vertices") * scale
    faces = np.asarray(raw_mesh["faces"])
    cloud, face_ids, barycentric = sample_surface(vertices, faces, count, seed)
    data = mujoco.MjData(model)
    open_hand = _hand_mesh(model, data, qpos[frame], side, profile["qopen"], profile)
    mid_hand = _hand_mesh(model, data, qpos[frame], side, profile["qmid"], profile)
    closed_hand = _hand_mesh(model, data, qpos[frame], side, profile["qclose"], profile)
    open_extent, open_offset = _box(open_hand, side)
    mid_extent, mid_offset = _box(mid_hand, side)
    description = dict(extents_open=open_extent, offset_open=open_offset, extents_mid=mid_extent,
                       offset_mid=mid_offset, gripper_type=2, fingertip_depth=open_offset[2] + open_extent[2] / 2)
    if description["fingertip_depth"] <= 0:
        raise ValueError("Actual native descriptor has nonpositive fingertip depth")
    data.qpos[:] = qpos[frame]
    mujoco.mj_forward(model, data)
    wrist = model.body(f"{side}_wrist_yaw_link").id
    world_wrist = np.eye(4)
    world_wrist[:3, :3], world_wrist[:3, 3] = data.xmat[wrist].reshape(3, 3), data.xpos[wrist]
    world_object = np.eye(4)
    world_object[:3, :3], world_object[:3, 3] = rotations[frame], translations[frame]
    wrist_object = np.linalg.inv(world_object) @ world_wrist
    hand_points, _, _ = sample_surface(open_hand["vertices"], open_hand["faces"], min(count, 3000), seed)
    with np.errstate(over="ignore"):
        inference_cloud = cloud.astype(np.float32)
    if not np.isfinite(inference_cloud).all():
        raise ValueError("Physical object surface cannot be represented as finite float32 inference points")
    packed = dict(object_pc=inference_cloud, object_face_ids=face_ids, object_barycentric=barycentric,
                  mesh_vertices=vertices, mesh_faces=faces.astype(np.int64), object_scale=scale,
                  obj_R=rotations[frame], obj_t=translations[frame], current_wrist_object=wrist_object,
                  current_qpos=qpos[frame].copy(), hand_open_points_wrist=hand_points,
                  frame=np.asarray(frame), side=np.asarray(side), fps=np.asarray(float(native["fps"])), **profile)
    report = dict(schema_version=1, status="PREPARED_UNREVIEWED", acceptance=False,
                  hand_profile="g1_dex3", side=side, frame=frame, fps=float(native["fps"]),
                  frame_count=len(qpos), point_count=count, seed=seed, bindings=bindings,
                  model_dependencies=dependencies, compiled_model_sha256=compiled_digest,
                  mujoco_version=mujoco.__version__, object_scale=scale.tolist(),
                  physical_vertices_sha256=_array_sha(vertices), W_G=profile["W_G"].tolist(),
                  coordinates="Meters; T_A_B maps B into A; physical object-local surface, no recentering",
                  scale_application="Original object vertices multiplied by native object_scale exactly once",
                  sampling="Deterministic original triangle area-weighted barycentric; zero-area faces have zero probability",
                  reference_semantics="Current bound native wrist at the requested frame, not the original GMR or source-video truth unless the caller explicitly binds that input",
                  hand_geometry="Actual MuJoCo visual group 2 mesh FK at open/mid/closed recipe states; wrist_yaw palm and descendant digits, excluding proximal wrist links",
                  hand_joint_binding=open_hand["joints"], hand_meshes={name: state["geoms"] for name, state in (("open", open_hand), ("mid", mid_hand), ("closed", closed_hand))},
                  checks_not_performed=["source-video agreement", "IK reachability", "contact correspondence", "whole-mesh collision", "sequence continuity", "force closure"])
    # Do not publish bindings if an external input changed while preparing it.
    for key, path in paths.items():
        if _sha(path) != bindings[key]["sha256"]:
            raise ValueError("Input changed during preparation: " + key)
    if any(_sha(path) != digest for path, digest in dependencies.items()):
        raise ValueError("Model dependency changed during preparation")
    with (output / "input.npz").open("xb") as stream:
        np.savez_compressed(stream, **packed)
    with (output / "hand_description.json").open("x") as stream:
        json.dump(description, stream, indent=2, allow_nan=False)
        stream.write("\n")
    report["files"] = {name: dict(path=str((output / name).resolve()), sha256=_sha(output / name)) for name in ("input.npz", "hand_description.json")}
    with (output / "inputs.json").open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    return report


def validate_proposals(grasps, confidence, inputs) -> dict:
    """Validate raw canonical-to-object proposals and derive native wrist poses.

    Inputs may be an already loaded mapping or the prepared ``input.npz`` path.
    Returned arrays preserve proposal order; no selection or edit is performed.
    """
    if isinstance(inputs, (str, Path)):
        from .contact_edit.native_io import load_npz
        inputs = load_npz(inputs)
    if not isinstance(inputs, Mapping):
        raise ValueError("inputs must be a prepared input mapping or NPZ path")
    raw = np.asarray(grasps)
    if raw.ndim != 3 or raw.shape[1:] != (4, 4) or len(raw) < 1:
        raise ValueError("grasps must contain a nonempty [N,4,4] transform batch")
    grasps = _rigid(raw, raw.shape, "grasps", 2e-5)
    _finite(confidence, (len(grasps),), "confidence")
    canonical_wrist = _rigid(inputs.get("W_G"), (4, 4), "W_G")
    reference = _rigid(inputs.get("current_wrist_object"), (4, 4), "current_wrist_object")
    world_object = np.eye(4)
    world_object[:3, :3] = _rotation(_finite(inputs.get("obj_R"), (3, 3), "obj_R"), "obj_R")
    world_object[:3, 3] = _finite(inputs.get("obj_t"), (3,), "obj_t")
    wrist_object = grasps @ np.linalg.inv(canonical_wrist)
    wrist_world = world_object[None] @ wrist_object
    shift = np.linalg.norm(wrist_object[:, :3, 3] - reference[:3, 3], axis=1)
    relative = wrist_object[:, :3, :3] @ reference[:3, :3].T
    angle = np.rad2deg(np.arccos(np.clip((np.trace(relative, axis1=1, axis2=2) - 1) * .5, -1, 1)))
    return dict(wrist_object=wrist_object, wrist_world=wrist_world,
                source_shift_m=shift, source_rotation_deg=angle)


def verify_bindings(report: dict) -> None:
    """Reject changed source/model assets before using or publishing proposals."""
    if not isinstance(report, dict) or report.get("schema_version") != 1:
        raise ValueError("Expected a prepared inputs schema_version 1 report")
    bindings = report.get("bindings")
    if not isinstance(bindings, dict) or set(bindings) != {"native", "model_xml", "object_mesh"}:
        raise ValueError("Prepared input bindings are incomplete")
    for key, binding in bindings.items():
        if _sha(binding["path"]) != binding["sha256"]:
            raise ValueError("Bound input changed: " + key)
    from .contact_edit.pose import _model_dependencies
    import mujoco

    dependencies = {str(path): _sha(path) for path in _model_dependencies(Path(bindings["model_xml"]["path"]))}
    if dependencies != report.get("model_dependencies"):
        raise ValueError("Bound model dependencies changed")
    model = mujoco.MjModel.from_xml_path(bindings["model_xml"]["path"])
    compiled = np.empty(mujoco.mj_sizeModel(model), dtype=np.uint8)
    mujoco.mj_saveModel(model, None, compiled)
    if hashlib.sha256(compiled.tobytes()).hexdigest() != report.get("compiled_model_sha256"):
        raise ValueError("Bound compiled model changed")
    files = report.get("files")
    if not isinstance(files, dict) or set(files) != {"input.npz", "hand_description.json"}:
        raise ValueError("Prepared file bindings are incomplete")
    for name, binding in files.items():
        if _sha(binding["path"]) != binding["sha256"]:
            raise ValueError("Prepared input file changed: " + name)
