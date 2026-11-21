"""Prepare explicit native-mesh pairs for SOMA's sampled clearance objective."""
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np


def _static_scene(config, config_dir):
    """Bind packed original geometry through an explicitly declared world map."""
    from .static_scene import static_scene_fingerprint, static_scene_payload

    if (not isinstance(config, dict)
            or set(config) != {"native_npz", "world_frame", "world_to_solver"}
            or not isinstance(config["native_npz"], str) or not config["native_npz"]
            or not isinstance(config["world_frame"], str) or not config["world_frame"]):
        raise ValueError("static_scene requires native_npz, world_frame and world_to_solver")
    matrix = np.asarray(config["world_to_solver"], dtype=float)
    if (matrix.shape != (4, 4) or not np.isfinite(matrix).all()
            or not np.array_equal(matrix[3], [0, 0, 0, 1])
            or not np.allclose(matrix[:3, :3] @ matrix[:3, :3].T, np.eye(3), rtol=0, atol=1e-8)
            or not np.isclose(np.linalg.det(matrix[:3, :3]), 1, rtol=0, atol=1e-8)):
        raise ValueError("static_scene world_to_solver must be an explicit rigid metre-space transform")
    path = (Path(config_dir) / config["native_npz"]).resolve(strict=True)
    with np.load(path, allow_pickle=False) as native:
        manifest, arrays = static_scene_payload(native, world_frame=config["world_frame"])
    if manifest["status"] == "UNKNOWN":
        raise ValueError("Requested static-scene collision geometry is UNKNOWN")
    meshes = []
    for mesh in manifest["meshes"]:
        vertices = arrays[mesh["vertices_key"]] @ matrix[:3, :3].T + matrix[:3, 3]
        meshes.append((vertices, arrays[mesh["faces_key"]], "static_scene:" + mesh["name"]))
    summary = dict(status=manifest["status"], native_npz=str(path),
        native_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        scene_sha256=static_scene_fingerprint(manifest, arrays), manifest=manifest,
        world_to_solver=matrix.tolist(), solver_world_frame="soma_retargeter_z_up_m")
    return meshes, summary


def compile_collision(config, model_xml, object_vertices, object_faces, rotations, positions, *, config_dir=".",
                      include_object_scene=False):
    import mujoco
    from scipy.spatial.transform import Rotation
    from .repair_geometry import TriangleMesh

    if type(include_object_scene) is not bool:
        raise ValueError("include_object_scene must be an explicit boolean")
    required = {"robot_geoms", "self_pairs", "samples_per_mesh", "margin_m", "weight"}
    optional = {"static_scene", "object_degenerate_policy", "witness_groups"}
    if not isinstance(config, dict) or required - set(config) or set(config) - required - optional:
        raise ValueError(f"collision_avoidance requires {sorted(required)}; optional {sorted(optional)}")
    object_policy = config.get("object_degenerate_policy", "strict")
    if not isinstance(object_policy, str) or object_policy not in ("strict", "exclude_zero_area_queries"):
        raise ValueError("Unknown object_degenerate_policy")
    count, margin, weight = (config[k] for k in ("samples_per_mesh", "margin_m", "weight"))
    if (type(count) is not int or not 16 <= count <= 1024
            or type(margin) not in (int, float) or not np.isfinite(margin) or not 0 <= margin <= .01
            or type(weight) not in (int, float) or not np.isfinite(weight) or weight <= 0):
        raise ValueError("Invalid collision sample count, clearance margin or weight")
    model = mujoco.MjModel.from_xml_path(str(model_xml))
    geoms = config["robot_geoms"]
    if (not isinstance(geoms, list) or not geoms or any(type(g) is not int for g in geoms)
            or len(set(geoms)) != len(geoms)):
        raise ValueError("robot_geoms must list unique explicit native mesh geom IDs")
    meshes = [(np.asarray(object_vertices, dtype=float), np.asarray(object_faces), "object")]
    labels = ["object"]
    body_types = ["object"]
    native_identities = {}
    for g in geoms:
        if not 0 <= g < model.ngeom or model.geom_type[g] != mujoco.mjtGeom.mjGEOM_MESH:
            raise ValueError("collision robot geom is not an original native mesh")
        mesh = model.geom_dataid[g]
        va, vn = model.mesh_vertadr[mesh], model.mesh_vertnum[mesh]
        fa, fn = model.mesh_faceadr[mesh], model.mesh_facenum[mesh]
        rotation = Rotation.from_quat(model.geom_quat[g][[1, 2, 3, 0]]).as_matrix()
        # As in motion_audit: MuJoCo retains triangle geometry in its compiled
        # mesh frame; geom transform restores original coordinates in the body.
        vertices = model.mesh_vert[va:va+vn].astype(float) @ rotation.T + model.geom_pos[g]
        faces = model.mesh_face[fa:fa+fn].copy()
        if "witness_groups" in config:
            from .soma_collision_witnesses import native_geom_fingerprint
            native_identities[g] = native_geom_fingerprint(
                model.mesh_vert[va:va+vn], faces, model.geom_pos[g], model.geom_quat[g])
        body = model.body(int(model.geom_bodyid[g])).name
        if not body:
            raise ValueError("Collision mesh must belong to a named native body")
        meshes.append((vertices, faces, body))
        labels.append(f"geom:{g}")
        body_types.append("robot")
    index = {g: i+1 for i, g in enumerate(geoms)}
    pairs = [(i, 0) for i in range(1, len(meshes))]
    self_pairs = config["self_pairs"]
    if not isinstance(self_pairs, list):
        raise ValueError("self_pairs must explicitly list pairs; [] means object-only")
    seen = set()
    for pair in self_pairs:
        if (not isinstance(pair, list) or len(pair) != 2 or any(type(g) is not int or g not in index for g in pair)
                or pair[0] == pair[1] or tuple(sorted(pair)) in seen):
            raise ValueError("Invalid or duplicate self collision pair")
        seen.add(tuple(sorted(pair)))
        pairs.append((index[pair[0]], index[pair[1]]))
    scene = dict(status="NOT_CONFIGURED")
    if "static_scene" in config:
        scene_meshes, scene = _static_scene(config["static_scene"], config_dir)
        for mesh in scene_meshes:
            scene_index = len(meshes)
            meshes.append(mesh)
            labels.append(mesh[2])
            body_types.append("static")
            pairs.extend((robot_index, scene_index) for robot_index in index.values())
            if include_object_scene:
                pairs.append((0, scene_index))
    vertices, faces, voff, foff, samples, records = [], [], [0], [0], [], []
    query_faces, query_face_ids, query_face_offsets = [], [], [0]
    for mesh_index, (v, f, body) in enumerate(meshes):
        geometry = TriangleMesh(v, f)
        valid = geometry.normal_valid.copy()
        cast = v.astype(np.float32)
        if not np.isfinite(cast).all():
            raise ValueError("Collision query vertices exceed float32 coordinates")
        cast_valid = TriangleMesh(cast, f).normal_valid
        if body_types[mesh_index] == "static":
            # Preserve full original geometry below. Query only original faces
            # with a usable normal, recording omissions and the exact face map.
            valid &= cast_valid
        else:
            if not valid.all():
                if body_types[mesh_index] != "object" or object_policy == "strict":
                    raise ValueError("Collision query mesh contains degenerate original triangles")
                triangles = geometry.triangles[~valid]
                cross = np.cross(triangles[:, 1]-triangles[:, 0], triangles[:, 2]-triangles[:, 0])
                if not np.all(cross == 0):
                    raise ValueError("Object query exclusions support only exact zero-area original faces")
            if body_types[mesh_index] == "object" and not cast_valid[valid].all():
                raise ValueError("Original object query faces are degenerate after float32 conversion")
        if not valid.any():
            raise ValueError("Collision query mesh has no usable original triangles")
        query_ids = np.flatnonzero(valid)
        query_f = f[query_ids]
        source_vertices = v[np.unique(query_f)] if not valid.all() else v
        # Original vertices and triangle centroids; no convexification or remesh.
        nv, nf = min(count//2, len(source_vertices)), min(count-count//2, len(query_f))
        sample = np.concatenate([source_vertices[np.linspace(0, len(source_vertices)-1, nv, dtype=int)],
            v[query_f[np.linspace(0, len(query_f)-1, nf, dtype=int)]].mean(axis=1)])
        samples.append(sample)
        vertices.append(v); faces.append(f); voff.append(voff[-1]+len(v)); foff.append(foff[-1]+len(f))
        query_faces.append(query_f); query_face_ids.append(query_ids)
        query_face_offsets.append(query_face_offsets[-1] + len(query_f))
        # Topology metadata only: weld exact coincident coordinates to recognize
        # STL seams without changing the actual query geometry.
        _, ids = np.unique(v, axis=0, return_inverse=True)
        tf = ids[query_f]
        edges = np.concatenate([tf[:, [0, 1]], tf[:, [1, 2]], tf[:, [2, 0]]])
        _, inv, counts = np.unique(np.sort(edges, axis=1), axis=0, return_inverse=True, return_counts=True)
        balance = np.zeros(len(counts), dtype=int)
        np.add.at(balance, inv, np.where(edges[:, 0] < edges[:, 1], 1, -1))
        records.append(dict(body=body, vertices=len(v), faces=len(f), samples=len(sample),
            query_faces=len(query_f), query_geometry_complete=bool(valid.all()),
            excluded_original_degenerate_faces=int((~geometry.normal_valid).sum()),
            excluded_float32_degenerate_faces=int((geometry.normal_valid & ~cast_valid & ~valid).sum()),
            max_float32_vertex_error_m=float(np.max(np.abs(v - cast.astype(float)))),
            topology_scope="queried original triangle subset",
            oriented_closed=bool(np.all(counts == 2) and np.all(balance == 0)),
            boundary_edges=int((counts == 1).sum()), nonmanifold_edges=int((counts > 2).sum())))
        if body_types[mesh_index] == "object":
            records[-1].update(
                query_policy=object_policy,
                excluded_original_zero_area_face_ids=np.flatnonzero(~valid).tolist(),
                query_face_to_original_sha256=hashlib.sha256(query_ids.astype("<i8").tobytes()).hexdigest(),
                original_geometry_sha256=hashlib.sha256(geometry.vertices.tobytes()+geometry.faces.tobytes()).hexdigest(),
                query_geometry_sha256=hashlib.sha256(geometry.vertices.tobytes()+query_f.astype(np.int64).tobytes()).hexdigest(),
                original_geometry_preserved=True)
    qpoints, qa, qb, qp = [], [], [], []
    offsets = [0]
    for pair_index, (a, b) in enumerate(pairs):
        for source, target in ((a, b), (b, a)):
            p = samples[source]
            qpoints.extend(p); qa.extend([source]*len(p)); qb.extend([target]*len(p)); qp.extend([pair_index]*len(p))
        offsets.append(len(qpoints))
    witness_reports = []
    query_weights = None
    if "witness_groups" in config:
        from .soma_collision_witnesses import compile_witness_groups
        original_weights = np.repeat(1./np.diff(offsets), np.diff(offsets))
        extra_weights = []
        groups = compile_witness_groups(config["witness_groups"], meshes=meshes,
            geom_to_mesh=index, existing_pairs=pairs, native_identities=native_identities)
        for group in groups:
            a, b, points = group['source'], group['target'], group['points']
            pair_index = len(pairs)
            pairs.append((a, b))
            qpoints.extend(points); qa.extend([a]*len(points)); qb.extend([b]*len(points)); qp.extend([pair_index]*len(points))
            offsets.append(len(qpoints))
            extra_weights.append(group['query_weights'])
            witness_reports.append(group['report'])
        query_weights = np.concatenate([original_weights, *extra_weights])
    omitted = [list(p) for p in itertools.combinations(sorted(geoms), 2) if p not in seen]
    summary = dict(status="SAMPLED_CLEARANCE_OBJECTIVE_NOT_AUDIT", backend="Warp winding-distance / Newton analytic IK",
        robot_geoms=geoms, mesh_labels=labels, meshes=records,
        static_scene=scene,
        object_degenerate_policy=object_policy,
        query_geometry_complete=all(record["query_geometry_complete"] for record in records),
        motion_accepted=False,
        include_object_scene=include_object_scene,
        omitted_object_scene_pairs=([["object", label] for label, kind in zip(labels, body_types)
                                     if kind == "static"] if not include_object_scene else []),
        pairs=[[labels[a], labels[b]] for a, b in pairs], bidirectional_queries=len(qpoints),
        margin_m=margin, weight=weight, omitted_self_pairs=omitted,
        omitted_model_mesh_geoms=[i for i in range(model.ngeom) if model.geom_type[i] == mujoco.mjtGeom.mjGEOM_MESH and i not in geoms],
        limitations=["Finite samples and winding signs are optimization heuristics, not clearance proof.",
            "No inherited overlap is waived; only explicitly listed self pairs enter the objective.",
            "Only bound static-scene meshes enter environment avoidance; absent geometry is unknown.",
            "Degenerate static-scene faces are retained as original evidence but omitted from queries with an original-face map; this is incomplete geometry coverage.",
            "Explicit zero-area object-query exclusions retain original faces and material IDs; omitted faces keep collision coverage incomplete, never cleared.",
            ("Moving-object versus scene pairs enter the same sampled clearance objective; support still needs an independent check."
             if include_object_scene else
             "Moving-object versus scene pairs are omitted; object motion edits require explicitly enabling this coverage."),
            "Check all relevant original triangles and subframes independently."])
    arrays = dict(collision_vertices=np.concatenate(vertices), collision_faces=np.concatenate(faces),
        collision_vertex_offsets=np.asarray(voff), collision_face_offsets=np.asarray(foff),
        collision_query_mesh_faces=np.concatenate(query_faces),
        collision_query_mesh_face_offsets=np.asarray(query_face_offsets),
        collision_query_original_face_ids=np.concatenate(query_face_ids),
        collision_bodies=np.asarray([x[2] for x in meshes]), collision_body_types=np.asarray(body_types),
        collision_query_points=np.asarray(qpoints),
        collision_query_a=np.asarray(qa), collision_query_b=np.asarray(qb), collision_query_pair=np.asarray(qp),
        collision_pair_offsets=np.asarray(offsets), collision_margin=np.asarray(margin), collision_weight=np.asarray(weight),
        collision_object_poses=np.c_[positions, Rotation.from_matrix(rotations).as_quat()],
        collision_metadata=np.asarray(json.dumps(summary)))
    if query_weights is not None:
        summary['witness_groups'] = witness_reports
        summary['witness_objective'] = 'Original pair means unchanged plus weight * mean(hinge squared) per explicitly added original-surface group'
        arrays['collision_query_weights'] = query_weights
        arrays['collision_metadata'] = np.asarray(json.dumps(summary))
    return arrays, summary
