"""Batched free-wrist surface-grasp proposals; no motion is accepted here."""
import argparse
import json
from pathlib import Path
import time

import numpy as np

from .contact_edit.pose import _integer, _keys
from .contact_patches import extract_contact_patch, validate_contact_patch, create_anchor_query_session
from .grasp_io import GraspInputs, code_binding, digest, indices, number, write_report
from .grasp_multistart import make_initial_deltas, validate_multistart
from .mesh_screen import MeshScreen
from .repair_geometry import TriangleMesh, closest_points


def propose(config, directory="."):
    import mujoco
    import torch
    from .repair_tensor import NativeTensorFK, WarpSignedDistance

    started = time.monotonic()
    _keys(config, {"inputs", "seed_frames", "device", "steps", "screen_backend"},
          {"samples_per_geom", "translation_limit_m", "rotation_limit_rad", "clearance_m", "learning_rate", "self_clearance_weight", "scene_clearance_weight", "retained_self_pairs", "contact_region_faces", "contact_patch", "multistart", "topo_objective", "contact_surface", "warm_start_bank", "anchor_queries"}, "bank")
    anchor_queries = create_anchor_query_session(config.get('anchor_queries'))
    robot_surface_query = anchor_queries.robot_surface if anchor_queries is not None else closest_points
    object_surface_query = anchor_queries.object_surface if anchor_queries is not None else closest_points
    ctx = GraspInputs(config["inputs"], directory)
    patch = validate_contact_patch(config.get("contact_patch"), ctx.hands)
    contact_surface = config.get("contact_surface", "all")
    if contact_surface not in ("all", "dex3_distal_pad"):
        raise ValueError("contact_surface must be all or dex3_distal_pad")
    if contact_surface == "dex3_distal_pad" and patch is None:
        raise ValueError("Distal-pad contact requires explicit per-member contact patches")
    topo_options = config.get("topo_objective")
    if topo_options is not None:
        _keys(topo_options, {"source_arrays"}, {"graph_weight", "direction_weight", "graph_decay",
              "object_sample_count", "fallback_neighbors", "node_set"}, "topo_objective")
        topo_options = dict(topo_options, method="topo",
                            source_arrays=str((Path(directory) / topo_options["source_arrays"]).resolve(strict=True)))
    warm_bank, warm_binding = None, None
    if "warm_start_bank" in config:
        from .grasp_bank_seed import load_bank_seed
        warm_bank, warm_binding = load_bank_seed(config["warm_start_bank"], directory, ctx)
    m, n = ctx.model, ctx.native
    if set(config["seed_frames"]) != set(ctx.hands):
        raise ValueError("seed_frames must cover exactly the configured hands")
    steps = _integer(config["steps"], "steps", 1)
    samples = _integer(config.get("samples_per_geom", 800), "samples_per_geom", 16)
    if steps > 2000 or samples > 4096:
        raise ValueError("Use a bounded bank: steps <= 2000, samples_per_geom <= 4096")
    limits = [number(config.get("translation_limit_m", .08), "translation limit", 1e-6, 1),
              number(config.get("rotation_limit_rad", .8), "rotation limit", 1e-6, np.pi)]
    multistart = validate_multistart(config.get("multistart"), *limits)
    margin = number(config.get("clearance_m", .0015), "clearance", 0, .01)
    rate = number(config.get("learning_rate", .008), "learning_rate", 1e-6, .1)
    self_weight = number(config.get("self_clearance_weight", 0), "self_clearance_weight")
    scene_weight = number(config.get("scene_clearance_weight", 0), "scene_clearance_weight")
    retained = config.get("retained_self_pairs", {})
    if not isinstance(retained, dict) or set(retained)-set(ctx.hands):
        raise ValueError("retained_self_pairs must map configured hands to explicit pairs")
    device = config["device"]
    if not isinstance(device, str) or not (device == "cpu" or device.startswith("cuda:")):
        raise ValueError("device must explicitly be cpu or cuda:N")
    fk = NativeTensorFK(m, device=device, dtype=torch.float32)
    object_query = WarpSignedDistance(ctx.object_query.vertices, ctx.object_query.faces, device=device)
    scene_clearance, scene_meshes = None, {}
    if scene_weight:
        from .hand_scene_clearance import HandSceneClearance
        from .static_scene import static_scene_payload
        scene_clearance = HandSceneClearance(ctx.native, device=device)
        if scene_clearance.enabled:
            manifest, arrays = static_scene_payload(ctx.native)
            scene_meshes = {"static_scene:" + item["name"]: TriangleMesh(
                arrays[item["vertices_key"]], arrays[item["faces_key"]])
                for item in manifest["meshes"]}
    d = mujoco.MjData(m)
    candidates, coverage = {}, {}
    contact_regions = config.get("contact_region_faces", {})
    if not isinstance(contact_regions, dict) or set(contact_regions)-set(ctx.hands):
        raise ValueError("contact_region_faces must map configured hands to contact groups")

    def synchronize():
        if device.startswith("cuda:"):
            torch.cuda.synchronize(device)

    synchronize()
    timings = dict(common_setup_seconds=time.monotonic()-started, hands={},
                   cuda_synchronized=device.startswith("cuda:"))
    for name, hand in ctx.hands.items():
        hand_started = time.monotonic()
        frames = indices(config["seed_frames"][name], len(ctx.q), "seed_frames")
        wrist, addresses, geoms = hand["wrist"], hand["addresses"], hand["robot_geoms"]
        initial_delta, start_records = None, None
        if multistart is not None:
            rows, initial_delta, start_records = make_initial_deltas(
                multistart, name, frames, ctx.q[frames][:, addresses],
                m.jnt_range[hand["joints"], 0], m.jnt_range[hand["joints"], 1])
            frames = [frames[row] for row in rows]
        retained_pairs, retained_reasons = [], []
        for item in retained.get(name, []):
            _keys(item, {"pair", "reason"}, set(), "retained self pair")
            pair = indices(item["pair"], m.ngeom, "retained self pair")
            if (len(pair) != 2 or not set(pair).issubset(geoms)
                    or not isinstance(item["reason"], str) or not item["reason"].strip()):
                raise ValueError("Retained self pairs need two bound hand geoms and a reason")
            retained_pairs.append(tuple(sorted(pair))); retained_reasons.append(item)
        if len(set(retained_pairs)) != len(retained_pairs):
            raise ValueError("Duplicate retained self pair")
        points, bodies, point_geoms = [], [], []
        for g in geoms:
            mesh = ctx.meshes[g]
            nv, nf = min(samples//2, len(mesh.vertices)), min(samples-samples//2, len(mesh.faces))
            p = np.concatenate([mesh.vertices[np.linspace(0, len(mesh.vertices)-1, nv, dtype=int)],
                                mesh.triangles[np.linspace(0, len(mesh.faces)-1, nf, dtype=int)].mean(axis=1)])
            points.extend(p); bodies.extend([ctx.bodies[g]]*len(p)); point_geoms.extend([g]*len(p))
        points = np.asarray(points)
        group_ids = {k: np.flatnonzero(np.isin(point_geoms, ids)) for k, ids in hand["contact_groups"].items()}
        groups = {k: torch.tensor(ids, device=device) for k, ids in group_ids.items()}
        contact_mask = np.ones(len(points), dtype=bool)
        pad_evidence = None
        if contact_surface == "dex3_distal_pad":
            from .grasp_distal_pad import dex3_distal_pad_mask
            contact_mask, pad_evidence = dex3_distal_pad_mask(ctx, name, points, point_geoms)
        member_ids = {}
        if patch is not None:
            for group, members in patch["members"][name].items():
                member_ids[group] = {}
                for member, member_geoms in members.items():
                    ids = np.flatnonzero(np.isin(point_geoms, member_geoms) & contact_mask)
                    if not len(ids):
                        raise ValueError("No contact material samples for declared member")
                    # Native visual/collision geoms may duplicate the same mesh.
                    # They must not multiply one material point's attraction.
                    keys = np.column_stack((np.asarray(bodies)[ids], np.round(points[ids], 10)))
                    _, unique = np.unique(keys, axis=0, return_index=True)
                    member_ids[group][member] = torch.tensor(ids[np.sort(unique)], device=device)
        pt, bi = fk.t(points), torch.tensor(bodies, device=device)
        # Bidirectional original-surface queries keep self clearance in the
        # proposal objective. World/free-wrist placement cancels from these
        # relative transforms. The independent triangle screen remains required.
        self_terms = []
        if self_weight:
            for target in geoms:
                tb = ctx.bodies[target]
                allowed = [g for g in geoms if ctx.bodies[g] != tb
                           and m.body_parentid[ctx.bodies[g]] != tb
                           and m.body_parentid[tb] != ctx.bodies[g]
                           and tuple(sorted((g, target))) not in retained_pairs]
                ids = np.flatnonzero(np.isin(point_geoms, allowed))
                if len(ids):
                    self_terms.append((tb, torch.tensor(ids, device=device),
                        WarpSignedDistance(ctx.meshes[target].vertices,
                                           ctx.meshes[target].faces, device=device)))
        qbase = fk.t(ctx.q[frames])
        object_R, object_t = fk.t(ctx.R[frames]), fk.t(ctx.t[frames])
        wp, wr = [], []
        for frame in frames:
            d.qpos[:] = ctx.q[frame]; mujoco.mj_forward(m, d)
            wp.append(ctx.R[frame].T @ (d.xpos[wrist]-ctx.t[frame]))
            wr.append(ctx.R[frame].T @ d.xmat[wrist].reshape(3, 3))
        wp, wr = fk.t(np.asarray(wp)), fk.t(np.asarray(wr))
        region_faces = np.asarray(hand["region_faces"])
        region_mesh = TriangleMesh(ctx.object.vertices, ctx.object.faces[region_faces])
        region = WarpSignedDistance(region_mesh.vertices, region_mesh.faces, device=device)
        overrides = contact_regions.get(name, {})
        if not isinstance(overrides, dict) or set(overrides)-set(groups):
            raise ValueError("Contact region overrides must name configured contact groups")
        group_regions = {}
        for group, faces in overrides.items():
            indices(faces, len(ctx.object.faces), "contact region faces")
            if not set(faces).issubset(region_faces):
                raise ValueError("Contact group region must be inside the declared hand region")
            gm = TriangleMesh(ctx.object.vertices, ctx.object.faces[faces])
            group_regions[group] = (np.asarray(faces), gm,
                WarpSignedDistance(gm.vertices, gm.faces, device=device))
        topo = None
        if topo_options is not None:
            from .grasp_paper_objectives import build_paper_objective
            topo = build_paper_objective(ctx, name, hand, frames, fk, topo_options)
        warm_reviews = None
        if warm_bank is not None:
            from .grasp_bank_seed import bank_seed_deltas
            initial_delta, warm_reviews = bank_seed_deltas(
                warm_bank, name, hand, frames, start_records, wp.detach().cpu().numpy(),
                wr.detach().cpu().numpy(), ctx.q[frames][:, addresses],
                m.jnt_range[hand["joints"]], limits)
        x = torch.nn.Parameter(torch.zeros((len(frames), 6+len(addresses)), device=device)
                               if initial_delta is None else fk.t(initial_delta))
        optimizer = torch.optim.Adam([x], lr=rate)
        basefinger = qbase[:, addresses].clone()
        low, high = fk.t(m.jnt_range[hand["joints"], 0]), fk.t(m.jnt_range[hand["joints"], 1])

        def evaluate():
            q = qbase.clone(); q[:, addresses] = basefinger+x[:, 6:]
            rot, pos = fk(q)
            local = torch.einsum("bji,bnj->bni", rot[:, wrist],
                torch.einsum("bnij,nj->bni", rot[:, bi], pt)+pos[:, bi]-pos[:, wrist, None])
            rv = x[:, 3:6]; zero = torch.zeros_like(rv[:, 0])
            skew = torch.stack([zero, -rv[:, 2], rv[:, 1], rv[:, 2], zero, -rv[:, 0],
                                -rv[:, 1], rv[:, 0], zero], -1).reshape(-1, 3, 3)
            rr, pp = torch.matrix_exp(skew) @ wr, wp+x[:, :3]
            p = torch.einsum("bij,bnj->bni", rr, local)+pp[:, None]
            dist, rd = object_query.distance(p), region.distance(p).abs()
            penetration = torch.relu(margin-dist)
            loss = 400*(penetration**2).mean(1)+20*penetration.max(1).values**2
            if scene_clearance is not None and scene_clearance.enabled:
                scene_loss, _ = scene_clearance.penalty(p, object_R, object_t, margin)
                loss = loss + scene_weight * scene_loss
            if self_terms:
                world = torch.einsum("bnij,nj->bni", rot[:, bi], pt)+pos[:, bi]
                violations = []
                for body, ids, query in self_terms:
                    relative = torch.einsum("bji,bnj->bni", rot[:, body],
                                           world[:, ids]-pos[:, body, None])
                    violation = torch.relu(margin-query.distance(relative))
                    violations.append(violation.topk(min(12, len(ids)), dim=1).values)
                merged = torch.cat(violations, dim=1)
                loss = loss+self_weight*merged.topk(min(12, merged.shape[1]), dim=1).values.square().mean(1)
            for group, ids in groups.items():
                if patch is None:
                    gd = group_regions[group][2].distance(p[:, ids]).abs() if group in group_regions else rd[:, ids]
                    loss = loss+2*((gd.topk(min(8, len(ids)), largest=False).values-margin)**2).mean(1)
                else:
                    members = member_ids[group]
                    for ids in members.values():
                        gd = group_regions[group][2].distance(p[:, ids]).abs() if group in group_regions else rd[:, ids]
                        selected = gd.topk(min(patch["points_per_member"], len(ids)), largest=False).values
                        loss = loss+2*((selected-margin)**2).mean(1)/len(members)
            loss = loss+.002*x[:, :3].square().sum(1)+.00001*x[:, 3:6].square().sum(1)+.000002*x[:, 6:].square().sum(1)
            if topo is not None:
                loss = loss + topo.evaluate(rr, pp, rot, pos)
            return loss, p, rr, pp, q, rot, pos, dist, rd

        synchronize()
        optimization_started = time.monotonic()
        hand_timing = dict(setup_seconds=optimization_started-hand_started,
                           candidate_count=len(frames), steps_per_candidate=steps,
                           screening_seconds=0., anchor_extraction_seconds=0.)
        if topo is not None:
            hand_timing["topo_objective"] = topo.metadata
        if pad_evidence is not None:
            hand_timing["contact_surface"] = pad_evidence
        if warm_binding is not None:
            hand_timing["warm_start"] = dict(warm_binding, row_count=len(frames))
        for _ in range(steps):
            optimizer.zero_grad()
            loss = evaluate()[0]
            if not torch.isfinite(loss).all():
                raise ValueError("Nonfinite surface-grasp objective")
            loss.sum().backward(); optimizer.step()
            with torch.no_grad():
                x[:, :3].clamp_(-limits[0], limits[0]); x[:, 3:6].clamp_(-limits[1], limits[1])
                x[:, 6:] = torch.minimum(torch.maximum(basefinger+x[:, 6:], low), high)-basefinger
        with torch.no_grad():
            values = [v.detach().cpu().numpy() for v in evaluate()]
        synchronize()
        hand_timing["optimization_seconds"] = time.monotonic()-optimization_started
        loss, pts, rr, pp, qq, rots, poss, distances, regional = values
        scene_minimum = None
        if scene_clearance is not None and scene_clearance.enabled:
            with torch.no_grad():
                _, scene_distances = scene_clearance.penalty(fk.t(pts), object_R, object_t, margin)
                scene_minimum = scene_distances.cpu().numpy()
        screening_setup_started = time.monotonic()
        bound = {f"geom:{g}": ctx.meshes[g] for g in geoms}
        bound["object"] = ctx.object
        pairs = [(f"geom:{g}", "object") for g in geoms]
        bound.update(scene_meshes)
        pairs.extend((f"geom:{g}", label) for g in geoms for label in scene_meshes)
        omitted = []
        for i, a in enumerate(geoms):
            for b in geoms[i+1:]:
                ba, bb = ctx.bodies[a], ctx.bodies[b]
                pair = [f"geom:{a}", f"geom:{b}"]
                if ba == bb or m.body_parentid[ba] == bb or m.body_parentid[bb] == ba:
                    omitted.append(pair)
                else:
                    pairs.append(tuple(pair))
        retained_labels = [(f"geom:{a}", f"geom:{b}") for a, b in retained_pairs]
        if not set(retained_labels).issubset(pairs):
            raise ValueError("Retained pair must be in the nonadjacent screened scope")
        screen = MeshScreen(bound, pairs, retained_pairs=retained_labels, backend=config["screen_backend"])
        # A retained overlap needs independently reproducible neutral-model
        # evidence; declaring a new crossing 'assembly' is not sufficient.
        neutral_evidence = None
        if retained_pairs:
            d.qpos[:] = m.qpos0; mujoco.mj_forward(m, d)
            neutral = {"object": np.eye(4)}
            neutral.update({label: np.eye(4) for label in scene_meshes})
            for g in geoms:
                b = ctx.bodies[g]; mat = np.eye(4)
                mat[:3, :3] = d.xmat[b].reshape(3, 3); mat[:3, 3] = d.xpos[b]
                neutral[f"geom:{g}"] = mat
            neutral_evidence = MeshScreen(bound, retained_labels, backend=config["screen_backend"]).query(neutral)
            found_pairs = {tuple(h["pair"]) for h in neutral_evidence["surface_hits"]}
            if found_pairs != set(retained_labels):
                raise ValueError("Retained pair has no confirmed neutral-model crossing")
        coverage[name] = dict(pairs=pairs, omitted_same_body_or_adjacent_pairs=omitted,
                              retained_self_pairs=retained_reasons, neutral_model_evidence=neutral_evidence,
                              omitted_robot_geoms=[g for g in range(m.ngeom) if g not in geoms],
                              environment="ORIGINAL_SURFACE_FIRST_CROSSING" if scene_meshes else "NOT_CHECKED",
                              containment="NOT_CHECKED")
        hand_timing["screening_setup_seconds"] = time.monotonic()-screening_setup_started
        records = []
        for i, frame in enumerate(frames):
            screening_started = time.monotonic()
            transforms = {"object": np.eye(4)}
            # The packed scene stays in native-world metres; the candidate
            # screen uses object coordinates for its actual seed frame.
            world_to_object = np.eye(4)
            world_to_object[:3, :3] = ctx.R[frame].T
            world_to_object[:3, 3] = -ctx.R[frame].T @ ctx.t[frame]
            transforms.update({label: world_to_object for label in scene_meshes})
            for g in geoms:
                b = ctx.bodies[g]; mat = np.eye(4)
                mat[:3, :3] = rr[i] @ rots[i, wrist].T @ rots[i, b]
                mat[:3, 3] = pp[i]+rr[i] @ rots[i, wrist].T @ (poss[i, b]-poss[i, wrist])
                transforms[f"geom:{g}"] = mat
            audit = screen.query(transforms, stop_on_first=True)
            screening_seconds = time.monotonic()-screening_started
            hand_timing["screening_seconds"] += screening_seconds
            anchors_started = time.monotonic()
            anchors = []
            patch_reports = []
            reserved_hand, reserved_object = [], []
            for group, ids in group_ids.items():
                gf, gm = group_regions[group][:2] if group in group_regions else (region_faces, region_mesh)
                if patch is not None:
                    members = patch["members"][name][group]
                    for member, member_geoms in members.items():
                        member_anchors, evidence = extract_contact_patch(
                            group=group, member=member, member_geoms=member_geoms,
                            points=points[contact_mask], point_geoms=np.asarray(point_geoms)[contact_mask],
                            geom_transforms={g: transforms[f"geom:{g}"] for g in member_geoms},
                            meshes=ctx.meshes, body_names={g: m.body(ctx.bodies[g]).name for g in member_geoms},
                            object_mesh=gm, object_face_ids=gf,
                            points_per_member=patch["points_per_member"], max_gap_m=patch["max_gap_m"],
                            min_spacing_m=patch["min_spacing_m"],
                            reserved_hand_points=reserved_hand, reserved_object_points=reserved_object,
                            anchor_queries=anchor_queries)
                        for anchor in member_anchors:
                            anchor["name"] = f"{name}/{anchor['name']}"
                            anchor["position_weight"] /= len(members)
                            reserved_hand.append(anchor["hand_point_object"])
                            reserved_object.append(anchor["object_point"])
                        evidence["position_weight_normalization"] = "1 / declared group members / selected member points"
                        anchors.extend(member_anchors)
                        patch_reports.append(evidence)
                    continue
                distances_to_group = object_surface_query(pts[i, ids], gm)["distances"] if group in group_regions else regional[i, ids]
                index = int(ids[np.argmin(distances_to_group)])
                nearest = object_surface_query(pts[i, index:index+1], gm)
                hand_surface = robot_surface_query(points[index:index+1], ctx.meshes[point_geoms[index]])
                anchors.append(dict(group=group, geom=int(point_geoms[index]), body=m.body(bodies[index]).name,
                    body_local_point=points[index].tolist(), robot_face=int(hand_surface["face_ids"][0]),
                    robot_barycentric=hand_surface["barycentric"][0].tolist(),
                    body_local_normal=hand_surface["normals"][0].tolist(), normal_weight=0,
                    object_face=int(gf[nearest["face_ids"][0]]),
                    object_barycentric=nearest["barycentric"][0].tolist(), gap_m=float(nearest["distances"][0])))
            anchor_seconds = time.monotonic()-anchors_started
            hand_timing["anchor_extraction_seconds"] += anchor_seconds
            record = dict(seed_frame=frame, position=pp[i].tolist(), rotation=rr[i].tolist(),
                finger_joints=hand["finger_joints"], finger_qpos=qq[i, addresses].tolist(),
                material_anchor_hypotheses=anchors, loss=float(loss[i]),
                sampled_clearance_min_m=float(distances[i].min()), screen=audit,
                rejected=not audit["no_unretained_crossing"], motion_accepted=False)
            if scene_minimum is not None:
                record["sampled_scene_min_signed_m"] = float(scene_minimum[i])
            if start_records is not None:
                record.update(start_records[i])
            if warm_reviews is not None:
                record["warm_start_seed"] = dict(warm_reviews[i], bank_sha256=warm_binding["sha256"])
                record["initial_delta"] = initial_delta[i].tolist()
            record["timing"] = dict(screening_seconds=screening_seconds,
                                    anchor_extraction_seconds=anchor_seconds)
            if patch is not None:
                complete = all(item["coverage_complete"] for item in patch_reports)
                record["contact_patch"] = dict(
                    status="COMPLETE" if complete else "SHORTFALL", coverage_complete=complete,
                    members=patch_reports, evidence="GEOMETRIC_PROXIMITY_HYPOTHESIS",
                    motion_accepted=False)
                record["rejected"] = record["rejected"] or not complete
                record["rejection_reasons"] = (
                    ([] if audit["no_unretained_crossing"] else ["ORIGINAL_SURFACE_SCREEN_FAILED"])
                    + ([] if complete else ["CONTACT_PATCH_COVERAGE_SHORTFALL"]))
            records.append(record)
        candidates[name] = records
        hand_timing["elapsed_seconds"] = time.monotonic()-hand_started
        timings["hands"][name] = hand_timing
    report = dict(schema_version=1, status="UNREVIEWED_FREE_GRASP_PROPOSALS", motion_accepted=False,
                binding=ctx.binding, config_sha256=digest(config), candidates=candidates,
                implementation_sha256=code_binding("grasp_io.py", "surface_grasp.py", "grasp_multistart.py", "contact_patches.py", "repair_tensor.py", "repair_geometry.py", "mesh_screen.py", "repair_numba.py", "grasp_bank_seed.py", "grasp_distal_pad.py", "grasp_paper_objectives.py", "grasp_interaction_graph.py"),
                coverage=coverage, device=device, normal_weight=0, steps=steps,
                multistart=multistart, timing=timings,
                elapsed_seconds=time.monotonic()-started, self_clearance_weight=self_weight,
                surface_query=dict(object_query.metadata(), original_geometry=ctx.object_query_provenance),
                limitations=["Finite sampled clearance is a proposal objective; original hand surface screening is not full robot clearance.",
                             "No reach, approach, temporal, source-image, environment, containment or dynamics acceptance.",
                             "Anchors are authored hypotheses frozen at this candidate, not measured contact forces."])
    if anchor_queries is not None:
        report['anchor_queries'] = anchor_queries.stats()
        report['implementation_sha256'].update(code_binding('contact_query_cache.py'))
        if anchor_queries.config['backend'] == 'numba_cached':
            report['implementation_sha256'].update(code_binding('contact_query_numba.py'))
    if scene_clearance is not None:
        report["scene_clearance"] = dict(scene_clearance.metadata, weight=scene_weight,
            margin_m=margin, candidate_scope="configured original hand meshes at each seed frame")
        report["implementation_sha256"].update(code_binding("hand_scene_clearance.py", "static_scene.py"))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        parser.error("--out must be a new directory")
    report = propose(json.loads(args.config.read_text()), args.config.resolve().parent)
    args.out.mkdir(parents=True)
    write_report(args.out/"bank.json", report)
    print(json.dumps(dict(output=str(args.out), elapsed_seconds=report["elapsed_seconds"], status=report["status"])))


if __name__ == "__main__":
    main()
