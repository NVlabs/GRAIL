"""Opt-in, hand-only, source-timed sequence retargeting.

The free wrist and native finger hinges are optimized at every source frame.
Body, arm and object trajectories remain immutable. Exports are free-hand
candidates, never full-robot qpos trajectories or accepted motions. Topo-inspired
source graphs and sampled original-surface distances are optimization guidance;
they are not original-whole-mesh collision certificates.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from .grasp_io import GraspInputs, number


def resolve_static_object_policy(config, rotations, translations):
    """Bind contact suppression to the complete original object trajectory.

    Exact equality matches SOMA's policy. A stationary prefix is still moving
    when any later source translation or rotation changes. An explicit stale
    flag is an input error, rather than permission to change contact semantics.
    """
    rotations, translations = np.asarray(rotations), np.asarray(translations)
    count = len(translations) if translations.ndim else 0
    if (not count or translations.shape != (count, 3)
            or rotations.shape != (count, 3, 3)
            or not np.isfinite(translations).all() or not np.isfinite(rotations).all()):
        raise ValueError('Static-object policy requires finite complete original R/t arrays')
    is_static = bool(np.array_equal(rotations, np.broadcast_to(rotations[:1], rotations.shape))
                     and np.array_equal(translations, np.broadcast_to(translations[:1], translations.shape)))
    declared = config.get('is_static_obj')
    if 'is_static_obj' in config and (type(declared) is not bool or declared != is_static):
        raise ValueError('is_static_obj must agree with exact full-source object motion')
    return dict(is_static_obj=is_static, declared_is_static_obj=declared,
                basis='exact full-source translation and rotation equality')


def phase_mask(intervals, count, *, static=False):
    """Inclusive explicit source-frame intervals; static objects have no contacts."""
    mask = np.zeros(count, dtype=bool)
    if not isinstance(intervals, list):
        raise ValueError('Contact intervals must be a list')
    if intervals and isinstance(intervals[0], int):
        intervals = [intervals]
    previous = -1
    for pair in intervals:
        if (not isinstance(pair, list) or len(pair) != 2
                or any(type(x) is not int for x in pair)
                or not 0 <= pair[0] <= pair[1] < count or pair[0] <= previous):
            raise ValueError('Contact intervals must be ordered disjoint inclusive frame pairs')
        mask[pair[0]:pair[1]+1] = True
        previous = pair[1]
    if static:
        mask[:] = False
    return mask


def masked_mean(values, mask):
    """An empty contact phase contributes exactly zero and has zero gradient."""
    weight = mask.to(dtype=values.dtype)
    return (values*weight).sum()/weight.sum().clamp_min(1)


def _difference(values, dt, order=1):
    for _ in range(order):
        values = (values[1:]-values[:-1])/dt
    return values


def rotation_velocity(rotation, dt):
    """Spatial angular velocity from consecutive SO(3) logarithms, radians/sec."""
    import torch
    relative = rotation[1:] @ rotation[:-1].transpose(-1, -2)
    vector = .5*torch.stack((relative[:, 2, 1]-relative[:, 1, 2],
                            relative[:, 0, 2]-relative[:, 2, 0],
                            relative[:, 1, 0]-relative[:, 0, 1]), dim=-1)
    sine = torch.linalg.vector_norm(vector, dim=-1)
    cosine = ((relative.diagonal(dim1=-2, dim2=-1).sum(-1)-1)*.5).clamp(-1, 1)
    angle = torch.atan2(sine, cosine)
    factor = torch.where(sine > 1e-6, angle/sine.clamp_min(1e-6), 1+sine.square()/6)
    return vector*factor[:, None]/dt


def temporal_loss(position_world, rotation_world, finger, position_object,
                  rotation_object, contact_active, dt, weights):
    """SI differences at original time steps; hold guidance is object-relative.

    Position acceleration/jerk are m/s^2,m/s^3; angular velocity and differences
    are rad/s,rad/s^2,rad/s^3. Finger derivatives have the same angular units.
    All source frames and acquisition/release joins contribute world terms.
    Only pairs wholly within declared contact intervals get hold-velocity terms.
    """
    import torch
    total = position_world.sum()*0
    angular = rotation_velocity(rotation_world, dt)
    terms = {}
    specs = [('wrist_acceleration', _difference(position_world, dt, 2)),
             ('wrist_jerk', _difference(position_world, dt, 3)),
             ('angular_acceleration', _difference(angular, dt)),
             ('angular_jerk', _difference(angular, dt, 2)),
             ('finger_acceleration', _difference(finger, dt, 2)),
             ('finger_jerk', _difference(finger, dt, 3))]
    for name, value in specs:
        term = value.square().mean() if len(value) else total*0
        terms[name] = term
        total = total+weights.get(name, 0)*term
    pair_active = contact_active[1:] & contact_active[:-1]
    for name, value in [('hold_translation_velocity', _difference(position_object, dt)),
                        ('hold_angular_velocity', rotation_velocity(rotation_object, dt)),
                        ('hold_finger_velocity', _difference(finger, dt))]:
        term = masked_mean(value.square().mean(-1), pair_active)
        terms[name] = term
        total = total+weights.get(name, 0)*term
    return total, terms


DEFAULT_TEMPORAL = dict(wrist_acceleration=2e-5, wrist_jerk=2e-8,
                       angular_acceleration=5e-7, angular_jerk=5e-10,
                       finger_acceleration=2e-7, finger_jerk=2e-10,
                       hold_translation_velocity=.01, hold_angular_velocity=.0001,
                       hold_finger_velocity=.00002)


def _write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _rotation(vector):
    import torch
    zero = torch.zeros_like(vector[:, 0])
    skew = torch.stack((zero, -vector[:, 2], vector[:, 1], vector[:, 2], zero,
                        -vector[:, 0], -vector[:, 1], vector[:, 0], zero), -1)
    return torch.matrix_exp(skew.reshape(-1, 3, 3))


def _samples(ctx, side, count):
    from .grasp_distal_pad import dex3_distal_pad_mask
    hand = ctx.hands[side]
    points, bodies, geom_ids = [], [], []
    for geom in hand['robot_geoms']:
        mesh = ctx.meshes[geom]
        nv, nf = min(count//2, len(mesh.vertices)), min(count-count//2, len(mesh.faces))
        material = np.concatenate((mesh.vertices[np.linspace(0, len(mesh.vertices)-1, nv, dtype=int)],
                  mesh.triangles[np.linspace(0, len(mesh.faces)-1, nf, dtype=int)].mean(1)))
        points.extend(material); bodies.extend([ctx.bodies[geom]]*len(material)); geom_ids.extend([geom]*len(material))
    points, bodies, geom_ids = np.asarray(points), np.asarray(bodies), np.asarray(geom_ids)
    mask, evidence = dex3_distal_pad_mask(ctx, side, points, geom_ids)
    members = {}
    for digit, suffix in [('thumb', 'thumb_2_link'), ('index', 'index_1_link'), ('middle', 'middle_1_link')]:
        body = int(ctx.model.body(f'{side}_hand_{suffix}').id)
        ids = np.flatnonzero((bodies == body) & mask)
        if not len(ids):
            raise ValueError(f'No sampled original distal pad for {side} {digit}; increase samples_per_geom')
        _, unique = np.unique(np.round(points[ids], 10), axis=0, return_index=True)
        members[digit] = ids[np.sort(unique)]
    return points, bodies, geom_ids, members, evidence


def solve(config, directory='.', output_directory='hand_sequence_output'):
    import mujoco
    import torch
    from scipy.spatial.transform import Rotation
    from .grasp_paper_objectives import build_paper_objective
    from .repair_geometry import TriangleMesh, closest_points
    from .repair_tensor import NativeTensorFK, WarpSignedDistance

    start = time.monotonic()
    base, out = Path(directory), Path(output_directory)
    out.mkdir(parents=True, exist_ok=True)
    ctx = GraspInputs(config['inputs'], base)
    input_config = config
    static_policy = resolve_static_object_policy(config, ctx.R, ctx.t)
    config = {**config, 'is_static_obj': static_policy['is_static_obj']}
    count = len(ctx.q)
    if count < 4:
        raise ValueError('Sequence requires at least four actual source frames')
    fps = number(config.get('fps', 24), 'fps', 1e-6)
    if 'fps' in ctx.native and not np.isclose(float(np.asarray(ctx.native['fps'])), fps):
        raise ValueError('Configured fps disagrees with original native timing')
    mode = config.get('mode', 'temporal')
    if mode not in ('independent', 'temporal', 'anchor_temporal'):
        raise ValueError('mode must be independent, temporal or anchor_temporal')
    device = config.get('device', 'cuda:0')
    if device != 'cpu' and not device.startswith('cuda:'):
        raise ValueError('Explicit cpu or cuda:N required')
    source_arrays = (base/config['source_arrays']).resolve(strict=True)
    samples = int(config.get('samples_per_geom', 128))
    if not 32 <= samples <= 4096:
        raise ValueError('samples_per_geom must be in [32,4096]')
    steps = {key: config.get(key, default) for key, default in
             [('topo_steps', 200), ('pad_steps', 0), ('temporal_steps', 200)]}
    if any(type(v) is not int or not 0 <= v <= 2000 for v in steps.values()):
        raise ValueError('Each stage requires an integer 0..2000 steps')
    if mode == 'independent':
        steps['temporal_steps'] = 0
    warm_path = config.get('warm_start_npz')
    warm = None
    if warm_path:
        warm_path = (base/warm_path).resolve(strict=True)
        with np.load(warm_path, allow_pickle=False) as saved:
            warm = {k: saved[k].copy() for k in saved.files}
        if (not np.array_equal(warm['source_qpos'], ctx.q)
                or not np.array_equal(warm['object_R'], ctx.R)
                or not np.array_equal(warm['object_trans'], ctx.t)):
            raise ValueError('Warm trajectory must retain the exact source and object binding')
        steps['topo_steps'] = steps['pad_steps'] = 0
    fk = NativeTensorFK(ctx.model, device=device, dtype=torch.float32)
    obj_query = WarpSignedDistance(ctx.object_query.vertices, ctx.object_query.faces, device=device)
    weights = {**DEFAULT_TEMPORAL, **config.get('temporal_weights', {})}
    if set(weights)-set(DEFAULT_TEMPORAL):
        raise ValueError('Unknown temporal weight')
    weights = {k: number(v, k) for k, v in weights.items()}
    translation_limit = number(config.get('translation_limit_m', .15), 'translation_limit_m', 1e-6, 1)
    rotation_limit = number(config.get('rotation_limit_rad', 1.5), 'rotation_limit_rad', 1e-6, np.pi)
    margin = number(config.get('clearance_m', .001), 'clearance_m', 0, .02)
    rate = number(config.get('learning_rate', .008), 'learning_rate', 1e-6, .1)
    source_prior = {**dict(position=.02, rotation=.0001, finger=.00002), **config.get('source_prior', {})}
    source_prior = {k: number(v, k) for k, v in source_prior.items()}
    if set(source_prior) != {'position', 'rotation', 'finger'}:
        raise ValueError('Unknown source prior')
    loss_weights = {**dict(collision=1., scene_collision=0., contact=4., fixed_anchor=4.), **config.get('loss_weights', {})}
    loss_weights = {k: number(v, k) for k, v in loss_weights.items()}
    from .hand_scene_clearance import HandSceneClearance
    scene_query = HandSceneClearance(ctx.native, device) if loss_weights['scene_collision'] else None
    scene_enabled = scene_query is not None and scene_query.enabled
    timestamps = np.arange(count, dtype=np.float64)/fps
    payload = dict(source_qpos=ctx.q.copy(), object_R=ctx.R.copy(), object_trans=ctx.t.copy(),
                   object_scale=np.asarray(ctx.native['object_scale']).copy(), timestamps=timestamps,
                   fps=np.asarray(fps), scope=np.asarray('free_hand_only_not_full_robot_trajectory'))
    stage_plan = [('topo', 'topo_steps'), ('pad', 'pad_steps'), ('temporal', 'temporal_steps')]
    executed_stages = [stage for stage, key in stage_plan if steps[key]]
    initial_stage = 'warm_start' if warm is not None else 'source'
    final_stage = executed_stages[-1] if executed_stages else initial_stage
    snapshots = {stage: dict(payload) for stage in dict.fromkeys(['source', initial_stage, *executed_stages])}
    report = dict(schema_version=1, mode=mode, frame_count=count, fps=fps, motion_accepted=False,
                  scope='free wrists and native finger joints only; no arm/body/object optimization',
                  binding=ctx.binding, input_config=input_config,
                  scene_clearance=(dict(scene_query.metadata, weight=loss_weights['scene_collision'],
                      contact_phase_masked=False, collocation_fractions=[]) if scene_query is not None
                      else dict(enabled=False, reason='scene_collision weight is zero; legacy configuration')),
                  static_object_policy=static_policy, source_arrays_sha256=_sha(source_arrays),
                  steps=steps, executed_stages=executed_stages,
                  skipped_stages=[dict(stage=stage, reason=f'{key}=0')
                                  for stage, key in stage_plan if not steps[key]],
                  temporal_weights=weights, source_prior=source_prior,
                  code_sha256={p:_sha(Path(__file__).with_name(p)) for p in
                               ['hand_sequence.py', 'hand_sequence_anchors.py', 'grasp_distal_pad.py', 'grasp_paper_objectives.py',
                                'grasp_interaction_graph.py', 'repair_tensor.py', 'hand_scene_clearance.py', 'static_scene.py']},
                  hands={}, original_object_query=ctx.object_query_provenance,
                  limitations=['Sampled signed-distance optimization is not whole-mesh collision validation.',
                               'Free wrists are not robot-mounted poses; arm reachability is untested.',
                               'Original surface geometry, body source, object trajectory and source timing remain fixed.',
                               'No dynamics, continuous collision, force closure, or motion acceptance.'])
    if warm is not None:
        report['warm_start'] = dict(path=str(warm_path), sha256=_sha(warm_path))

    def sync():
        if device.startswith('cuda:'):
            torch.cuda.synchronize(device)

    for side, hand in ctx.hands.items():
        hand_start = time.monotonic()
        mask_np = phase_mask(config.get('contact_frames', {}).get(side, []), count,
                             static=bool(config.get('is_static_obj', False)))
        if side in config.get('contact_mask', {}):
            supplied = np.asarray(config['contact_mask'][side])
            if supplied.shape != (count,) or supplied.dtype.kind != 'b':
                raise ValueError('contact_mask must be exactly T booleans')
            mask_np = supplied.copy()
            if config.get('is_static_obj', False):
                mask_np[:] = False
        labels = list(config.get('phase_labels', {}).get(side, ['contact' if v else 'free' for v in mask_np]))
        if len(labels) != count or any(not isinstance(v, str) for v in labels):
            raise ValueError('phase_labels must contain T strings')
        # Source plans may encode state='contact' separately from its role.
        for phase in config.get('contact_phases', {}).get(side, []):
            if phase.get('state') == 'contact' and phase.get('role'):
                first, end = phase['frames']  # Original contact plan is half-open.
                labels[first:end] = [phase['role'].replace('-', '_')]*(end-first)
        fixed_mask_np = mask_np & np.asarray([v.lower() in ('hold', 'grasp', 'stable_hold', 'contact') for v in labels])
        active = torch.tensor(mask_np, device=device)
        hold_active = torch.tensor(fixed_mask_np, device=device)
        points, bodies, geoms, members, pad_metadata = _samples(ctx, side, samples)
        pt, bi = fk.t(points), fk.t(bodies, torch.long)
        declared_regions = config.get('contact_region_faces', {}).get(side, {})
        if declared_regions:
            members = {digit: ids for digit, ids in members.items()
                       if ('thumb' if digit == 'thumb' else 'long_fingers') in declared_regions}
        member_t = {name: fk.t(ids, torch.long) for name, ids in members.items()}
        self_weight = number(config.get('self_clearance_weight', 1.), 'self_clearance_weight')
        self_terms, omitted_self = [], []
        if self_weight:
            for target in hand['robot_geoms']:
                target_body = ctx.bodies[target]
                allowed = []
                for other in hand['robot_geoms']:
                    body = ctx.bodies[other]
                    if body == target_body or ctx.model.body_parentid[body] == target_body or ctx.model.body_parentid[target_body] == body:
                        omitted_self.append([int(other), int(target)])
                    else:
                        allowed.append(other)
                ids = np.flatnonzero(np.isin(geoms, allowed))
                if len(ids):
                    self_terms.append((target_body, fk.t(ids, torch.long),
                                       WarpSignedDistance(ctx.meshes[target].vertices, ctx.meshes[target].faces, device=device)))
        qbase, address = fk.t(ctx.q.copy()), hand['addresses']
        finger_source = qbase[:, address].clone()
        wrist = hand['wrist']
        low, high = fk.t(ctx.model.jnt_range[hand['joints'], 0]), fk.t(ctx.model.jnt_range[hand['joints'], 1])
        with torch.no_grad():
            source_rot, source_pos = fk(qbase)
            object_R, object_t = fk.t(ctx.R), fk.t(ctx.t)
            wr = object_R.transpose(-1, -2) @ source_rot[:, wrist]
            wp = (object_R.transpose(-1, -2) @ (source_pos[:, wrist]-object_t)[..., None]).squeeze(-1)
        region_ids = np.asarray(hand['region_faces'])
        region_mesh = TriangleMesh(ctx.object.vertices, ctx.object.faces[region_ids])
        region_query = WarpSignedDistance(region_mesh.vertices, region_mesh.faces, device=device)
        digit_queries = {}
        digit_meshes = {}
        for digit in members:
            group = 'thumb' if digit == 'thumb' else 'long_fingers'
            faces = np.asarray(declared_regions.get(group, region_ids), dtype=np.int64)
            if not len(faces) or np.any(faces < 0) or np.any(faces >= len(ctx.object.faces)):
                raise ValueError('Invalid original per-digit object region')
            mesh = TriangleMesh(ctx.object.vertices, ctx.object.faces[faces])
            digit_meshes[digit] = (faces, mesh)
            digit_queries[digit] = WarpSignedDistance(mesh.vertices, mesh.faces, device=device)
        scheduled_queries = []
        for schedule in config.get('region_schedule', {}).get(side, []):
            first, end = schedule['frames']
            if not 0 <= first < end <= count:
                raise ValueError('region_schedule uses original half-open source frames')
            configured = schedule.get('contact_region_faces', {})
            rows = {}
            for digit in members:
                group = 'thumb' if digit == 'thumb' else 'long_fingers'
                faces = np.asarray(configured.get(group, schedule['region_faces']), dtype=np.int64)
                mesh = TriangleMesh(ctx.object.vertices, ctx.object.faces[faces])
                rows[digit] = (faces, mesh, WarpSignedDistance(mesh.vertices, mesh.faces, device=device))
            scheduled_queries.append((first, end, rows))
        options = dict(method='topo', source_arrays=str(source_arrays), graph_weight=config.get('graph_weight', 500.),
                       object_sample_count=config.get('object_sample_count', 50), direction_weight=config.get('direction_weight', 0.))
        # Source-only export and warm temporal refinement do not use a Topo
        # objective, so they must not require its source mapping or node budget.
        graph = (build_paper_objective(ctx, side, hand, list(range(count)), fk, options)
                 if steps['topo_steps'] else None)
        initial = np.zeros((count, 6+len(address)), dtype=np.float32)
        if warm is not None:
            if list(warm[f'{side}_finger_joint_names']) != hand['finger_joints']:
                raise ValueError('Warm finger joint identity mismatch')
            initial[:, :3] = warm[f'{side}_wrist_position_object']-wp.cpu().numpy()
            initial[:, 3:6] = Rotation.from_matrix(warm[f'{side}_wrist_rotation_object'] @
                                                 wr.cpu().numpy().transpose(0, 2, 1)).as_rotvec()
            initial[:, 6:] = warm[f'{side}_finger_qpos']-ctx.q[:, address]
        x = torch.nn.Parameter(fk.t(initial))
        anchors = []
        anchor_selection = []
        history = []

        def evaluate(graph_on=False, temporal_on=False):
            q = qbase.clone(); q[:, address] = finger_source+x[:, 6:]
            rr, pp = _rotation(x[:, 3:6]) @ wr, wp+x[:, :3]
            rots, poss = fk(q)
            world = torch.einsum('bnij,nj->bni', rots[:, bi], pt)+poss[:, bi]
            local = torch.einsum('bji,bnj->bni', rots[:, wrist], world-poss[:, wrist, None])
            posed = torch.einsum('bij,bnj->bni', rr, local)+pp[:, None]
            signed = obj_query.distance(posed)
            violation = torch.relu(margin-signed)
            collision = (400*violation.square().mean(1)+20*violation.max(1).values.square()).mean()
            contact_frame = torch.zeros(count, dtype=fk.dtype, device=device)
            gaps = {}
            for digit, ids in member_t.items():
                distance = digit_queries[digit].distance(posed[:, ids]).abs()
                for first, end, rows in scheduled_queries:
                    replacement = rows[digit][2].distance(posed[first:end, ids]).abs()
                    distance = torch.cat((distance[:first], replacement, distance[end:]), dim=0)
                chosen = distance.topk(min(4, len(ids)), largest=False, dim=1).values
                gaps[digit] = chosen
                contact_frame = contact_frame+(chosen-margin).square().mean(1)
            contact = masked_mean(contact_frame, active)
            prior = (source_prior['position']*x[:, :3].square().sum(1)
                     +source_prior['rotation']*x[:, 3:6].square().sum(1)
                     +source_prior['finger']*x[:, 6:].square().sum(1)).mean()
            loss = loss_weights['collision']*collision+loss_weights['contact']*contact+prior
            components = dict(collision=collision, contact=contact, source_prior=prior)
            if scene_enabled:
                scene_loss, _ = scene_query.penalty(posed, object_R, object_t, margin)
                loss = loss + loss_weights['scene_collision']*scene_loss.mean()
                components['scene_collision'] = scene_loss.mean()
            self_min = torch.zeros(count, dtype=fk.dtype, device=device)
            if self_terms:
                violations = []
                for body, ids, query in self_terms:
                    relative = torch.einsum('bji,bnj->bni', rots[:, body], world[:, ids]-poss[:, body, None])
                    self_distance = query.distance(relative)
                    self_min = torch.minimum(self_min, self_distance.min(1).values)
                    violation = torch.relu(margin-self_distance)
                    violations.append(violation.topk(min(12, len(ids)), dim=1).values)
                merged = torch.cat(violations, dim=1)
                self_loss = merged.topk(min(12, merged.shape[1]), dim=1).values.square().mean()
                loss = loss+self_weight*self_loss
                components['sampled_self_clearance'] = self_loss
            if graph_on:
                graph_loss = graph.evaluate(rr, pp, rots, poss).mean()
                loss = loss+graph_loss; components['topo'] = graph_loss
            if temporal_on:
                position_world = torch.einsum('bij,bj->bi', object_R, pp)+object_t
                rotation_world = object_R @ rr
                temporal, temporal_terms = temporal_loss(position_world, rotation_world, q[:, address],
                                                        pp, rr, hold_active, 1/fps, weights)
                loss = loss+temporal; components['temporal'] = temporal
                if anchors:
                    anchor_loss = loss*0
                    for item in anchors:
                        residual = posed[:, item['sample_index']]-fk.t(item['object_point'])
                        phase = torch.zeros_like(active); phase[item['start']:item['end']+1] = True
                        anchor_loss = anchor_loss+masked_mean(residual.square().sum(-1), phase)
                    anchor_loss = anchor_loss/len(anchors)
                    loss = loss+loss_weights['fixed_anchor']*anchor_loss
                    components['fixed_anchor'] = anchor_loss
            return loss, components, dict(wrist_position_object=pp, wrist_rotation_object=rr,
                      finger_qpos=q[:, address], posed_surface_points=posed,
                      signed_distances=signed, sampled_self_min=self_min, gap_by_digit=gaps)

        def snapshot(stage):
            with torch.no_grad():
                _, components, values = evaluate()
                for key in ['wrist_position_object', 'wrist_rotation_object', 'finger_qpos']:
                    snapshots[stage][f'{side}_{key}'] = values[key].cpu().numpy().copy()
                for key in ['wrist_position_object', 'wrist_rotation_object', 'finger_qpos']:
                    snapshots[stage][f'source_{side}_{key}'] = {'wrist_position_object':wp,
                        'wrist_rotation_object':wr,'finger_qpos':finger_source}[key].cpu().numpy().copy()
                snapshots[stage][f'{side}_finger_joint_names'] = np.asarray(hand['finger_joints'])
                snapshots[stage][f'{side}_finger_addresses'] = np.asarray(address)
                snapshots[stage][f'{side}_contact_active'] = mask_np
                snapshots[stage][f'{side}_phase_labels'] = np.asarray(labels)
                return {k: float(v) for k, v in components.items()}

        # True source remains independent of the selected warm start.
        with torch.no_grad():
            saved = x.clone(); x.zero_(); snapshot('source')
            source_self_min = evaluate()[2]['sampled_self_min'].cpu().numpy().copy()
            x.copy_(saved)
            if warm is not None:
                snapshot('warm_start')
        stage_times = {}
        for stage, key in stage_plan:
            if not steps[key]:
                continue
            if stage == 'temporal' and mode == 'anchor_temporal' and fixed_mask_np.any():
                from .hand_sequence_anchors import extract_frame_anchors
                gate = config.get('anchor_selection', {})
                maximum_gap = number(gate.get('maximum_gap_m', .005), 'maximum_gap_m', 0, .1)
                minimum_spacing = number(gate.get('minimum_spacing_m', .003), 'minimum_spacing_m', 0, .05)
                penetration_limit = number(gate.get('sampled_object_penetration_limit_m', .002), 'sampled_object_penetration_limit_m', 0, .1)
                self_excess_limit = number(gate.get('sampled_self_excess_limit_m', .002), 'sampled_self_excess_limit_m', 0, .1)
                frame_limit = int(gate.get('maximum_reference_frames', 16))
                if not 1 <= frame_limit <= count:
                    raise ValueError('maximum_reference_frames must be 1..T')
                with torch.no_grad():
                    candidate = evaluate()[2]
                    posed = candidate['posed_surface_points'].cpu().numpy()
                    object_min = candidate['signed_distances'].min(1).values.cpu().numpy()
                    self_min = candidate['sampled_self_min'].cpu().numpy()
                    nearest = {digit:value.cpu().numpy() for digit,value in candidate['gap_by_digit'].items()}
                starts = np.flatnonzero(fixed_mask_np & ~np.r_[False, fixed_mask_np[:-1]])
                ends = np.flatnonzero(fixed_mask_np & ~np.r_[fixed_mask_np[1:], False])
                for first, last in zip(starts, ends):
                    frame_ids = np.arange(first, last+1)
                    nearest_all = np.concatenate(list(nearest.values()), axis=1)
                    count_near = (nearest_all <= maximum_gap).sum(1)
                    object_bad = np.maximum(0, -object_min-penetration_limit)
                    self_bad = np.maximum(0, source_self_min-self_min-self_excess_limit)
                    ordering = sorted(frame_ids.tolist(), key=lambda f:(bool(object_bad[f] or self_bad[f]),
                                      -int(count_near[f]),float(nearest_all[f].mean()),f))[:frame_limit]
                    trials=[]
                    best=None
                    for frame in ordering:
                        regions = dict(digit_meshes)
                        for first_schedule, end_schedule, rows in scheduled_queries:
                            if first_schedule <= frame < end_schedule:
                                regions.update({digit:row[:2] for digit,row in rows.items()})
                        pairs, coverage = extract_frame_anchors(points,posed[frame],members,geoms,bodies,regions,
                                                              maximum_gap=maximum_gap,spacing=minimum_spacing)
                        complete = all(row['complete'] for row in coverage.values())
                        proxy_eligible = not bool(object_bad[frame] or self_bad[frame])
                        score = (proxy_eligible,complete,len(pairs),-float(np.mean([a['initial_gap_m'] for a in pairs])) if pairs else -1.)
                        trial = dict(frame=frame,coverage=coverage,complete=complete,proxy_eligible=proxy_eligible,
                                     anchor_count=len(pairs),sampled_object_min_signed_m=float(object_min[frame]),
                                     sampled_self_min_signed_m=float(self_min[frame]),
                                     source_sampled_self_min_signed_m=float(source_self_min[frame]),
                                     sampled_self_excess_m=float(max(0,source_self_min[frame]-self_min[frame])))
                        trials.append(trial)
                        if best is None or score > best[0]:
                            best=(score,frame,pairs,trial)
                        if complete and proxy_eligible:
                            break
                    selected = bool(best and best[3]['proxy_eligible'] and best[2])
                    if selected:
                        for item in best[2]:
                            item.update(start=int(first),end=int(last),reference_frame=best[1])
                        anchors.extend(best[2])
                    anchor_selection.append(dict(start=int(first),end=int(last),selected=selected,
                        selected_reference_frame=best[1] if selected else None,
                        complete=bool(selected and best[3]['complete']),
                        maximum_gap_m=maximum_gap,minimum_spacing_m=minimum_spacing,
                        sampled_object_penetration_limit_m=penetration_limit,sampled_self_excess_limit_m=self_excess_limit,
                        evaluated_frames=trials,eligible_source_frame_count=len(frame_ids),maximum_reference_frames=frame_limit,
                        rule='Original material pairs within gap and two-sided spacing; missing coverage retained.',
                        limitations='Signed sampled object/self proxies are not whole-mesh collision certification; source-min self reference cannot identify individual inherited pairs.'))
            sync(); stage_start = time.monotonic()
            optimizer = torch.optim.Adam([x], lr=rate if stage != 'temporal' else config.get('temporal_learning_rate', .003))
            for iteration in range(steps[key]):
                optimizer.zero_grad()
                loss, components, _ = evaluate(graph_on=stage == 'topo', temporal_on=stage == 'temporal')
                if not torch.isfinite(loss):
                    raise ValueError(f'Nonfinite hand objective {side}/{stage}/{iteration}')
                loss.backward(); torch.nn.utils.clip_grad_norm_([x], 10.); optimizer.step()
                with torch.no_grad():
                    x[:, :3].clamp_(-translation_limit, translation_limit)
                    x[:, 3:6].clamp_(-rotation_limit, rotation_limit)
                    x[:, 6:] = torch.minimum(torch.maximum(finger_source+x[:, 6:], low), high)-finger_source
                if iteration % 50 == 0 or iteration+1 == steps[key]:
                    row = dict(stage=stage, iteration=iteration+1, loss=float(loss.detach()),
                               components={k:float(v.detach()) for k,v in components.items()})
                    history.append(row)
                    print(json.dumps(dict(hand=side, **row)), flush=True)
            sync(); stage_times[stage] = time.monotonic()-stage_start
            snapshot(stage)
        with torch.no_grad():
            _, _, final = evaluate()
            gap_summary = {digit: dict(active_mean_m=float(value[active].mean()) if active.any() else None,
                                      active_max_m=float(value[active].max()) if active.any() else None)
                           for digit, value in final['gap_by_digit'].items()}
            signed = final['signed_distances']
            metric = dict(sampled_min_signed_distance_m=float(signed.min()),
                          sampled_penetration_fraction=float((signed < 0).float().mean()),
                          gap_by_digit=gap_summary)
        report['hands'][side] = dict(contact_intervals=config.get('contact_frames', {}).get(side, []),
                                    phase_labels=labels, fixed_anchor_eligible_frames=int(fixed_mask_np.sum()),
                                    sampled_self_adjacent_same_body_omissions=omitted_self,
                                    contact_active_frames=int(mask_np.sum()), static_contact_disabled=bool(config.get('is_static_obj', False)),
                                    original_surface_sample_count=len(points), samples_per_geom=samples,
                                    distal_pad=pad_metadata,
                                    graph=graph.metadata if graph is not None else dict(enabled=False, reason='topo_steps=0'),
                                    fixed_anchors=anchors, anchor_selection=anchor_selection,
                                    history=history, stage_seconds=stage_times, total_seconds=time.monotonic()-hand_start,
                                    final_sampled_metrics=metric)
        _write(out/'progress.json', report)
    for stage, arrays in snapshots.items():
        np.savez_compressed(out/f'{stage}.npz', **arrays)
    np.savez_compressed(out/'hand_sequence.npz', **snapshots[final_stage])
    report.update(elapsed_seconds=time.monotonic()-start, final_stage=final_stage,
                  outputs={str(p.name):dict(sha256=_sha(p), bytes=p.stat().st_size) for p in out.glob('*.npz')})
    _write(out/'report.json', report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('config'); parser.add_argument('--output', required=True)
    args = parser.parse_args()
    path = Path(args.config).resolve(strict=True)
    solve(json.loads(path.read_text()), path.parent, args.output)


if __name__ == '__main__':
    main()
