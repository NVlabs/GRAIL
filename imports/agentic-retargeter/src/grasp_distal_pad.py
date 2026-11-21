"""Explicit Dex3 material-surface prior; does not alter collision geometry.

This is an experimental anatomy heuristic, not an implementation of a paper.
Input points and original mesh vertices are in their native body coordinates.
Only attraction/anchor sample IDs may be filtered using the returned mask.
"""
import numpy as np


def distal_pad_point_mask(points, tip_offset, hinge_axis, joint_range,
                          mesh_vertices, *, axial_fraction=.4, side_fraction=.5):
    p, tip, axis, limits, vertices = map(lambda a: np.asarray(a, dtype=float),
                                        (points, tip_offset, hinge_axis,
                                         joint_range, mesh_vertices))
    if p.ndim != 2 or p.shape[1] != 3 or vertices.ndim != 2 or vertices.shape[1] != 3:
        raise ValueError('Points and original vertices must be Nx3')
    if tip.shape != (3,) or axis.shape != (3,) or limits.shape != (2,):
        raise ValueError('Expected original fingertip offset, hinge axis, limits')
    if not all(np.isfinite(a).all() for a in (p,tip,axis,limits,vertices)):
        raise ValueError('Non-finite anatomy inputs')
    if not (0 <= axial_fraction <= 1 and 0 <= side_fraction < 1):
        raise ValueError('Invalid anatomy mask fractions')
    length = np.linalg.norm(tip)
    if length <= 0 or np.linalg.norm(axis) <= 0:
        raise ValueError('Degenerate original fingertip/hinge axis')
    tip_axis = tip/length
    if limits[0] < 0 and limits[1] == 0:
        closing_sign = -1
    elif limits[0] == 0 and limits[1] > 0:
        closing_sign = 1
    else:
        raise ValueError('Ambiguous distal closing direction: require one-sided hinge')
    side = closing_sign*np.cross(axis,tip_axis)
    if np.linalg.norm(side) < 1e-8:
        raise ValueError('Distal hinge axis parallel to fingertip axis')
    side /= np.linalg.norm(side)
    extent = float((vertices@side).max())
    if extent <= 0:
        raise ValueError('No original mesh surface on designated pad side')
    axial_cutoff, side_cutoff = axial_fraction*length, side_fraction*extent
    mask = (p@tip_axis >= axial_cutoff) & (p@side >= side_cutoff)
    return mask, dict(tip_axis=tip_axis.tolist(),closing_side=side.tolist(),
        tip_length_m=float(length),axial_min_m=float(axial_cutoff),
        closing_side_min_m=float(side_cutoff),sample_count=len(p),
        selected_count=int(mask.sum()),axial_fraction=axial_fraction,
        side_fraction=side_fraction,normal_alignment_used=False,
        original_collision_geometry_preserved=True,
        scope='Original distal material surfaces used for attraction/anchor hypotheses only')


def dex3_distal_pad_mask(ctx, hand_name, points, point_geoms, **options):
    """Return global bool mask plus exact body/geom/point-index evidence.

    Parent objective must keep the unfiltered points for all collision queries.
    Parent anchor extractor must filter its points by the same material IDs;
    otherwise proximal anchors can still satisfy a nominal pad objective.
    """
    points, point_geoms = np.asarray(points), np.asarray(point_geoms)
    if points.shape != (len(point_geoms),3):
        raise ValueError('Sample/geometry binding mismatch')
    m = ctx.model
    hand = ctx.hands[hand_name]
    result = np.zeros(len(points), dtype=bool)
    evidence = []
    for digit, suffix in [('thumb','thumb_2_link'),('index','index_1_link'),('middle','middle_1_link')]:
        name = f'{hand_name}_hand_{suffix}'
        body = m.body(name).id
        tip = m.body(f'{hand_name}_hand_{digit}_finger_tip').id
        if int(m.body_parentid[tip]) != body or int(m.body_jntnum[body]) != 1:
            raise ValueError('Original Dex3 distal body/tip binding changed: '+name)
        joint = int(m.body_jntadr[body])
        geoms = [g for g in hand['robot_geoms'] if ctx.bodies[g] == body]
        if not geoms:
            raise ValueError('Missing original distal surface: '+name)
        for geom in geoms:
            ids = np.flatnonzero(point_geoms == geom)
            mask, record = distal_pad_point_mask(points[ids],m.body_pos[tip],
                m.jnt_axis[joint],m.jnt_range[joint],ctx.meshes[geom].vertices,**options)
            result[ids[mask]] = True
            evidence.append(dict(record,body=name,geom=int(geom),digit=digit,
                original_sample_indices=ids[mask].tolist()))
    return result, dict(hand=hand_name,records=evidence,
        total_original_samples=len(points),total_pad_samples=int(result.sum()),
        heuristic='Distal 60 percent; inner half-thickness on original closing side',
        geometry_changed=False,normal_alignment_used=False,motion_accepted=False)
