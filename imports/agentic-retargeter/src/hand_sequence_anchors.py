"""Quality-gated original-material anchors for free-hand sequence candidates.

Closest points are evaluated on exact original triangles. This module chooses
material correspondences, not a physical grasp or whole-mesh acceptance.
"""
from __future__ import annotations

import numpy as np


def choose_spaced_pairs(body_points, object_points, distances, *, maximum_gap=.005,
                        spacing=.003, count=4):
    """Closest eligible pair first, then maximize minimum two-sided spread.

    A pair must satisfy spacing independently on robot and object surfaces.
    No pair beyond maximum_gap can be returned, even for missing coverage.
    """
    body_points, object_points, distances = map(np.asarray, (body_points,object_points,distances))
    if body_points.shape != object_points.shape or body_points.shape != (len(distances),3):
        raise ValueError('Material pairs must be matching N x 3 points and N distances')
    if not all(np.isfinite(value).all() for value in (body_points,object_points,distances)):
        raise ValueError('Material pair inputs must be finite')
    if maximum_gap < 0 or spacing < 0 or type(count) is not int or count < 1:
        raise ValueError('Invalid anchor thresholds')
    eligible = np.flatnonzero(distances <= maximum_gap)
    selected=[]
    while len(eligible) and len(selected) < count:
        if not selected:
            winner=int(eligible[np.lexsort((eligible,distances[eligible]))[0]])
        else:
            robot_spread=np.linalg.norm(body_points[eligible,None]-body_points[selected],axis=-1).min(-1)
            object_spread=np.linalg.norm(object_points[eligible,None]-object_points[selected],axis=-1).min(-1)
            keep=(robot_spread >= spacing)&(object_spread >= spacing)
            eligible=eligible[keep]
            spread=np.minimum(robot_spread[keep],object_spread[keep])
            if not len(eligible):
                break
            winner=int(eligible[np.lexsort((eligible,distances[eligible],-spread))[0]])
        selected.append(winner)
        eligible=eligible[eligible != winner]
    return selected


def extract_frame_anchors(points, posed, members, geoms, bodies, region_for_digit,
                          *, maximum_gap=.005, spacing=.003, points_per_digit=4):
    """Bind fixed native robot points to original face IDs and barycentrics.

    region_for_digit maps each member to (original_face_ids, TriangleMesh).
    Distal surface sample deduplication is done by the caller before extraction.
    """
    from .repair_geometry import closest_points
    anchors=[]; coverage={}
    for digit, ids in members.items():
        original_faces,mesh=region_for_digit[digit]
        result=closest_points(posed[ids],mesh)
        selected=choose_spaced_pairs(points[ids],result['points'],result['distances'],
                                     maximum_gap=maximum_gap,spacing=spacing,count=points_per_digit)
        selected_samples=ids[selected]
        coverage[digit]=dict(required=points_per_digit,selected=len(selected),complete=len(selected)==points_per_digit,
                             eligible_within_gap=int((result['distances'] <= maximum_gap).sum()),
                             nearest_gap_m=float(result['distances'].min()),
                             maximum_selected_gap_m=float(result['distances'][selected].max()) if selected else None)
        for local,sample in zip(selected,selected_samples):
            anchors.append(dict(digit=digit,sample_index=int(sample),geom=int(geoms[sample]),body=int(bodies[sample]),
                body_local_point=points[sample].tolist(),object_point=result['points'][local].tolist(),
                object_face=int(original_faces[result['face_ids'][local]]),
                object_barycentric=result['barycentric'][local].tolist(),initial_gap_m=float(result['distances'][local]),
                motion_accepted=False))
    return anchors,coverage
