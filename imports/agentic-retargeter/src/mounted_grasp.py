"""Fixed original-surface contacts for joint hand/arm candidate refinement.

No nearest-point queries run during fitting. The incoming material pairs stay
fixed while robot joints and the explicitly bounded shared object pose move.
"""
from collections import Counter

import numpy as np
from scipy.spatial.transform import Rotation

from .contact_edit.pose import _finite, _integer, _keys
from .grasp_io import digest, number


def refinement_config(value):
    if value is None:
        return None
    _keys(value, {"max_delta_rad", "contact_weight", "wrist_weight", "posture_weight", "max_gap_m"},
          set(), "finger_refinement")
    return dict(max_delta_rad=number(value["max_delta_rad"], "finger max_delta_rad", 1e-9, np.pi),
                contact_weight=number(value["contact_weight"], "finger contact_weight", 1e-9),
                wrist_weight=number(value["wrist_weight"], "finger wrist_weight"),
                posture_weight=number(value["posture_weight"], "finger posture_weight"),
                max_gap_m=number(value["max_gap_m"], "finger max_gap_m", 1e-9, .1))


def _barycentric(value, name):
    bary = _finite(value, (3,), name)
    if np.any(bary < 0) or np.any(bary > 1) or not np.isclose(bary.sum(), 1, atol=1e-8, rtol=0):
        raise ValueError(f"{name} must be normalized nonnegative triangle coordinates")
    return bary


def bind_material_anchors(ctx, hand_name, candidate):
    """Bind a complete descriptor to native robot/object triangles and anatomy.

    Equal total weight per group, then per declared member, prevents a densely
    sampled digit from replacing another. Patch shortfalls never get padded.
    """
    hand = ctx.hands[hand_name]
    raw = candidate.get("material_anchor_hypotheses")
    if not isinstance(raw, list) or not raw:
        raise ValueError("Joint finger refinement requires complete material_anchor_hypotheses")
    patch = candidate.get("contact_patch")
    declared = {}
    if patch is not None:
        if not isinstance(patch, dict) or patch.get("coverage_complete") is not True or patch.get("status") != "COMPLETE":
            raise ValueError("Incomplete contact patches cannot be mounted")
        members = patch.get("members")
        if not isinstance(members, list) or not members:
            raise ValueError("Complete contact patch must declare its members")
        for member in members:
            if not isinstance(member, dict):
                raise ValueError("Contact patch members must be records")
            group, label = member.get("group"), member.get("member")
            if group not in hand["contact_groups"] or not isinstance(label, str) or not label:
                raise ValueError("Contact patch group/member binding mismatch")
            key = (group, label)
            if key in declared:
                raise ValueError("Duplicate contact patch member")
            expected = _integer(member.get("requested_count"), "requested patch count", 1)
            if expected > 64 or member.get("coverage_complete") is not True or member.get("selected_count") != expected:
                raise ValueError("Contact patch member has incomplete material coverage")
            geoms = member.get("member_geoms")
            if (not isinstance(geoms, list) or not geoms or any(type(g) is not int for g in geoms)
                    or len(set(geoms)) != len(geoms) or not set(geoms) <= set(hand["contact_groups"][group])):
                raise ValueError("Contact patch member geom binding mismatch")
            declared[key] = (expected, set(geoms))
        if {key[0] for key in declared} != set(hand["contact_groups"]):
            raise ValueError("Contact patch must cover every participating group")
        for group, geoms in hand["contact_groups"].items():
            groups = [value[1] for key, value in declared.items() if key[0] == group]
            if set().union(*groups) != set(geoms) or sum(map(len, groups)) != len(set(geoms)):
                raise ValueError("Contact patch members must partition original group geoms")
    result = []; names = set(); robot_identities = set(); object_identities = set()
    for index, anchor in enumerate(raw):
        if not isinstance(anchor, dict):
            raise ValueError("Material anchors must be records")
        group = anchor.get("group")
        if group not in hand["contact_groups"]:
            raise ValueError("Material anchor group is outside the configured hand")
        member = anchor.get("member", group)
        if not isinstance(member, str) or not member:
            raise ValueError("Material anchor member must be a nonempty name")
        key = (group, member)
        geom = _integer(anchor.get("geom"), "anchor geom")
        if geom not in hand["contact_groups"][group]:
            raise ValueError("Material anchor geom is outside its hand/contact group")
        if patch is not None and (key not in declared or geom not in declared[key][1]):
            raise ValueError("Material anchor is outside its declared patch member")
        body = ctx.bodies[geom]
        if anchor.get("body") != ctx.model.body(body).name:
            raise ValueError("Material anchor body/geom binding mismatch")
        mesh = ctx.meshes[geom]
        face = _integer(anchor.get("robot_face"), "anchor robot_face")
        if face >= len(mesh.faces):
            raise ValueError("Material anchor robot_face is outside original topology")
        bary = _barycentric(anchor.get("robot_barycentric"), "robot_barycentric")
        local = bary @ mesh.vertices[mesh.faces[face]]
        supplied = _finite(anchor.get("body_local_point"), (3,), "anchor body_local_point")
        if not np.allclose(local, supplied, atol=2e-7, rtol=0):
            raise ValueError("Material anchor body point does not match original robot face/barycentric")
        object_face = _integer(anchor.get("object_face"), "anchor object_face")
        if object_face >= len(ctx.object.faces) or object_face not in hand["region_faces"]:
            raise ValueError("Material anchor object_face is outside the original eligible region")
        obary = _barycentric(anchor.get("object_barycentric"), "object_barycentric")
        objpoint = obary @ ctx.object.vertices[ctx.object.faces[object_face]]
        if "object_point" in anchor and not np.allclose(
                objpoint, _finite(anchor["object_point"], (3,), "anchor object_point"), atol=2e-7, rtol=0):
            raise ValueError("Material anchor object point does not match original face/barycentric")
        name = anchor.get("name", f"{hand_name}/{group}/{member}/{index}")
        if not isinstance(name, str) or not name or name in names:
            raise ValueError("Material anchor names must be unique nonempty IDs")
        names.add(name)
        robot_identity = (body, tuple(np.round(local, 10)))
        object_identity = tuple(np.round(objpoint, 10))
        if robot_identity in robot_identities or object_identity in object_identities:
            raise ValueError("Duplicate hand or object material points do not provide contact coverage")
        robot_identities.add(robot_identity); object_identities.add(object_identity)
        result.append(dict(name=name, group=group, member=member, geom=geom, body=body,
                           robot_face=face, robot_barycentric=bary, body_local_point=local,
                           object_face=object_face, object_barycentric=obary, object_point=objpoint))
    if {a["group"] for a in result} != set(hand["contact_groups"]):
        raise ValueError("Material anchors must cover every participating hand group")
    counts = Counter((a["group"], a["member"]) for a in result)
    if patch is not None and (set(counts) != set(declared) or any(counts[k] != declared[k][0] for k in counts)):
        raise ValueError("Material anchor count does not match complete patch descriptor")
    group_members = Counter(key[0] for key in counts)
    for anchor in result:
        anchor["weight"] = 1 / len(group_members) / group_members[anchor["group"]] / counts[(anchor["group"], anchor["member"])]
    return result, dict(descriptor_sha256=digest(raw), anchor_count=len(result),
                        weighting="equal group totals, equal member totals, equal points within member",
                        fixed_material_ids=[a["name"] for a in result])


def material_residual(model, data, anchors, object_rotation, object_translation, offset,
                      dofs, nvariables, *, move_object=False, weight=1., jacobian=False):
    """Residual/Jacobian in world metres for unchanged material correspondences."""
    import mujoco
    offset = np.asarray(offset)
    delta = Rotation.from_rotvec(offset[3:]).as_matrix()
    count = len(dofs)
    rotation_derivatives = []
    if jacobian and move_object:
        for eps in np.eye(3)*1e-6:
            rotation_derivatives.append((Rotation.from_rotvec(offset[3:]+eps).as_matrix()
                -Rotation.from_rotvec(offset[3:]-eps).as_matrix())/2e-6)
    output = []
    for anchor in anchors:
        body = anchor["body"]
        actual = data.xmat[body].reshape(3, 3) @ anchor["body_local_point"] + data.xpos[body]
        before = object_rotation @ anchor["object_point"]
        desired = delta @ before + object_translation + offset[:3]
        scale = weight * np.sqrt(anchor["weight"])
        if not jacobian:
            output.extend(scale*(actual-desired))
            continue
        jp = np.zeros((3, model.nv)); jr = np.zeros_like(jp)
        mujoco.mj_jac(model, data, jp, jr, actual, body)
        row = np.zeros((3, nvariables)); row[:, :count] = scale*jp[:, dofs]
        if move_object:
            row[:, count:count+3] = -scale*np.eye(3)
            for axis, dr in enumerate(rotation_derivatives):
                row[:, count+3+axis] = -scale*dr @ before
        output.append(row)
    return np.vstack(output) if jacobian else np.asarray(output)


def material_diagnostics(model, data, anchors, rotation, translation, max_gap_m):
    records = []
    for a in anchors:
        actual = data.xmat[a["body"]].reshape(3, 3) @ a["body_local_point"] + data.xpos[a["body"]]
        gap = float(np.linalg.norm(actual-(rotation @ a["object_point"]+translation)))
        records.append(dict(name=a["name"], group=a["group"], member=a["member"], gap_m=gap,
                            passed=gap <= max_gap_m, weight=a["weight"]))
    return dict(max_gap_m=max(r["gap_m"] for r in records), limit_m=max_gap_m,
                passed=all(r["passed"] for r in records), anchors=records)
