"""Mount a selected surface grasp with bounded native joint/object corrections.

Each authored hold starts from one object-relative wrist pose and finger layout.
Optional material refinement fits active fingers with the native arm joints.
This does not generate transitions or accept a resulting animation.
"""
import argparse
import json
from pathlib import Path
import time

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from .contact_edit import native_io
from .contact_edit.pose import _finite, _integer, _is_derived, _keys, _sha
from .grasp_io import GraspInputs, code_binding, digest, indices, number, rigid, write_report
from .shared_grasp_scene import SceneClearance
from .joint_velocity import JointVelocityGate, snapshot_from_model
from .mounted_grasp import (refinement_config, bind_material_anchors,
                            material_residual, material_diagnostics)
from .grasp_body_reference import BodyReference
from .grasp_object_clearance import ObjectClearance


def interval(value, count, name):
    if (not isinstance(value, list) or len(value) != 2 or any(type(x) is not int for x in value)
            or not 0 <= value[0] < value[1] <= count):
        raise ValueError(f"{name} must be a nonempty half-open frame interval")
    return value


def solve(config, directory="."):
    import mujoco
    started = time.monotonic()
    _keys(config, {"inputs", "bank", "holds", "frames", "editable_joints", "collision_pairs"},
          {"object_motion", "max_nfev", "max_joint_step_rad", "posture_weight", "temporal_weight", "collision_weight", "body_guides", "scene_clearance", "object_clearance", "temporal", "finger_refinement", "body_reference"}, "hold solve")
    ctx = GraspInputs(config["inputs"], directory)
    refinement = refinement_config(config.get("finger_refinement"))
    body_reference = BodyReference(ctx, config["body_reference"], directory) if "body_reference" in config else None
    bank_path = (Path(directory)/config["bank"]).resolve(strict=True)
    bank = json.loads(bank_path.read_text())
    if bank.get("schema_version") != 1 or bank.get("binding") != ctx.binding:
        raise ValueError("Grasp bank does not match the original native/model/geometry/hand binding")
    m = ctx.model; d = mujoco.MjData(m)
    temporal = config.get("temporal", {})
    _keys(temporal, set(), {"max_velocity_rad_s", "joint_velocity_limits_rad_s"}, "hold temporal")
    velocity_gate = JointVelocityGate(temporal, snapshot_from_model(m))
    frames = indices(config["frames"], len(ctx.q), "frames")
    if frames != sorted(frames):
        raise ValueError("frames must increase")
    holds = config["holds"]
    if not isinstance(holds, dict) or set(holds) != set(ctx.hands):
        raise ValueError("holds must cover exactly the configured hands")
    targets = {}; material = {}; material_binding = {}
    for name, hold in holds.items():
        _keys(hold, {"candidate", "window"}, set(), name)
        interval(hold["window"], len(ctx.q), name)
        index = _integer(hold["candidate"], "candidate")
        if index >= len(bank["candidates"][name]):
            raise ValueError("Candidate index outside bank")
        c = bank["candidates"][name][index]
        if c.get("rejected", True):
            raise ValueError("A crossing/unknown free-grasp candidate must not be mounted")
        h = ctx.hands[name]
        fingers = _finite(c["finger_qpos"], (len(h["joints"]),), "finger_qpos")
        limits = m.jnt_range[h["joints"]]
        if c["finger_joints"] != h["finger_joints"] or np.any(fingers < limits[:, 0]-1e-6) or np.any(fingers > limits[:, 1]+1e-6):
            raise ValueError("Candidate finger binding/limits mismatch")
        targets[name] = (_finite(c["position"], (3,), "grasp position"), rigid(c["rotation"], "grasp rotation"), fingers)
        if refinement is not None:
            material[name], material_binding[name] = bind_material_anchors(ctx, name, c)
    edits = config["editable_joints"]
    if not isinstance(edits, dict) or not edits:
        raise ValueError("editable_joints must declare native correction corridors")
    js = ctx.joints(list(edits)); addresses = m.jnt_qposadr[js]; dofs = m.jnt_dofadr[js]
    if any(set(js).intersection(h["joints"]) for h in ctx.hands.values()):
        raise ValueError("Finger joints are owned by holds/refinement and cannot also be editable arm variables")
    scene_joints = list(js)
    if refinement is not None:
        scene_joints.extend(j for h in ctx.hands.values() for j in h["joints"])
    scene_clearance = SceneClearance(m, ctx.native, scene_joints, config.get("scene_clearance"))
    object_clearance = ObjectClearance(ctx, scene_joints, config.get(
        "object_clearance", None if refinement is not None else {"enabled": False}))
    corridors = np.array([_finite(value, (2,), name) for name, value in edits.items()])
    if np.any(corridors[:, 0] > 0) or np.any(corridors[:, 1] < 0) or np.any(corridors[:, 0] >= corridors[:, 1]):
        raise ValueError("Each correction corridor must contain zero with positive width")
    pairs = config["collision_pairs"]
    if not isinstance(pairs, list):
        raise ValueError("collision_pairs must explicitly list native proxy geom pairs")
    seen = set()
    for pair in pairs:
        indices(pair, m.ngeom, "collision pair")
        key = tuple(sorted(pair))
        if len(pair) != 2 or key in seen:
            raise ValueError("Expected unique two-geom collision pairs")
        seen.add(key)
    max_nfev = _integer(config.get("max_nfev", 90), "max_nfev", 1)
    if max_nfev > 1000:
        raise ValueError("max_nfev must not exceed 1000")
    step_joint = number(config.get("max_joint_step_rad", .13), "joint step", 1e-6, np.pi)
    pw = number(config.get("posture_weight", .0002), "posture weight")
    tw = number(config.get("temporal_weight", .012), "temporal weight")
    cw = number(config.get("collision_weight", 20), "collision weight")
    body_guides = config.get("body_guides", {})
    if not isinstance(body_guides, dict):
        raise ValueError("body_guides must map named bodies to source-position weights")
    guides = {}
    for name, weight in body_guides.items():
        b = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, name)
        if b <= 0:
            raise ValueError("body_guides must name native bodies")
        guides[b] = number(weight, name)
    obj = config.get("object_motion")
    if obj is not None:
        _keys(obj, {"window", "translation_limit_m", "rotation_limit_rad", "max_translation_step_m", "max_rotation_step_rad", "ramp_frames"}, set(), "object_motion")
        interval(obj["window"], len(ctx.q), "object_motion.window")
        limit = np.array([number(obj["translation_limit_m"], "object translation", 1e-9)]*3+
                         [number(obj["rotation_limit_rad"], "object rotation", 1e-9, np.pi)]*3)/np.sqrt(3)
        step = np.array([number(obj["max_translation_step_m"], "translation step", 1e-9)]*3+
                        [number(obj["max_rotation_step_rad"], "rotation step", 1e-9, np.pi)]*3)/np.sqrt(3)
        ramp = _integer(obj["ramp_frames"], "ramp_frames", 1)
    q = ctx.q.copy(); new_r, new_t = ctx.R.copy(), ctx.t.copy()
    offsets = np.zeros((len(q), 6)); logs = []; allowed = np.zeros(q.shape, dtype=bool)
    virtual = np.array([[0, 0, 0], [.1, 0, 0], [0, .1, 0], [0, 0, .1]])
    frame_set = set(frames)
    arm_js, arm_addresses = list(js), addresses.copy()
    for f in frames:
        d.qpos[:] = ctx.q[f]; mujoco.mj_forward(m, d)
        guide_positions = {b: d.xpos[b].copy() for b in guides}
        active = [name for name, hold in holds.items() if hold["window"][0] <= f < hold["window"][1]]
        if not active:
            raise ValueError(f"No declared stable hold at requested frame {f}; use local repair for approach")
        for name in active:
            a = ctx.hands[name]["addresses"]
            q[f, a] = targets[name][2]; allowed[f, a] = True
        js = list(arm_js)
        finger_seed = np.concatenate([targets[name][2] for name in active]) if refinement is not None else np.empty(0)
        if refinement is not None:
            js.extend(j for name in active for j in ctx.hands[name]["joints"])
        addresses = m.jnt_qposadr[js]; dofs = m.jnt_dofadr[js]
        reference = np.r_[ctx.q[f, arm_addresses], finger_seed]
        lower = ctx.q[f, arm_addresses]+corridors[:, 0]
        upper = ctx.q[f, arm_addresses]+corridors[:, 1]
        if refinement is not None:
            lower = np.r_[lower, finger_seed-refinement["max_delta_rad"]]
            upper = np.r_[upper, finger_seed+refinement["max_delta_rad"]]
        lower = np.maximum(lower, m.jnt_range[js, 0])
        upper = np.minimum(upper, m.jnt_range[js, 1])
        seed = q[f-1, addresses].copy() if f-1 in frame_set else reference.copy()
        # Hold propagation couples consecutive solved samples. Unsolved approach
        # joins need local repair; constraining a static grasp to the raw preceding
        # frame would incorrectly rule out a reachable initialization.
        if f-1 in frame_set:
            def ancestry(body):
                result = set()
                while body:
                    result.add(int(body)); body = int(m.body_parentid[body])
                return result
            previous_active = [name for name, hold in holds.items()
                               if hold["window"][0] <= f-1 < hold["window"][1]]
            continued_bodies = set().union(*(ancestry(ctx.hands[name]["wrist"])
                                             for name in previous_active))
            new_bodies = set().union(*(ancestry(ctx.hands[name]["wrist"])
                                      for name in active if name not in previous_active))
            # A newly acquired hand has no fitted preceding hold pose. Solve
            # its first key freely inside the original corridor, then repair
            # the approach. Shared torso joints still obey an existing hold.
            continuous = np.array([int(m.jnt_bodyid[j]) not in new_bodies-continued_bodies for j in js])
            if refinement is not None:
                cursor = len(arm_js)
                for name in active:
                    n = len(ctx.hands[name]["joints"])
                    continuous[cursor:cursor+n] = name in previous_active
                    if name not in previous_active:
                        seed[cursor:cursor+n] = targets[name][2]
                    cursor += n
            lower[continuous] = np.maximum(lower[continuous], q[f-1, addresses[continuous]]-step_joint)
            upper[continuous] = np.minimum(upper[continuous], q[f-1, addresses[continuous]]+step_joint)
        move_object = obj is not None and obj["window"][0] <= f < obj["window"][1]
        prev_obj = offsets[f-1].copy() if f else np.zeros(6)
        if move_object:
            start, end = obj["window"]
            remaining = end-f if end < len(q) else np.inf
            # A sparse solve also has a protected join at the next unsolved
            # sample. Reserve the return corridor throughout the segment,
            # rather than discovering an impossible jump on its final frame.
            next_unsolved = f+1
            while next_unsolved < end and next_unsolved in frame_set:
                next_unsolved += 1
            if next_unsolved < len(q):
                remaining = min(remaining, next_unsolved-f)
            envelope = min((f-start+1)/ramp, remaining/ramp, 1.)
            # A strictly interior reserve avoids zero-width SciPy bounds when
            # a previous solution reaches the exact return-speed boundary.
            bound = np.minimum(limit*envelope, step*remaining*.9999)
            olow, ohigh = np.maximum(-bound, prev_obj-step), np.minimum(bound, prev_obj+step)
            if f+1 < len(q) and f+1 not in frame_set:
                olow, ohigh = np.maximum(olow, -step), np.minimum(ohigh, step)
            lower, upper = np.r_[lower, olow], np.r_[upper, ohigh]
            seed = np.r_[seed, prev_obj]
        if np.any(lower >= upper):
            raise ValueError(f"Empty native/object corridor at frame {f}; revise the explicit proposal")
        count = len(js)
        wrist_weight = refinement["wrist_weight"] if refinement is not None else 1.

        def evaluate(y, jacobian=False):
            d.qpos[:] = q[f]; d.qpos[addresses] = y[:count]; mujoco.mj_forward(m, d)
            offset = y[count:] if move_object else np.zeros(6)
            delta = Rotation.from_rotvec(offset[3:]).as_matrix()
            residual, rows = [], []
            for name in active:
                hand = ctx.hands[name]; b = hand["wrist"]
                p, r, _ = targets[name]
                local = virtual @ r.T+p
                before = local @ ctx.R[f].T
                desired = before @ delta.T+ctx.t[f]+offset[:3]
                actual = virtual @ d.xmat[b].reshape(3, 3).T+d.xpos[b]
                residual.extend(wrist_weight*(actual-desired).ravel())
                if jacobian:
                    for k, point in enumerate(actual):
                        jp = np.zeros((3, m.nv)); jr = np.zeros_like(jp)
                        mujoco.mj_jac(m, d, jp, jr, point, b)
                        row = np.zeros((3, len(y))); row[:, :count] = jp[:, dofs]
                        if move_object:
                            row[:, count:count+3] = -np.eye(3)
                            for axis in range(3):
                                eps = np.eye(3)[axis]*1e-6
                                dr = (Rotation.from_rotvec(offset[3:]+eps).as_matrix()-Rotation.from_rotvec(offset[3:]-eps).as_matrix())/2e-6
                                row[:, count+3+axis] = -dr @ before[k]
                        rows.append(wrist_weight*row)
            if refinement is not None:
                for name in active:
                    value = material_residual(m, d, material[name], ctx.R[f], ctx.t[f], offset,
                        dofs, len(y), move_object=move_object,
                        weight=refinement["contact_weight"], jacobian=jacobian)
                    if jacobian:
                        rows.append(value)
                    else:
                        residual.extend(value)
            if body_reference is not None:
                value = body_reference.evaluate(d, f, dofs, len(y), jacobian=jacobian)
                if jacobian:
                    rows.append(value)
                else:
                    residual.extend(value)
            for b, weight in guides.items():
                residual.extend(weight*(d.xpos[b]-guide_positions[b]))
                if jacobian:
                    jp = np.zeros((3, m.nv)); jr = np.zeros_like(jp)
                    mujoco.mj_jac(m, d, jp, jr, d.xpos[b], b)
                    row = np.zeros((3, len(y))); row[:, :count] = weight*jp[:, dofs]; rows.append(row)
            posture_weights = np.r_[np.full(len(arm_js), pw),
                                    np.full(len(finger_seed), refinement["posture_weight"])] if refinement is not None else np.full(count, pw)
            for weight, ref in ((posture_weights, reference), (np.full(count, tw), seed[:count])):
                residual.extend(weight*(y[:count]-ref))
                if jacobian:
                    row = np.zeros((count, len(y))); row[:, :count] = np.diag(weight); rows.append(row)
            for g, h in pairs:
                witness = np.zeros(6)
                dist = mujoco.mj_geomDistance(m, d, g, h, .1, witness)
                residual.append(cw*min(dist, 0))
                if jacobian:
                    row = np.zeros((1, len(y)))
                    if dist < -1e-10:
                        jp, jr, kp, kr = [np.zeros((3, m.nv)) for _ in range(4)]
                        mujoco.mj_jac(m, d, jp, jr, witness[:3], int(m.geom_bodyid[g]))
                        mujoco.mj_jac(m, d, kp, kr, witness[3:], int(m.geom_bodyid[h]))
                        row[0, :count] = cw*((witness[:3]-witness[3:])/dist) @ (jp[:, dofs]-kp[:, dofs])
                    rows.append(row)
            if jacobian:
                scene_rows = scene_clearance.evaluate(m, d, dofs, len(y), True)
                if len(scene_rows):
                    rows.append(scene_rows)
            else:
                residual.extend(scene_clearance.evaluate(m, d, dofs, len(y)))
            object_value = object_clearance.evaluate(d, ctx.R[f], ctx.t[f], offset, dofs, len(y),
                                                    move_object=move_object, jacobian=jacobian)
            if jacobian:
                if len(object_value):
                    rows.append(object_value)
            else:
                residual.extend(object_value)
            if move_object:
                for weights, ref in ((np.array([.04]*3+[.004]*3), np.zeros(6)),
                                     (np.array([.3]*3+[.03]*3), prev_obj)):
                    residual.extend(weights*(offset-ref))
                    if jacobian:
                        row = np.zeros((6, len(y))); row[:, count:] = np.diag(weights); rows.append(row)
            return np.vstack(rows) if jacobian else np.asarray(residual)

        seed = np.clip(seed, lower, upper)
        fit = least_squares(evaluate, seed, jac=lambda y: evaluate(y, True), bounds=(lower, upper),
                            max_nfev=max_nfev, ftol=1e-9, xtol=1e-9, gtol=1e-9)
        q[f, addresses] = fit.x[:count]; allowed[f, addresses] = True
        if move_object:
            offsets[f] = fit.x[count:]
            new_r[f] = Rotation.from_rotvec(offsets[f, 3:]).as_matrix() @ ctx.R[f]
            new_t[f] = ctx.t[f]+offsets[f, :3]
        evaluate(fit.x)  # Restore FK to the exact exported native solution.
        wrist_errors = []
        for name in active:
            p, r, _ = targets[name]; b = ctx.hands[name]["wrist"]
            actual = virtual @ d.xmat[b].reshape(3, 3).T+d.xpos[b]
            desired = (virtual @ r.T+p) @ new_r[f].T+new_t[f]
            wrist_errors.extend(np.linalg.norm(actual-desired, axis=1).tolist())
        record = dict(frame=f, solver_success=bool(fit.success), nfev=fit.nfev,
                      max_wrist_guide_error_m=max(wrist_errors),
                      object_offset=offsets[f].tolist(), scene_clearance=scene_clearance.diagnostics(d),
                      object_clearance=object_clearance.diagnostics(d, ctx.R[f], ctx.t[f], offsets[f]),
                      frozen_inactive_finger_joints=[j for name, hand in ctx.hands.items()
                                                     if name not in active for j in hand["finger_joints"]])
        if refinement is not None:
            record["material_contact"] = {name: material_diagnostics(
                m, d, material[name], new_r[f], new_t[f], refinement["max_gap_m"]) for name in active}
            record["finger_changes"] = {}
            for name in active:
                h = ctx.hands[name]; a = h["addresses"]
                changes = q[f, a]-targets[name][2]
                record["finger_changes"][name] = dict(joints=h["finger_joints"],
                    seed_delta_rad=changes.tolist(), original_delta_rad=(q[f, a]-ctx.q[f, a]).tolist(),
                    max_seed_delta_rad=float(np.abs(changes).max()))
        if body_reference is not None:
            record["body_reference"] = body_reference.diagnostics(d, f)
        logs.append(record)
    if not np.array_equal(q[~allowed], ctx.q[~allowed]):
        raise AssertionError("Frozen native channels changed")
    velocity_review = velocity_gate.evaluate(
        q, fps=float(ctx.native["fps"]), timestamps=ctx.native.get("timestamps"),
        editable_mask=allowed,
    )
    reasons = []
    if velocity_review["passed"] is False:
        reasons.append("JOINT_VELOCITY_LIMIT_EXCEEDED")
    if refinement is not None and any(not item["passed"] for row in logs for item in row["material_contact"].values()):
        reasons.append("MATERIAL_CONTACT_GAP_EXCEEDED")
    status = ("REJECTED_BY_JOINT_VELOCITY" if "JOINT_VELOCITY_LIMIT_EXCEEDED" in reasons
              else "REJECTED_BY_MATERIAL_CONTACT" if reasons else "UNREVIEWED_SHARED_GRASP")
    if body_reference is not None:
        body_reference.check_unchanged()
    invalidated = sorted(k for k in ctx.native if _is_derived(k) or k in native_io.INVALIDATED_FIELDS
                         or k in native_io.CANDIDATE_STATE_FIELDS or k == "motion_accepted")
    candidate = {k: v.copy() for k, v in ctx.native.items() if k not in invalidated}
    candidate.update(qpos=q, obj_R=new_r, obj_t=new_t, model_xml_path=np.asarray(str(ctx.paths["model_xml"])),
                     status=np.asarray(status), motion_accepted=np.asarray(False),
                     fk_cache_valid=np.asarray(False), shared_grasp_parent_sha256=np.asarray(ctx.binding["native_sha256"]))
    report = dict(schema_version=1, status=status, motion_accepted=False,
                  rejection_reasons=reasons, finger_refinement=refinement,
                  material_anchor_binding=material_binding,
                  body_reference=body_reference.binding if body_reference is not None else None,
                  frozen_native_channels_preserved=True,
                  editable_qpos_addresses_by_frame={str(f): np.flatnonzero(allowed[f]).tolist() for f in frames},
                  joint_velocity=velocity_review,
                  binding=ctx.binding, bank_sha256=_sha(bank_path), config_sha256=digest(config),
                  implementation_sha256=code_binding("grasp_io.py", "shared_grasp.py", "mounted_grasp.py", "grasp_body_reference.py", "grasp_object_clearance.py", "shared_grasp_scene.py", "joint_velocity.py", "repair_geometry.py", "static_scene.py"),
                  elapsed_seconds=time.monotonic()-started, frames=logs, holds=holds,
                  invalidated_fields=invalidated, normal_weight=0, collision_proxy_pairs=pairs,
                  scene_clearance=scene_clearance.report, object_clearance=object_clearance.report,
                  max_object_translation_m=float(np.linalg.norm(offsets[:, :3], axis=1).max()),
                  max_object_rotation_rad=float(np.linalg.norm(offsets[:, 3:], axis=1).max()),
                  max_object_translation_step_m=float(np.linalg.norm(np.diff(offsets[:, :3], axis=0), axis=1).max(initial=0)),
                  max_object_rotvec_step_rad=float(np.linalg.norm(np.diff(offsets[:, 3:], axis=0), axis=1).max(initial=0)),
                  limitations=[("Joint hand/arm material fitting is a soft kinematic objective with a post-fit gap gate, not physical stability or sequence acceptance."
                                if refinement is not None else "Fixed finger layout and shared wrist guides are kinematic objectives, not hard no-slip constraints."),
                               "Native geom-distance proxies cover only explicit robot pairs; the scene adds finite unsigned surface barriers, not an original-mesh collision audit.",
                               "Object clearance is sampled unsigned surface guidance; material anchors and this penalty do not certify nonpenetration or containment. Independent original-mesh checks remain required.",
                               "Sparse keys and approach/hold joins need local repair, original-mesh/subframe audit and source/visual review."])
    return candidate, report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        parser.error("--out must be a new directory")
    candidate, report = solve(json.loads(args.config.read_text()), args.config.resolve().parent)
    args.out.mkdir(parents=True)
    np.savez_compressed(args.out/"candidate.npz", **candidate)
    write_report(args.out/"report.json", report)
    print(json.dumps(dict(output=str(args.out), elapsed_seconds=report["elapsed_seconds"], status=report["status"])))


if __name__ == "__main__":
    main()
