"""Authored waypoint seeds and batched native transition repair.

A route is an explicit robot adaptation, not recovered source evidence. Original
geometry audits remain independent of convex seed and sampled GPU objectives.
"""

import argparse
import json
import time
from pathlib import Path

import mujoco
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation, Slerp

from .contact_edit import native_io
from .contact_edit.pose import _keys, _finite, _sha
from .grasp_io import number
from .motion_audit import AuditContext
from .motion_diagnostics import MotionDiagnostics, timestamps
from .transition_scope import inspect_collision_scope, options as scope_options


def _interval(value, count, label):
    if (
        not isinstance(value, list)
        or len(value) != 2
        or any(type(x) is not int for x in value)
        or not 0 <= value[0] < value[1] < count
    ):
        raise ValueError(label + " requires increasing in-range integer endpoints")
    return value


def _solver_options(config):
    opt = config["solver"]
    _keys(
        opt,
        {"device"},
        {
            "steps",
            "learning_rate",
            "samples",
            "margin_m",
            "allow_degenerate_subset",
            "proposal_pairs",
            "weights",
            "fit_substeps",
            "max_seconds",
            "velocity_projection",
            "route_reference",
            "velocity_projection_interval",
            "temporal_projection",
            "derivative_units",
            "spline_initializer",
            "contact_mode",
            "projection_scope",
        },
        "transition solver",
    )
    if not isinstance(opt["device"], str) or not (
        opt["device"] == "cpu" or opt["device"].startswith("cuda:")
    ):
        raise ValueError("Declare cpu or cuda:N")
    for key, default, high in [
        ("steps", 300, 10000),
        ("fit_substeps", 2, 8),
        ("samples", 96, 4096),
    ]:
        value = opt.get(key, default)
        if type(value) is not int or not 1 <= value <= high:
            raise ValueError(key + " must be an in-range integer")
    number(opt.get("max_seconds", 120), "max seconds", 0.01, 960)
    if type(opt.get("velocity_projection", False)) is not bool:
        raise ValueError("velocity_projection must be boolean")
    if opt.get("route_reference", "seed") not in ("seed", "parent"):
        raise ValueError("route_reference must be seed or parent")
    interval = opt.get("velocity_projection_interval", 0)
    if type(interval) is not int or not 0 <= interval <= 10000:
        raise ValueError("velocity_projection_interval must be a nonnegative integer")
    if interval and not opt.get("velocity_projection", False):
        raise ValueError("velocity_projection_interval requires velocity_projection")
    if opt.get("derivative_units", "frames") not in ("frames", "seconds"):
        raise ValueError("derivative_units must be frames or seconds")
    if opt.get("contact_mode", "within_gates") not in ("target", "within_gates"):
        raise ValueError("contact_mode must be target or within_gates")
    if opt.get("projection_scope", "all_hinges") not in ("all_hinges", "editable_hinges"):
        raise ValueError("projection_scope must be all_hinges or editable_hinges")
    if opt.get("temporal_projection") and not opt.get("velocity_projection", False):
        raise ValueError("temporal_projection requires velocity_projection")
    return opt


def prepare(context, config, directory="."):
    """Validate the route and tighten (never expand) the common repair scope."""
    c = context
    _keys(
        config,
        {"repair_config", "joint_windows", "routes", "solver"},
        {"joint_blends", "seed", "collision_scope", "pose_keyframes", "posture_reference"},
        "transition",
    )
    _solver_options(config)
    scope_options(config)
    from .posture_reference import PostureReference
    c.bindings.pop("posture_reference", None)
    c.posture_reference = (PostureReference(c, config["posture_reference"], directory)
                           if "posture_reference" in config else None)
    if c.contact_normal_policy or any(
        x["normal_weight"] != 0 or x["max_normal_deg"] < 180 for x in c.contacts
    ):
        raise ValueError(
            "Transition workflow requires explicit normal-free contact objectives and gates"
        )
    if not isinstance(config["joint_windows"], dict) or set(
        config["joint_windows"]
    ) != set(c.names):
        raise ValueError("joint_windows must cover exactly the named editable joints")
    mask = c.mask.copy()
    for name, window in config["joint_windows"].items():
        start, end = _interval(window, c.T, "joint window")
        if start < c.start or end > c.end:
            raise ValueError("Joint window exceeds the common repair window")
        a = c.qids[c.names.index(name)]
        mask[: start + 1, a] = False
        mask[end:, a] = False
    if not mask.any():
        raise ValueError("No editable transition coordinates")
    routes = []
    if not isinstance(config["routes"], list) or not config["routes"]:
        raise ValueError("Declare at least one route")
    for route in config["routes"]:
        _keys(
            route,
            {"body", "joints", "window", "waypoints"},
            {"rotation_window", "orientation_radius_m"},
            "route",
        )
        start, end = _interval(route["window"], c.T, "route window")
        js = route["joints"]
        if (
            not isinstance(js, list)
            or not js
            or len(set(js)) != len(js)
            or not set(js) <= set(c.names)
        ):
            raise ValueError("Route joints must be unique editable joints")
        if any(config["joint_windows"][j] != [start, end] for j in js):
            raise ValueError("Route must match each joint window")
        body = mujoco.mj_name2id(c.model, mujoco.mjtObj.mjOBJ_BODY, route["body"])
        if body <= 0:
            raise ValueError("Unknown route body")
        waypoints = route["waypoints"]
        last = start
        if not isinstance(waypoints, list):
            raise ValueError("waypoints must be a list")
        for pt in waypoints:
            _keys(pt, {"frame", "position_world"}, set(), "waypoint")
            if type(pt["frame"]) is not int or not last < pt["frame"] < end:
                raise ValueError("Waypoints must increase strictly inside route")
            _finite(pt["position_world"], (3,), "waypoint")
            last = pt["frame"]
        rw = _interval(
            route.get("rotation_window", [start, end]), c.T, "rotation window"
        )
        if rw[0] < start or rw[1] > end:
            raise ValueError("Rotation window must lie inside route")
        routes.append(
            dict(
                route,
                body_id=body,
                rotation_window=rw,
                orientation_radius_m=number(
                    route.get("orientation_radius_m", 0.1),
                    "orientation radius",
                    0.001,
                    1,
                ),
            )
        )
    claimed = set()
    for route in routes:
        if claimed.intersection(route["joints"]):
            raise ValueError("Routes cannot independently overwrite shared joints")
        claimed.update(route["joints"])
    for blend in config.get("joint_blends", []):
        _keys(blend, {"joints", "window"}, set(), "joint blend")
        window = _interval(blend["window"], c.T, "blend window")
        js = blend["joints"]
        if (
            not isinstance(js, list)
            or not js
            or len(set(js)) != len(js)
            or not set(js) <= set(c.names)
            or claimed.intersection(js)
        ):
            raise ValueError("Joint blends must own distinct editable joints")
        if any(
            config["joint_windows"][j][0] > window[0]
            or config["joint_windows"][j][1] < window[1]
            for j in js
        ):
            raise ValueError("Blend exceeds joint window")
        claimed.update(js)
    if claimed != set(c.names):
        raise ValueError("Every editable joint needs a route or blend")
    return mask, routes


def seed_routes(c, config, mask, routes, *, deadline, progress=None):
    opt = config.get("seed", {})
    _keys(
        opt,
        set(),
        {
            "max_evaluations",
            "posture_weight",
            "continuity_weight",
            "collision_pairs",
            "clearance_m",
            "clearance_weight",
            "native_input",
        },
        "seed",
    )
    if type(opt.get("native_input", False)) is not bool:
        raise ValueError("seed.native_input must be boolean")
    if opt.get("native_input", False):
        return c.initial_qpos.copy(), [dict(mode="NATIVE_INPUT", waypoint_ik=False)], "COMPLETE"
    evaluations = int(
        number(opt.get("max_evaluations", 55), "max evaluations", 1, 1000)
    )
    pose_w = number(opt.get("posture_weight", 0.004), "seed posture")
    continuity = number(opt.get("continuity_weight", 0.012), "seed continuity")
    margin = number(opt.get("clearance_m", 0.002), "seed margin", 0, 0.02)
    cw = number(opt.get("clearance_weight", 20), "seed clearance")
    pairs = opt.get("collision_pairs", [])
    allowed = {frozenset((p["a"], p["b"])) for p in c.pairs}
    gp = []
    for pair in pairs:
        if (
            len(pair) != 2
            or frozenset(pair) not in allowed
            or any(c.meshes[p]["geom"] is None for p in pair)
        ):
            raise ValueError("Seed pairs must be declared native robot geom pairs")
        gp.append(tuple(c.meshes[p]["geom"] for p in pair))
    q = c.initial_qpos.copy()
    original = q.copy()
    ts = timestamps(c.parent)
    m = c.model
    d = mujoco.MjData(m)
    lo = q.copy()
    hi = q.copy()
    lo[np.ix_(c.frames, c.qids)] = c.lower
    hi[np.ix_(c.frames, c.qids)] = c.upper
    logs = []
    for blend in config.get("joint_blends", []):
        start, end = blend["window"]
        ids = [c.qids[c.names.index(j)] for j in blend["joints"]]
        for f in range(start + 1, end):
            u = (ts[f] - ts[start]) / (ts[end] - ts[start])
            u = u * u * (3 - 2 * u)
            for a in ids:
                if mask[f, a]:
                    q[f, a] = np.clip(
                        (1 - u) * original[start, a] + u * original[end, a],
                        lo[f, a],
                        hi[f, a],
                    )

    def pose(frame, body):
        d.qpos[:] = original[frame]
        mujoco.mj_forward(m, d)
        return d.xpos[body].copy(), d.xmat[body].reshape(3, 3).copy()

    for route in routes:
        start, end = route["window"]
        body = route["body_id"]
        p0, r0 = pose(start, body)
        pe, re = pose(end, body)
        v0 = (
            (p0 - pose(start - 1, body)[0]) / (ts[start] - ts[start - 1])
            if start
            else np.zeros(3)
        )
        ve = (
            (pose(end + 1, body)[0] - pe) / (ts[end + 1] - ts[end])
            if end + 1 < c.T
            else np.zeros(3)
        )
        frames = [start, *[w["frame"] for w in route["waypoints"]], end]
        spline = CubicSpline(
            ts[frames],
            [p0, *[w["position_world"] for w in route["waypoints"]], pe],
            bc_type=((1, v0), (1, ve)),
        )
        spin = Slerp([0, 1], Rotation.from_matrix([r0, re]))
        cloud = np.vstack([np.zeros(3), np.eye(3) * route["orientation_radius_m"]])
        ids = [c.qids[c.names.index(j)] for j in route["joints"]]
        dofs = [c.dids[c.names.index(j)] for j in route["joints"]]
        for f in range(start + 1, end):
            if time.monotonic() >= deadline:
                return q, logs, "TIME_LIMIT"
            active = [i for i, a in enumerate(ids) if mask[f, a]]
            if not active:
                continue
            ads = np.array(ids)[active]
            ds = np.array(dofs)[active]
            rs, rt = route["rotation_window"]
            u = np.clip((ts[f] - ts[rs]) / (ts[rt] - ts[rs]), 0, 1)
            u = u * u * (3 - 2 * u)
            target = cloud @ spin(u).as_matrix().T + spline(ts[f])
            prior = c.parent["qpos"][f, ads]
            previous = q[f - 1, ads]

            def fun(x, jac=False):
                if time.monotonic() >= deadline:
                    raise TimeoutError("Transition seed time limit")
                d.qpos[:] = q[f]
                d.qpos[ads] = x
                mujoco.mj_forward(m, d)
                world = cloud @ d.xmat[body].reshape(3, 3).T + d.xpos[body]
                residual = [
                    (world - target).ravel(),
                    pose_w * (x - prior),
                    continuity * (x - previous),
                ]
                rows = []
                if jac:
                    for pt in world:
                        jp = np.zeros((3, m.nv))
                        jr = np.zeros_like(jp)
                        mujoco.mj_jac(m, d, jp, jr, pt, body)
                        rows.append(jp[:, ds])
                    rows.extend(
                        [pose_w * np.eye(len(ds)), continuity * np.eye(len(ds))]
                    )
                for a, b in gp:
                    witness = np.zeros(6)
                    distance = mujoco.mj_geomDistance(m, d, a, b, 0.06, witness)
                    residual.append([cw * min(distance - margin, 0)])
                    if jac:
                        row = np.zeros(len(ds))
                        if distance < margin and abs(distance) > 1e-9:
                            jp, jr, kp, kr = [np.zeros((3, m.nv)) for _ in range(4)]
                            mujoco.mj_jac(
                                m, d, jp, jr, witness[:3], int(m.geom_bodyid[a])
                            )
                            mujoco.mj_jac(
                                m, d, kp, kr, witness[3:], int(m.geom_bodyid[b])
                            )
                            row = (
                                cw
                                * ((witness[:3] - witness[3:]) / distance)
                                @ (jp[:, ds] - kp[:, ds])
                            )
                        rows.append(row[None])
                return np.vstack(rows) if jac else np.concatenate(residual)

            try:
                fit = least_squares(
                    fun,
                    np.clip(previous, lo[f, ads], hi[f, ads]),
                    jac=lambda x: fun(x, True),
                    bounds=(lo[f, ads], hi[f, ads]),
                    max_nfev=evaluations,
                )
            except TimeoutError:
                return q, logs, "TIME_LIMIT"
            q[f, ads] = fit.x
            # scipy's final residual already corresponds to fit.x; avoid
            # another query after the cooperative deadline has expired.
            err = float(np.linalg.norm(fit.fun[:12].reshape(4, 3), axis=1).max())
            logs.append(
                dict(
                    body=route["body"], frame=f, guide_error_m=err, evaluations=fit.nfev
                )
            )
        if progress:
            progress(dict(stage="route_seed", body=route["body"], frames=len(logs)))
    assert np.array_equal(q[~mask], original[~mask])
    return q, logs, "COMPLETE"


def refine(c, config, seed, mask, routes, *, deadline, progress=None):
    import torch
    from .clearance_objective import BatchedClearance
    from .repair_tensor import NativeTensorFK
    from .repair_material import (
        _prepare_contact_terms,
        _prepared_contact_residual_squared,
    )

    opt = _solver_options(config)
    projection_logs = []
    def bounds(values):
        lower, upper = values.copy(), values.copy()
        lower[np.ix_(c.frames, c.qids)] = c.lower
        upper[np.ix_(c.frames, c.qids)] = c.upper
        return lower, upper
    def project_speed(values, stage="unspecified"):
        from .velocity_projection import project
        lower, upper = bounds(values)
        gate = c.joint_velocity_gate
        temporal_options = opt.get("temporal_projection")
        projection_scope = opt.get("projection_scope", "all_hinges")
        if projection_scope == "editable_hinges":
            import copy
            gate = copy.copy(gate)
            gate.hinges = [j for j in gate.hinges if mask[:, j["address"]].any()]
            if temporal_options:
                temporal_options = copy.deepcopy(temporal_options)
                names = {j["joint"] for j in gate.hinges}
                for key in ("joint_acceleration_limits_rad_s2", "joint_jerk_limits_rad_s3"):
                    if key in temporal_options:
                        temporal_options[key] = {k: v for k, v in temporal_options[key].items() if k in names}
        if opt.get("temporal_projection"):
            from .temporal_projection import project as project_temporal
            projected, receipt = project_temporal(values, mask, lower, upper,
                gate, timestamps(c.parent), temporal_options, deadline=deadline)
        else:
            projected, receipt = project(values, mask, lower, upper, gate,
                                         timestamps(c.parent), deadline=deadline)
        receipt["stage"] = stage
        receipt["projection_scope"] = projection_scope
        receipt["projected_hinges"] = [j["joint"] for j in gate.hinges]
        if projection_scope == "editable_hinges":
            receipt["full_motion_velocity"] = c.joint_velocity_gate.evaluate(
                projected, float(c.parent["fps"]), timestamps(c.parent), mask)
            if "temporal" in receipt:
                receipt["temporal"]["scope"] = "projected editable hinges over all frames and joins; other hinges require the final full-motion gate"
        projection_logs.append(receipt)
        return projected
    spline_receipt = None
    if opt.get("spline_initializer"):
        from .spline_smoothing import smooth
        lower, upper = bounds(seed)
        seed, spline_receipt = smooth(seed, mask, lower, upper, timestamps(c.parent),
                                     **opt["spline_initializer"])
    if opt.get("velocity_projection", False):
        seed = project_speed(seed, "initializer")
    if time.monotonic() >= deadline:
        return seed.copy(), dict(stop_reason="TIME_LIMIT", history=[])
    device = opt["device"]
    if not isinstance(device, str) or not (
        device == "cpu" or device.startswith("cuda:")
    ):
        raise ValueError("Declare cpu or cuda:N")
    steps = int(number(opt.get("steps", 300), "steps", 1, 10000))
    sub = int(number(opt.get("fit_substeps", 2), "fit substeps", 1, 8))
    w = dict(
        clearance=1500.0,
        joint_acceleration=2.0,
        joint_velocity=0.01,
        posture=0.002,
        route=1.0,
        wrist_acceleration=0.00003,
        contact=100.0,
        velocity_violation=100.0,
        joint_jerk=0.0,
        wrist_jerk=0.0,
    )
    _keys(opt.get("weights", {}), set(), set(w), "transition weights")
    w.update(opt.get("weights", {}))
    w = {k: number(v, k) for k, v in w.items()}
    pairs = opt.get("proposal_pairs", [[p["a"], p["b"]] for p in c.pairs])
    allowed = {frozenset((p["a"], p["b"])) for p in c.pairs}
    if any(len(p) != 2 or frozenset(p) not in allowed for p in pairs):
        raise ValueError("Proposal pairs must be declared audit pairs")
    engine = (
        BatchedClearance(
            {n: v["mesh"] for n, v in c.meshes.items()},
            pairs,
            device=device,
            samples=opt.get("samples", 96),
            margin_m=opt.get("margin_m", 0.003),
            allow_degenerate_subset=opt.get("allow_degenerate_subset", []),
        )
        if pairs
        else None
    )
    fk = NativeTensorFK(c.model, device, dtype=torch.float32)
    base = fk.t(seed)
    parent_prior = fk.t(c.parent["qpos"])
    posture_reference = getattr(c, "posture_reference", None)
    if "posture_reference" in config and posture_reference is None:
        raise ValueError("Bind the posture reference with prepare before refinement")
    prior = fk.t(posture_reference.target(c.parent["qpos"])) if posture_reference else parent_prior
    A = c.qids
    tm = fk.t(mask[:, A])
    ts = fk.t(timestamps(c.parent))
    speed_limits = None
    if c.joint_velocity_gate.configured:
        gate = c.joint_velocity_gate
        by_address = {j["address"]: gate.overrides.get(j["joint"], gate.default_limit)
                      for j in gate.hinges}
        speed_limits = fk.t([by_address[a] for a in A])
    # Exact same interpolation convention as the independent native audit.
    times = np.arange(c.start * sub, c.end * sub + 1) / sub
    left = np.floor(times).astype(int)
    right = np.minimum(left + 1, c.T - 1)
    alpha = fk.t((times - left)[:, None])
    oR = fk.t(np.array([c.sample(seed, f)[1] for f in times]))
    ot = fk.t(np.array([c.sample(seed, f)[2] for f in times]))
    terms = _prepare_contact_terms(
        c.contacts, times, base, contact_weight=w["contact"], normal_weight=0
    )
    route_reference = opt.get("route_reference", "seed")
    rotation0, position0 = fk(parent_prior if route_reference == "parent" else base)
    bodies = sorted({r["body_id"] for r in routes})
    position0 = position0[:, bodies].detach()
    lower = seed[:, A].copy()
    upper = lower.copy()
    lower[c.frames] = c.lower
    upper[c.frames] = c.upper
    lo = fk.t(lower)
    hi = fk.t(upper)
    x = torch.nn.Parameter(base[:, A].clone())
    optimizer = torch.optim.Adam(
        [x], lr=number(opt.get("learning_rate", 0.006), "learning rate", 1e-6, 0.1)
    )
    eye = fk.t(np.broadcast_to(np.eye(3), (len(times), 3, 3)).copy())
    zero = fk.t(np.zeros((len(times), 3)))
    geom = {
        k: (
            fk.t(
                Rotation.from_quat(
                    c.model.geom_quat[v["geom"]][[1, 2, 3, 0]]
                ).as_matrix()
            ),
            fk.t(c.model.geom_pos[v["geom"]]),
        )
        for k, v in c.meshes.items()
        if v["geom"] is not None
    }

    def diff(values):
        return torch.diff(values, dim=0) / torch.diff(ts).reshape(
            (-1,) + (1,) * (values.ndim - 1)
        )

    history = []
    reason = "STEPS_COMPLETE"
    for step in range(steps):
        if time.monotonic() >= deadline:
            reason = "TIME_LIMIT"
            break
        optimizer.zero_grad()
        q = base.clone()
        q[:, A] = base[:, A] + tm * (x - base[:, A])
        rot, pos = fk(q)
        interp = q[left] + alpha * (q[right] - q[left])
        sr, sp = fk(interp)
        trans = {}
        if engine:
            for name in engine.points:
                item = c.meshes[name]
                if item.get("static"):
                    trans[name] = (eye, zero)
                elif name == "object":
                    trans[name] = (oR, ot)
                else:
                    R, p = geom[name]
                    b = item["body"]
                    trans[name] = (
                        sr[:, b] @ R,
                        sp[:, b] + (sr[:, b] @ p[:, None]).squeeze(-1),
                    )
            clear = engine.loss(trans).mean()
        else:
            clear = q.sum() * 0
        vel = torch.diff(q[:, A], dim=0)
        acc = torch.diff(q[:, A], n=2, dim=0)
        from .temporal_derivatives import torch_finite_difference
        jerk = q.sum() * 0
        if w["joint_jerk"]:
            jerk, _ = torch_finite_difference(q[:, A], ts, order=3)
        if opt.get("derivative_units", "frames") == "seconds":
            vel, _ = torch_finite_difference(q[:, A], ts, order=1)
            acc, _ = torch_finite_difference(q[:, A], ts, order=2)
        speed_violation = (torch.relu(torch.abs(diff(q[:, A])) - 0.98*speed_limits).square().mean()
                           if speed_limits is not None else q.sum()*0)
        wp = pos[:, bodies]
        wv = diff(wp)
        mid = (ts[:-1] + ts[1:]) * 0.5
        wa = torch.diff(wv, dim=0) / torch.diff(mid)[:, None, None]
        wj = wp.sum(-1) * 0
        if w["wrist_jerk"]:
            wj, _ = torch_finite_difference(wp, ts, order=3)
        contact = _prepared_contact_residual_squared(
            terms, sr, sp, oR, ot, mode=opt.get("contact_mode", "within_gates")
        ) / max(1, len(times))
        loss = (
            w["clearance"] * clear
            + w["joint_acceleration"] * acc.square().mean()
            + w["joint_velocity"] * vel.square().mean()
            + w["velocity_violation"] * speed_violation
            + w["posture"] * (q[:, A] - prior[:, A]).square().mean()
            + w["route"] * (wp - position0).square().mean()
            + w["wrist_acceleration"] * wa.square().sum(-1).mean()
            + w["joint_jerk"] * jerk.square().mean()
            + w["wrist_jerk"] * wj.square().sum(-1).mean()
            + contact
        )
        if not torch.isfinite(loss):
            raise ValueError("Nonfinite transition loss")
        loss.backward()
        if x.grad is None or not torch.isfinite(x.grad).all():
            raise ValueError("Nonfinite transition gradient")
        optimizer.step()
        with torch.no_grad():
            x.copy_(torch.minimum(torch.maximum(x, lo), hi))
            interval = opt.get("velocity_projection_interval", 0)
            if interval and (step+1) % interval == 0 and time.monotonic() < deadline:
                proposal = seed.copy()
                values = x.detach().cpu().numpy()
                for column, address in enumerate(A):
                    proposal[mask[:, address], address] = values[mask[:, address], column]
                feasible = project_speed(proposal, "iteration_" + str(step+1))
                x.copy_(fk.t(feasible[:, A]))
        if step % 50 == 0 or step == steps - 1:
            row = dict(
                step=step,
                loss=float(loss.detach()),
                clearance=float(clear.detach()),
                wrist_acceleration_rms_m_s2=float(
                    wa.detach().square().sum(-1).mean().sqrt()
                ),
            )
            history.append(row)
            if progress:
                progress(dict(stage="batched_refinement", **row))
    out = seed.copy()
    values = x.detach().cpu().numpy()
    for k, a in enumerate(A):
        out[mask[:, a], a] = values[mask[:, a], k]
    # Clamp in native float64 too: a float32 cast can otherwise exceed the corridor.
    out[np.ix_(c.frames, A)] = np.minimum(
        np.maximum(out[np.ix_(c.frames, A)], c.lower), c.upper
    )
    out[~mask] = seed[~mask]
    if opt.get("velocity_projection", False):
        out = project_speed(out, "final")
    return out, dict(
        stop_reason=reason,
        history=history,
        weights=w,
        derivative_units=opt.get("derivative_units", "frames"),
        spline_initializer=spline_receipt,
        normal_weight=0,
        contact_mode=opt.get("contact_mode", "within_gates"),
        velocity_projections=projection_logs,
        route_reference=route_reference,
        posture_reference=(posture_reference.binding if posture_reference else None),
        device=device,
        geometry=engine.metadata() if engine else {"status": "NO_PROPOSAL_PAIRS"},
        omitted_proposal_pairs=[
            [p["a"], p["b"]]
            for p in c.pairs
            if frozenset((p["a"], p["b"])) not in {frozenset(x) for x in pairs}
        ],
    )


def audit_candidate(context, candidate, *, progress=None):
    """Reject a confirmed crossing early; retain explicit partial coverage."""
    context.check_inputs_unchanged()
    required = context.times()
    qpos = native_io._native_qpos(candidate)
    checked = []
    rejected = False
    for frame in required:
        row = context._row(qpos, frame)
        checked.append(frame)
        rejected = any(f["kind"] == "collision" for f in row["failures"])
        if progress and (len(checked) == 1 or len(checked) % 10 == 0 or rejected):
            progress(
                dict(
                    stage="original_mesh_audit",
                    checked=len(checked),
                    required=len(required),
                    crossing_found=rejected,
                )
            )
        if rejected:
            break
    # Cached rows are reused; invariants, joins and configured acceptance gates
    # still run. Unknown geometry never counts as a confirmed crossing/pass.
    report = context.audit(candidate, times=checked)
    report["early_rejection"] = dict(
        confirmed_crossing=rejected,
        sample_coverage_complete=checked == required,
        required_source_frames=required,
        untested_source_frames=required[len(checked) :],
    )
    return report


def _run(config_path, output, *, cache_dir=None, proposal_only=False, progress=None):
    started = time.monotonic()
    path = Path(config_path).resolve(strict=True)
    config = json.loads(path.read_text())
    out = Path(output)
    repair_path = (path.parent / config["repair_config"]).resolve(strict=True)
    repair = json.loads(repair_path.read_text())
    c = AuditContext(repair, repair_path.parent, cache_dir=cache_dir)
    mask, routes = prepare(c, config, path.parent)
    seconds = number(
        config.get("solver", {}).get("max_seconds", 120), "max seconds", 0.01, 960
    )
    source_hashes = {
        str(p): _sha(p)
        for p in [
            path,
            repair_path,
            Path(__file__),
            Path(__file__).with_name("clearance_objective.py"),
            Path(__file__).with_name("repair_tensor.py"),
            Path(__file__).with_name("repair_material.py"),
            Path(__file__).with_name("motion_diagnostics.py"),
            Path(__file__).with_name("transition_scope.py"),
            Path(__file__).with_name("velocity_projection.py"),
            Path(__file__).with_name("temporal_projection.py"),
            Path(__file__).with_name("temporal_derivatives.py"),
            Path(__file__).with_name("spline_smoothing.py"),
            Path(__file__).with_name("pose_keyframes.py"),
            Path(__file__).with_name("posture_reference.py"),
        ]
    }
    (out / "config.json").write_text(json.dumps(config, indent=2))
    (out / "repair_config.json").write_text(json.dumps(repair, indent=2))
    preflight = inspect_collision_scope(c, config, mask, routes, progress=progress)
    (out / "collision_scope_preflight.json").write_text(
        json.dumps(preflight, indent=2, allow_nan=False)
    )
    diagnostic = preflight["mode"] == "diagnostic"
    if preflight["blocked"] and not diagnostic:
        report = dict(
            status="BLOCKED_EDIT_SCOPE", motion_accepted=False, selected=False,
            audit_after=None, audit_status="NOT_RUN", candidate_sha256=None,
            collision_scope_preflight=preflight,
            solver=dict(stop_reason="BLOCKED_EDIT_SCOPE", history=[]),
            bindings=c.bindings, implementation_and_config_sha256=source_hashes,
            next_action="Read collision_scope_preflight.json; revise local joint windows, "
                        "protected keys, proposal pair coverage or parent explicitly before solving.",
            elapsed_seconds=time.monotonic() - started,
        )
        if any(_sha(p) != sha for p, sha in source_hashes.items()):
            raise ValueError("Transition code/config changed during preflight")
        (out / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False))
        return report
    # The cached baseline audit has its own recorded walltime. The surrounding
    # case/launcher budget still bounds the complete operation.
    deadline = time.monotonic() + seconds
    seed, seed_log, seed_status = seed_routes(
        c, config, mask, routes, deadline=deadline, progress=progress
    )
    keyframe_receipt = None
    fit_mask = mask
    if seed_status == "COMPLETE" and config.get("pose_keyframes"):
        from .pose_keyframes import apply
        lower, upper = seed.copy(), seed.copy()
        lower[np.ix_(c.frames, c.qids)] = c.lower
        upper[np.ix_(c.frames, c.qids)] = c.upper
        seed, fit_mask, keyframe_receipt = apply(
            seed, mask, lower, upper, timestamps(c.parent), c.model_snapshot,
            config["pose_keyframes"])
    if seed_status == "COMPLETE":
        q, detail = refine(
            c, config, seed, fit_mask, routes, deadline=deadline, progress=progress
        )
    else:
        q, detail = seed, dict(stop_reason="TIME_LIMIT", history=[])
    if not np.array_equal(q[~mask], c.initial_qpos[~mask]):
        raise RuntimeError("Transition changed protected poses")
    pinned = mask & ~fit_mask
    if not np.array_equal(q[pinned], seed[pinned]):
        raise RuntimeError("Transition changed authored pose keys")
    plan = dict(
        schema_version=1,
        channels=[
            dict(
                joint=name,
                window=[c.start, c.end],
                keys=[
                    [c.start, float(c.parent["qpos"][c.start, a])],
                    [c.end, float(c.parent["qpos"][c.end, a])],
                ],
                interpolation="LINEAR",
            )
            for name, a in zip(c.names, c.qids)
        ],
        derived_fields=repair.get("derived_fields", []),
    )
    candidate, merge = native_io.merge_curve_samples(
        c.parent, c.model_snapshot, plan, q
    )
    candidate["selected"] = np.asarray(False)
    candidate["status"] = np.asarray("UNREVIEWED_TRANSITION")
    candidate["motion_accepted"] = np.asarray(False)
    candidate_path = out / "candidate_UNREVIEWED.npz"
    np.savez_compressed(candidate_path, **candidate)
    loaded = native_io.load_npz(candidate_path)
    if any(not np.array_equal(v, loaded[k]) for k, v in candidate.items()):
        raise RuntimeError("Candidate serialization changed arrays")
    if not np.array_equal(loaded["qpos"][pinned], seed[pinned]):
        raise RuntimeError("Native export changed authored pose keys")
    # Reuse exact unchanged baseline rows; changed poses get independent queries.
    audit = None if proposal_only else audit_candidate(c, loaded, progress=progress)
    derivative_gate = None
    tp = config["solver"].get("temporal_projection")
    if tp:
        from .temporal_derivatives import JointDerivativeGate
        policy = dict(max_acceleration_rad_s2=tp["acceleration_limit_rad_s2"],
                      max_jerk_rad_s3=tp["jerk_limit_rad_s3"])
        for name in ("joint_acceleration_limits_rad_s2", "joint_jerk_limits_rad_s3"):
            if name in tp:
                policy[name] = tp[name]
        derivative_gate = JointDerivativeGate(policy, c.model_snapshot).evaluate(
            loaded["qpos"], timestamps(loaded), mask)
    posture_gate = (c.posture_reference.evaluate(loaded["qpos"])
                    if c.posture_reference else None)
    status = (
        "REJECTED_BY_POSTURE_REFERENCE" if posture_gate is not None and not posture_gate["passed"] else
        "REJECTED_BY_DERIVATIVE_GATE" if derivative_gate is not None and derivative_gate["passed"] is not True else
        "REJECTED_BY_AUDIT" if audit is not None and not audit["passed"] else
        "DIAGNOSTIC_SCOPE_OVERRIDE" if diagnostic else "UNREVIEWED_TRANSITION"
    )
    candidate["status"] = np.asarray(status)
    np.savez_compressed(candidate_path, **candidate)
    loaded = native_io.load_npz(candidate_path)
    if any(not np.array_equal(v, loaded[k]) for k, v in candidate.items()):
        raise RuntimeError("Final candidate serialization changed arrays")
    diag = c.motion_diagnostics or MotionDiagnostics(
        c.model, c.parent, {"bodies": list(dict.fromkeys(r["body"] for r in routes))}
    )
    report = dict(
        status=status,
        motion_accepted=False,
        selected=False,
        audit_after=audit,
        audit_status="NOT_RUN" if proposal_only else audit["status"],
        collision_scope_preflight=preflight,
        diagnostic_only=diagnostic,
        seed=seed_log,
        pose_keyframes=keyframe_receipt,
        solver=detail,
        derivative_gate=derivative_gate,
        posture_reference_gate=posture_gate,
        checks=dict(protected_qpos_exact=True, serialized_exact=True,
                    authored_pose_keys_exact=True),
        native_merge=merge,
        changed_integer_frames=np.flatnonzero(
            np.any(loaded["qpos"] != c.initial_qpos, axis=1)
        ).tolist(),
        motion_diagnostics=(audit or {}).get("motion_diagnostics") or diag.compare(loaded["qpos"]),
        bindings=c.bindings,
        implementation_and_config_sha256=source_hashes,
        candidate_sha256=_sha(candidate_path),
        elapsed_seconds=time.monotonic() - started,
        limitations=[
            "Authored waypoint route; no automatic scene-edge inference or source-pose acceptance.",
            "Convex seed queries and sampled GPU distances guide proposals; original-triangle audit and visual review remain required.",
            "max_seconds bounds seed/refinement with cooperative checks; setup, one in-flight solve and final audit/export need outer case budget.",
        ],
    )
    if audit is not None and not audit["passed"]:
        report["next_action"] = (
            "Candidate rejected by the complete configured audit status, including inherited "
            "failures. Inspect failed pairs/frames, author the affected route or revise the "
            "scope explicitly, and retain the prior checkpoint. An affected_scope pass is not acceptance."
        )
    c.check_inputs_unchanged()
    if any(_sha(p) != sha for p, sha in source_hashes.items()):
        raise ValueError("Transition code/config changed during run")
    (out / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False))
    return report


def run(config_path, output, **kwargs):
    out = Path(output)
    out.mkdir(parents=True, exist_ok=False)
    try:
        return _run(config_path, out, **kwargs)
    except Exception as exc:
        (out / "error.json").write_text(
            json.dumps(
                dict(
                    status="FAILED",
                    motion_accepted=False,
                    type=type(exc).__name__,
                    message=str(exc),
                ),
                indent=2,
            )
        )
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("config", "output"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--cache")
    parser.add_argument(
        "--proposal-only",
        action="store_true",
        help="Explicitly skip audit; never an accepted result",
    )
    args = parser.parse_args(argv)
    report = run(
        args.config,
        args.output,
        cache_dir=args.cache,
        proposal_only=args.proposal_only,
        progress=lambda r: print(json.dumps(r), flush=True),
    )
    print(
        json.dumps(
            {k: report[k] for k in ("status", "audit_status", "elapsed_seconds")}
        )
    )
    return (
        0
        if report["audit_after"] is not None and report["audit_after"]["passed"]
        and not report.get("diagnostic_only", False)
        and (report.get("derivative_gate") is None or report["derivative_gate"]["passed"] is True)
        and (report.get("posture_reference_gate") is None or report["posture_reference_gate"]["passed"] is True)
        else 2
    )


if __name__ == "__main__":
    raise SystemExit(main())
