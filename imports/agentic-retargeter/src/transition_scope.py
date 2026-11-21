"""Bind original-mesh collision evidence to actual transition edit controls.

This is a feasibility guard, not a collision solver or acceptance certificate.
It never expands windows or unlocks a floating root. Rows come directly from the
current AuditContext's exact-state cache, not a caller-supplied audit summary.
"""

import copy
import hashlib
import time

import mujoco
import numpy as np

from .contact_edit.pose import _keys


def options(config):
    opt = config.get("collision_scope", {})
    _keys(opt, set(), {"mode", "reason"}, "collision scope")
    mode = opt.get("mode", "required")
    if mode not in ("required", "diagnostic"):
        raise ValueError("collision_scope.mode must be required or diagnostic")
    if mode == "diagnostic" and (
        not isinstance(opt.get("reason"), str) or not opt["reason"].strip()
    ):
        raise ValueError("Diagnostic collision scope requires an explicit reason")
    return dict(mode=mode, reason=opt.get("reason"))


def _array_sha(value):
    value = np.ascontiguousarray(value)
    return hashlib.sha256(
        str(value.dtype).encode() + str(value.shape).encode() + value.tobytes()
    ).hexdigest()


def _topology(model, routes):
    parents = [int(x) for x in model.body_parentid]
    children = [[] for _ in parents]
    for body in range(1, model.nbody):
        children[parents[body]].append(body)
    articulated = set(int(x) for x in model.jnt_bodyid)
    for body in reversed(range(1, model.nbody)):
        if body in articulated:
            articulated.add(parents[body])

    def chain(body):
        result = []
        while body:
            result.append(body)
            body = parents[body]
        return result

    def limb_root(body):
        # The named wrist/effector is the route endpoint. Include its serial
        # arm chain up to (but not including) the first articulated fork, and
        # its distal digit subtrees. This keeps waist/torso ancestors shared.
        while parents[body] > 0:
            parent = parents[body]
            if sum(child in articulated for child in children[parent]) > 1:
                break
            body = parent
        return body

    roots = {limb_root(route["body_id"]) for route in routes}

    def controls(body):
        ancestry = chain(body)
        candidates = [root for root in roots if root in ancestry]
        # A forearm is in the same root subtree as its route's wrist, while
        # digits inherit that wrist's arm instead of mistaking the palm fork
        # for a shared torso. Unrouted limbs use their own nearest fork.
        root = max(candidates, key=ancestry.index) if candidates else limb_root(body)
        local_bodies = set(ancestry[: ancestry.index(root) + 1])
        ancestors = set(ancestry)
        local, shared = [], []
        for joint in range(model.njnt):
            owner = int(model.jnt_bodyid[joint])
            if owner in ancestors and model.jnt_type[joint] == mujoco.mjtJoint.mjJNT_HINGE:
                (local if owner in local_bodies else shared).append(joint)
        return root, local, shared

    return controls


def _support(frame, count):
    lo = int(np.floor(frame))
    alpha = float(frame) - lo
    return [(lo, 1.0 - alpha)] + (
        [(min(lo + 1, count - 1), alpha)] if alpha > 0 else []
    )


def inspect_collision_scope(context, config, mask, routes, *, progress=None):
    """Audit the current seed on the declared grid, then check local controls.

    A local hinge with a nonzero interpolation coefficient is necessary, not
    proof of a usable clearance gradient/reachable route. The independent
    candidate audit still tests every configured gate after fitting.
    """
    started = time.monotonic()
    c = context
    opt = options(config)
    scene = sorted(k for k, item in c.meshes.items() if item.get("static"))
    report = dict(
        schema_version=1, mode=opt["mode"], diagnostic_reason=opt["reason"],
        status="NOT_REQUIRED_NO_STATIC_SCENE", blocked=False,
        motion_accepted=False, scene_meshes=scene, findings=[],
        coverage=dict(source_frames=[], continuous_time=False),
        scope_policy="native_route_limb_chain_below_first_articulated_fork",
        checked_failure_scope="pairs_involving_static_scene_meshes",
        limitations=[
            "Available local hinges do not prove clearance gradients or a feasible path.",
            "Shared ancestor movement alone is insufficient local limb repair coverage.",
            "Original-triangle evidence is finite sampled; missing or degenerate geometry is not clearance.",
        ],
    )
    if not scene:
        return report
    c.check_inputs_unchanged()
    seed = c.initial_qpos
    report["bindings"] = dict(
        inputs=copy.deepcopy(c.bindings), repair_config_sha256=c.config_digest,
        initial_qpos_sha256=_array_sha(seed), effective_edit_mask_sha256=_array_sha(mask),
        audit_code_sha256=copy.deepcopy(c.code_bindings),
        model_dependencies=copy.deepcopy(c.dependencies),
        compiled_model_sha256=c.compiled_digest,
    )
    robot = [k for k, item in c.meshes.items() if item["geom"] is not None]
    configured = {frozenset((p["a"], p["b"])) for p in c.pairs}
    missing = [[name, table] for name in robot for table in scene
               if frozenset((name, table)) not in configured]
    proposal = config["solver"].get(
        "proposal_pairs", [[p["a"], p["b"]] for p in c.pairs]
    )
    proposed = {frozenset(p) for p in proposal}
    clearance_enabled = config["solver"].get("weights", {}).get("clearance", 1500) > 0
    controls = _topology(c.model, routes)
    mesh_controls = {
        name: controls(item["body"]) for name, item in c.meshes.items()
        if item["geom"] is not None and item["body"] > 0
    }
    can_move = {
        name: any(mask[:, int(c.model.jnt_qposadr[j])].any()
                  for j in local + shared)
        for name, (_, local, shared) in mesh_controls.items()
    }
    # A clean parent pose does not excuse omitting a mesh that can move in the
    # proposal. Entirely frozen meshes still belong to the independent audit;
    # retaining their zero-gradient terms in every solve is unnecessary.
    omitted = [[name, table] for name in robot for table in scene
               if can_move.get(name, False) and frozenset((name, table)) not in proposed]
    frozen_omitted = [[name, table] for name in robot for table in scene
                      if not can_move.get(name, False) and frozenset((name, table)) not in proposed]
    problems = []
    loaded_geoms = {item["geom"] for item in c.meshes.values() if item["geom"] is not None}
    unlisted = [g for g in range(c.model.ngeom)
                if c.model.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH
                and c.model.geom_bodyid[g] > 0 and g not in loaded_geoms]
    if unlisted:
        problems.append(dict(kind="unlisted_robot_scene_meshes", geom_ids=unlisted))
    if missing:
        problems.append(dict(kind="missing_scene_audit_pairs", pairs=missing))
    if omitted:
        problems.append(dict(kind="missing_scene_proposal_pairs", pairs=omitted))
    if not clearance_enabled:
        problems.append(dict(kind="scene_clearance_weight_disabled"))
    hits, misses = c.cache_hits, c.cache_misses
    required = c.times()
    unknown = []
    for index, frame in enumerate(required):
        row = c._row(seed, frame)
        for failure in row["failures"]:
            if not any(name in scene for name in failure.get("pair", [])):
                continue
            if failure["kind"] == "geometry_incomplete":
                unknown.append(dict(source_frame=frame, pair=failure["pair"]))
            if failure["kind"] != "collision":
                continue
            pair = failure["pair"]
            ancestry = [set(sum(mesh_controls[name][1:], []))
                        if name in mesh_controls else set() for name in pair]
            common = ancestry[0] & ancestry[1]
            moving = []
            for name in pair:
                if name not in mesh_controls:
                    continue
                root, local, shared = mesh_controls[name]
                record = dict(
                    mesh=name, body=c.model.body(c.meshes[name]["body"]).name,
                    local_chain_root=c.model.body(root).name,
                    local_hinges=[], shared_ancestor_hinges=[],
                )
                for kind, joints in (("local_hinges", local), ("shared_ancestor_hinges", shared)):
                    for joint in joints:
                        jname = c.model.joint(joint).name
                        address = int(c.model.jnt_qposadr[joint])
                        support = [dict(frame=f, coefficient=w,
                                        editable=bool(mask[f, address]))
                                   for f, w in _support(frame, c.T)]
                        record[kind].append(dict(
                            joint=jname, qpos_address=address,
                            declared_window=config["joint_windows"].get(jname),
                            interpolation_support=support,
                            shared_by_collision_pair=joint in common,
                            can_change_relative_pose=(joint not in common and
                                                      any(s["editable"] for s in support)),
                        ))
                moving.append(record)
            local = [j["joint"] for m in moving for j in m["local_hinges"]
                     if j["can_change_relative_pose"]]
            remote = [j["joint"] for m in moving for j in m["shared_ancestor_hinges"]
                      if j["can_change_relative_pose"]]
            classification = ("LOCAL_CONTROL_AVAILABLE" if local else
                              "SHARED_ANCESTOR_ONLY" if remote else "NO_EDITABLE_CONTROL")
            actionable = (
                "Author/review a clearing route and rerun the original-mesh audit."
                if local else
                "Revise the declared local limb windows/hinges at these support frames, "
                "or choose another parent/scope. Do not automatically unlock root or protected keys."
            )
            report["findings"].append(dict(
                source_frame=frame, pair=pair, classification=classification,
                blocked=not bool(local), moving_meshes=moving,
                local_editable_hinges=local, shared_editable_hinges=remote,
                next_action=actionable,
            ))
        if progress and (index == 0 or (index + 1) % 10 == 0 or index + 1 == len(required)):
            progress(dict(stage="collision_scope_preflight", checked=index + 1,
                          required=len(required), collisions=len(report["findings"])))
    c.check_inputs_unchanged()
    report.update(
        blocked=bool(problems or any(f["blocked"] for f in report["findings"])),
        configuration_problems=problems,
        frozen_omitted_proposal_pairs=frozen_omitted,
        geometry_incomplete=unknown,
        coverage=dict(source_frames=required, pairs=[[p["a"], p["b"]] for p in c.pairs],
                      continuous_time=False, complete_declared_grid=True),
        cache=dict(hits=c.cache_hits-hits, misses=c.cache_misses-misses),
        elapsed_seconds=time.monotonic()-started,
    )
    report["status"] = "BLOCKED_EDIT_SCOPE" if report["blocked"] else "LOCAL_CONTROL_SCOPE_READY"
    return report
