"""Topology-only inspection of a declared original-mesh collision scope.

This module does not load numerical or GPU runtimes on import. Enumeration is
not collision detection: omitted pairs remain untested, including articulated
parent/child pairs and omissions with a user-provided reason.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
import hashlib
from itertools import combinations
import json
from numbers import Integral
from pathlib import Path


def _digest(value):
    try:
        encoded = json.dumps(value, sort_keys=True, allow_nan=False, separators=(",", ":")).encode()
    except (TypeError, ValueError) as exc:
        raise ValueError("Coverage metadata must be finite JSON data") from exc
    return hashlib.sha256(encoded).hexdigest()


def _integer(value, label, maximum):
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"{label} must be an integer")
    value = int(value)
    if not 0 <= value < maximum:
        raise ValueError(f"{label} outside model range")
    return value


def _sequence(value, label):
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"{label} must be a sequence")
    return value


def _name(value, label):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a nonempty string")
    return value


def _pair(item, labels, label, *, omission=False):
    allowed = {"a", "b", "reason"} if omission else {
        "a", "b", "clearance_m", "reference_frame"
    }
    required = {"a", "b", "reason"} if omission else {"a", "b"}
    if not isinstance(item, Mapping) or set(item) - allowed or required - set(item):
        raise ValueError(f"{label} has unknown or missing fields")
    a, b = _name(item["a"], label), _name(item["b"], label)
    if a not in labels or b not in labels or a == b:
        raise ValueError(f"{label} has unknown or identical mesh labels")
    result = {"a": a, "b": b}
    if omission:
        result["reason"] = _name(item["reason"], "omission reason")
    return result, tuple(sorted((a, b)))


def geometry_coverage(
    model,
    loaded_meshes,
    configured_pairs,
    editable_joint_names,
    *,
    omission_reasons=(),
):
    """Enumerate scope from an existing native model without querying geometry.

    ``loaded_meshes`` accepts AuditContext descriptors with ``geom`` and ``body``
    fields. A native descriptor must identify a mesh geom and its actual owner.
    An external object/environment has both fields ``None``; ``static=True``
    labels an environment descriptor but never establishes physical clearance.
    The optional triangle payload is not inspected or retained.

    All unordered loaded pairs are expected except robot pairs on the same body
    or directly parented bodies. Those two exclusions are listed explicitly;
    neither is a surface-clearance certificate. A named omission remains in the
    missing list and carries no proof. Existing pair orientation is preserved.

    Empty relative edit ancestry means geometry is unchanged *at matching source
    times* under that edit mask. It says nothing about variation in the original
    motion, external object trajectories, or pre-existing intersections.
    """
    if not isinstance(loaded_meshes, Mapping) or not loaded_meshes:
        raise ValueError("loaded_meshes must be a nonempty mapping")
    editable_joint_names = list(_sequence(editable_joint_names, "editable joints"))
    editable = []
    joint_names = {}
    for jid in range(model.njnt):
        name = model.joint(jid).name
        if name:
            if name in joint_names:
                raise ValueError("Duplicate native joint name")
            joint_names[name] = jid
    for name in editable_joint_names:
        _name(name, "editable joint")
        if name not in joint_names or name in editable:
            raise ValueError("Unknown or duplicate editable joint: " + name)
        # MuJoCo's public enum values: mjJNT_HINGE=3, mjGEOM_MESH=7.
        if int(model.jnt_type[joint_names[name]]) != 3:
            raise ValueError("Editable joint must be a native hinge: " + name)
        editable.append(name)
    editable = sorted(editable)
    on_body = {body: set() for body in range(model.nbody)}
    for name in editable:
        body = _integer(model.jnt_bodyid[joint_names[name]], "joint body", model.nbody)
        on_body[body].add(name)
    parents = [
        _integer(model.body_parentid[b], "parent body", model.nbody)
        for b in range(model.nbody)
    ]
    if parents[0] != 0:
        raise ValueError("World body must parent itself")
    ancestry = {0: set(on_body[0])}

    def ancestors(body):
        if body in ancestry:
            return ancestry[body]
        chain, seen, cursor = [], set(), body
        while cursor not in ancestry:
            if cursor in seen:
                raise ValueError("Native body parent cycle")
            seen.add(cursor)
            chain.append(cursor)
            cursor = parents[cursor]
        inherited = set(ancestry[cursor])
        for node in reversed(chain):
            inherited = inherited | on_body[node]
            ancestry[node] = set(inherited)
        return ancestry[body]

    inventory, used_geoms = {}, set()
    for label, descriptor in loaded_meshes.items():
        _name(label, "mesh label")
        if not isinstance(descriptor, Mapping) or not {"geom", "body"} <= set(descriptor):
            raise ValueError("Mesh descriptor requires geom and body")
        geom, body = descriptor["geom"], descriptor["body"]
        static = descriptor.get("static", False)
        if not isinstance(static, bool):
            raise ValueError("Mesh static flag must be boolean")
        if geom is None:
            if body is not None:
                raise ValueError("External mesh body must be None")
            kind = "environment" if static else "object"
            ancestor_joints, body_name, geom_name = [], None, None
        else:
            geom = _integer(geom, "mesh geom", model.ngeom)
            body = _integer(body, "mesh body", model.nbody)
            if geom in used_geoms:
                raise ValueError("Duplicate native mesh geom")
            if int(model.geom_type[geom]) != 7:
                raise ValueError("Native geom must be a mesh")
            if int(model.geom_bodyid[geom]) != body:
                raise ValueError("Mesh body does not own native geom")
            used_geoms.add(geom)
            kind = "robot"
            ancestor_joints = sorted(ancestors(body))
            body_name, geom_name = model.body(body).name, model.geom(geom).name
        inventory[label] = dict(
            label=label, kind=kind, geom=geom, geom_name=geom_name,
            body=body, body_name=body_name,
            editable_ancestor_joints=ancestor_joints,
            affected_by_edit=bool(ancestor_joints),
            external_transform="frozen_to_edit_not_assumed_constant_in_source_time" if geom is None else None,
        )
    labels = set(inventory)
    configured, configured_keys = [], set()
    for item in _sequence(configured_pairs, "configured pairs"):
        pair, key = _pair(item, labels, "configured pair")
        if key in configured_keys:
            raise ValueError("Duplicate configured unordered pair")
        configured_keys.add(key)
        configured.append(pair)
    omissions, omission_keys = [], set()
    for item in _sequence(omission_reasons, "omission reasons"):
        pair, key = _pair(item, labels, "omission", omission=True)
        if key in omission_keys or key in configured_keys:
            raise ValueError("Duplicate omission or omission for configured pair")
        omission_keys.add(key)
        omissions.append(pair)
    reasons = {tuple(sorted((p["a"], p["b"]))): p["reason"] for p in omissions}

    def pair_record(a, b):
        left, right = inventory[a], inventory[b]
        relative = sorted(set(left["editable_ancestor_joints"]) ^ set(right["editable_ancestor_joints"]))
        robot_count = int(left["kind"] == "robot") + int(right["kind"] == "robot")
        category = {0: "external_external", 1: "robot_external", 2: "robot_robot"}[robot_count]
        topology = None
        if robot_count == 2:
            ba, bb = left["body"], right["body"]
            if ba == bb:
                topology = "same_body"
            # World-attached original mesh obstacles are not articulated links.
            # Excluding world/root merely because body0 owns the tree would hide
            # collisions with scene geometry, including a floating robot base.
            elif ba != 0 and bb != 0 and (parents[ba] == bb or parents[bb] == ba):
                topology = "direct_parent"
        return dict(
            a=a, b=b, category=category, topology_exclusion=topology,
            bodies=[left["body"], right["body"]], geoms=[left["geom"], right["geom"]],
            editable_relative_joints=relative,
            fixed_relative_to_edit_mask=not relative,
        )

    missing, exclusions = [], []
    for a, b in combinations(sorted(labels), 2):
        pair = pair_record(a, b)
        key = (a, b)
        if pair["topology_exclusion"]:
            pair["configured"] = key in configured_keys
            if key in reasons:
                pair["declared_reason"] = reasons[key]
            exclusions.append(pair)
        elif key not in configured_keys:
            if key in reasons:
                pair["declared_reason"] = reasons[key]
            missing.append(pair)
    configured = [pair_record(p["a"], p["b"]) for p in configured]
    omission_records = [dict(pair_record(p["a"], p["b"]), reason=p["reason"], classification=pair_record(p["a"], p["b"])["topology_exclusion"] or "missing_expected_pair") for p in omissions]
    unlisted = []
    for geom in range(model.ngeom):
        if int(model.geom_type[geom]) == 7 and geom not in used_geoms:
            body = int(model.geom_bodyid[geom])
            unlisted.append(dict(geom=geom, geom_name=model.geom(geom).name, body=body, body_name=model.body(body).name, editable_ancestor_joints=sorted(ancestors(body))))
    policy = dict(
        expected="all_unordered_loaded_pairs_except_same_body_or_direct_parent_robot_pairs",
        topology_exclusions_are_clearance_proof=False,
        named_omissions_are_clearance_proof=False,
        missing_pairs_are_automatically_added=False,
        configured_pairs_are_retained_even_when_topology_excluded=True,
    )
    result = dict(
        schema_version=1, status="ENUMERATION_ONLY", geometry_status="NOT_RUN",
        policy=policy, editable_joint_names=editable,
        loaded_meshes=[inventory[k] for k in sorted(inventory)],
        configured_pairs=configured, missing_pairs=missing,
        topology_exclusions=exclusions, named_omissions=omission_records,
        unlisted_model_mesh_geoms=unlisted,
        environment_status="CONFIGURED" if any(x["kind"] == "environment" for x in inventory.values()) else "NOT_CONFIGURED",
        counts=dict(
            loaded_robot_meshes=sum(x["kind"] == "robot" for x in inventory.values()),
            loaded_external_meshes=sum(x["kind"] != "robot" for x in inventory.values()),
            configured_pairs=len(configured), missing_pairs=len(missing),
            missing_robot_robot_pairs=sum(x["category"] == "robot_robot" for x in missing),
            missing_robot_external_pairs=sum(x["category"] == "robot_external" for x in missing),
            missing_external_external_pairs=sum(x["category"] == "external_external" for x in missing),
            same_body_exclusions=sum(x["topology_exclusion"] == "same_body" for x in exclusions),
            direct_parent_exclusions=sum(x["topology_exclusion"] == "direct_parent" for x in exclusions),
            named_omissions=len(omissions), unlisted_model_mesh_geoms=len(unlisted),
        ),
        limitations=[
            "Enumeration covers only loaded mesh labels; unlisted native meshes and absent environment remain unchecked.",
            "Same-body/direct-parent and named omissions are scope policy, not physical clearance or inherited-seam proof.",
            "No editable relative joints means invariant relative geometry at matching source times under this mask, not constant geometry over the source trajectory.",
            "No triangle query, trajectory/contact validation, containment, dynamics, or continuous-time collision check was run; audit acceptance is unchanged.",
        ],
    )
    result["topology_scope_sha256"] = _digest(result)
    return result


def preflight_config(config, config_dir="."):
    """Compile only native model topology/dependencies for a repair config.

    Motion, object/environment NPZ files and contact definitions are not loaded
    or validated. MuJoCo necessarily reads its model's own referenced assets.
    The result is diagnostic metadata, never validation of the repair config.
    """
    if not isinstance(config, Mapping):
        raise ValueError("Repair config must be a mapping")
    config = deepcopy(dict(config))
    logical_digest = _digest(config)
    if "schema_version" in config and (type(config["schema_version"]) is not int or config["schema_version"] != 1):
        raise ValueError("Unsupported repair schema_version")
    model_value = _name(config.get("model_xml"), "model_xml")
    coverage = config.get("coverage", {})
    if not isinstance(coverage, Mapping) or set(coverage) - {"omissions"}:
        raise ValueError("coverage accepts only omissions")
    edits = config.get("editable_joints", {})
    if not isinstance(edits, Mapping):
        raise ValueError("editable_joints must be a mapping")
    robot_geoms = _sequence(config.get("robot_geoms", []), "robot_geoms")
    environment = _sequence(config.get("environment", []), "environment")
    # Imported only when the caller explicitly requests native-model preflight.
    import mujoco
    from .contact_edit import pose

    _model_dependencies, _sha = pose._model_dependencies, pose._sha
    code_paths = (Path(__file__).resolve(), Path(pose.__file__).resolve())
    code_before = {str(p): _sha(p) for p in code_paths}

    path = (Path(config_dir) / model_value).resolve(strict=True)
    before = {str(p): _sha(p) for p in _model_dependencies(path)}
    model = mujoco.MjModel.from_xml_path(str(path))
    descriptors = {"object": {"geom": None, "body": None}}
    seen = set()
    for geom in robot_geoms:
        geom = _integer(geom, "robot geom", model.ngeom)
        if geom in seen:
            raise ValueError("Duplicate robot geom")
        seen.add(geom)
        descriptors[f"geom:{geom}"] = dict(geom=geom, body=int(model.geom_bodyid[geom]))
    for item in environment:
        if not isinstance(item, Mapping):
            raise ValueError("Environment entry must be a mapping")
        label = _name(item.get("name"), "environment name")
        if label in descriptors:
            raise ValueError("Duplicate environment mesh label")
        descriptors[label] = dict(geom=None, body=None, static=True)
    result = geometry_coverage(
        model, descriptors, config.get("collision_pairs", []), list(edits),
        omission_reasons=coverage.get("omissions", ()),
    )
    after = {str(p): _sha(p) for p in _model_dependencies(path)}
    if before != after:
        raise ValueError("Native model dependencies changed during coverage preflight")
    code_after = {str(p): _sha(p) for p in code_paths}
    if code_before != code_after:
        raise ValueError("Coverage code changed during preflight")
    expected = config.get("expected_input_sha256", {})
    if not isinstance(expected, Mapping):
        raise ValueError("expected_input_sha256 must be a mapping")
    if "model_xml" in expected and expected["model_xml"] != before[str(path)]:
        raise ValueError("Input hash mismatch: model_xml")
    result["bindings"] = dict(
        config_logical_sha256=logical_digest,
        model_xml=dict(path=str(path), sha256=before[str(path)]),
        model_dependencies=before, model_dependencies_sha256=_digest(before),
        code=code_before, runtime=dict(mujoco_version=str(mujoco.__version__)),
    )
    result["unchecked_fields"] = [
        "trajectory_npz and native trajectories/timing/frozen fields",
        "object_mesh_npz and external environment mesh payloads/transforms",
        "contacts, source observations and contact/temporal acceptance gates",
        "editable correction bounds, solver/runtime/backend settings and pair numeric settings",
        "expected hashes for non-model inputs and full repair configuration validity",
    ]
    return result
