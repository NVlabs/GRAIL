"""Export an explicitly selected free-hand sequence to hash-bound SOMA targets.

No grasp is selected here, no anchors are promoted, and no robot solve runs.
The consumer is exercised before publishing the new, unreviewed target bundle.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path
import tempfile

import numpy as np

from .grasp_io import GraspInputs, number
from .hand_sequence import phase_mask, resolve_static_object_policy
from .hand_sequence_seed import normalized_roles
from .soma_robot_hand_targets import load_robot_hand_targets, _CONTACT_PHASES, _OTHER_PHASES


def _sha(value):
    return hashlib.sha256(value).hexdigest()


def _json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def _exact(actual, expected, name):
    actual, expected = np.asarray(actual), np.asarray(expected)
    if (actual.shape != expected.shape or actual.dtype != expected.dtype
            or actual.tobytes() != expected.tobytes()):
        raise ValueError("Canonical source reference bytes changed: " + name)


def export_hand_targets(*, config, sequence, report, selection, output):
    """Validate selected solver evidence and create a new SOMA handoff directory.

    ``selection`` is an explicit caller-authored JSON with schema
    ``hand_sequence_selection.v1``, config/sequence/report SHA256 values,
    nonempty ``reason``, and ``motion_accepted: false``. Selection for downstream
    initialization does not assert collision clearance or whole-motion quality.
    """
    target = Path(output).absolute()
    if target.exists() or target.is_symlink():
        raise FileExistsError("Preserve existing handoff output: " + str(target))
    paths = {key: Path(value).resolve(strict=True) for key, value in
             dict(config=config, sequence=sequence, report=report, selection=selection).items()}
    raw = {key: path.read_bytes() for key, path in paths.items()}
    hashes = {key: _sha(value) for key, value in raw.items()}
    cfg, solver, chosen = [json.loads(raw[key]) for key in ("config", "report", "selection")]
    required = {"schema", "config_sha256", "sequence_sha256", "report_sha256", "reason", "motion_accepted"}
    if (not isinstance(chosen, dict) or set(chosen) != required
            or chosen["schema"] != "hand_sequence_selection.v1"
            or chosen["motion_accepted"] is not False
            or not isinstance(chosen["reason"], str) or not chosen["reason"].strip()):
        raise ValueError("An explicit unaccepted hand_sequence_selection.v1 selection is required")
    for key in ("config", "sequence", "report"):
        if chosen[key + "_sha256"] != hashes[key]:
            raise ValueError("Selected input hash differs: " + key)
    if (solver.get("motion_accepted") is not False or solver.get("input_config") != cfg
            or solver.get("outputs", {}).get(paths["sequence"].name, {}).get("sha256") != hashes["sequence"]):
        raise ValueError("Selected sequence/config is not bound by the unaccepted solver report")
    ctx = GraspInputs(cfg["inputs"], paths["config"].parent)
    if solver.get("binding") != ctx.binding:
        raise ValueError("Solver/input geometry binding differs")
    frames = len(ctx.q)
    fps = number(cfg["fps"], "fps", 1e-12)
    if solver.get("frame_count") != frames or solver.get("fps") != fps:
        raise ValueError("Solver frame count/FPS differs from original inputs")
    times = np.arange(frames, dtype=np.float64) / fps
    if "fps" in ctx.native and float(np.asarray(ctx.native["fps"])) != fps:
        raise ValueError("Native FPS differs from selected configuration")
    if "timestamps" in ctx.native:
        _exact(ctx.native["timestamps"], times, "native timestamps")
    with np.load(io.BytesIO(raw["sequence"]), allow_pickle=False) as archive:
        seq = {key: archive[key].copy() for key in archive.files}
    expected = dict(source_qpos=ctx.q, object_R=ctx.R, object_trans=ctx.t,
                    object_scale=np.asarray(ctx.native["object_scale"]), timestamps=times)
    for key, value in expected.items():
        if key not in seq:
            raise ValueError("Selected sequence lacks original reference: " + key)
        _exact(seq[key], value, key)
    static_policy = resolve_static_object_policy(cfg, ctx.R, ctx.t)
    static = static_policy["is_static_obj"]
    if "static_object_policy" in solver and solver["static_object_policy"] != static_policy:
        raise ValueError("Executed solver static-object policy differs from bound original inputs")
    arrays = {key: seq[key].copy() for key in ("timestamps", "object_R", "object_trans")}
    arrays["object_scale"] = np.broadcast_to(seq["object_scale"], (3,)).copy()
    arrays["source_qpos_reference_only"] = seq["source_qpos"].copy()
    hands, anchor_evidence = {}, {}
    for side, hand in ctx.hands.items():
        roles = normalized_roles(cfg, side, frames)
        source_roles = seq.get(side + "_phase_labels")
        if (source_roles is None or source_roles.shape != (frames,)
                or source_roles.dtype.kind not in "US" or source_roles.tolist() != roles
                or solver.get("hands", {}).get(side, {}).get("phase_labels") != roles):
            raise ValueError("Source phase labels differ: " + side)
        configured = cfg.get("contact_mask", {}).get(side)
        if configured is None:
            mask = phase_mask(cfg.get("contact_frames", {}).get(side, []), frames, static=static)
        else:
            mask = np.asarray(configured)
            if mask.shape != (frames,) or mask.dtype.kind != "b":
                raise ValueError("Contact masks must contain original frame booleans")
            mask = np.zeros(frames, dtype=bool) if static else mask.copy()
        active = seq.get(side + "_contact_active")
        if active is None:
            raise ValueError("Selected sequence lacks source contact mask: " + side)
        _exact(active, mask, side + " contact mask")
        target_roles = ["no_contact"] * frames if static else roles
        if (set(target_roles) - _CONTACT_PHASES - _OTHER_PHASES
                or not np.array_equal(mask, [role in _CONTACT_PHASES for role in target_roles])):
            raise ValueError("Canonical SOMA phase labels and contact mask must agree: " + side)
        for key in ("wrist_position_object", "wrist_rotation_object", "finger_qpos", "finger_joint_names"):
            arrays[side + "_" + key] = seq[side + "_" + key].copy()
        names = arrays[side + "_finger_joint_names"]
        fingers = arrays[side + "_finger_qpos"]
        if names.tolist() != hand["finger_joints"] or fingers.shape != (frames, len(names)):
            raise ValueError("Selected native finger order/shape differs: " + side)
        limits = ctx.model.jnt_range[hand["joints"]]
        if (not np.isfinite(fingers).all() or (fingers < limits[:, 0] - 1e-6).any()
                or (fingers > limits[:, 1] + 1e-6).any()):
            raise ValueError("Selected finger angles exceed original native limits: " + side)
        p, r = [arrays[side + "_wrist_" + key + "_object"] for key in ("position", "rotation")]
        world_p = np.einsum("tij,tj->ti", ctx.R, p) + ctx.t
        world_r = ctx.R @ r
        for key, value in (("wrist_position_world", world_p), ("wrist_rotation_world", world_r)):
            if side + "_" + key in seq and not np.allclose(seq[side + "_" + key], value, atol=2e-6, rtol=0):
                raise ValueError("Selected world/object target inconsistency: " + side + " " + key)
            arrays[side + "_" + key] = value
        hands[side] = dict(wrist_body=hand["wrist_body"], finger_joints=hand["finger_joints"],
                           phase_labels=target_roles, contact_mask=mask.tolist(), source_phase_labels=roles)
        evidence = solver["hands"][side]
        anchor_evidence[side] = {key: evidence[key] for key in
                               ("fixed_anchors", "anchor_rejections", "anchor_selection") if key in evidence}
    provenance = {key: dict(path=str(paths[key]), sha256=hashes[key]) for key in paths}
    binding = dict(ctx.binding, object_scale=arrays["object_scale"].tolist())
    reference_dtypes = {key: dict(native=str(np.asarray(ctx.native[native_key]).dtype),
                                 canonical=str(expected[key].dtype)) for key, native_key in
                        (("source_qpos", "qpos"), ("object_R", "obj_R"),
                         ("object_trans", "obj_t"), ("object_scale", "object_scale"))}
    metadata = dict(schema="unaccepted_hand_sequence_soma_handoff.v1", frames=frames, fps=fps,
                    hand_only=True, motion_accepted=False, whole_body_soma_executed=False,
                    binding=binding, hands=hands, provenance=provenance,
                    canonical_source_arrays_verified_byte_identical=True,
                    source_reference_dtypes=reference_dtypes, current_case_is_static=static,
                    static_object_policy=static_policy,
                    static_contact_policy="Exactly static full-source objects use no contact mask/anchors; collision and source costs remain.",
                    material_anchor_status="Separate solver evidence only; no anchors configured or accepted by this exporter.")
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".hand_handoff_", dir=target.parent) as temporary:
        staged = Path(temporary)
        np.savez_compressed(staged / "hand_targets.npz", **arrays)
        _json(staged / "handoff.json", metadata)
        options = dict(handoff_json=str(staged / "handoff.json"),
                       handoff_sha256=_sha((staged / "handoff.json").read_bytes()),
                       targets_npz=str(staged / "hand_targets.npz"),
                       targets_sha256=_sha((staged / "hand_targets.npz").read_bytes()),
                       expected_object_mesh_sha256=ctx.binding["object_mesh_sha256"],
                       enabled_hands=list(hands), source_hand_weight_multiplier=0.)
        loaded = load_robot_hand_targets(options, fps=fps, frame_count=frames,
            object_positions=ctx.t, object_rotations=ctx.R, object_scale=ctx.native["object_scale"],
            available_joint_names=[ctx.model.joint(i).name for i in range(ctx.model.njnt)],
            available_body_names=[ctx.model.body(i).name for i in range(ctx.model.nbody)])
        options = loaded["config"]
        options.update(handoff_json=str(target / "handoff.json"), targets_npz=str(target / "hand_targets.npz"))
        _json(staged / "robot_hand_targets_config.json", options)
        _json(staged / "anchor_evidence.json", dict(hands=anchor_evidence, motion_accepted=False,
              contacts_configured=False, source_report_sha256=hashes["report"],
              policy="Evidence only. Independently select/revalidate material contacts and supply contact_constraints separately."))
        for key, filename in (("config", "input_config.json"), ("report", "solver_report.json"), ("selection", "selection.json")):
            (staged / filename).write_bytes(raw[key])
        # Fail on changed inputs instead of recording mixed snapshots.
        for key, path in paths.items():
            if _sha(path.read_bytes()) != hashes[key]:
                raise ValueError("Input changed during export: " + key)
        if GraspInputs(cfg["inputs"], paths["config"].parent).binding != ctx.binding:
            raise ValueError("Native/model/object inputs changed during export")
        receipt = dict(schema="hand_sequence_handoff_export.v1", status="UNREVIEWED_SOFT_TARGETS",
            motion_accepted=False, whole_body_soma_executed=False, inputs=provenance,
            output=str(target), frames=frames, fps=fps, static_object=static, static_object_policy=static_policy,
            consumer_validation_passed=True, canonical_source_reference_bytes_unchanged=True,
            source_reference_dtypes=reference_dtypes, original_native_sha256=ctx.binding["native_sha256"],
            fixed_anchors_loaded=False, selection_reason=chosen["reason"],
            code_sha256={name: _sha(Path(__file__).with_name(name).read_bytes()) for name in
                         ("hand_sequence_handoff.py", "soma_robot_hand_targets.py", "grasp_io.py")},
            outputs={path.name: dict(sha256=_sha(path.read_bytes()), bytes=path.stat().st_size)
                     for path in sorted(staged.iterdir())})
        _json(staged / "export_receipt.json", receipt)
        if target.exists() or target.is_symlink():
            raise FileExistsError("Output appeared during export: " + str(target))
        staged.rename(target)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("config", "sequence", "report", "selection", "output"):
        parser.add_argument("--" + name, required=True)
    receipt = export_hand_targets(**vars(parser.parse_args()))
    print(json.dumps(receipt, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
