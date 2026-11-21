"""Hash-bound free-hand banks to explicit, unreviewed phase seed plans.

No optimization or geometry modification occurs here. A shortlist is a visible
ordering of saved evidence, not an acceptance decision. Callers choose the bank
stage and candidate ID, including a reason for any rejected diagnostic seed.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from .hand_sequence_seed import normalized_roles
from .hand_sequence_path_seed import validate_candidate

HOLD_ROLES = {"hold", "grasp", "stable_hold"}
PATH_ROLES = {"support", "slide", "sliding"}
RANK_COLUMNS = ["coverage_incomplete", "missing_declared_points",
                "sampled_penetration_m", "normalized_source_deviation"]


def load_bound_bank(path, *, sha256, expected_binding, stage, bank_id=None):
    """Require exact bytes and caller-verified original input/model binding."""
    if stage not in {"topo", "pad"}:
        raise ValueError("Bank stage must explicitly be topo or pad")
    path = Path(path).resolve(strict=True)
    raw = path.read_bytes()
    actual = hashlib.sha256(raw).hexdigest()
    if actual != sha256:
        raise ValueError("Candidate bank SHA256 mismatch")
    bank = json.loads(raw)
    if bank.get("schema_version") != 1 or bank.get("binding") != expected_binding:
        raise ValueError("Candidate bank original input binding mismatch")
    if not isinstance(bank.get("candidates"), dict) or not bank["candidates"]:
        raise ValueError("Candidate bank needs explicit hands")
    for side, rows in bank["candidates"].items():
        if side not in {"left", "right"} or not isinstance(rows, list):
            raise ValueError("Invalid candidate bank hand rows")
        ids = [row.get("candidate_id") for row in rows]
        if any(not isinstance(i, str) or not i for i in ids) or len(set(ids)) != len(ids):
            raise ValueError("Candidate IDs must be present and unique per hand")
        if any(type(row.get("rejected")) is not bool for row in rows):
            raise ValueError("Every candidate must retain explicit rejection status")
    return dict(bank_id=bank_id or stage, stage=stage, path=str(path),
                sha256=actual, data=bank, motion_accepted=False)


def _coverage(row):
    patch = row.get("contact_patch", {})
    members = patch.get("members", [])
    known = bool(members) and all(type(m.get("missing_count")) is int
                                 and m["missing_count"] >= 0 for m in members)
    missing = sum(m["missing_count"] for m in members) if known else None
    return dict(coverage_complete=bool(known and patch.get("coverage_complete") is True
                                      and missing == 0),
                missing_count=missing, members=deepcopy(members),
                original_patch=deepcopy(patch), evidence_known=known)


def _source_deviation(row, side, arrays):
    if arrays is None:
        saved = row.get("source_pose_deviation")
        return deepcopy(saved) if isinstance(saved, dict) else None
    f = row["seed_frame"]
    p = np.asarray(row["position"]) - arrays[side + "_wrist_position_object"][f]
    r = np.asarray(row["rotation"]) @ arrays[side + "_wrist_rotation_object"][f].T
    q = np.asarray(row["finger_qpos"]) - arrays[side + "_finger_qpos"][f]
    dp, dr, dq = float(np.linalg.norm(p)), float(Rotation.from_matrix(r).magnitude()), float(np.sqrt(np.mean(q*q)))
    return dict(wrist_translation_m=dp, wrist_rotation_rad=dr, finger_rms_rad=dq,
                normalized_score=dp/.05+dr/.5+dq/.5,
                normalization=dict(translation_m=.05, rotation_rad=.5, finger_rad=.5))


def shortlist_candidates(banks, side, *, limit=8, source_arrays=None):
    """Return transparent lexicographic columns; missing evidence sorts last.

    Saved patch thresholds are displayed unchanged. Scores from differing
    contact-region/coverage policies are not an equal-contract comparison.
    Original screen failures remain visible even though they are not collapsed
    into the proximity ranking. This function never selects a phase seed.
    """
    if type(limit) is not int or limit < 1:
        raise ValueError("Shortlist limit must be a positive integer")
    rows = []
    for bank_id, bank in banks.items():
        if bank_id != bank["bank_id"]:
            raise ValueError("Bank registry identity mismatch")
        for row in bank["data"]["candidates"].get(side, []):
            coverage = _coverage(row)
            signed = row.get("sampled_clearance_min_m")
            penetration = max(0., -signed) if isinstance(signed, (int, float)) and np.isfinite(signed) else None
            source = _source_deviation(row, side, source_arrays)
            deviation = source.get("normalized_score") if source else None
            if deviation is not None and (not np.isfinite(deviation) or deviation < 0):
                raise ValueError("Invalid saved source deviation")
            columns = dict(coverage_incomplete=int(not coverage["coverage_complete"]),
                           missing_declared_points=coverage["missing_count"],
                           sampled_penetration_m=penetration, normalized_source_deviation=deviation)
            rows.append(dict(bank_id=bank_id, stage=bank["stage"],
                candidate_id=row["candidate_id"], source_frame=row["seed_frame"],
                rank_columns=columns, declared_coverage=coverage, source_deviation=source,
                raw_rejected=row["rejected"], rejection_reasons=deepcopy(row.get("rejection_reasons", [])),
                screen=deepcopy(row.get("screen")), inherited_rejection=deepcopy(row.get("warm_start_seed")),
                bank_sha256=bank["sha256"], motion_accepted=False))
    def key(row):
        values = [row["rank_columns"][name] for name in RANK_COLUMNS]
        return tuple(float("inf") if v is None else v for v in values) + (row["bank_id"], row["candidate_id"])
    rows.sort(key=key)
    return dict(ordering=RANK_COLUMNS, candidates=rows[:limit], candidate_count=len(rows),
                selection_performed=False, motion_accepted=False,
                limitations=["Saved sampled distances and first-witness screens do not prove clearance.",
                             "Coverage uses each saved policy unchanged; no threshold is relaxed."])


def _collision_evidence(row, coverage):
    """Count saved ordered screen prefixes; unvisited pairs remain unresolved."""
    screen = row.get("screen", {})
    pairs = coverage.get("pairs", [])
    checked = screen.get("checked_pairs")
    valid = (bool(pairs) and type(checked) is int and 0 <= checked <= len(pairs)
             and screen.get("configured_pairs") == len(pairs))
    unknown = {tuple(sorted(p)) for p in screen.get("unknown_pairs", [])}
    result = {}
    for label, object_pair in [("hand_object", True), ("nonadjacent_self", False)]:
        belongs = lambda p: ("object" in p) == object_pair
        configured = [p for p in pairs if belongs(p)]
        done = [p for p in pairs[:checked] if belongs(p)] if valid else []
        hits = [h for h in screen.get("surface_hits", []) if belongs(h.get("pair", []))]
        unresolved = len(configured)-len(done)+sum(tuple(sorted(p)) in unknown for p in done)
        result[label] = dict(hits=len(hits), unresolved=unresolved if valid else max(100000, unresolved),
                             configured=len(configured), checked=len(done), raw_hits=deepcopy(hits))
    result["complete_declared_screen"] = bool(valid and checked == len(pairs)
        and screen.get("pair_scope_exhausted") is True and not unknown)
    result["containment_and_scene_checked"] = False
    return result


def _choice(banks, side, choice, names, limits):
    bank_id = choice.get("bank_id", choice.get("stage"))
    if bank_id not in banks:
        raise ValueError("Unknown candidate bank identity")
    bank = banks[bank_id]
    if bank_id != bank["bank_id"]:
        raise ValueError("Bank registry identity mismatch")
    if choice.get("stage") != bank["stage"]:
        raise ValueError("Selected candidate stage mismatch")
    matches = [row for row in bank["data"]["candidates"].get(side, [])
               if row["candidate_id"] == choice.get("candidate_id")]
    if len(matches) != 1:
        raise ValueError("Selected candidate ID must address exactly one bound hand row")
    row = matches[0]
    if type(row.get("seed_frame")) is not int or row["seed_frame"] < 0:
        raise ValueError("Candidate source frame must be an original nonnegative integer")
    reason = choice.get("diagnostic_reason")
    if row["rejected"] and (not isinstance(reason, str) or not reason.strip()):
        raise ValueError("Rejected diagnostic seed requires an explicit reason")
    provenance = dict(bank_id=bank_id, stage=bank["stage"], bank_path=bank["path"],
        bank_sha256=bank["sha256"], source_binding=deepcopy(bank["data"]["binding"]),
        candidate_id=row["candidate_id"], raw_strict_rejected=row["rejected"],
        raw_rejection_reasons=deepcopy(row.get("rejection_reasons", [])),
        parent_topo_evidence=deepcopy(row.get("warm_start_seed")),
        raw_screen=deepcopy(row.get("screen")), raw_contact_patch=deepcopy(row.get("contact_patch")),
        sampled_clearance_min_m=row.get("sampled_clearance_min_m"),
        diagnostic_reason=reason, motion_accepted=False, status="UNREVIEWED_DIAGNOSTIC_SEED")
    selected = dict(candidate_id=row["candidate_id"], source_frame=row["seed_frame"],
        wrist_position_object=deepcopy(row["position"]), wrist_rotation_object=deepcopy(row["rotation"]),
        finger_qpos=deepcopy(row["finger_qpos"]), finger_joint_names=deepcopy(row["finger_joints"]),
        provenance=provenance)
    validate_candidate(selected, names, np.asarray(limits, dtype=float))
    return selected, row, bank


def _path_quality(row, bank, side):
    coverage = _coverage(row)
    # The existing DP names this field five_mm_spaced_coverage. A looser old
    # bank must not acquire that meaning just by going through this adapter.
    if not coverage["evidence_known"] or any(
        not isinstance(m.get("max_gap_m"), (int, float)) or not np.isfinite(m["max_gap_m"])
        or not 0 <= m["max_gap_m"] <= .005
        or not isinstance(m.get("min_spacing_m"), (int, float)) or not np.isfinite(m["min_spacing_m"])
        or m["min_spacing_m"] < .003 for m in coverage["members"]):
        raise ValueError("Varying path requires saved coverage at <=5 mm gap and >=3 mm spacing; no relabeling")
    gaps = [a.get("gap_m") for a in row.get("material_anchor_hypotheses", [])]
    if any(not isinstance(g, (int, float)) or not np.isfinite(g) or g < 0 for g in gaps):
        raise ValueError("Invalid saved material anchor gap")
    coverage["available_anchor_mean_gap_m"] = float(np.mean(gaps)) if gaps else None
    quality = dict(collision_evidence=_collision_evidence(row, bank["data"].get("coverage", {}).get(side, {})),
                   five_mm_spaced_coverage=coverage, raw_strict_rejected=row["rejected"])
    if isinstance(row.get("source_pose_deviation"), dict):
        quality["source_pose_deviation"] = deepcopy(row["source_pose_deviation"])
    return quality


def build_seed_plans(banks, sequence_config, phase_selections, joint_limits):
    """Build existing helper inputs. Phase start/end are inclusive source frames.

    Holds name one candidate; support/sliding phases explicitly name key layers
    and their alternatives. The existing path helper performs DP and varying
    interpolation later. This adapter does not synthesize an unprovided key.
    """
    if sequence_config.get("is_static_obj", False) and any(phase_selections.values()):
        raise ValueError("Static objects do not receive contact seed plans")
    holds, paths, records = {}, {}, []
    configured = sequence_config["inputs"]["hands"]
    for side, phases in phase_selections.items():
        if side not in configured or side not in joint_limits:
            raise ValueError("Selection hand must have explicit native bindings and limits")
        names = configured[side]["finger_joints"]
        limits = np.asarray(joint_limits[side], dtype=float)
        if limits.shape != (len(names), 2) or not np.isfinite(limits).all() or np.any(limits[:, 0] > limits[:, 1]):
            raise ValueError("Invalid native finger limits")
        active = np.asarray(sequence_config["contact_mask"][side])
        if active.ndim != 1 or active.dtype.kind != "b":
            raise ValueError("Exact boolean source contact mask required")
        roles = normalized_roles(sequence_config, side, len(active))
        touched = set()
        for phase in phases:
            first, last, role = phase["start"], phase["end"], phase["role"]
            if type(first) is not int or type(last) is not int or not 0 <= first <= last < len(active):
                raise ValueError("Invalid inclusive source phase interval")
            frames = set(range(first, last+1))
            if touched & frames:
                raise ValueError("Selected phases overlap")
            if not active[first:last+1].all():
                raise ValueError("Selection contradicts original active contact mask")
            if role in HOLD_ROLES:
                if any(r not in HOLD_ROLES for r in roles[first:last+1]) or "key_layers" in phase:
                    raise ValueError("Constant seed requires a strict hold, never support or slide")
                selected, _, _ = _choice(banks, side, phase["candidate"], names, limits)
                if not first <= selected["source_frame"] <= last:
                    raise ValueError("Hold candidate source frame outside selected interval")
                selected["start"], selected["end"], selected["frame"] = first, last, selected["source_frame"]
                holds.setdefault(side, []).append(selected)
            elif role in PATH_ROLES:
                canonical = lambda r: "slide" if r == "sliding" else r
                if any(canonical(r) != canonical(role) for r in roles[first:last+1]) or "candidate" in phase:
                    raise ValueError("Varying path contradicts original support/sliding roles")
                layers = []
                for layer in phase.get("key_layers", []):
                    frame = layer["frame"]
                    if type(frame) is not int or not first <= frame <= last or not layer.get("candidates"):
                        raise ValueError("Varying path requires explicit in-phase key candidates")
                    choices = []
                    for choice in layer["candidates"]:
                        selected, row, bank = _choice(banks, side, choice, names, limits)
                        if selected["source_frame"] != frame:
                            raise ValueError("Path candidate source frame differs from key layer")
                        selected["quality"] = _path_quality(row, bank, side)
                        choices.append(selected)
                    layers.append(dict(frame=frame, candidates=choices))
                layer_frames = [layer["frame"] for layer in layers]
                if not layers or layer_frames != sorted(set(layer_frames)):
                    raise ValueError("Varying paths need ordered distinct explicit key layers")
                if (first == 0 and layer_frames[0] != 0) or (last == len(active)-1 and layer_frames[-1] != last):
                    raise ValueError("Boundary phase needs explicit endpoint key; no extrapolation")
                paths.setdefault(side, []).append(dict(start=first, end=last, role=role, key_layers=layers))
            else:
                raise ValueError("Only declared strict holds or varying support/sliding can be seeded")
            touched |= frames
            records.append(dict(side=side, start=first, end=last, role=role, motion_accepted=False))
    return holds, paths, dict(schema_version=1, status="UNREVIEWED_EXPLICIT_PHASE_SEED_PLANS",
        phases=records, motion_accepted=False, source_clock_edited=False, creates_fixed_material_anchors=False,
        ordering="Apply hold seeds first, then varying path DP; preserve original frame zero separately.",
        bank_bindings={key:{k:deepcopy(bank[k]) for k in ("stage", "path", "sha256")} for key,bank in banks.items()})
