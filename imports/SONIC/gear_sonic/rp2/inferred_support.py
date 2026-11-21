"""Boundary support inference for object-interaction motions (general, dataset-agnostic; stdlib + numpy).

References the archived InterMimic IsaacGym task (``intermimic_g1_retarget.py``, ``_compute_table_positions``)
and integrates "table support" into the shared training data path. Verified reference semantics: at the first and the last frame of a motion, if the object's lowest point
(rotated, scaled object bounding box) is above the floor (``bottom_z > 0.2 m``) and no hand is in reference contact
with the object at that frame, a fixed slab (0.5 x 0.5 x 0.06 m, centered at the object's x/y, top at the object's
lowest point) is placed under it; the archived runs used contact margins of 10000 frames, i.e. a placed slab persisted
for the whole motion. This is synthetic inferred support, not a reconstruction of real furniture.

Human corrections applied (same day, in order): one GENERAL mechanism for every dataset, no dataset-name branches;
existing authored support geometry has precedence (a motion whose static context already carries a support - e.g. an
authored table - keeps that ONE surface and gets no synthetic slab); near-coincident start/end surfaces (same scene
coordinates: position, top height, orientation, dimensions within noise) are merged into ONE representative surface
(never a bounding-box union, never two colliders); genuinely distinct supports are kept; a merged surface covers the
whole motion so it is never removed mid-use.

Classification per boundary frame (``start`` = frame 0, ``end`` = last frame), kept distinct in the output:
  floor          lowest point <= FLOOR_MAX_BOTTOM_Z: the floor supports it (no slab)
  authored       an authored static-context surface supports the object there (no slab; precedence)
  handheld       object floating and a hand is in reference contact (explicit hand-held; no slab)
  unknown        object floating, hand-contact evidence unavailable (no slab: never give every floating object a board)
  inferred       object floating, no hand contact, no authored support -> slab
Coordinates: everything is in the motion's scene frame (object ``root_pos``/``root_quat`` as stored in the payload,
quaternion WXYZ, local geometry = the pack's ``object_local_aabb_*`` scaled by ``object_scale``); the per-env origin is
added once by the simulator when the static context prim is spawned, exactly as for the existing authored boxes.
Nothing here touches rewards, observations, sampling or the robot; the slab is a kinematic collider the OBJECT rests on.
"""
from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from gear_sonic.rp2.static_context import StaticBox, _quaternion_wxyz_matrix

CONTRACT = "inferred_boundary_support_v1"
SLAB_SIZE_XY_M = 0.5            # archived InterMimic _TABLE_SIZE_XY
SLAB_THICKNESS_M = 0.06         # archived _TABLE_HALF_H * 2
FLOOR_MAX_BOTTOM_Z_M = 0.2      # archived "bottom_z > 0.2" rule
SUPPORT_TOP_TOLERANCE_BELOW_M = 0.03   # object lowest point may sit slightly inside an authored surface (contact offset)
SUPPORT_TOP_TOLERANCE_ABOVE_M = 0.05   # ... or hover slightly above it (data noise)
SUPPORT_XY_MARGIN_M = 0.02              # footprint test margin for "the surface is under the object"
MERGE_XY_TOLERANCE_M = 0.15             # near-coincident surfaces: center distance in the surface plane
MERGE_TOP_TOLERANCE_M = 0.01            # ... and top height
MERGE_ANGLE_TOLERANCE_RAD = math.radians(5.0)
MERGE_SIZE_TOLERANCE_M = 0.02
LEFT_HAND_GROUP = "left_hand"
RIGHT_HAND_GROUP = "right_hand"
ACTIVE_ENTITY_CLASS = "active_entity"
BOUNDARIES = ("start", "end")
INFERRED_PRIM_PREFIX = "InferredSupport"
INFERRED_ATTR = "hoi:inferredSupport"          # custom USD attribute marking a synthetic slab prim


@dataclass(frozen=True)
class SupportDecision:
    boundary: str
    frame: int
    kind: str                      # floor | authored | handheld | unknown | inferred
    bottom_z: float
    object_xy: tuple[float, float]
    hand_contact: bool | None      # None = no evidence
    authored_surface: str | None   # name of the authored box that supports the object (kind == authored)
    slab: StaticBox | None

    def as_dict(self) -> dict[str, Any]:
        out = asdict(self)
        out["slab"] = self.slab.as_dict() if self.slab is not None else None
        return out


@dataclass(frozen=True)
class SupportPlan:
    motion_key: str
    decisions: tuple[SupportDecision, ...]
    slabs: tuple[StaticBox, ...]        # after deduplication; these are authored into the static context
    merged_boundaries: tuple[tuple[str, ...], ...] = field(default_factory=tuple)
    authored_boxes: int = 0
    notes: tuple[str, ...] = field(default_factory=tuple)

    @property
    def needs_slabs(self) -> bool:
        return bool(self.slabs)

    def as_dict(self) -> dict[str, Any]:
        return {
            "contract": CONTRACT,
            "motion_key": self.motion_key,
            "decisions": [d.as_dict() for d in self.decisions],
            "slabs": [s.as_dict() for s in self.slabs],
            "merged_boundaries": [list(m) for m in self.merged_boundaries],
            "authored_boxes": int(self.authored_boxes),
            "notes": list(self.notes),
            "policy": {
                "slab_size_m": [SLAB_SIZE_XY_M, SLAB_SIZE_XY_M, SLAB_THICKNESS_M],
                "floor_max_bottom_z_m": FLOOR_MAX_BOTTOM_Z_M,
                "support_top_tolerance_m": [SUPPORT_TOP_TOLERANCE_BELOW_M, SUPPORT_TOP_TOLERANCE_ABOVE_M],
                "merge_tolerance": {"xy_m": MERGE_XY_TOLERANCE_M, "top_m": MERGE_TOP_TOLERANCE_M,
                                    "angle_rad": MERGE_ANGLE_TOLERANCE_RAD, "size_m": MERGE_SIZE_TOLERANCE_M},
                "lifetime": "whole motion (archived contact margins 10000 frames)",
            },
        }


# ----------------------------------------------------------------------------------------------------------------------
# geometry
# ----------------------------------------------------------------------------------------------------------------------
def object_lowest_z(root_pos: Sequence[float], root_quat_wxyz: Sequence[float], aabb_min: Sequence[float],
                    aabb_max: Sequence[float], scale: Sequence[float] | float = 1.0) -> float:
    """Lowest world z of the object's scaled local bounding box rotated by the object's root rotation (WXYZ)."""
    lo = np.asarray(aabb_min, dtype=np.float64).reshape(3)
    hi = np.asarray(aabb_max, dtype=np.float64).reshape(3)
    s = np.broadcast_to(np.asarray(scale, dtype=np.float64), (3,))
    if not (np.isfinite(lo).all() and np.isfinite(hi).all() and np.isfinite(s).all()) or np.any(hi < lo):
        raise ValueError(f"invalid object bounding box {lo} .. {hi} scale {s}")
    corners = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])]) * s
    rotation = _quaternion_wxyz_matrix(np.asarray(root_quat_wxyz, dtype=np.float64))
    world = corners @ rotation.T + np.asarray(root_pos, dtype=np.float64).reshape(3)
    return float(world[:, 2].min())


def box_top_at_xy(box: StaticBox, xy: Sequence[float], *, margin: float = SUPPORT_XY_MARGIN_M) -> float | None:
    """Top z of ``box`` under the point ``xy`` (scene frame) if the point lies within the box footprint, else None.
    Uses the box's own orientation: the point is expressed in the box frame; the top face is +half-size along the
    box z axis, evaluated for a box whose z axis is (close to) vertical. Tilted surfaces are not supports."""
    rotation = _quaternion_wxyz_matrix(np.asarray(box.quaternion_wxyz, dtype=np.float64))
    up = rotation[:, 2]
    if up[2] < math.cos(math.radians(10.0)):      # surface normal more than 10 deg from vertical: not a resting surface
        return None
    center = np.asarray(box.position, dtype=np.float64)
    half = 0.5 * np.asarray(box.size, dtype=np.float64)
    local_xy = (np.array([xy[0], xy[1], center[2]]) - center) @ rotation
    if abs(local_xy[0]) > half[0] + margin or abs(local_xy[1]) > half[1] + margin:
        return None
    return float(center[2] + half[2] * up[2])


def surface_supports(box: StaticBox, xy: Sequence[float], bottom_z: float) -> bool:
    top = box_top_at_xy(box, xy)
    if top is None:
        return False
    return (top - SUPPORT_TOP_TOLERANCE_BELOW_M) <= bottom_z <= (top + SUPPORT_TOP_TOLERANCE_ABOVE_M)


def make_slab(boundary: str, object_xy: Sequence[float], bottom_z: float) -> StaticBox:
    """The archived reference slab: 0.5 x 0.5 x 0.06, centered under the object, top at the object's lowest point."""
    return StaticBox(
        name=f"{INFERRED_PRIM_PREFIX}_{boundary}",
        position=(float(object_xy[0]), float(object_xy[1]), float(bottom_z - 0.5 * SLAB_THICKNESS_M)),
        quaternion_wxyz=(1.0, 0.0, 0.0, 0.0),
        size=(SLAB_SIZE_XY_M, SLAB_SIZE_XY_M, SLAB_THICKNESS_M),
    )


def _quat_angle(a: Sequence[float], b: Sequence[float]) -> float:
    qa = np.asarray(a, dtype=np.float64); qa /= np.linalg.norm(qa)
    qb = np.asarray(b, dtype=np.float64); qb /= np.linalg.norm(qb)
    return 2.0 * math.acos(min(1.0, abs(float(np.dot(qa, qb)))))


def surfaces_near_coincident(a: StaticBox, b: StaticBox) -> bool:
    """Approximately the same support in the same scene coordinates: position in the surface plane, top height,
    orientation and dimensions all within tolerance. Center distance alone is never sufficient."""
    if _quat_angle(a.quaternion_wxyz, b.quaternion_wxyz) > MERGE_ANGLE_TOLERANCE_RAD:
        return False
    if np.max(np.abs(np.asarray(a.size) - np.asarray(b.size))) > MERGE_SIZE_TOLERANCE_M:
        return False
    top_a = a.position[2] + 0.5 * a.size[2]
    top_b = b.position[2] + 0.5 * b.size[2]
    if abs(top_a - top_b) > MERGE_TOP_TOLERANCE_M:
        return False
    return math.hypot(a.position[0] - b.position[0], a.position[1] - b.position[1]) <= MERGE_XY_TOLERANCE_M


def deduplicate_surfaces(labelled: Sequence[tuple[str, StaticBox]]) -> tuple[list[StaticBox], list[tuple[str, ...]]]:
    """Merge near-coincident surfaces: the FIRST surface of a group is the representative (no union geometry);
    its name carries the merged boundaries. Returns (surfaces, merged boundary groups)."""
    kept: list[tuple[list[str], StaticBox]] = []
    for boundary, box in labelled:
        for group in kept:
            if surfaces_near_coincident(group[1], box):
                group[0].append(boundary)
                break
        else:
            kept.append(([boundary], box))
    out: list[StaticBox] = []
    merged: list[tuple[str, ...]] = []
    for boundaries, box in kept:
        name = f"{INFERRED_PRIM_PREFIX}_{'_'.join(boundaries)}"
        out.append(StaticBox(name=name, position=box.position, quaternion_wxyz=box.quaternion_wxyz, size=box.size))
        if len(boundaries) > 1:
            merged.append(tuple(boundaries))
    return out, merged


# ----------------------------------------------------------------------------------------------------------------------
# evidence from the existing payloads
# ----------------------------------------------------------------------------------------------------------------------
def authored_boxes_from_record(static_context: Mapping[str, Any] | None) -> list[StaticBox]:
    """Authored static-context boxes of a pack record (``canonical_static_boxes_v1``); other kinds -> []."""
    if not isinstance(static_context, Mapping) or static_context.get("kind") != "canonical_static_boxes_v1":
        return []
    boxes: list[StaticBox] = []
    for raw in static_context.get("boxes") or []:
        try:
            boxes.append(StaticBox(
                name=str(raw.get("name", "surface")),
                position=tuple(float(v) for v in raw["position"]),
                quaternion_wxyz=tuple(float(v) for v in raw.get("quaternion_wxyz", (1.0, 0.0, 0.0, 0.0))),
                size=tuple(float(v) for v in raw["size"]),
            ))
        except (KeyError, TypeError, ValueError):
            continue
    return boxes


def authored_boxes_from_usd(path: str | Path) -> list[StaticBox]:
    """Box colliders authored in an existing static-context USD (Cube prims under /StaticContext), via pxr.
    Returns [] when pxr is unavailable or the file has no cube colliders (mesh contexts are not boxes)."""
    try:
        from pxr import Usd, UsdGeom
    except Exception:  # noqa: BLE001
        return []
    try:
        stage = Usd.Stage.Open(str(path))
    except Exception:  # noqa: BLE001
        return []
    if stage is None:
        return []
    boxes: list[StaticBox] = []
    cache = UsdGeom.XformCache()
    for prim in stage.Traverse():
        if prim.GetTypeName() != "Cube":
            continue
        if prim.GetAttribute(INFERRED_ATTR).IsValid() and prim.GetAttribute(INFERRED_ATTR).Get():
            continue   # a previously inferred slab is never "authored" evidence
        matrix = np.asarray(cache.GetLocalToWorldTransform(prim), dtype=np.float64)   # row-vector convention
        m = matrix.T
        size_attr = prim.GetAttribute("size").Get() or 1.0
        rotation = m[:3, :3]
        scale = np.linalg.norm(rotation, axis=0)
        if not np.all(scale > 0):
            continue
        rotation = rotation / scale
        w = math.sqrt(max(0.0, 1.0 + rotation[0, 0] + rotation[1, 1] + rotation[2, 2])) / 2.0
        if w > 1e-6:
            x = (rotation[2, 1] - rotation[1, 2]) / (4 * w); y = (rotation[0, 2] - rotation[2, 0]) / (4 * w); z = (rotation[1, 0] - rotation[0, 1]) / (4 * w)
        else:
            x, y, z = 0.0, 0.0, 1.0
        boxes.append(StaticBox(
            name=prim.GetParent().GetName() if prim.GetName() == "geometry" else prim.GetName(),
            position=tuple(float(v) for v in m[:3, 3]),
            quaternion_wxyz=(float(w), float(x), float(y), float(z)),
            size=tuple(float(v) for v in scale * float(size_attr)),
        ))
    return boxes


def local_aabb_from_usd(path: str | Path) -> tuple[list[float], list[float]] | None:
    """Object-local axis-aligned bounds of an asset USD (default prim, default purpose) via pxr; None if unavailable.
    Used for packs whose index lacks ``object_local_aabb_*`` (the geometry itself is the same the simulator spawns)."""
    try:
        import os
        os.environ.setdefault("PXR_WORK_THREAD_LIMIT", "1")   # login nodes may refuse USD's worker threads (pthread_create)
        from pxr import Usd, UsdGeom
        stage = Usd.Stage.Open(str(path))
        if stage is None:
            return None
        prim = stage.GetDefaultPrim() or stage.GetPseudoRoot()
        cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render])
        box = cache.ComputeWorldBound(prim).ComputeAlignedRange()
        lo, hi = box.GetMin(), box.GetMax()
        if box.IsEmpty():
            return None
        return [float(lo[0]), float(lo[1]), float(lo[2])], [float(hi[0]), float(hi[1]), float(hi[2])]
    except Exception:  # noqa: BLE001
        return None


def record_local_aabb(record: Mapping[str, Any], asset_usd: str | Path | None = None) -> tuple[list[float], list[float]] | None:
    """``object_local_aabb_min/max`` from the pack index when present, else measured from the asset USD."""
    lo, hi = record.get("object_local_aabb_min"), record.get("object_local_aabb_max")
    if lo is not None and hi is not None:
        return [float(v) for v in lo], [float(v) for v in hi]
    return local_aabb_from_usd(asset_usd) if asset_usd else None


HAND_EVIDENCE_ARRAY_FIELDS = (
    "object_contact_mask_{}", "object_in_contact_{}", "source_hand_contact_window_{}", "source_hand_contact_label_{}",
)


def hand_contact_evidence(objects_payload: Mapping[str, Any], frame: int) -> bool | None:
    """True/False when the payload tells whether a hand touches the active object at ``frame``; None when unknown.
    Primary: canonical ``reference_contact_state[frame, active_entity, {left,right}_hand]`` (1 contact, 0 none,
    -1 unknown). Fallbacks when unknown: the per-frame ``contact_points_{left,right}_hand`` dictionaries (a frame
    with >= 1 point = contact; dictionaries present but no entry for the frame = no contact), then the per-frame source
    hand-contact arrays listed in ``HAND_EVIDENCE_ARRAY_FIELDS``."""
    state = objects_payload.get("reference_contact_state")
    classes = list(objects_payload.get("reference_contact_entity_classes") or [])
    groups = list(objects_payload.get("reference_contact_body_groups") or [])
    decided: bool | None = None
    if state is not None and ACTIVE_ENTITY_CLASS in classes and LEFT_HAND_GROUP in groups and RIGHT_HAND_GROUP in groups:
        arr = np.asarray(state)
        if arr.ndim == 3 and 0 <= frame < arr.shape[0]:
            row = arr[frame, classes.index(ACTIVE_ENTITY_CLASS)]
            vals = [int(row[groups.index(LEFT_HAND_GROUP)]), int(row[groups.index(RIGHT_HAND_GROUP)])]
            if any(v == 1 for v in vals):
                return True
            if all(v == 0 for v in vals):
                return False
            decided = None   # -1 somewhere: fall through to the point evidence
    left = objects_payload.get("contact_points_left_hand")
    right = objects_payload.get("contact_points_right_hand")
    if isinstance(left, Mapping) or isinstance(right, Mapping):
        for table in (left, right):
            if isinstance(table, Mapping):
                pts = table.get(frame, table.get(str(frame)))
                if pts is not None and len(np.asarray(pts).reshape(-1, 3)) > 0:
                    return True
        return False
    # Per-frame source hand-contact arrays of the other packs (FORM-HOI / RP2 / OmniContact object_contact_mask_*,
    # OmniContact/InterAct object_in_contact_*, source_hand_contact_window_*, and the ternary
    # source_hand_contact_label_*): > 0.5 = contact, 0 = no contact, negative = unknown. Without these, packs whose
    # canonical hand groups are -1 at the clip boundaries (FORM-HOI) were classified "unknown" and never got a slab.
    for template in HAND_EVIDENCE_ARRAY_FIELDS:
        vals = []
        for side in ("left", "right"):
            arr = objects_payload.get(template.format(side))
            if arr is None:
                continue
            arr = np.asarray(arr)
            if arr.ndim >= 1 and 0 <= frame < arr.shape[0]:
                vals.append(float(np.asarray(arr[frame]).reshape(-1)[0]))
        if vals:
            if any(v > 0.5 for v in vals):
                return True
            if all(v == 0 for v in vals):
                return False
    return decided


def _first_object(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    """``objects_with_masks`` files are ``{motion_key: {...}}``; accept either level."""
    if "root_pos" in payload:
        return payload
    if len(payload) == 1:
        inner = next(iter(payload.values()))
        if isinstance(inner, Mapping) and "root_pos" in inner:
            return inner
    raise KeyError("objects payload without root_pos")


# ----------------------------------------------------------------------------------------------------------------------
# the plan
# ----------------------------------------------------------------------------------------------------------------------
def plan_boundary_support(
    motion_key: str,
    objects_payload: Mapping[str, Any],
    aabb_min: Sequence[float],
    aabb_max: Sequence[float],
    *,
    authored_boxes: Sequence[StaticBox] = (),
    object_index: int = 0,
    start_motion_gate: bool = False,
    motion_window_frames: int = 10,
    motion_threshold_m: float = 0.01,
) -> SupportPlan:
    """Classify the start and end frames and return the deduplicated slabs to author (possibly none).

    ``start_motion_gate`` is the archived optional rule (default off there and here): a hand-held object at the start
    that does not move more than ``motion_threshold_m`` within ``motion_window_frames`` still receives a slab."""
    obj = _first_object(objects_payload)
    root_pos = np.asarray(obj["root_pos"], dtype=np.float64)
    root_quat = np.asarray(obj["root_quat"], dtype=np.float64)
    if root_pos.ndim == 3:
        root_pos = root_pos[:, object_index]
        root_quat = root_quat[:, object_index]
    frames = int(root_pos.shape[0])
    if frames == 0:
        raise ValueError(f"{motion_key}: empty object trajectory")
    scale = np.asarray(obj.get("object_scale", 1.0), dtype=np.float64)
    if scale.ndim == 2:
        scale = scale[object_index]
    decisions: list[SupportDecision] = []
    labelled: list[tuple[str, StaticBox]] = []
    notes: list[str] = []
    for boundary, frame in (("start", 0), ("end", frames - 1)):
        pos = root_pos[frame]
        bottom_z = object_lowest_z(pos, root_quat[frame], aabb_min, aabb_max, scale)
        xy = (float(pos[0]), float(pos[1]))
        contact = hand_contact_evidence(obj, frame)
        supporter = next((b.name for b in authored_boxes if surface_supports(b, xy, bottom_z)), None)
        slab = None
        if bottom_z <= FLOOR_MAX_BOTTOM_Z_M:
            kind = "floor"
        elif supporter is not None:
            kind = "authored"
        elif contact is None:
            kind = "unknown"
        elif contact:
            kind = "handheld"
            if boundary == "start" and start_motion_gate and frames > 1:
                probe = min(motion_window_frames, frames - 1)
                if float(np.linalg.norm(root_pos[probe] - root_pos[0])) < motion_threshold_m:
                    kind = "inferred"
                    notes.append("start slab restored by the motion gate (hand contact but no motion)")
        else:
            kind = "inferred"
        if kind == "inferred":
            slab = make_slab(boundary, xy, bottom_z)
            labelled.append((boundary, slab))
        decisions.append(SupportDecision(boundary=boundary, frame=frame, kind=kind, bottom_z=float(bottom_z),
                                         object_xy=xy, hand_contact=contact, authored_surface=supporter, slab=slab))
    # authored surfaces also take precedence over an inferred slab that would duplicate them (same support)
    filtered: list[tuple[str, StaticBox]] = []
    for boundary, slab in labelled:
        twin = next((b for b in authored_boxes if surfaces_near_coincident(b, slab)), None)
        if twin is not None:
            notes.append(f"{boundary}: slab coincides with authored surface {twin.name}; authored surface kept")
            continue
        filtered.append((boundary, slab))
    slabs, merged = deduplicate_surfaces(filtered)
    return SupportPlan(motion_key=motion_key, decisions=tuple(decisions), slabs=tuple(slabs),
                       merged_boundaries=tuple(merged), authored_boxes=len(authored_boxes), notes=tuple(notes))


# ----------------------------------------------------------------------------------------------------------------------
# USD authoring (pxr)
# ----------------------------------------------------------------------------------------------------------------------
def write_support_context_usd(path: str | Path, slabs: Sequence[StaticBox], *, base_usd: str | Path | None = None) -> None:
    """Author ``path``: the existing static context (``base_usd``, referenced, never copied or modified) plus the
    inferred slabs as kinematic box colliders under the same ``/StaticContext`` root. Each slab prim carries
    ``hoi:inferredSupport = true`` so the runtime can exclude the robot from colliding with it."""
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    stage = Usd.Stage.CreateNew(str(path))
    root = UsdGeom.Xform.Define(stage, "/StaticContext").GetPrim()
    stage.SetDefaultPrim(root)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    if base_usd is not None:
        base = Path(base_usd).resolve()
        root.GetReferences().AddReference(str(base).replace("@", "%40"))
    rigid = UsdPhysics.RigidBodyAPI.Apply(root)
    rigid.CreateRigidBodyEnabledAttr(True)
    rigid.CreateKinematicEnabledAttr(True)
    UsdPhysics.MassAPI.Apply(root).CreateMassAttr(1.0)
    for index, box in enumerate(slabs):
        safe = "".join(c if c.isalnum() or c == "_" else "_" for c in box.name)
        body = UsdGeom.Xform.Define(stage, f"/StaticContext/{safe}").GetPrim()
        xform = UsdGeom.Xformable(body)
        xform.AddTranslateOp().Set(Gf.Vec3d(*box.position))
        q = box.quaternion_wxyz
        xform.AddOrientOp().Set(Gf.Quatf(q[0], Gf.Vec3f(*q[1:])))
        cube = UsdGeom.Cube.Define(stage, f"/StaticContext/{safe}/geometry")
        cube.CreateSizeAttr(1.0)
        cube.AddScaleOp().Set(Gf.Vec3f(*box.size))
        cube.CreateDisplayColorAttr([(0.55, 0.45, 0.25)])
        UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
        for prim in (body, cube.GetPrim()):
            prim.CreateAttribute(INFERRED_ATTR, Sdf.ValueTypeNames.Bool, custom=True).Set(True)
        body.CreateAttribute("hoi:supportIndex", Sdf.ValueTypeNames.Int, custom=True).Set(int(index))
    stage.GetRootLayer().Save()


def write_plan_json(path: str | Path, plan: SupportPlan, *, extra: Mapping[str, Any] | None = None) -> None:
    payload = plan.as_dict()
    if extra:
        payload.update(dict(extra))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=1, sort_keys=True))
    tmp.replace(path)
