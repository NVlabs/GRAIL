"""Canonical static-context geometry shared by data and simulator adapters."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from gear_sonic.rp2.usd_physics import require_resolved_usd_stage


@dataclass(frozen=True)
class StaticBox:
    """One world-space oriented box used as kinematic scene context."""

    name: str
    position: tuple[float, float, float]
    quaternion_wxyz: tuple[float, float, float, float]
    size: tuple[float, float, float]

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "position": list(self.position),
            "quaternion_wxyz": list(self.quaternion_wxyz),
            "size": list(self.size),
        }


def table_static_box(metadata: Mapping[str, Any]) -> StaticBox | None:
    """Resolve the canonical table box declared by a source record."""
    if metadata.get("table_needed") is False:
        return None
    position = np.asarray(metadata.get("table_pos", []), dtype=np.float64).reshape(-1)
    quaternion = np.asarray(
        metadata.get("table_quat", [1.0, 0.0, 0.0, 0.0]),
        dtype=np.float64,
    ).reshape(-1)
    size = np.asarray(metadata.get("table_size", []), dtype=np.float64).reshape(-1)
    if (
        position.shape != (3,)
        or quaternion.shape != (4,)
        or size.shape != (3,)
        or not np.isfinite(position).all()
        or not np.isfinite(quaternion).all()
        or not np.isfinite(size).all()
        or np.linalg.norm(quaternion) <= 1.0e-8
        or np.any(size <= 0.0)
        or position[2] <= -5.0
    ):
        return None
    quaternion = quaternion / np.linalg.norm(quaternion)
    return StaticBox(
        name="table",
        position=tuple(float(value) for value in position),
        quaternion_wxyz=tuple(float(value) for value in quaternion),
        size=tuple(float(value) for value in size),
    )


def _quaternion_wxyz_matrix(quaternion: np.ndarray) -> np.ndarray:
    quaternion = np.asarray(quaternion, dtype=np.float64)
    quaternion = quaternion / np.linalg.norm(quaternion)
    w, x, y, z = quaternion
    return np.asarray(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def points_to_static_box_contact_distance(
    points_world: np.ndarray,
    box: StaticBox,
) -> np.ndarray:
    """Return exact outside distance to an OBB, with penetration counted as contact."""
    points = np.asarray(points_world, dtype=np.float64)
    if points.shape[-1] != 3 or not np.isfinite(points).all():
        raise ValueError(f"points_world must be finite [..., 3], got {points.shape}")
    center = np.asarray(box.position, dtype=np.float64)
    rotation = _quaternion_wxyz_matrix(
        np.asarray(box.quaternion_wxyz, dtype=np.float64)
    )
    local = (points - center) @ rotation
    outside = np.maximum(
        np.abs(local) - 0.5 * np.asarray(box.size, dtype=np.float64),
        0.0,
    )
    return np.linalg.norm(outside, axis=-1).astype(np.float32)


def write_static_context_usd(
    path: Path,
    boxes: tuple[StaticBox, ...],
) -> None:
    """Author one canonical kinematic body containing all context colliders."""
    from pxr import Gf, Usd, UsdGeom, UsdPhysics

    path.parent.mkdir(parents=True, exist_ok=True)
    stage = Usd.Stage.CreateNew(str(path))
    root = UsdGeom.Xform.Define(stage, "/StaticContext").GetPrim()
    stage.SetDefaultPrim(root)
    rigid = UsdPhysics.RigidBodyAPI.Apply(root)
    rigid.CreateRigidBodyEnabledAttr(True)
    rigid.CreateKinematicEnabledAttr(True)
    UsdPhysics.MassAPI.Apply(root).CreateMassAttr(1.0)
    for index, box in enumerate(boxes):
        safe_name = "".join(
            character if character.isalnum() or character == "_" else "_"
            for character in box.name
        )
        prim_path = (
            "/StaticContext/Table"
            if box.name == "table"
            else f"/StaticContext/{index:02d}_{safe_name}"
        )
        body = UsdGeom.Xform.Define(stage, prim_path).GetPrim()
        xform = UsdGeom.Xformable(body)
        xform.AddTranslateOp().Set(Gf.Vec3d(*box.position))
        quaternion = box.quaternion_wxyz
        xform.AddOrientOp().Set(
            Gf.Quatf(
                quaternion[0],
                Gf.Vec3f(*quaternion[1:]),
            )
        )
        cube = UsdGeom.Cube.Define(stage, f"{prim_path}/geometry")
        cube.CreateSizeAttr(1.0)
        cube.AddScaleOp().Set(Gf.Vec3f(*box.size))
        cube.CreateDisplayColorAttr([(0.5, 0.35, 0.2)])
        UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
    stage.GetRootLayer().Save()


def enforce_static_context_single_body(path: str | Path) -> dict[str, Any]:
    """Collapse composed context colliders under one kinematic root body."""
    from pxr import Usd, UsdPhysics

    usd_path = Path(path).expanduser().resolve(strict=True)
    stage = Usd.Stage.Open(str(usd_path))
    if stage is None:
        raise ValueError(f"{usd_path}: USD stage did not open")
    require_resolved_usd_stage(stage, usd_path)
    root = stage.GetPrimAtPath("/StaticContext")
    if not root.IsValid():
        raise ValueError(f"{usd_path}: missing /StaticContext")

    before = static_context_rigid_body_states(usd_path)
    stage.SetEditTarget(stage.GetRootLayer())
    root_api = UsdPhysics.RigidBodyAPI.Apply(root)
    root_api.CreateRigidBodyEnabledAttr(True).Set(True)
    root_api.CreateKinematicEnabledAttr(True).Set(True)
    collapsed = []
    for prim in list(stage.Traverse()):
        if prim.GetPath() == root.GetPath():
            continue
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            prim.RemoveAPI(UsdPhysics.RigidBodyAPI)
            collapsed.append(prim.GetPath().pathString)
    stage.GetRootLayer().Save()

    after = static_context_rigid_body_states(usd_path)
    expected = [
        {
            "prim": "/StaticContext",
            "enabled": True,
            "kinematic": True,
        }
    ]
    if after != expected:
        raise ValueError(
            f"{usd_path}: canonical static context must have exactly one "
            f"kinematic root body, found {after}"
        )
    return {
        "path": str(usd_path),
        "source_rigid_body_count": len(before),
        "collapsed_rigid_bodies": collapsed,
        "rigid_body_count": 1,
        "remaining_dynamic_rigid_bodies": [],
    }


def static_context_rigid_body_states(
    path: str | Path,
) -> list[dict[str, Any]]:
    """Inspect composed rigid bodies in one normalized static-context USD."""
    from pxr import Usd, UsdPhysics

    usd_path = Path(path).expanduser().resolve(strict=True)
    stage = Usd.Stage.Open(str(usd_path))
    if stage is None:
        raise ValueError(f"{usd_path}: USD stage did not open")
    require_resolved_usd_stage(stage, usd_path)
    root = stage.GetPrimAtPath("/StaticContext")
    if not root.IsValid():
        raise ValueError(f"{usd_path}: missing /StaticContext")

    states = []
    for prim in stage.Traverse():
        if not prim.HasAPI(UsdPhysics.RigidBodyAPI):
            continue
        api = UsdPhysics.RigidBodyAPI(prim)
        states.append(
            {
                "prim": prim.GetPath().pathString,
                "enabled": api.GetRigidBodyEnabledAttr().Get() is not False,
                "kinematic": api.GetKinematicEnabledAttr().Get() is True,
            }
        )
    return states


def enforce_static_context_kinematics(path: str | Path) -> dict[str, Any]:
    """Author root-layer kinematic opinions for every composed rigid body."""
    from pxr import Usd, UsdPhysics

    usd_path = Path(path).expanduser().resolve(strict=True)
    stage = Usd.Stage.Open(str(usd_path))
    if stage is None:
        raise ValueError(f"{usd_path}: USD stage did not open")
    require_resolved_usd_stage(stage, usd_path)
    root = stage.GetPrimAtPath("/StaticContext")
    if not root.IsValid():
        raise ValueError(f"{usd_path}: missing /StaticContext")

    stage.SetEditTarget(stage.GetRootLayer())
    patched = []
    for prim in list(stage.Traverse()):
        if not prim.HasAPI(UsdPhysics.RigidBodyAPI):
            continue
        api = UsdPhysics.RigidBodyAPI(prim)
        enabled = api.GetRigidBodyEnabledAttr().Get() is not False
        was_kinematic = api.GetKinematicEnabledAttr().Get() is True
        if enabled and not was_kinematic:
            api.CreateKinematicEnabledAttr().Set(True)
            patched.append(prim.GetPath().pathString)
    stage.GetRootLayer().Save()

    states = static_context_rigid_body_states(usd_path)
    remaining_dynamic = [
        state["prim"]
        for state in states
        if state["enabled"] and not state["kinematic"]
    ]
    if remaining_dynamic:
        raise ValueError(
            f"{usd_path}: static context still has dynamic rigid bodies "
            f"{remaining_dynamic[:8]}"
        )
    return {
        "path": str(usd_path),
        "rigid_body_count": len(states),
        "patched_rigid_bodies": patched,
        "remaining_dynamic_rigid_bodies": remaining_dynamic,
    }


def static_context_record(boxes: tuple[StaticBox, ...]) -> dict[str, Any]:
    """Serialize the normalized static-context contract into pack metadata."""
    return {
        "kind": "canonical_static_boxes_v1",
        "boxes": [box.as_dict() for box in boxes],
        "has_table": any(box.name == "table" for box in boxes),
        "table_position": (
            list(next(box.position for box in boxes if box.name == "table"))
            if any(box.name == "table" for box in boxes)
            else None
        ),
        "table_quaternion_wxyz": (
            list(
                next(
                    box.quaternion_wxyz for box in boxes if box.name == "table"
                )
            )
            if any(box.name == "table" for box in boxes)
            else None
        ),
        "table_size": (
            list(next(box.size for box in boxes if box.name == "table"))
            if any(box.name == "table" for box in boxes)
            else None
        ),
    }
