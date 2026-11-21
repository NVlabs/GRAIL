"""Small kinematic support block that replaces the reconstructed table during tracking.

The block is the object's frame-0 world-space AABB footprint (scaled, rotated and translated
mesh) plus 3 cm on each XY side, 8 cm thick, with its top at the object's minimum world Z.
It replaces only the active static-context USD of a table motion; the native motion, object,
BPS and physics bytes are unchanged. Floor motions keep their native support context.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

MARGIN_M = 0.03
THICKNESS_M = 0.08


def _require(value, message):
    if not value:
        raise ValueError(message)


def block_geometry(native_path, mesh_path) -> dict:
    with np.load(native_path, allow_pickle=False) as native, np.load(mesh_path, allow_pickle=False) as mesh:
        scale = np.asarray(native["object_scale"], dtype=np.float64)
        rotation = np.asarray(native["obj_R"][0], dtype=np.float64)
        translation = np.asarray(native["obj_t"][0], dtype=np.float64)
        vertices = np.asarray(mesh["vertices"], dtype=np.float64)
    _require(scale.shape == (3,) and np.all(scale > 0), "Invalid native scale")
    _require(rotation.shape == (3, 3) and translation.shape == (3,), "Invalid frame0 pose")
    _require(vertices.ndim == 2 and vertices.shape[1] == 3 and len(vertices) >= 4, "Invalid mesh")
    _require(all(np.isfinite(x).all() for x in (scale, rotation, translation, vertices)), "Nonfinite geometry")
    _require(np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5) and np.linalg.det(rotation) > .999,
             "Invalid frame0 rotation")
    world = (vertices * scale) @ rotation.T + translation
    low, high = world.min(0), world.max(0)
    block_low = np.r_[low[:2] - MARGIN_M, low[2] - THICKNESS_M]
    block_high = np.r_[high[:2] + MARGIN_M, low[2]]
    return {"recipe": "native_frame0_scaled_mesh_world_AABB_margin_0.03_each_XY_side_thickness_0.08_top_minZ",
            "object_world_aabb": [low.tolist(), high.tolist()],
            "block_world_aabb": [block_low.tolist(), block_high.tolist()],
            "center": ((block_low + block_high) / 2).tolist(), "size": (block_high - block_low).tolist(),
            "quaternion_wxyz": [1, 0, 0, 0], "margin_each_side_m": MARGIN_M, "thickness_m": THICKNESS_M,
            "minimum_footprint_floor_applied": False, "object_scale_applied_once": scale.tolist(),
            "object_frame0_rotation": rotation.tolist(), "object_frame0_translation": translation.tolist(),
            "frame0_support_gap_m": 0., "grounding_or_motion_shift_applied": False}


def write_block_usd(path, geometry) -> None:
    """Author the block with the vendored SONIC static-context writer (one kinematic cube)."""
    from gear_sonic.rp2.static_context import StaticBox, write_static_context_usd
    from pxr import Usd, UsdGeom
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    write_static_context_usd(Path(path), (StaticBox(name="table", position=tuple(geometry["center"]),
                                                    quaternion_wxyz=(1., 0., 0., 0.), size=tuple(geometry["size"])),))
    # Same stage metadata as the converted pack USDs: meters, Z up.
    stage = Usd.Stage.Open(str(path))
    UsdGeom.SetStageMetersPerUnit(stage, 1.)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    stage.GetRootLayer().Save()


def validate_context(path, geometry) -> dict:
    from pxr import Usd, UsdGeom, UsdPhysics, UsdUtils
    layers, assets, unresolved = UsdUtils.ComputeAllDependencies(str(path))
    _require(len(layers) == 1 and not assets and not unresolved, "Context has external dependencies")
    stage = Usd.Stage.Open(str(path))
    _require(stage and UsdGeom.GetStageMetersPerUnit(stage) == 1 and UsdGeom.GetStageUpAxis(stage) == UsdGeom.Tokens.z,
             "Context must be meters/Z-up")
    prims = list(stage.Traverse())
    cubes = [p for p in prims if p.IsA(UsdGeom.Cube)]
    meshes = [p for p in prims if p.IsA(UsdGeom.Mesh)]
    bodies = [p for p in prims if p.HasAPI(UsdPhysics.RigidBodyAPI)]
    colliders = [p for p in prims if p.HasAPI(UsdPhysics.CollisionAPI)]
    _require(len(cubes) == len(bodies) == len(colliders) == 1 and not meshes and cubes == colliders,
             "Require exactly one cube collider, one body, zero table meshes")
    body = UsdPhysics.RigidBodyAPI(bodies[0])
    _require(str(bodies[0].GetPath()) == "/StaticContext" and body.GetKinematicEnabledAttr().Get() is True
             and body.GetRigidBodyEnabledAttr().Get() is True, "Context body must be enabled/kinematic")
    corners = np.asarray([[x, y, z] for x in (-.5, .5) for y in (-.5, .5) for z in (-.5, .5)])
    cube = cubes[0]
    matrix = np.asarray(UsdGeom.XformCache().GetLocalToWorldTransform(cube), dtype=float).T
    points = corners * UsdGeom.Cube(cube).GetSizeAttr().Get()
    world = points @ matrix[:3, :3].T + matrix[:3, 3]
    actual = np.asarray([world.min(0), world.max(0)])
    wanted = np.asarray(geometry["block_world_aabb"])
    _require(np.allclose(actual, wanted, rtol=0, atol=1e-6), "Authored USD differs from the block recipe")
    return {"status": "passed", "cube_count": 1, "kinematic_body_count": 1, "collider_count": 1,
            "active_table_mesh_count": 0, "external_layer_or_asset_count": 0,
            "actual_world_aabb": actual.tolist(), "max_float_writer_error_m": float(np.abs(actual - wanted).max())}
