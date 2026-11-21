"""Disposable Blender process: actual addon controls, stale guard and reopen."""

import sys
from pathlib import Path

import bpy
import numpy as np
from bpy_extras.object_utils import world_to_camera_view
from mathutils import Quaternion, Vector

staging, checkpoint, edited = map(Path, sys.argv[sys.argv.index("--") + 1 :])
sys.path.insert(0, str(staging))
import grail_hoi_refinement as addon

addon.register()
bpy.ops.wm.open_mainfile(filepath=str(checkpoint))
runtime = addon.stage2.runtime()
arrays = runtime.unpack("scene")
point = np.array([0.2, -0.1, 0.3, 1])
cv = np.linalg.inv(arrays["camera_matrix_world"]) @ point
pixel = arrays["camera_intrinsics"] @ cv[:3]
pixel = pixel[:2] / pixel[2]
projected = world_to_camera_view(
    bpy.context.scene, bpy.context.scene.objects[runtime.PREFIX + "SourceCamera"], Vector(point[:3])
)
actual = np.array([projected.x * arrays["image_width"], (1 - projected.y) * arrays["image_height"]])
assert np.max(np.abs(actual - pixel)) < 1e-4, (actual, pixel)
oblique = bpy.context.scene.objects[runtime.PREFIX + "ObliqueCamera"]
orientation = np.asarray(oblique.matrix_world)[:3, :3]
source_up = -arrays["camera_matrix_world"][:3, 1]
projected_up = source_up - np.dot(source_up, orientation[:, 2]) * orientation[:, 2]
projected_up /= np.linalg.norm(projected_up)
assert np.allclose(orientation[:, 1], projected_up, atol=1e-6)
assert abs(np.dot(orientation[:, 0], source_up)) < 1e-6
human = bpy.context.scene.objects[runtime.PREFIX + "AfterHuman"]
assert human.hide_viewport, "Saved/reopened snapshots must start disabled"
assert bpy.ops.grail_hoi.stage2_enable() == {"FINISHED"}
assert not human.hide_viewport
runtime.disable_snapshots()
assert human.hide_viewport and human.hide_render
assert bpy.ops.grail_hoi.stage2_enable() == {"FINISHED"}
assert not human.hide_viewport
camera = bpy.context.scene.objects[runtime.PREFIX + "SourceCamera"]
camera_matrix = camera.matrix_world.copy()
camera.location.x += 0.1
bpy.context.view_layer.update()
runtime.snapshot_update(bpy.context.scene)
assert human.hide_viewport and "camera" in bpy.context.scene["grail_stage2_snapshot_status"]
camera.matrix_world = camera_matrix
bpy.context.view_layer.update()
runtime.snapshot_update(bpy.context.scene)
assert not human.hide_viewport
object_mesh = bpy.context.scene.objects[runtime.PREFIX + "Object"].data
original_vertex = object_mesh.vertices[0].co.copy()
object_mesh.vertices[0].co.x += 0.1
runtime.snapshot_update(bpy.context.scene)
assert human.hide_viewport
assert "object mesh changed" in bpy.context.scene["grail_stage2_snapshot_status"]
object_mesh.vertices[0].co = original_vertex
runtime.snapshot_update(bpy.context.scene)
assert not human.hide_viewport
bpy.context.scene.frame_set(2)  # Source frame 1 is absent from cache.
assert human.hide_viewport
assert "uncached" in bpy.context.scene["grail_stage2_snapshot_status"]
bpy.context.scene.frame_set(3)
assert not human.hide_viewport
control = bpy.context.scene.objects[runtime.PREFIX + "Key_w000_j001_k001"]
control.rotation_quaternion = Quaternion((0, 0, 1), 0.24)
bpy.context.view_layer.update()
runtime.snapshot_update(bpy.context.scene)
assert human.hide_viewport
assert "keys changed" in bpy.context.scene["grail_stage2_snapshot_status"]
assert abs(runtime.authored_plan()["windows"][0]["keys"][1]["delta_rotvec"][2] - 0.24) < 1e-6
bpy.context.scene.grail_hoi_stage2.plan_export = str(edited.with_suffix(".json"))
assert bpy.ops.grail_hoi.stage2_export_plan() == {"FINISHED"}
bpy.ops.wm.save_as_mainfile(filepath=str(edited), check_existing=False)
bpy.ops.wm.open_mainfile(filepath=str(edited))
assert bpy.ops.grail_hoi.stage2_enable() == {"FINISHED"}
assert bpy.context.scene.objects[runtime.PREFIX + "AfterHuman"].hide_viewport
endpoint = bpy.context.scene.objects[runtime.PREFIX + "Key_w000_j001_k000"]
endpoint.rotation_quaternion = Quaternion((0, 0, 1), 0.1)
try:
    runtime.authored_plan()
except ValueError as error:
    assert "Protected" in str(error)
else:
    raise AssertionError("Protected endpoint edit was accepted")
addon.unregister()
print("STAGE2_ADDON_SMOKE_OK")
