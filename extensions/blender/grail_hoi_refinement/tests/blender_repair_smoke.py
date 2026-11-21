"""Exercise reversible repair UI with synthetic bound native snapshots."""

import json
import sys
from pathlib import Path

import bpy
from mathutils import Quaternion

staging, seed, directory = map(Path, sys.argv[sys.argv.index("--") + 1 :])
sys.path.insert(0, str(staging))
import grail_hoi_refinement as addon

addon.register()
bpy.ops.wm.open_mainfile(filepath=str(seed))
runtime = addon.stage2.runtime()
bpy.context.scene.grail_hoi_stage2.repair_round = str(directory)


def rejected(call, text):
    try:
        call()
    except (ValueError, RuntimeError) as error:
        assert text in str(error), str(error)
    else:
        raise AssertionError("Invalid preview/apply was not rejected")


# A geometrically valid scene with the same plan but different source lineage
# must not preview a proposal from another case.
contract = runtime.contract()
wrong = dict(contract, bindings={"different": "case"})
runtime.put_text(runtime.CONTRACT, json.dumps(wrong))
rejected(lambda: bpy.ops.grail_hoi.stage2_preview_repair(), "different source")
runtime.put_text(runtime.CONTRACT, json.dumps(contract))
assert bpy.ops.grail_hoi.stage2_preview_repair() == {"FINISHED"}
assert "grail_stage2_repair_preview" in bpy.context.scene
control_name = runtime.PREFIX + "Key_w000_j001_k001"
control = bpy.context.scene.objects[control_name]
candidate_quaternion = control.rotation_quaternion.copy()
control.rotation_quaternion = Quaternion((0, 0, 1), 0.24)
rejected(lambda: bpy.ops.grail_hoi.stage2_apply_repair(), "keys changed")
assert bpy.ops.grail_hoi.stage2_restore_repair() == {"FINISHED"}
assert "grail_stage2_repair_preview" not in bpy.context.scene
assert abs(runtime.authored_plan()["windows"][0]["keys"][1]["delta_rotvec"][2] - 0.2) < 1e-6
assert list(directory.glob("restored_preview_*.blend"))
assert bpy.ops.grail_hoi.stage2_preview_repair() == {"FINISHED"}
assert bpy.context.scene.objects[control_name].rotation_quaternion == candidate_quaternion
assert bpy.ops.grail_hoi.stage2_apply_repair() == {"FINISHED"}
output = directory / "APPLIED_UNREVIEWED.blend"
assert output.is_file()
bpy.ops.wm.open_mainfile(filepath=str(output))
assert "grail_stage2_repair_preview" not in bpy.context.scene
record = json.loads(bpy.context.scene["grail_stage2_repair_applied"])
assert record["motion_status"] == "UNREVIEWED" and record["selected"] is False
assert bpy.ops.grail_hoi.stage2_enable() == {"FINISHED"}
print("STAGE2_REPAIR_PREVIEW_RESTORE_APPLY_OK")
addon.unregister()
