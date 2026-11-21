"""Thin dual-model Stage 2 UI: snapshot authoring plus external native refresh."""

import importlib.util
import json
import os
import subprocess
import sys
import uuid
from pathlib import Path

import bpy
from bpy.app.handlers import persistent
from bpy.props import PointerProperty, StringProperty

_RUNTIME = None
_PREVIEW = None


def preview_tools():
    global _PREVIEW
    if _PREVIEW is None:
        source = Path(__file__).with_name("stage2_preview.py")
        if not source.is_file():
            source = (
                Path(__file__).resolve().parents[3]
                / "grail"
                / "agentic_kinematic"
                / "hoi_refinement"
                / "preview.py"
            )
        spec = importlib.util.spec_from_file_location(__package__ + ".stage2_preview", source)
        _PREVIEW = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(_PREVIEW)
    return _PREVIEW


def runtime():
    global _RUNTIME
    if _RUNTIME is None:
        source = Path(__file__).with_name("stage2_worker.py")
        if not source.is_file():
            source = (
                Path(__file__).resolve().parents[3]
                / "grail"
                / "agentic_kinematic"
                / "hoi_refinement"
                / "blender_worker.py"
            )
        if not source.is_file():
            raise RuntimeError("Install the matching built GRAIL extension with Stage 2 worker")
        spec = importlib.util.spec_from_file_location(__package__ + ".stage2_worker", source)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        _RUNTIME = module
    return _RUNTIME


def _path(value, label, exists=True, executable=False):
    if not value.strip():
        raise ValueError("Set " + label)
    path = Path(bpy.path.abspath(value)).expanduser()
    # Keep a virtual environment's executable entry path: resolving its symlink
    # can select the base interpreter and discard the environment's packages.
    path = Path(os.path.abspath(path)) if executable else path.resolve()
    if exists and not path.exists():
        raise FileNotFoundError(f"{label} not found: {path}")
    if executable and (not path.is_file() or not os.access(path, os.X_OK)):
        raise ValueError(f"{label} must be an executable file: {path}")
    return path


class GRAILHOI_Stage2Settings(bpy.types.PropertyGroup):
    case_path: StringProperty(name="Native case JSON", subtype="FILE_PATH")
    native_python: StringProperty(name="External GRAIL Python", subtype="FILE_PATH")
    repo_path: StringProperty(name="GRAIL repository", subtype="DIR_PATH")
    output_dir: StringProperty(name="New round directory", subtype="DIR_PATH")
    plan_export: StringProperty(name="New plan JSON", subtype="FILE_PATH")
    status: StringProperty(name="Status", default="UNREVIEWED; native evaluation runs externally")
    refreshed_blend: StringProperty(name="Refreshed scene", subtype="FILE_PATH")
    repair_round: StringProperty(name="Repair round", subtype="DIR_PATH")


class _Operation:
    def execute(self, context):
        try:
            message = self.run(context)
            context.scene.grail_hoi_stage2.status = message
            self.report({"INFO"}, message)
            return {"FINISHED"}
        except Exception as error:
            context.scene.grail_hoi_stage2.status = str(error)
            self.report({"ERROR"}, str(error))
            return {"CANCELLED"}


class GRAILHOI_OT_stage2_enable(_Operation, bpy.types.Operator):
    bl_idname = "grail_hoi.stage2_enable"
    bl_label = "Enable Verified Native Snapshots"

    def run(self, context):
        runtime().enable_snapshots()
        return context.scene.get("grail_stage2_snapshot_status", "UNREVIEWED")


class GRAILHOI_OT_stage2_jump(_Operation, bpy.types.Operator):
    bl_idname = "grail_hoi.stage2_jump"
    bl_label = "Jump to Source Key"

    def run(self, context):
        ob = context.active_object
        if ob is None or "blender_frame" not in ob:
            raise ValueError("Select a GRAILStage2_Key control")
        frame = float(ob["blender_frame"])
        context.scene.frame_set(int(frame), subframe=frame % 1)
        return f"Source {ob['source_frame']} = Blender {ob['blender_frame']}"


class GRAILHOI_OT_stage2_export(_Operation, bpy.types.Operator):
    bl_idname = "grail_hoi.stage2_export_plan"
    bl_label = "Export Actual Authored Plan"

    def run(self, context):
        path = _path(context.scene.grail_hoi_stage2.plan_export, "new plan JSON", False)
        plan = runtime().authored_plan()
        with path.open("x", encoding="utf8") as stream:
            json.dump(plan, stream, indent=2, allow_nan=False)
            stream.write("\n")
        return "UNREVIEWED plan exported; Refresh evaluates the native candidate"


class GRAILHOI_OT_stage2_refresh(_Operation, bpy.types.Operator):
    bl_idname = "grail_hoi.stage2_refresh"
    bl_label = "Refresh via External Native Evaluator"
    bl_description = "Save a separate scene copy; run bounded native evaluation and Blender review"
    command = "run"

    def run(self, context):
        settings = context.scene.grail_hoi_stage2
        python = _path(settings.native_python, "external GRAIL Python", executable=True)
        repo = _path(settings.repo_path, "GRAIL repository")
        case = _path(settings.case_path, "case JSON")
        out = _path(settings.output_dir, "new round directory", False)
        if out.exists():
            raise FileExistsError("Use a new round directory")
        out.parent.mkdir(parents=True, exist_ok=True)
        # Export/validate actual controls before writing a snapshot of user work.
        has_controls = runtime().CONTRACT in bpy.data.texts
        if self.command == "repair" and not has_controls:
            raise ValueError("Open and review a native Stage 2 scene before local repair")
        if has_controls:
            runtime().authored_plan()
        elif len(context.scene.objects):
            raise ValueError("Create Stage 2 from an empty document, or open a Stage 2 scene")
        token = uuid.uuid4().hex
        saved_copy = out.parent / f"stage2_input_{token}.blend"
        log = out.parent / f"stage2_refresh_{token}.log"
        command = [
            str(python),
            "-m",
            "grail.agentic_kinematic.hoi_refinement",
            self.command,
            "--config",
            str(case),
            "--output",
            str(out),
        ]
        if has_controls:
            bpy.ops.wm.save_as_mainfile(filepath=str(saved_copy), copy=True, check_existing=False)
            command.extend(["--input-blend", str(saved_copy)])
        with log.open("wb") as stream:
            try:
                result = subprocess.run(
                    command,
                    cwd=repo,
                    stdin=subprocess.DEVNULL,
                    stdout=stream,
                    stderr=subprocess.STDOUT,
                    check=False,
                    timeout=600,
                )
            except subprocess.TimeoutExpired as error:
                raise RuntimeError(f"External refresh timed out; inspect {log}") from error
        if result.returncode:
            raise RuntimeError(f"External refresh failed; inspect {log}")
        checkpoints = list(out.rglob("UNREVIEWED_authoring.blend"))
        if len(checkpoints) != 1:
            raise RuntimeError(f"Refresh did not produce one authoring scene; inspect {log}")
        settings.refreshed_blend = str(checkpoints[0])
        if self.command == "repair":
            settings.repair_round = str(out)
            check = preview_tools().validate_preview(out)
            return "Repair saved, UNREVIEWED; " + (
                "Preview to inspect"
                if check["can_apply"]
                else "Preview only: " + "; ".join(check["reasons"])
            )
        return "UNREVIEWED round saved; Open Refreshed Scene to inspect native observations"


class GRAILHOI_OT_stage2_repair(_Operation, bpy.types.Operator):
    bl_idname = "grail_hoi.stage2_repair"
    bl_label = "Propose Local Repair"
    command = "repair"

    def run(self, context):
        return GRAILHOI_OT_stage2_refresh.run(self, context)


def _settings(scene):
    settings = scene.grail_hoi_stage2
    return {
        key: getattr(settings, key)
        for key in (
            "native_python",
            "repo_path",
            "case_path",
            "output_dir",
            "repair_round",
            "refreshed_blend",
        )
    }


def _open_preserving_settings(path, values):
    bpy.ops.wm.open_mainfile(filepath=str(path))
    for key, value in values.items():
        setattr(bpy.context.scene.grail_hoi_stage2, key, value)


class GRAILHOI_OT_stage2_preview(_Operation, bpy.types.Operator):
    bl_idname = "grail_hoi.stage2_preview_repair"
    bl_label = "Preview Repair"

    def run(self, context):
        if context.scene.get("grail_stage2_repair_preview"):
            raise ValueError("Restore or apply the current preview first")
        values = _settings(context.scene)
        record = preview_tools().validate_preview(values["repair_round"])
        receipt = json.loads((Path(record["round_directory"]) / "iteration.json").read_text())
        seed = receipt.get("input_blend")
        if not seed:
            raise ValueError("Reversible Blender preview requires a saved input scene")
        # Pending local edits invalidate the trial, even if the saved seed is unchanged.
        expected = json.loads(Path(receipt["seed_plan"]["path"]).read_text())
        if runtime().contract().get("bindings") != record["bindings"]:
            raise ValueError("Current scene belongs to a different source, model or camera")
        runtime().validate_scene_binding(runtime().contract())
        runtime().validate_original_meshes(
            runtime().unpack("before"), runtime().unpack("after"), runtime().unpack("scene")
        )
        if not runtime()._plan_equal(runtime().authored_plan(), expected):
            raise ValueError("Current keys changed since repair; run a new proposal")
        preview_tools().verify_records(seed)
        record["seed_blend"] = seed
        # Preserve all current UI/scene work before opening the candidate.
        backup = Path(record["round_directory"]) / f"preview_return_{uuid.uuid4().hex}.blend"
        bpy.ops.wm.save_as_mainfile(filepath=str(backup), copy=True, check_existing=False)
        record["return_blend"] = {"path": str(backup), "sha256": preview_tools().digest(backup)}
        _open_preserving_settings(record["checkpoint"]["path"], values)
        bpy.context.scene["grail_stage2_repair_preview"] = json.dumps(record)
        runtime().enable_snapshots()
        return "UNREVIEWED repair preview; Restore Seed keeps the earlier motion"


class GRAILHOI_OT_stage2_restore(_Operation, bpy.types.Operator):
    bl_idname = "grail_hoi.stage2_restore_repair"
    bl_label = "Restore Seed"

    def run(self, context):
        raw = context.scene.get("grail_stage2_repair_preview")
        if not raw:
            raise ValueError("No active repair preview")
        record, values = json.loads(raw), _settings(context.scene)
        preview_tools().verify_records(record["return_blend"])
        # Save even rejected/manual preview changes before restoring, never discard them.
        backup = Path(record["round_directory"]) / f"restored_preview_{uuid.uuid4().hex}.blend"
        bpy.ops.wm.save_as_mainfile(filepath=str(backup), copy=True, check_existing=False)
        _open_preserving_settings(record["return_blend"]["path"], values)
        return "Seed restored; rejected preview and diagnostics retained"


class GRAILHOI_OT_stage2_apply(_Operation, bpy.types.Operator):
    bl_idname = "grail_hoi.stage2_apply_repair"
    bl_label = "Apply Repair as Unreviewed"

    def run(self, context):
        raw = context.scene.get("grail_stage2_repair_preview")
        if not raw:
            raise ValueError("Preview the repair before applying")
        record = json.loads(raw)
        check = preview_tools().validate_preview(record["round_directory"], require_apply=True)
        if check["receipt_sha256"] != record["receipt_sha256"]:
            raise ValueError("Repair receipt changed since preview")
        if runtime().contract().get("bindings") != check["bindings"]:
            raise ValueError("Preview belongs to a different source, model or camera")
        runtime().validate_scene_binding(runtime().contract())
        runtime().validate_original_meshes(
            runtime().unpack("before"), runtime().unpack("after"), runtime().unpack("scene")
        )
        expected = json.loads(Path(check["plan"]["path"]).read_text())
        if not runtime()._plan_equal(runtime().authored_plan(), expected):
            raise ValueError("Preview keys changed; refresh and re-audit before applying")
        runtime().enable_snapshots()
        output = Path(record["round_directory"]) / "APPLIED_UNREVIEWED.blend"
        if output.exists():
            raise FileExistsError("Applied checkpoint already exists; use a new round")
        context.scene["grail_stage2_repair_applied"] = json.dumps(check)
        del context.scene["grail_stage2_repair_preview"]
        try:
            result = bpy.ops.wm.save_as_mainfile(filepath=str(output), check_existing=False)
            if "FINISHED" not in result:
                raise RuntimeError("Applied checkpoint was not saved")
        except Exception:
            context.scene["grail_stage2_repair_preview"] = raw
            del context.scene["grail_stage2_repair_applied"]
            raise
        return "Saved applied UNREVIEWED checkpoint; source/contact review still required"


class GRAILHOI_OT_stage2_open(_Operation, bpy.types.Operator):
    bl_idname = "grail_hoi.stage2_open"
    bl_label = "Open Refreshed Scene"

    def run(self, context):
        settings = context.scene.grail_hoi_stage2
        path = _path(settings.refreshed_blend, "refreshed authoring scene")
        # Save the current document copy before the explicit open operation.
        if bpy.data.is_dirty:
            raise ValueError("Save your current document before opening the refreshed scene")
        values = {
            key: getattr(settings, key)
            for key in ("native_python", "repo_path", "case_path", "output_dir")
        }
        bpy.ops.wm.open_mainfile(filepath=str(path))
        for key, value in values.items():
            setattr(bpy.context.scene.grail_hoi_stage2, key, value)
        return "Enable verified snapshots; inspect source and oblique evidence before acceptance"


class GRAILHOI_PT_stage2(bpy.types.Panel):
    bl_idname = "GRAILHOI_PT_stage2"
    bl_label = "Stage 2 SOMA Native Round"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "GRAIL HOI"

    def draw(self, context):
        layout, settings = self.layout, context.scene.grail_hoi_stage2
        layout.label(text="UNREVIEWED • external native model")
        layout.label(text="Source 0 = Blender 1")
        layout.operator("grail_hoi.stage2_enable")
        layout.label(text=context.scene.get("grail_stage2_snapshot_status", "Snapshots disabled"))
        ob = context.active_object
        if ob and ob.name.startswith("GRAILStage2_Key_"):
            layout.label(text=f"Source {ob['source_frame']} / joint {ob['native_joint']}")
            row = layout.row()
            row.enabled = not ob["protected"]
            row.prop(ob, "rotation_quaternion", text="Native delta WXYZ")
            layout.operator("grail_hoi.stage2_jump")
        layout.label(text="Fixed key markers; rotate only. No IK.")
        layout.prop(settings, "plan_export")
        layout.operator("grail_hoi.stage2_export_plan")
        for name in ("case_path", "native_python", "repo_path", "output_dir"):
            layout.prop(settings, name)
        layout.operator("grail_hoi.stage2_refresh")
        layout.prop(settings, "refreshed_blend")
        layout.operator("grail_hoi.stage2_open")
        layout.separator()
        layout.operator("grail_hoi.stage2_repair")
        layout.prop(settings, "repair_round")
        row = layout.row()
        row.operator("grail_hoi.stage2_preview_repair")
        row.operator("grail_hoi.stage2_restore_repair")
        layout.operator("grail_hoi.stage2_apply_repair")
        layout.label(text=settings.status)


@persistent
def before_load(unused):
    if _RUNTIME is not None:
        _RUNTIME.disable_snapshots()


@persistent
def after_load(unused):
    if "GRAILStage2_Contract.json" in bpy.data.texts:
        runtime()._hide(bpy.context.scene, "DISABLED after open: enable verified native snapshots")


_CLASSES = (
    GRAILHOI_Stage2Settings,
    GRAILHOI_OT_stage2_enable,
    GRAILHOI_OT_stage2_jump,
    GRAILHOI_OT_stage2_export,
    GRAILHOI_OT_stage2_refresh,
    GRAILHOI_OT_stage2_open,
    GRAILHOI_OT_stage2_repair,
    GRAILHOI_OT_stage2_preview,
    GRAILHOI_OT_stage2_restore,
    GRAILHOI_OT_stage2_apply,
    GRAILHOI_PT_stage2,
)


def register():
    for cls in _CLASSES:
        bpy.utils.register_class(cls)
    bpy.types.Scene.grail_hoi_stage2 = PointerProperty(type=GRAILHOI_Stage2Settings)
    if before_load not in bpy.app.handlers.load_pre:
        bpy.app.handlers.load_pre.append(before_load)
    if after_load not in bpy.app.handlers.load_post:
        bpy.app.handlers.load_post.append(after_load)
    # Blender's real extension installer restricts data access during register.
    # Existing scenes are verified when the user explicitly enables snapshots.
    if hasattr(bpy.data, "texts"):
        after_load(None)


def unregister():
    before_load(None)
    if before_load in bpy.app.handlers.load_pre:
        bpy.app.handlers.load_pre.remove(before_load)
    if after_load in bpy.app.handlers.load_post:
        bpy.app.handlers.load_post.remove(after_load)
    if hasattr(bpy.types.Scene, "grail_hoi_stage2"):
        del bpy.types.Scene.grail_hoi_stage2
    for cls in reversed(_CLASSES):
        bpy.utils.unregister_class(cls)
