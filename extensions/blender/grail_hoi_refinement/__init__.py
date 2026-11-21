"""SOMA03 Stage 2 controls and explicit review notes for Blender."""
from pathlib import Path

import bpy
import numpy as np
from bpy.props import EnumProperty, PointerProperty, StringProperty

from . import _support, stage2

bl_info = {
    "name": "GRAIL HOI Recon Refinement",
    "author": "NVIDIA",
    "version": (0, 3, 0),
    "blender": (4, 2, 0),
    "location": "View3D > Sidebar > GRAIL HOI",
    "category": "Animation",
    "description": "Author native human keys and export unreviewed HOI candidates",
}

def _path(value, label, existing=True):
    if not value.strip():
        raise ValueError(f"Set {label}")
    path = Path(bpy.path.abspath(value)).expanduser().resolve()
    if existing and not path.is_file():
        raise FileNotFoundError(f"{label} not found: {path}")
    return path


class GRAILHOI_Settings(bpy.types.PropertyGroup):
    status: StringProperty(name="Status")
    review_candidate_path: StringProperty(name="Actual candidate NPZ", subtype="FILE_PATH")
    review_note_path: StringProperty(name="New review-note JSON", subtype="FILE_PATH")
    reviewer: StringProperty(name="Reviewer")
    review_status: EnumProperty(
        name="Explicit decision",
        items=[
            ("UNREVIEWED", "Unreviewed / diagnostic", "No acceptance is implied"),
            ("PARTIAL", "Partial", "Review completed only for a limited scope"),
            ("REJECTED", "Rejected", "Candidate rejected for the stated scope"),
            (
                "ACCEPTED_FOR_STATED_SCOPE",
                "Accepted for stated scope",
                "Explicit independent reviewer decision, not plugin validation",
            ),
        ],
    )
    review_scope: StringProperty(name="Stated review scope")
    review_evidence: StringProperty(
        name="Actual review evidence",
        description="Paths/references for source, original geometry, contact, collision, temporal review as applicable",
    )
    review_findings: StringProperty(name="Findings / unresolved issues")


class _Operation:
    """Report actionable errors and keep failed operations out of FINISHED."""

    def execute(self, context):
        try:
            message = self.run(context)
            context.scene.grail_hoi.status = message
            self.report({"INFO"}, message)
            return {"FINISHED"}
        except Exception as error:
            message = f"{type(error).__name__}: {error}"
            context.scene.grail_hoi.status = message
            self.report({"ERROR"}, message)
            return {"CANCELLED"}


class GRAILHOI_OT_review(_Operation, bpy.types.Operator):
    bl_idname = "grail_hoi.export_review_note"
    bl_label = "Save Explicit Review Note"
    bl_description = "Write a separate manual decision with actual candidate SHA256; candidate status is never changed automatically"

    def run(self, context):
        settings = context.scene.grail_hoi
        candidate = _path(settings.review_candidate_path, "actual exported candidate NPZ")
        with np.load(candidate, allow_pickle=False) as native:
            if "model_type" not in native or native["model_type"].item() != "soma03":
                raise ValueError("Review notes require a native SOMA03 candidate")
        out = _path(settings.review_note_path, "new review-note JSON path", existing=False)
        if out.suffix.lower() != ".json":
            raise ValueError("Review note output must end in .json")
        _support.export_review_note(
            candidate,
            out,
            reviewer=settings.reviewer,
            status=settings.review_status,
            scope=settings.review_scope,
            evidence=settings.review_evidence,
            notes=settings.review_findings,
        )
        return f"Saved explicit {settings.review_status} review note; candidate unchanged"


class GRAILHOI_PT_review(bpy.types.Panel):
    bl_label = "Separate Manual Review Note"
    bl_idname = "GRAILHOI_PT_review"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "GRAIL HOI"
    bl_options = {"DEFAULT_CLOSED"}

    def draw(self, context):
        layout, settings = self.layout, context.scene.grail_hoi
        for name in (
            "review_candidate_path",
            "review_note_path",
            "reviewer",
            "review_status",
            "review_scope",
            "review_evidence",
            "review_findings",
        ):
            layout.prop(settings, name)
        layout.operator("grail_hoi.export_review_note")
        layout.label(text="Decision and hash saved separately.")
        layout.label(text="Plugin does not infer approval.")


_CLASSES = (GRAILHOI_Settings, GRAILHOI_OT_review, GRAILHOI_PT_review)


def register():
    if not (4, 2, 0) <= bpy.app.version < (4, 4, 0):
        raise RuntimeError("This extension supports Blender 4.2–4.3 (tested with 4.3.2)")
    for cls in _CLASSES:
        bpy.utils.register_class(cls)
    bpy.types.Scene.grail_hoi = PointerProperty(type=GRAILHOI_Settings)
    stage2.register()


def unregister():
    stage2.unregister()
    if hasattr(bpy.types.Scene, "grail_hoi"):
        del bpy.types.Scene.grail_hoi
    for cls in reversed(_CLASSES):
        bpy.utils.unregister_class(cls)
