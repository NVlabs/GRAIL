"""Task-local active-panel mass copies for a single-mass ablation.

The source USD is copied byte-for-byte and only ``/ActiveArticulation/panel``
``physics:mass`` is re-authored. Inertia, centre of mass and density stay
unauthored (zero / sentinel), so PhysX recomputes the panel inertia from the
same collider for the new mass: same mass distribution, inertia scaled by the
mass ratio. The frame helper body and every other opinion are left unchanged.
Source packs are never written.
"""

from __future__ import annotations

import json
import math
import shutil
from pathlib import Path
from typing import Any

PANEL_PRIM = "/ActiveArticulation/panel"
FRAME_PRIM = "/ActiveArticulation/frame"
MASS_ATTR = "physics:mass"


def _layer_opinions(layer) -> dict[str, Any]:
    """Every authored field of every spec, keyed by ``path|field``."""
    opinions: dict[str, Any] = {}

    def visit(path):
        spec = layer.GetObjectAtPath(path)
        if spec is None:
            return
        for field in spec.ListInfoKeys():
            value = spec.GetInfo(field)
            opinions[f"{path}|{field}"] = repr(value)

    layer.Traverse("/", visit)
    for field in layer.pseudoRoot.ListInfoKeys():
        opinions[f"/|{field}"] = repr(layer.pseudoRoot.GetInfo(field))
    return opinions


def _mass_api_fields(layer, prim: str) -> dict[str, Any]:
    out = {}
    for name in ("physics:mass", "physics:density", "physics:diagonalInertia", "physics:centerOfMass", "physics:principalAxes"):
        attr = layer.GetAttributeAtPath(f"{prim}.{name}")
        out[name] = None if attr is None else attr.default
    return out


def _check_inertia_unauthored(fields: dict[str, Any], prim: str) -> None:
    inertia = fields["physics:diagonalInertia"]
    if inertia is not None and any(float(v) != 0.0 for v in inertia):
        raise ValueError(f"{prim}: authored inertia {inertia} would not scale with mass")
    com = fields["physics:centerOfMass"]
    if com is not None and not all(math.isinf(float(v)) for v in com):
        raise ValueError(f"{prim}: authored centre of mass {com}")
    density = fields["physics:density"]
    if density is not None and float(density) != 0.0:
        raise ValueError(f"{prim}: authored density {density}")


def write_panel_mass_copy(source: Path, destination: Path, panel_mass_kg: float) -> dict[str, Any]:
    """Create (or re-validate) one task-local copy and return its report."""
    from pxr import Sdf, Usd

    if not panel_mass_kg > 0.0:
        raise ValueError("panel mass must be positive")
    source = source.resolve(strict=True)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not destination.exists():
        shutil.copyfile(source, destination)
        destination.chmod(0o644)
        layer = Sdf.Layer.FindOrOpen(str(destination))
        attr = layer.GetAttributeAtPath(f"{PANEL_PRIM}.{MASS_ATTR}")
        if attr is None:
            raise ValueError(f"{source}: no authored {PANEL_PRIM}.{MASS_ATTR}")
        attr.default = float(panel_mass_kg)
        layer.Save()
    src_layer = Sdf.Layer.OpenAsAnonymous(str(source))
    dst_layer = Sdf.Layer.OpenAsAnonymous(str(destination))
    src_ops = _layer_opinions(src_layer)
    dst_ops = _layer_opinions(dst_layer)
    changed = sorted(k for k in set(src_ops) | set(dst_ops) if src_ops.get(k) != dst_ops.get(k))
    expected_change = f"{PANEL_PRIM}.{MASS_ATTR}|default"
    if changed != [expected_change]:
        raise ValueError(f"{destination}: unexpected opinion differences {changed[:10]}")
    src_panel = _mass_api_fields(src_layer, PANEL_PRIM)
    dst_panel = _mass_api_fields(dst_layer, PANEL_PRIM)
    src_frame = _mass_api_fields(src_layer, FRAME_PRIM)
    dst_frame = _mass_api_fields(dst_layer, FRAME_PRIM)
    for prim, fields in ((PANEL_PRIM, dst_panel), (FRAME_PRIM, dst_frame)):
        _check_inertia_unauthored(fields, prim)
    if not math.isclose(float(dst_panel[MASS_ATTR]), panel_mass_kg, rel_tol=1e-6):
        raise ValueError(f"{destination}: panel mass {dst_panel[MASS_ATTR]} != {panel_mass_kg}")
    stage = Usd.Stage.Open(dst_layer)
    report = {
        "source_usd": str(source),
        "override_usd": str(destination),
        "changed_opinions": changed,
        "kilograms_per_unit": stage.GetMetadata("kilogramsPerUnit"),
        "meters_per_unit": stage.GetMetadata("metersPerUnit"),
        "authored_panel_mass_kg": {"source": float(src_panel[MASS_ATTR]), "override": float(dst_panel[MASS_ATTR])},
        "authored_frame_mass_kg": {"source": float(src_frame[MASS_ATTR]), "override": float(dst_frame[MASS_ATTR])},
        "authored_assembly_mass_kg": {
            "source": float(src_panel[MASS_ATTR]) + float(src_frame[MASS_ATTR]),
            "override": float(dst_panel[MASS_ATTR]) + float(dst_frame[MASS_ATTR]),
        },
        "panel_inertia_com_density": "unauthored; PhysX computes from the unchanged collider",
    }
    return report


def write_report(path: Path, panel_mass_kg: float, reports: list[dict[str, Any]]) -> None:
    path.write_text(json.dumps({"panel_mass_kg": panel_mass_kg, "assets": reports}, indent=2) + "\n")
