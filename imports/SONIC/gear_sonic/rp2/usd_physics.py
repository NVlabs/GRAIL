"""Source-independent USD mass and density resolution for active entities."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any


NORMALIZED_ACTIVE_BODY_PATHS = (
    "/ActiveArticulation/frame",
    "/ActiveArticulation/panel",
)
USD_DEPENDENCY_CONTRACT = "usd_file_layer_content_closure_v1"


def usd_stage_composition_errors(stage: Any) -> list[str]:
    """Return unresolved layers/references and other USD composition failures."""
    getter = getattr(stage, "GetCompositionErrors", None)
    if getter is None:
        raise RuntimeError(
            "USD runtime does not expose Stage.GetCompositionErrors; "
            "dependency validation cannot fail closed"
        )
    return [str(error) for error in getter()]


def require_resolved_usd_stage(stage: Any, path: str | Path) -> None:
    """Fail when a USD stage opened but any composed dependency did not."""
    errors = usd_stage_composition_errors(stage)
    if errors:
        raise ValueError(
            f"{Path(path).expanduser()}: unresolved USD composition: "
            + "; ".join(errors)
        )


def _sha256(
    path: Path,
    cache: dict[Path, str] | None = None,
) -> str:
    resolved = path.resolve(strict=True)
    if cache is not None and resolved in cache:
        return cache[resolved]
    digest = hashlib.sha256()
    with resolved.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    value = digest.hexdigest()
    if cache is not None:
        cache[resolved] = value
    return value


def usd_dependency_fingerprint(
    path: str | Path,
    *,
    hash_cache: dict[Path, str] | None = None,
) -> dict[str, Any]:
    """Hash the root USD and every file-backed layer it composes."""
    from pxr import Usd

    root_path = Path(path).expanduser().resolve(strict=True)
    stage = Usd.Stage.Open(str(root_path))
    if stage is None:
        raise ValueError(f"{root_path}: USD stage did not open")
    require_resolved_usd_stage(stage, root_path)

    layers: dict[Path, str] = {}
    for layer in stage.GetUsedLayers():
        if bool(getattr(layer, "anonymous", False)):
            continue
        real_path = str(getattr(layer, "realPath", "") or "").strip()
        if not real_path:
            raise ValueError(
                f"{root_path}: composed layer {layer.identifier!r} has no "
                "file-backed real path"
            )
        layer_path = Path(real_path).expanduser().resolve(strict=True)
        layers[layer_path] = _sha256(layer_path, hash_cache)
    if root_path not in layers:
        layers[root_path] = _sha256(root_path, hash_cache)

    layer_hashes = sorted(layers.values())
    payload = {
        "contract": USD_DEPENDENCY_CONTRACT,
        "root_sha256": _sha256(root_path, hash_cache),
        "layer_count": len(layer_hashes),
        "used_layer_sha256": layer_hashes,
    }
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    payload["closure_sha256"] = hashlib.sha256(encoded).hexdigest()
    return payload


def nearest_rigid_body(
    prim: Any,
    root: Any,
    UsdPhysics: Any,
) -> Any | None:
    """Return the closest rigid-body ancestor within ``root``."""
    root_path = root.GetPath()
    current = prim
    while current.IsValid() and current.GetPath().HasPrefix(root_path):
        if current.HasAPI(UsdPhysics.RigidBodyAPI):
            return current
        if current == root:
            break
        current = current.GetParent()
    return None


def _property_provenance(attribute: Any) -> list[dict[str, str]]:
    provenance: list[dict[str, str]] = []
    for specification in attribute.GetPropertyStack():
        layer = getattr(specification, "layer", None)
        spec_path = getattr(specification, "path", None)
        provenance.append(
            {
                "layer": str(getattr(layer, "identifier", "")),
                "property": str(
                    getattr(spec_path, "pathString", spec_path or "")
                ),
            }
        )
    return provenance


def _authored_numeric_property(
    prim: Any,
    attribute_name: str,
    *,
    api_applied: bool,
) -> dict[str, Any] | None:
    attribute = prim.GetAttribute(attribute_name)
    if not attribute.IsValid() or not attribute.HasAuthoredValueOpinion():
        return None
    value = attribute.Get()
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        numeric = math.nan
    return {
        "prim": prim.GetPath().pathString,
        "attribute": attribute_name,
        "value": numeric,
        "mass_api_applied": bool(api_applied),
        "provenance": _property_provenance(attribute),
    }


def dynamic_body_mass_density(
    stage: Any,
    root: Any,
    *,
    UsdPhysics: Any,
) -> tuple[list[str], list[dict[str, Any]]]:
    """Resolve explicit mass/density for every dynamic body in a USD stage."""
    errors: list[str] = []
    dynamic_bodies = []
    for prim in stage.Traverse():
        if not prim.HasAPI(UsdPhysics.RigidBodyAPI):
            continue
        api = UsdPhysics.RigidBodyAPI(prim)
        enabled = api.GetRigidBodyEnabledAttr().Get() is not False
        kinematic = api.GetKinematicEnabledAttr().Get() is True
        if enabled and not kinematic:
            dynamic_bodies.append(prim)

    records: list[dict[str, Any]] = []
    for body in dynamic_bodies:
        mass_opinions: list[dict[str, Any]] = []
        density_opinions: list[dict[str, Any]] = []
        for prim in stage.Traverse():
            owner = nearest_rigid_body(prim, root, UsdPhysics)
            if owner is None or owner != body:
                continue
            api_applied = prim.HasAPI(UsdPhysics.MassAPI)
            mass = _authored_numeric_property(
                prim,
                "physics:mass",
                api_applied=api_applied,
            )
            density = _authored_numeric_property(
                prim,
                "physics:density",
                api_applied=api_applied,
            )
            if mass is not None:
                mass_opinions.append(mass)
            if density is not None:
                density_opinions.append(density)

        body_path = body.GetPath().pathString
        for opinion in mass_opinions:
            if not opinion["mass_api_applied"]:
                errors.append(
                    f"{body_path} has authored physics:mass without "
                    f"MassAPI at {opinion['prim']}"
                )
            value = float(opinion["value"])
            if not math.isfinite(value) or value <= 0.0:
                errors.append(
                    f"{body_path} has non-positive/nonfinite authored mass "
                    f"{value} at {opinion['prim']}"
                )
        for opinion in density_opinions:
            if not opinion["mass_api_applied"]:
                errors.append(
                    f"{body_path} has authored physics:density without "
                    f"MassAPI at {opinion['prim']}"
                )
            value = float(opinion["value"])
            if not math.isfinite(value) or value < 0.0:
                errors.append(
                    f"{body_path} has negative/nonfinite authored density "
                    f"{value} at {opinion['prim']}"
                )

        positive_mass = [
            float(opinion["value"])
            for opinion in mass_opinions
            if math.isfinite(float(opinion["value"]))
            and float(opinion["value"]) > 0.0
        ]
        positive_density = [
            float(opinion["value"])
            for opinion in density_opinions
            if math.isfinite(float(opinion["value"]))
            and float(opinion["value"]) > 0.0
        ]
        if positive_mass:
            effective_mode = "authored_mass"
            effective_values = positive_mass
        elif positive_density:
            effective_mode = "authored_density"
            effective_values = positive_density
        else:
            effective_mode = "missing"
            effective_values = []
            errors.append(
                f"{body_path} has no positive authored mass or density; "
                "implicit simulator defaults are forbidden"
            )

        effective_value = (
            effective_values[0]
            if effective_values
            and len(set(effective_values)) == 1
            else math.nan
        )
        if effective_values and not math.isfinite(effective_value):
            label = (
                "masses"
                if effective_mode == "authored_mass"
                else "densities"
            )
            errors.append(
                f"{body_path} has ambiguous positive authored {label} "
                f"{effective_values}"
            )
        records.append(
            {
                "body": body_path,
                "effective_mode": effective_mode,
                "effective_value": effective_value,
                "mass_opinions": mass_opinions,
                "density_opinions": density_opinions,
            }
        )
    return errors, records


def normalized_active_body_physics(
    path: str | Path,
) -> dict[str, dict[str, Any]]:
    """Read the canonical frame/panel physics contract from normalized USD."""
    from pxr import Usd, UsdPhysics

    usd_path = Path(path).expanduser().resolve(strict=True)
    stage = Usd.Stage.Open(str(usd_path))
    if stage is None:
        raise ValueError(f"{usd_path}: USD stage did not open")
    require_resolved_usd_stage(stage, usd_path)
    root = stage.GetPrimAtPath("/ActiveArticulation")
    if not root.IsValid():
        raise ValueError(f"{usd_path}: missing /ActiveArticulation")

    errors, records = dynamic_body_mass_density(
        stage,
        root,
        UsdPhysics=UsdPhysics,
    )
    if errors:
        raise ValueError(f"{usd_path}: {'; '.join(errors)}")
    by_body = {str(record["body"]): record for record in records}
    expected_paths = set(NORMALIZED_ACTIVE_BODY_PATHS)
    if set(by_body) != expected_paths:
        raise ValueError(
            f"{usd_path}: dynamic bodies must be exactly "
            f"{sorted(expected_paths)}, found {sorted(by_body)}"
        )

    contract: dict[str, dict[str, Any]] = {}
    for body_path in NORMALIZED_ACTIVE_BODY_PATHS:
        record = by_body[body_path]
        mode = str(record["effective_mode"])
        opinions_key = (
            "mass_opinions"
            if mode == "authored_mass"
            else "density_opinions"
        )
        opinions = [
            opinion
            for opinion in record[opinions_key]
            if math.isfinite(float(opinion["value"]))
            and float(opinion["value"]) > 0.0
        ]
        provenance = ",".join(
            sorted(
                f"usd:{opinion['prim']}.{opinion['attribute']}"
                for opinion in opinions
            )
        )
        if not provenance:
            raise ValueError(
                f"{usd_path}: {body_path} has no explicit physics provenance"
            )
        contract[body_path] = {
            "mode": mode,
            "value": float(record["effective_value"]),
            "provenance": provenance,
        }
    return contract
