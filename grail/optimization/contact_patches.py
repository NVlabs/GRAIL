"""Balance explicit authored vertex patches before compiling a v2 contact review.

This is an authoring helper, not a correspondence search. It never adds, snaps,
cross-joins or changes vertex pairs. Keep the returned diagnostics with the
review: the resulting ``confidence`` values are relative loss weights, not
calibrated visibility probabilities. Source confidence is retained separately.
"""

from __future__ import annotations

import copy
import math
from urllib.parse import quote

import numpy as np

from .fixed_contact import _integer, _number, _text

_PAIR_FIELDS = {
    "id",
    "human_vertex_id",
    "object_vertex_id",
    "hand_surface",
    "object_surface",
}


def _positive(value, label):
    value = _number(value, label)
    if value <= 0:
        raise ValueError(f"{label} must be positive")
    return value


def _normalized(values):
    # Avoid overflow when finite authored areas/weights have very large units.
    largest = max(values)
    scaled = [value / largest for value in values]
    total = math.fsum(scaled)
    return [value / total for value in scaled]


def _geometry(value, count, label):
    if value is None:
        return None
    vertices = np.asarray(value)
    if (
        vertices.shape != (count, 3)
        or vertices.dtype.kind not in "fiu"
        or not np.isfinite(vertices).all()
    ):
        raise ValueError(f"{label} must be finite original [{count},3] vertex positions")
    return vertices


def _coverage(pairs, human_vertices, object_vertices):
    result = {"pair_count": len(pairs)}
    for side, geometry in (("human", human_vertices), ("object", object_vertices)):
        groups = {}
        for pair in pairs:
            groups.setdefault(pair[f"{side}_vertex_id"], []).append(pair["id"])
        ids = sorted(groups)
        result[f"unique_{side}_vertex_count"] = len(ids)
        result[f"{side}_vertex_ids"] = ids
        result[f"shared_{side}_vertices"] = [
            {"vertex_id": vertex, "pair_ids": groups[vertex]}
            for vertex in ids
            if len(groups[vertex]) > 1
        ]
        if geometry is None:
            result[f"unique_{side}_position_count"] = None
            result[f"coincident_{side}_vertex_groups"] = None
        else:
            _, inverse = np.unique(geometry[ids], axis=0, return_inverse=True)
            positions = {}
            for vertex, group in zip(ids, inverse.tolist()):
                positions.setdefault(group, []).append(vertex)
            result[f"unique_{side}_position_count"] = len(positions)
            result[f"coincident_{side}_vertex_groups"] = [
                group for group in positions.values() if len(group) > 1
            ]
    result["collapsed_object_target"] = len(pairs) > 1 and (
        result["unique_object_vertex_count"] == 1 or result["unique_object_position_count"] == 1
    )
    return result


def weight_contact_patches(
    patches,
    *,
    hand_confidence,
    human_vertex_count,
    object_vertex_count,
    human_vertices=None,
    object_vertices=None,
):
    """Return ``(v2_contacts, diagnostics)`` for ONE hand in ONE reviewed interval.

    Each patch is ``{id, contacts, weight?}`` (positive weight defaults to one).
    Each contact supplies the normal v2 pair fields except ``confidence``, plus
    an optional positive ``surface_weight`` (default one; e.g. represented area).
    Output IDs are stable escaped ``patch_id:pair_id`` names in input order.

    Pair confidence is ``hand_confidence * patch_share * within_patch_share``.
    The total per active hand/frame is therefore ``hand_confidence``, independent
    of sample count. Use the same source confidence and patch masses when
    changing density. Relative sampling weights affect quadrature, not source
    visibility. The caller still authors source intervals and compiles/validates
    the model/topology binding. This helper never activates contact by itself.

    Optional native vertex arrays report EXACT coincident-position aliases;
    distinct OBJ indices are not assumed to add surface coverage. Diagnostics
    do not certify contact quality, spatial extent, normals or collision safety.
    """
    _integer(human_vertex_count, "human_vertex_count", minimum=1)
    _integer(object_vertex_count, "object_vertex_count", minimum=1)
    confidence = _positive(hand_confidence, "hand_confidence")
    if confidence > 1:
        raise ValueError("hand_confidence must be <= 1")
    human_geometry = _geometry(human_vertices, human_vertex_count, "human_vertices")
    object_geometry = _geometry(object_vertices, object_vertex_count, "object_vertices")
    if not isinstance(patches, list) or not patches:
        raise ValueError("patches must be a nonempty list")
    patch_ids, seen_pairs, prepared = set(), set(), []
    for patch in patches:
        if (
            not isinstance(patch, dict)
            or not {"id", "contacts"} <= set(patch)
            or set(patch) - {"id", "contacts", "weight"}
        ):
            raise ValueError("Each patch needs id and contacts, with optional weight")
        name = _text(patch["id"], "patch.id")
        if name in patch_ids:
            raise ValueError(f"Duplicate patch id: {name}")
        patch_ids.add(name)
        patch_weight = _positive(patch.get("weight", 1.0), "patch.weight")
        if not isinstance(patch["contacts"], list) or not patch["contacts"]:
            raise ValueError(f"Patch {name} needs a nonempty explicit contact list")
        pair_ids, records, surface_weights = set(), [], []
        for pair in patch["contacts"]:
            if (
                not isinstance(pair, dict)
                or not _PAIR_FIELDS <= set(pair)
                or set(pair) - (_PAIR_FIELDS | {"surface_weight"})
            ):
                raise ValueError(
                    "Each pair needs explicit v2 vertex/surface fields, no confidence; "
                    "only surface_weight is optional"
                )
            pair_id = _text(pair["id"], "pair.id")
            if pair_id in pair_ids:
                raise ValueError(f"Duplicate pair id in patch {name}: {pair_id}")
            pair_ids.add(pair_id)
            for side, count in (("human", human_vertex_count), ("object", object_vertex_count)):
                vertex = _integer(pair[f"{side}_vertex_id"], f"pair.{side}_vertex_id")
                if vertex >= count:
                    raise ValueError(f"{side} vertex is out of range")
            indices = (pair["human_vertex_id"], pair["object_vertex_id"])
            if indices in seen_pairs:
                raise ValueError(f"Duplicate exact human/object vertex pair: {indices}")
            seen_pairs.add(indices)
            for field in ("hand_surface", "object_surface"):
                _text(pair[field], f"pair.{field}")
            surface_weights.append(
                _positive(pair.get("surface_weight", 1.0), "pair.surface_weight")
            )
            record = {field: copy.deepcopy(pair[field]) for field in sorted(_PAIR_FIELDS)}
            record["id"] = f"{quote(name, safe='')}:{quote(pair_id, safe='')}"
            records.append(record)
        prepared.append((name, patch_weight, records, surface_weights))

    contacts, patch_reports = [], []
    shares = _normalized([item[1] for item in prepared])
    for (name, raw_weight, records, surface_weights), share in zip(prepared, shares):
        within = _normalized(surface_weights)
        for record, sample_share in zip(records, within):
            record["confidence"] = confidence * share * sample_share
            if record["confidence"] <= 0:
                raise ValueError(
                    "Sampling weights underflow; rescale or remove negligible pairs explicitly"
                )
        contacts.extend(records)
        patch_reports.append(
            dict(
                id=name,
                authored_patch_weight=raw_weight,
                normalized_patch_share=share,
                authored_surface_weights=surface_weights,
                normalized_surface_shares=within,
                pair_ids=[record["id"] for record in records],
                confidence_sum=math.fsum(record["confidence"] for record in records),
                coverage=_coverage(records, human_geometry, object_geometry),
            )
        )
    diagnostics = dict(
        method="normalize explicit patch masses and within-patch surface weights",
        source_hand_confidence=confidence,
        confidence_sum=math.fsum(record["confidence"] for record in contacts),
        patches=patch_reports,
        coverage=_coverage(contacts, human_geometry, object_geometry),
        position_alias_semantics="Exact coordinate equality only; no geometric coverage claim",
        warning="Output confidence is a derived relative loss weight, not calibrated visibility. "
        "Save this sidecar to preserve source confidence and authored patch masses.",
    )
    return contacts, diagnostics
