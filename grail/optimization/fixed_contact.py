"""Frozen human-vertex / original object-vertex contact correspondences.

The JSON prediction supplies material correspondences, confidence and source-frame
intervals once. Optimization only moves the live human and rigid object; it never
selects a closer point or changes a contact's activation from its current gap.
Version 2 records select explicit vertices on both meshes. Historical version 1
triangle/barycentric records are read without converting or relabeling them.

Array SHA-256 is computed over the shape as contiguous little-endian int64 bytes,
then the C-order payload (little-endian int64 faces or float32 metre vertices).
Faces and vertices must be two-dimensional, so the shape prefix has two entries.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import re
from pathlib import Path

import numpy as np
import torch
from torch import nn

SCHEMA = "grail.fixed_contact.v2"
LEGACY_SCHEMA = "grail.fixed_contact.v1"
SUPPORTED_SCHEMAS = (LEGACY_SCHEMA, SCHEMA)
_BINDING_FIELDS = {
    "sequence",
    "source_video_sha256",
    "frame_count",
    "fps",
    "human_model_type",
    "human_vertex_count",
    "human_topology_sha256",
    "object_vertex_count",
    "object_face_count",
    "object_topology_sha256",
    "object_vertices_sha256",
}
_CONTACT_FIELDS = {
    "id",
    "body_part",
    "human_vertex_id",
    "active_frames",
    "confidence",
}


def anchor_fields(schema):
    """Exact object-anchor fields; mixed vertex/triangle records are invalid."""
    if schema == SCHEMA:
        return {"object_vertex_id"}
    if schema == LEGACY_SCHEMA:
        return {"object_face_id", "object_barycentric"}
    raise ValueError(f"Fixed contact schema must be one of {SUPPORTED_SCHEMAS}")


def validate_anchor(record, binding, schema, label):
    """Return a detached, index-validated anchor descriptor in its own schema."""
    if schema == SCHEMA:
        vertex = _integer(record["object_vertex_id"], f"{label}.object_vertex_id")
        if vertex >= binding["object_vertex_count"]:
            raise ValueError(f"{label}: original object vertex is out of range")
        return {"object_vertex_id": vertex}
    if schema != LEGACY_SCHEMA:
        anchor_fields(schema)  # Reject unknown schemas, including empty reviews.
    face = _integer(record["object_face_id"], f"{label}.object_face_id")
    if face >= binding["object_face_count"]:
        raise ValueError(f"{label}: original object face is out of range")
    bary = record["object_barycentric"]
    if not isinstance(bary, list) or len(bary) != 3:
        raise ValueError(f"{label}.object_barycentric must have three numbers")
    bary = [_number(value, f"{label}.object_barycentric") for value in bary]
    if any(value < 0 or value > 1 for value in bary) or abs(sum(bary) - 1) > 1e-6:
        raise ValueError(f"{label}: barycentrics must be nonnegative and sum to one")
    return {"object_face_id": face, "object_barycentric": bary}


def file_sha256(path):
    """Hash a file's bytes without loading the entire file into memory."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _array(value):
    if torch.is_tensor(value):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _faces(value, label):
    faces = _array(value)
    if faces.ndim != 2 or faces.shape[1] != 3 or len(faces) == 0:
        raise ValueError(f"{label} must be a nonempty [F, 3] triangle array")
    if faces.dtype.kind not in "iu" or np.any(faces < 0):
        raise ValueError(f"{label} must contain nonnegative integer indices")
    if np.any(faces > np.iinfo(np.int64).max):
        raise ValueError(f"{label} exceeds the int64 index range")
    return np.array(faces, dtype="<i8", order="C", copy=True)


def _vertices(value):
    vertices = _array(value)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or len(vertices) == 0:
        raise ValueError("object_vertices must be a nonempty [V, 3] array")
    if vertices.dtype.kind not in "fiu" or not np.isfinite(vertices).all():
        raise ValueError("object_vertices must contain finite metre coordinates")
    with np.errstate(over="ignore"):
        vertices = np.array(vertices, dtype="<f4", order="C", copy=True)
    if not np.isfinite(vertices).all():
        raise ValueError("object_vertices must be representable as finite float32")
    return vertices


def _array_sha256(array):
    digest = hashlib.sha256(np.asarray(array.shape, dtype="<i8").tobytes())
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def topology_sha256(faces):
    """Hash ordered triangle indices after the documented canonical i8 cast."""
    return _array_sha256(_faces(faces, "faces"))


def vertices_sha256(vertices):
    """Hash ordered local metre vertices after the documented canonical f4 cast."""
    return _array_sha256(_vertices(vertices))


def _text(value, label):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a nonempty string")
    return value


def _integer(value, label, minimum=0):
    if type(value) is not int or value < minimum:
        raise ValueError(f"{label} must be an integer >= {minimum}")
    return value


def _number(value, label):
    if type(value) not in (int, float):
        raise ValueError(f"{label} must be a finite number")
    try:
        number = float(value)
    except OverflowError as exc:
        raise ValueError(f"{label} must be a finite number") from exc
    if not math.isfinite(number):
        raise ValueError(f"{label} must be a finite number")
    return number


def _sha256(value, label):
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256 hex digest")
    return value


def _fields(value, expected, label):
    if not isinstance(value, dict) or set(value) != expected:
        raise ValueError(f"{label} must have exactly these fields: {sorted(expected)}")


def _validate_binding(binding):
    _fields(binding, _BINDING_FIELDS, "binding")
    for key in ("sequence", "human_model_type"):
        _text(binding[key], f"binding.{key}")
    for key in ("frame_count", "human_vertex_count", "object_vertex_count", "object_face_count"):
        _integer(binding[key], f"binding.{key}", minimum=1)
    if _number(binding["fps"], "binding.fps") <= 0:
        raise ValueError("binding.fps must be positive")
    for key in _BINDING_FIELDS:
        if key.endswith("sha256"):
            _sha256(binding[key], f"binding.{key}")


def build_binding(
    *,
    sequence,
    source_video_sha256,
    frame_count,
    fps,
    human_model_type,
    human_vertex_count,
    human_faces,
    object_vertices,
    object_faces,
):
    """Bind original topology/geometry and the complete source timeline.

    ``object_vertices`` are ORIGINAL object-local vertices already scaled into
    metres, never the separate remeshed/simplified tracking object.
    """
    vertices = _vertices(object_vertices)
    obj_faces = _faces(object_faces, "object_faces")
    hum_faces = _faces(human_faces, "human_faces")
    _integer(human_vertex_count, "human_vertex_count", minimum=1)
    if np.any(obj_faces >= len(vertices)):
        raise ValueError("object_faces references an out-of-range vertex")
    if np.any(hum_faces >= human_vertex_count):
        raise ValueError("human_faces references an out-of-range vertex")
    binding = {
        "sequence": sequence,
        "source_video_sha256": source_video_sha256,
        "frame_count": frame_count,
        "fps": _number(fps, "fps"),
        "human_model_type": human_model_type,
        "human_vertex_count": human_vertex_count,
        "human_topology_sha256": _array_sha256(hum_faces),
        "object_vertex_count": len(vertices),
        "object_face_count": len(obj_faces),
        "object_topology_sha256": _array_sha256(obj_faces),
        "object_vertices_sha256": _array_sha256(vertices),
    }
    _validate_binding(binding)
    return binding


def _json_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON field: {key}")
        result[key] = value
    return result


def _reject_json_constant(value):
    raise ValueError(f"Nonfinite JSON constant: {value}")


class FixedContactLoss(nn.Module):
    """Confidence-weighted isotropic Huber distance for frozen contact pairs.

    Construct with :meth:`from_file`. Prediction data are detached copies; only
    live human vertices and object rotation/translation receive gradients.
    ``active_frames`` is a source-frame interval with an exclusive end.
    """

    @classmethod
    def from_file(
        cls,
        path,
        *,
        binding,
        object_vertices,
        object_faces,
        human_faces,
        device,
        dtype=torch.float32,
        robust_delta_m=0.01,
        require_video_review=False,
    ):
        _validate_binding(binding)
        delta = _number(robust_delta_m, "robust_delta_m")
        if delta <= 0:
            raise ValueError("robust_delta_m must be positive")
        if not dtype.is_floating_point:
            raise ValueError("Fixed contact buffers require a floating-point dtype")
        current = build_binding(
            **{
                key: binding[key]
                for key in (
                    "sequence",
                    "source_video_sha256",
                    "frame_count",
                    "fps",
                    "human_model_type",
                    "human_vertex_count",
                )
            },
            human_faces=human_faces,
            object_vertices=object_vertices,
            object_faces=object_faces,
        )
        if current != binding:
            raise ValueError(
                "Fixed contact binding differs from supplied original geometry/topology"
            )
        path = Path(path).expanduser().resolve()
        payload = path.read_bytes()  # One read: hash and parse the identical snapshot.
        spec = json.loads(
            payload, object_pairs_hook=_json_object, parse_constant=_reject_json_constant
        )
        fields = {"schema", "binding", "prediction", "contacts"}
        if isinstance(spec, dict) and "video_review" in spec:
            fields.add("video_review")
        _fields(spec, fields, "fixed contact JSON")
        schema = spec["schema"]
        fields = _CONTACT_FIELDS | anchor_fields(schema)
        _validate_binding(spec["binding"])
        if spec["binding"] != binding:
            raise ValueError(
                "Fixed contact prediction binding differs from the current sequence/model/timeline/geometry"
            )
        _fields(spec["prediction"], {"method", "evidence"}, "prediction")
        for key in ("method", "evidence"):
            _text(spec["prediction"][key], f"prediction.{key}")
        contacts = spec["contacts"]
        review = spec.get("video_review")
        if require_video_review and review is None:
            raise ValueError(
                "Stage 2 requires video_review: inspect the source video and compile "
                "per-hand contact intervals (docs/fixed_contact_points.md)"
            )
        if "video_review" in spec:
            from grail.optimization.contact_review import compile_review, fixed_schema_for_review

            if fixed_schema_for_review(review) != schema:
                raise ValueError("Fixed contact and video review schema versions must match")
            if compile_review(review, binding) != contacts:
                raise ValueError("Fixed contact pairs differ from their reviewed video intervals")
        if not isinstance(contacts, list) or (not contacts and review is None):
            raise ValueError("contacts must be a nonempty list")
        # Hashing uses canonical f4; fitting retains the precision of the
        # supplied original coordinates, detached from any reference graph.
        vertices = np.array(_array(object_vertices), dtype=np.float64, copy=True)
        faces = _faces(object_faces, "object_faces")
        seen, anchors = set(), []
        frame_ids, human_ids, pair_ids, weights = [], [], [], []
        for pair_id, record in enumerate(contacts):
            _fields(record, fields, f"contacts[{pair_id}]")
            contact_id = _text(record["id"], "contact.id")
            if contact_id in seen:
                raise ValueError(f"Duplicate contact id: {contact_id}")
            seen.add(contact_id)
            if record["body_part"] not in ("L_Hand", "R_Hand"):
                raise ValueError("contact.body_part must be L_Hand or R_Hand")
            human_id = _integer(record["human_vertex_id"], "contact.human_vertex_id")
            if human_id >= binding["human_vertex_count"]:
                raise ValueError("Contact human vertex is out of range")
            anchor = validate_anchor(record, binding, schema, "contact")
            if schema == SCHEMA:
                anchors.append(vertices[anchor["object_vertex_id"]].copy())
            else:
                triangle = vertices[faces[anchor["object_face_id"]]]
                edge_a, edge_b = triangle[1] - triangle[0], triangle[2] - triangle[0]
                scale = np.linalg.norm(edge_a) * np.linalg.norm(edge_b)
                if scale == 0 or np.linalg.norm(np.cross(edge_a, edge_b)) <= 1e-12 * scale:
                    raise ValueError(
                        f"Contact {contact_id} references a degenerate object triangle"
                    )
                anchors.append(np.asarray(anchor["object_barycentric"]) @ triangle)
            interval = record["active_frames"]
            if not isinstance(interval, list) or len(interval) != 2:
                raise ValueError("contact.active_frames must be [start, end_exclusive]")
            start, end = [_integer(x, "contact.active_frames") for x in interval]
            if not 0 <= start < end <= binding["frame_count"]:
                raise ValueError(
                    "Contact interval must be nonempty and inside the full source timeline"
                )
            confidence = _number(record["confidence"], "contact.confidence")
            if not 0 <= confidence <= 1:
                raise ValueError("contact.confidence must be between zero and one")
            if confidence > 0:
                count = end - start
                frame_ids.extend(range(start, end))
                human_ids.extend([human_id] * count)
                pair_ids.extend([pair_id] * count)
                weights.extend([confidence] * count)
        if not weights and contacts:
            raise ValueError("At least one contact must have positive confidence")
        module = cls()
        module.frame_count = binding["frame_count"]
        module.human_vertex_count = binding["human_vertex_count"]
        module.robust_delta_m = delta
        module._contacts = copy.deepcopy(contacts)
        module._anchor_records = np.asarray(anchors, dtype=np.float64).tolist()
        module._provenance = {
            "path": str(path),
            "sha256": hashlib.sha256(payload).hexdigest(),
            "schema": schema,
            "binding": copy.deepcopy(binding),
            "prediction": copy.deepcopy(spec["prediction"]),
        }
        if review is not None:
            from grail.optimization.contact_review import review_sha256

            module._provenance["video_review"] = copy.deepcopy(review)
            module._provenance["video_review_sha256"] = review_sha256(review)
            module._provenance["phase_frame_counts"] = {
                hand: {
                    state: sum(
                        interval["frames"][1] - interval["frames"][0]
                        for interval in intervals
                        if interval["state"] == state
                    )
                    for state in ("contact", "no_contact", "unknown")
                }
                for hand, intervals in review["hands"].items()
            }
        for name, values, buffer_dtype in (
            ("_frame_ids", frame_ids, torch.long),
            ("_human_ids", human_ids, torch.long),
            ("_pair_ids", pair_ids, torch.long),
            ("_anchors", np.asarray(anchors).reshape(-1, 3), dtype),
            ("_weights", weights, dtype),
        ):
            module.register_buffer(
                name, torch.as_tensor(values, device=device, dtype=buffer_dtype).detach().clone()
            )
        return module

    @property
    def provenance(self):
        """A JSON-safe copy, isolated from the cached prediction state."""
        return copy.deepcopy(self._provenance)

    def export_contacts(self):
        """Original records plus their fixed original-object local metre point."""
        records = copy.deepcopy(self._contacts)
        for record, point in zip(records, self._anchor_records):
            record["object_local_point_m"] = list(point)
        return records

    def contact_points_by_hand(self):
        """Positive-confidence local anchors by hand and active source frame."""
        result = {"left_hand": {}, "right_hand": {}}
        for record, point in zip(self._contacts, self._anchor_records):
            if record["confidence"] <= 0:
                continue
            hand = "left_hand" if record["body_part"] == "L_Hand" else "right_hand"
            for frame in range(*record["active_frames"]):
                result[hand].setdefault(frame, []).append(list(point))
        return result

    def forward(self, human_vertices, object_R, object_t):
        expected = (
            (self.frame_count, self.human_vertex_count, 3),
            (self.frame_count, 3, 3),
            (self.frame_count, 3),
        )
        inputs = (human_vertices, object_R, object_t)
        for name, value, shape in zip(("human_vertices", "object_R", "object_t"), inputs, expected):
            if not torch.is_tensor(value) or tuple(value.shape) != shape:
                raise ValueError(f"{name} must have full-sequence shape {shape}")
            if not value.is_floating_point():
                raise ValueError(f"{name} must be floating point")
            if value.device != human_vertices.device:
                raise ValueError("Live fixed contact tensors must share a device")
        dtype = torch.promote_types(
            human_vertices.dtype, torch.promote_types(object_R.dtype, object_t.dtype)
        )
        device = human_vertices.device
        weights = self._weights.to(device=device, dtype=dtype)
        if not weights.numel() or not bool(weights.sum() > 0):
            return sum(value.to(dtype=dtype).sum() * 0 for value in inputs)
        frames = self._frame_ids.to(device=device)
        human_ids = self._human_ids.to(device=device)
        pair_ids = self._pair_ids.to(device=device)
        anchors = self._anchors.to(device=device, dtype=dtype)[pair_ids]
        targets = torch.bmm(object_R[frames].to(dtype=dtype), anchors.unsqueeze(-1)).squeeze(
            -1
        ) + object_t[frames].to(dtype=dtype)
        distance = torch.linalg.vector_norm(
            human_vertices[frames, human_ids].to(dtype=dtype) - targets, dim=-1
        )
        delta = self.robust_delta_m
        robust = torch.where(
            distance <= delta, 0.5 * distance.square(), delta * (distance - 0.5 * delta)
        )
        return (weights * robust).sum() / weights.sum()
