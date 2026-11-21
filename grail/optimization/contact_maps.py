"""Source-phase contact-region maps on native vertices, without correspondences.

Human and object sets have independent cardinality. Strength is authored regional
evidence, not calibrated visibility or a statement that all vertices touch. This
format is a review/export artifact; explicit Stage 2 consumption is implemented
separately in phase_contact_map_loss. Export alone creates no point assignment
or optimization result.
"""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import numpy as np

from .fixed_contact import (
    _fields,
    _integer,
    _json_object,
    _number,
    _reject_json_constant,
    _text,
    _validate_binding,
    build_binding,
    file_sha256,
)

SCHEMA = "grail.contact_map.v1"
_HANDS = {"left_hand": "left", "right_hand": "right"}
_INTERVAL_FIELDS = {"id", "frames", "state", "description", "evidence_frames", "regions"}
_REGION_FIELDS = {
    "id",
    "description",
    "human_vertex_ids",
    "human_strength",
    "object_vertex_ids",
    "object_strength",
}


def _vertex_field(ids, strength, count, label):
    if not isinstance(ids, list) or not ids:
        raise ValueError(f"{label}.vertex_ids must be a nonempty list")
    checked = [_integer(value, f"{label}.vertex_ids") for value in ids]
    if len(set(checked)) != len(checked) or max(checked) >= count:
        raise ValueError(f"{label}.vertex_ids must be unique and inside original topology")
    if not isinstance(strength, list) or len(strength) != len(ids):
        raise ValueError(f"{label}.strength must have one value per vertex on its OWN surface")
    values = [_number(value, f"{label}.strength") for value in strength]
    if any(value < 0 or value > 1 for value in values) or not any(value > 0 for value in values):
        raise ValueError(f"{label}.strength must be in [0,1], with some positive support")
    return checked, values


class PhaseContactMap:
    """Validated immutable snapshot with independent per-frame surface fields."""

    @classmethod
    def from_file(cls, path, **kwargs):
        path = Path(path).resolve()
        payload = path.read_bytes()
        spec = json.loads(
            payload, object_pairs_hook=_json_object, parse_constant=_reject_json_constant
        )
        result = cls.from_spec(spec, **kwargs)
        result._provenance["manifest"] = {
            "path": str(path),
            "sha256": hashlib.sha256(payload).hexdigest(),
        }
        return result

    @classmethod
    def from_spec(
        cls,
        spec,
        *,
        binding,
        human_faces,
        object_vertices,
        object_faces,
        source_video_path=None,
        hand_vertex_ids=None,
    ):
        """Validate against independently supplied source/model/geometry binding.

        If supplied, ``hand_vertex_ids`` maps left_hand/right_hand to native hand
        segmentation IDs and enforces membership. It is never a map generator.
        ``source_video_path`` additionally verifies the source bytes themselves.
        Overlapping authored regions combine with max, not additive evidence.
        """
        _validate_binding(binding)
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
            raise ValueError("Contact map binding differs from current original geometry/topology")
        _fields(spec, {"schema", "binding", "reviewed_by", "hands"}, "contact map")
        if spec["schema"] != SCHEMA:
            raise ValueError(f"Contact map schema must be {SCHEMA}; never reinterpret fixed pairs")
        _validate_binding(spec["binding"])
        if spec["binding"] != binding:
            raise ValueError("Contact map binding differs from source/model/timeline/geometry")
        _text(spec["reviewed_by"], "reviewed_by")
        _fields(spec["hands"], set(_HANDS), "hands")
        video = None
        if source_video_path is not None:
            video = dict(
                path=str(Path(source_video_path).resolve()), sha256=file_sha256(source_video_path)
            )
            if video["sha256"] != binding["source_video_sha256"]:
                raise ValueError("Contact map source video hash mismatch")
        segments = None
        if hand_vertex_ids is not None:
            _fields(hand_vertex_ids, set(_HANDS), "hand_vertex_ids")
            segments = {}
            for hand, ids in hand_vertex_ids.items():
                values = np.asarray(ids)
                if (
                    values.ndim != 1
                    or values.dtype.kind not in "iu"
                    or not len(values)
                    or np.any(values < 0)
                    or np.any(values >= binding["human_vertex_count"])
                    or len(np.unique(values)) != len(values)
                ):
                    raise ValueError(f"{hand} segmentation must contain unique valid original IDs")
                segments[hand] = set(values.tolist())
        frame_count = binding["frame_count"]
        fields, phase_counts, interval_counts = {}, {}, {}
        for hand, side in _HANDS.items():
            intervals = spec["hands"][hand]
            if not isinstance(intervals, list) or not intervals:
                raise ValueError(f"{hand} requires nonempty complete timeline intervals")
            next_frame, seen_intervals = 0, set()
            validated, unions = [], {"human": set(), "object": set()}
            phase_counts[hand] = dict(contact=0, no_contact=0, unknown=0)
            interval_counts[hand] = []
            for interval in intervals:
                _fields(interval, _INTERVAL_FIELDS, f"{hand}.interval")
                name = _text(interval["id"], "interval.id")
                if name in seen_intervals:
                    raise ValueError(f"Duplicate interval id in {hand}: {name}")
                seen_intervals.add(name)
                frames = interval["frames"]
                if not isinstance(frames, list) or len(frames) != 2:
                    raise ValueError("interval.frames must be [start,end_exclusive]")
                start, end = [_integer(value, "interval.frames") for value in frames]
                if start != next_frame or not 0 <= start < end <= frame_count:
                    raise ValueError(
                        "Hand intervals must cover timeline in order without gaps or overlaps"
                    )
                next_frame = end
                state = interval["state"]
                if state not in ("contact", "no_contact", "unknown"):
                    raise ValueError("interval.state must be contact, no_contact or unknown")
                phase_counts[hand][state] += end - start
                _text(interval["description"], "interval.description")
                evidence = interval["evidence_frames"]
                if not isinstance(evidence, list) or not evidence:
                    raise ValueError("evidence_frames must contain inspected source frames")
                evidence = [_integer(value, "evidence_frames") for value in evidence]
                if len(set(evidence)) != len(evidence) or any(
                    not start <= value < end for value in evidence
                ):
                    raise ValueError("evidence_frames must be unique and inside the interval")
                regions = interval["regions"]
                if not isinstance(regions, list) or (state == "contact") != bool(regions):
                    raise ValueError(
                        "Contact intervals need regions; no_contact/unknown must have none"
                    )
                seen_regions, prepared = set(), []
                interval_ids = {"human": set(), "object": set()}
                for region in regions:
                    _fields(region, _REGION_FIELDS, "contact map region")
                    region_id = _text(region["id"], "region.id")
                    if region_id in seen_regions:
                        raise ValueError(f"Duplicate region id in interval {name}: {region_id}")
                    seen_regions.add(region_id)
                    _text(region["description"], "region.description")
                    sample = {}
                    for surface in ("human", "object"):
                        ids, values = _vertex_field(
                            region[f"{surface}_vertex_ids"],
                            region[f"{surface}_strength"],
                            binding[f"{surface}_vertex_count"],
                            surface,
                        )
                        if (
                            surface == "human"
                            and segments is not None
                            and not set(ids) <= segments[hand]
                        ):
                            raise ValueError(
                                f"Human map includes vertices outside native {hand} segmentation"
                            )
                        sample[surface] = (ids, values)
                        unions[surface].update(ids)
                        interval_ids[surface].update(
                            vertex for vertex, value in zip(ids, values) if value > 0
                        )
                    prepared.append(sample)
                validated.append((start, end, prepared))
                interval_counts[hand].append(
                    dict(
                        id=name,
                        frames=[start, end],
                        state=state,
                        positive_human_vertex_count=len(interval_ids["human"]),
                        positive_object_vertex_count=len(interval_ids["object"]),
                        region_count=len(regions),
                    )
                )
            if next_frame != frame_count:
                raise ValueError(f"{hand} intervals must cover the complete source timeline")
            for surface, role in (("human", "hand"), ("object", "object")):
                ids = np.asarray(sorted(unions[surface]), dtype=np.int64)
                strength = np.zeros((frame_count, len(ids)), dtype=np.float32)
                for start, end, prepared in validated:
                    for region in prepared:
                        region_ids, values = region[surface]
                        columns = np.searchsorted(ids, region_ids)
                        strength[start:end, columns] = np.maximum(
                            strength[start:end, columns], values
                        )
                ids.flags.writeable = strength.flags.writeable = False
                fields[f"{side}_{role}_vertex_ids"] = ids
                fields[f"{side}_{role}_strength"] = strength
        result = cls()
        result._spec = copy.deepcopy(spec)
        result._fields = fields
        payload = json.dumps(spec, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
        result._provenance = dict(
            schema=SCHEMA,
            binding=copy.deepcopy(binding),
            canonical_spec_sha256=hashlib.sha256(payload).hexdigest(),
            source_video=video,
            native_hand_membership_validated=segments is not None,
            optimizer_ready=False,
            contains_point_correspondences=False,
            field_semantics="Independent authored regional strength, not calibrated visibility or simultaneous contact",
            overlap_reduction="max",
            id_order="ascending original vertex IDs",
            phase_frame_counts=phase_counts,
            interval_counts=interval_counts,
        )
        return result

    @property
    def provenance(self):
        return copy.deepcopy(self._provenance)

    def export_spec(self):
        return copy.deepcopy(self._spec)

    def compact_fields(self):
        """Independent IDs plus [source frame, vertex] strengths; detached copies."""
        return {key: value.copy() for key, value in self._fields.items()}

    def dense_fields(self):
        """Expand to full original vertex order, keeping human/object axes separate."""
        binding = self._provenance["binding"]
        dense = {}
        for side in _HANDS.values():
            for role, surface in (("hand", "human"), ("object", "object")):
                prefix = f"{side}_{role}"
                field = np.zeros(
                    (binding["frame_count"], binding[f"{surface}_vertex_count"]), dtype=np.float32
                )
                field[:, self._fields[f"{prefix}_vertex_ids"]] = self._fields[f"{prefix}_strength"]
                dense[f"{prefix}_strength"] = field
        return dense

    def export_npz(self, path):
        """Save descriptive compact maps only; never overwrite a previous map."""
        path = Path(path)
        fields = self.compact_fields()
        fields["frame_indices"] = np.arange(
            self._provenance["binding"]["frame_count"], dtype=np.int32
        )
        with path.open("xb") as stream:
            np.savez_compressed(stream, **fields)
        return path
