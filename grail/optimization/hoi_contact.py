"""Visually reviewed material correspondences for HOI optimization.

The reviewer authors contact hypotheses *before* optimization. A pair keeps one
human vertex and one barycentric point on an original object triangle throughout
its inclusive source-frame phase. These are hypotheses supported by visible
evidence, not measurements of hidden 3-D truth. There is no nearest-point search,
contact discovery, distance gate, or fallback to legacy contact labels here.

JSON uses ``schema_version: 1`` and binds the source video, original scaled object
mesh, human topology, case, frame count and fps. Both hands must be labeled once
at every source frame. Only ``contact`` phases carry attraction pairs; unknown,
no_contact and body_contact phases never attract the hand to the object.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from numbers import Integral, Real
from pathlib import Path

import numpy as np

from grail.core.contact import mesh_digest

_STATES = frozenset({"contact", "no_contact", "unknown", "body_contact"})


def _integer(value, name, *, minimum=0):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be an integer")
    if value < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    return int(value)


def _number(value, name, *, positive=False):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite number")
    try:
        value = float(value)
    except (OverflowError, ValueError) as error:
        raise ValueError(f"{name} must be a finite number") from error
    if not math.isfinite(value) or (value <= 0 if positive else value < 0):
        qualifier = "positive" if positive else "nonnegative"
        raise ValueError(f"{name} must be finite and {qualifier}")
    return value


def _text(value, name):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string")
    return value


def _digest(value, name):
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(c not in "0123456789abcdef" for c in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _fields(data, required, optional, name):
    if not isinstance(data, dict):
        raise ValueError(f"{name} must be a JSON object")
    missing, extra = required - data.keys(), data.keys() - required - optional
    if missing or extra:
        raise ValueError(f"{name} fields: missing={sorted(missing)}, unexpected={sorted(extra)}")


def _faces(faces, vertex_count, name):
    raw = np.asarray(faces)
    if raw.ndim != 2 or raw.shape[1] != 3 or raw.dtype.kind not in "iu" or not len(raw):
        raise ValueError(f"{name} must be nonempty integer (F, 3)")
    if raw.min() < 0 or raw.max() >= vertex_count:
        raise ValueError(f"{name} must index the original vertices")
    return np.asarray(raw, dtype=np.int64)


def topology_digest(faces, vertex_count):
    """Hash human vertex count and exact face order/winding, without posed geometry."""
    count = _integer(vertex_count, "human_vertex_count", minimum=1)
    faces = _faces(faces, count, "human_faces")
    digest = hashlib.sha256(b"grail-human-topology-v1\0")
    digest.update(np.asarray([count, *faces.shape], dtype="<i8").tobytes())
    digest.update(np.asarray(faces, dtype="<i8").tobytes())
    return digest.hexdigest()


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON field: {key}")
        result[key] = value
    return result


def _invalid_constant(value):
    raise ValueError(f"Nonfinite JSON constant: {value}")


@dataclass(frozen=True)
class ContactPair:
    human_vertex_id: int
    object_face_id: int
    object_barycentric: tuple[float, float, float]
    weight: float
    tolerance_m: float = 0.0


@dataclass(frozen=True)
class ContactPhase:
    side: str
    start_frame: int
    end_frame: int
    state: str
    evidence_frames: tuple[int, ...]
    evidence: str
    pairs: tuple[ContactPair, ...]


@dataclass(frozen=True)
class ReviewedContact:
    case_id: str
    source_sha256: str
    object_mesh_sha256: str
    human_topology_sha256: str
    frame_count: int
    fps: float
    reviewer: str
    phases: tuple[ContactPhase, ...]
    human_vertex_count: int
    object_vertex_count: int
    object_faces: tuple[tuple[int, int, int], ...]
    contact_path: str
    contact_sha256: str

    @classmethod
    def load(
        cls,
        path,
        *,
        case_id,
        source_sha256,
        object_mesh_sha256,
        human_faces,
        human_vertex_count,
        frame_count,
        fps,
        object_vertices,
        object_faces,
    ):
        """Validate a review against the actual original geometry and source timeline.

        ``object_vertices`` must already include the scale used by HOI. Pass the
        original local mesh, before per-frame object pose transforms. Geometry is
        checked again against ``object_mesh_sha256``; trusting a caller-supplied
        digest alone would allow stale or reordered face IDs to slip through.
        """
        case_id = _text(case_id, "case_id")
        source_sha256 = _digest(source_sha256, "source_sha256")
        object_mesh_sha256 = _digest(object_mesh_sha256, "object_mesh_sha256")
        human_vertex_count = _integer(human_vertex_count, "human_vertex_count", minimum=1)
        frame_count = _integer(frame_count, "frame_count", minimum=1)
        fps = _number(fps, "fps", positive=True)
        human_topology_sha256 = topology_digest(human_faces, human_vertex_count)
        actual_object_digest = mesh_digest(object_vertices, object_faces)
        if actual_object_digest != object_mesh_sha256:
            raise ValueError(
                "Original scaled object mesh changed; review its correspondences again"
            )
        object_vertex_count = len(object_vertices)
        object_faces = _faces(object_faces, object_vertex_count, "object_faces")

        path = Path(path).resolve(strict=True)
        raw = path.read_bytes()
        data = json.loads(raw, object_pairs_hook=_unique_object, parse_constant=_invalid_constant)
        required = {
            "schema_version",
            "case_id",
            "source_sha256",
            "object_mesh_sha256",
            "human_topology_sha256",
            "frame_count",
            "fps",
            "reviewer",
            "phases",
        }
        _fields(data, required, set(), "contact review")
        if _integer(data["schema_version"], "schema_version", minimum=1) != 1:
            raise ValueError("Unsupported contact review schema_version")
        if _text(data["case_id"], "case_id") != case_id:
            raise ValueError("Contact review belongs to another case_id")
        for name, expected in (
            ("source_sha256", source_sha256),
            ("object_mesh_sha256", object_mesh_sha256),
            ("human_topology_sha256", human_topology_sha256),
        ):
            if _digest(data[name], name) != expected:
                raise ValueError(f"Contact review {name} binding changed; review again")
        if _integer(data["frame_count"], "frame_count", minimum=1) != frame_count:
            raise ValueError("Contact review frame_count disagrees with the source timeline")
        if _number(data["fps"], "fps", positive=True) != fps:
            raise ValueError("Contact review fps disagrees with the source timeline")
        reviewer = _text(data["reviewer"], "reviewer")
        if not isinstance(data["phases"], list) or not data["phases"]:
            raise ValueError("phases must be a nonempty JSON array covering both hands")
        phases = tuple(
            cls._phase(phase, frame_count, human_vertex_count, len(object_faces))
            for phase in data["phases"]
        )
        for side in ("left", "right"):
            cursor = 0
            for phase in sorted((p for p in phases if p.side == side), key=lambda p: p.start_frame):
                if phase.start_frame != cursor:
                    raise ValueError(
                        f"{side} phases have a gap or overlap at source frame {cursor}"
                    )
                cursor = phase.end_frame + 1
            if cursor != frame_count:
                raise ValueError(f"{side} phases must cover every source frame exactly once")
        return cls(
            case_id,
            source_sha256,
            object_mesh_sha256,
            human_topology_sha256,
            frame_count,
            fps,
            reviewer,
            phases,
            human_vertex_count,
            object_vertex_count,
            tuple(tuple(int(index) for index in face) for face in object_faces),
            str(path),
            hashlib.sha256(raw).hexdigest(),
        )

    @staticmethod
    def _phase(data, frame_count, human_vertex_count, object_face_count):
        _fields(
            data,
            {"side", "start_frame", "end_frame", "state", "evidence_frames", "evidence"},
            {"pairs"},
            "phase",
        )
        side, state = data["side"], data["state"]
        if not isinstance(side, str) or side not in {"left", "right"}:
            raise ValueError("phase side must be left or right")
        if not isinstance(state, str) or state not in _STATES:
            raise ValueError(f"phase state must be one of {sorted(_STATES)}")
        start = _integer(data["start_frame"], "start_frame")
        end = _integer(data["end_frame"], "end_frame")
        if not start <= end < frame_count:
            raise ValueError(
                "Inclusive phase start_frame/end_frame lie outside the source timeline"
            )
        frames = data["evidence_frames"]
        if not isinstance(frames, list) or not frames:
            raise ValueError("Every phase needs nonempty evidence_frames")
        frames = tuple(_integer(frame, "evidence frame") for frame in frames)
        if len(set(frames)) != len(frames) or any(not start <= frame <= end for frame in frames):
            raise ValueError("evidence_frames must be unique and inside their inclusive phase")
        evidence = _text(data["evidence"], "evidence")
        pairs = data.get("pairs", [])
        if not isinstance(pairs, list):
            raise ValueError("phase pairs must be a JSON array")
        if state == "contact" and not pairs:
            raise ValueError("contact phases require explicit correspondence pairs")
        if state != "contact" and pairs:
            raise ValueError("Only contact phases may carry object-attraction pairs")
        parsed_pairs = []
        for pair in pairs:
            _fields(
                pair,
                {"human_vertex_id", "object_face_id", "object_barycentric", "weight"},
                {"tolerance_m"},
                "contact pair",
            )
            human_id = _integer(pair["human_vertex_id"], "human_vertex_id")
            face_id = _integer(pair["object_face_id"], "object_face_id")
            if human_id >= human_vertex_count or face_id >= object_face_count:
                raise ValueError("contact pair must index an original human vertex and object face")
            bary = pair["object_barycentric"]
            if not isinstance(bary, list) or len(bary) != 3:
                raise ValueError("object_barycentric must contain three finite weights")
            bary = tuple(_number(value, "object_barycentric") for value in bary)
            if not math.isclose(sum(bary), 1.0, rel_tol=0, abs_tol=1e-8):
                raise ValueError("object_barycentric must be nonnegative and sum to one")
            weight = _number(pair["weight"], "weight", positive=True)
            tolerance = _number(pair.get("tolerance_m", 0), "tolerance_m")
            parsed_pairs.append(ContactPair(human_id, face_id, bary, weight, tolerance))
        return ContactPhase(side, start, end, state, frames, evidence, tuple(parsed_pairs))

    @property
    def interaction_start_frame(self):
        return min((p.start_frame for p in self.phases if p.state == "contact"), default=0)

    @property
    def interaction_end_frame(self):
        """Inclusive last authored contact frame, or final source frame if none."""
        return max(
            (p.end_frame for p in self.phases if p.state == "contact"),
            default=self.frame_count - 1,
        )

    def labels_for_frame(self, frame):
        """Display labels only; the correspondence loss does not consume these labels."""
        frame = _integer(frame, "frame")
        if frame >= self.frame_count:
            raise ValueError("frame lies outside the source timeline")
        return {
            phase.side: phase.state
            for phase in self.phases
            if phase.start_frame <= frame <= phase.end_frame
        }

    def report(self):
        """Serializable review, source bindings and inclusive phase accounting."""
        counts = {side: {state: 0 for state in sorted(_STATES)} for side in ("left", "right")}
        for phase in self.phases:
            counts[phase.side][phase.state] += phase.end_frame - phase.start_frame + 1
        return {
            "schema_version": 1,
            "case_id": self.case_id,
            "source_sha256": self.source_sha256,
            "object_mesh_sha256": self.object_mesh_sha256,
            "human_topology_sha256": self.human_topology_sha256,
            "frame_count": self.frame_count,
            "fps": self.fps,
            "reviewer": self.reviewer,
            "phases": [asdict(phase) for phase in self.phases],
            "contact_path": self.contact_path,
            "contact_sha256": self.contact_sha256,
            "state_frame_counts": counts,
            "active_pair_frames": sum(
                (p.end_frame - p.start_frame + 1) * len(p.pairs) for p in self.phases
            ),
            "interaction_start_frame": self.interaction_start_frame,
            "interaction_end_frame": self.interaction_end_frame,
            "interpretation": "Visually reviewed material hypotheses; hidden 3-D contact is not observed truth.",
        }

    def loss(
        self,
        human_verts_seq,
        object_verts_seq,
        *,
        huber_delta_m=0.02,
        ramp_frames=0,
        phase_loss_transform=None,
    ):
        """Weighted mean Huber loss over all authored pair-frame correspondences.

        Inputs are floating ``(source_frames, original_vertices, 3)`` tensors in
        the same world space, metres, dtype and device. The residual is
        ``max(norm(human_vertex - barycentric_object_point) - tolerance_m, 0)``.
        Standard Huber is quadratic up to ``huber_delta_m`` and linear beyond;
        distant authored pairs therefore still pull. Weights normalize across
        the whole source sequence, never by the current distance or a window.
        """
        import torch

        delta = _number(huber_delta_m, "huber_delta_m", positive=True)
        ramp_frames = _integer(ramp_frames, "ramp_frames")
        for tensor, vertex_count, name in (
            (human_verts_seq, self.human_vertex_count, "human_verts_seq"),
            (object_verts_seq, self.object_vertex_count, "object_verts_seq"),
        ):
            if (
                not isinstance(tensor, torch.Tensor)
                or not tensor.is_floating_point()
                or tuple(tensor.shape) != (self.frame_count, vertex_count, 3)
            ):
                raise ValueError(f"{name} must be floating ({self.frame_count}, {vertex_count}, 3)")
        if (
            human_verts_seq.device != object_verts_seq.device
            or human_verts_seq.dtype != object_verts_seq.dtype
        ):
            raise ValueError("Human and object vertices must share dtype and device")
        # Empty sums connect both inputs without reducing all original vertices.
        loss = human_verts_seq[:0].sum() + object_verts_seq[:0].sum()
        contact_phases = [phase for phase in self.phases if phase.state == "contact"]
        if not contact_phases:
            return loss
        # Scaling weights before their sum avoids overflow for large valid weights.
        weight_scale = max(pair.weight for phase in contact_phases for pair in phase.pairs)
        total_weight = 0.0
        device, dtype = human_verts_seq.device, human_verts_seq.dtype
        for phase in contact_phases:
            frames = slice(phase.start_frame, phase.end_frame + 1)
            human_ids = [pair.human_vertex_id for pair in phase.pairs]
            triangles = torch.tensor(
                [self.object_faces[pair.object_face_id] for pair in phase.pairs],
                dtype=torch.long,
                device=device,
            )
            barycentric = torch.tensor(
                [pair.object_barycentric for pair in phase.pairs], dtype=dtype, device=device
            )
            tolerance = torch.tensor(
                [pair.tolerance_m for pair in phase.pairs], dtype=dtype, device=device
            )
            weights = [pair.weight / weight_scale for pair in phase.pairs]
            weight_tensor = torch.tensor(weights, dtype=dtype, device=device)
            object_points = (
                object_verts_seq[frames, triangles] * barycentric[None, :, :, None]
            ).sum(dim=-2)
            distance = torch.linalg.vector_norm(
                human_verts_seq[frames, human_ids] - object_points, dim=-1
            )
            residual = torch.clamp_min(distance - tolerance, 0)
            # Clipping the quadratic branch avoids squaring a far residual that
            # would overflow even though the linear Huber value remains finite.
            quadratic = torch.clamp_max(residual, delta)
            huber = 0.5 * quadratic.square() + delta * (residual - quadratic)
            # Taper only within declared contact. Keep the original denominator:
            # normalizing by tapered mass would cancel the effect on short phases.
            ramp = torch.ones(huber.shape[0], dtype=dtype, device=device)
            if ramp_frames:
                # Equal-pair contiguous annotations form one physical contact
                # run, including when an annotation splits inside the taper.
                start, end = phase.start_frame, phase.end_frame
                changed = True
                while changed:
                    changed = False
                    for adjacent in contact_phases:
                        if adjacent.side != phase.side or adjacent.pairs != phase.pairs:
                            continue
                        if adjacent.end_frame + 1 == start:
                            start = adjacent.start_frame
                            changed = True
                        if end + 1 == adjacent.start_frame:
                            end = adjacent.end_frame
                            changed = True
                index = torch.arange(
                    phase.start_frame, phase.end_frame + 1, dtype=dtype, device=device
                )
                if start > 0:
                    ramp *= 0.5 * (
                        1 - torch.cos(math.pi * ((index - start + 1) / ramp_frames).clamp(max=1))
                    )
                if end < self.frame_count - 1:
                    ramp *= 0.5 * (
                        1 - torch.cos(math.pi * ((end - index + 1) / ramp_frames).clamp(max=1))
                    )
            phase_loss = (huber * weight_tensor * ramp[:, None]).sum()
            if phase_loss_transform is not None:
                phase_loss = phase_loss_transform(phase_loss, phase.side)
            loss = loss + phase_loss
            total_weight += (phase.end_frame - phase.start_frame + 1) * math.fsum(weights)
        return loss / total_weight
