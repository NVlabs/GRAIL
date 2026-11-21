"""Authored contact establishment while the object is still supported.

This is a trajectory fitting boundary condition, not a physical release model.
All authored pairs are measured again from serialized native geometry.
"""

import math

import numpy as np

from .hoi_stationary import validate_stationary_intervals


def validate_acquisition_reduction(reduction):
    if not isinstance(reduction, str) or reduction not in ("legacy_event", "phase_balanced"):
        raise ValueError("contact_acquisition_reduction must be legacy_event or phase_balanced")
    return reduction


def validate_acquisition_events(events, frame_count=None):
    if not isinstance(events, list):
        raise ValueError("contact_acquisition_events must be a list")
    if frame_count is not None and (
        isinstance(frame_count, bool) or not isinstance(frame_count, int) or frame_count < 1
    ):
        raise ValueError("frame_count must be a positive integer")
    result = []
    for event in events:
        if (
            not isinstance(event, dict)
            or not {"side", "start_frame", "end_frame", "max_gap_m"} <= set(event)
            or set(event) - {"side", "start_frame", "end_frame", "max_gap_m", "hold_until_frame"}
        ):
            raise ValueError("Acquisition requires side/start_frame/end_frame/max_gap_m")
        if event["side"] not in ("left", "right"):
            raise ValueError("Acquisition side must be left or right")
        for key in ("start_frame", "end_frame"):
            value = event[key]
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError("Acquisition frame indices must be nonnegative integers")
        start, end = event["start_frame"], event["end_frame"]
        hold = event.get("hold_until_frame", end)
        if isinstance(hold, bool) or not isinstance(hold, int) or hold < end:
            raise ValueError("hold_until_frame must be an integer at or after acquisition end")
        gap = event["max_gap_m"]
        if (
            isinstance(gap, bool)
            or not isinstance(gap, (float, int))
            or not math.isfinite(gap)
            or gap <= 0
        ):
            raise ValueError("Acquisition max_gap_m must be positive and finite")
        if start > end or (frame_count is not None and hold >= frame_count):
            raise ValueError("Acquisition interval exceeds source timeline")
        if any(
            e["side"] == event["side"]
            and start <= e.get("hold_until_frame", e["end_frame"])
            and e["start_frame"] <= hold
            for e in result
        ):
            raise ValueError("Acquisition intervals overlap for one hand")
        result.append(dict(event))
    return result


class ContactAcquisition:
    def __init__(
        self, review, events, stationary_intervals, human_faces=None, *, reduction="legacy_event"
    ):
        self.review = review
        self.reduction = validate_acquisition_reduction(reduction)
        self.human_faces = (
            human_faces.detach().cpu().numpy() if hasattr(human_faces, "detach") else human_faces
        )
        self.events = validate_acquisition_events(events, review.frame_count)
        intervals = validate_stationary_intervals(stationary_intervals, review.frame_count)
        self.phases = []
        for event in self.events:
            start, end = event["start_frame"], event["end_frame"]
            phases = [
                p
                for p in review.phases
                if p.side == event["side"]
                and p.state == "contact"
                and p.start_frame
                <= start
                <= end
                <= event.get("hold_until_frame", end)
                <= p.end_frame
            ]
            if len(phases) != 1 or not phases[0].pairs:
                raise ValueError("Acquisition must lie within one reviewed contact phase")
            if not any(i["start_frame"] <= start <= end <= i["end_frame"] for i in intervals):
                raise ValueError("Acquisition requires hard stationary coverage")
            self.phases.append(phases[0])

    def reduction_windows(self, event):
        """Disjoint optimization windows; measurement always retains the full hold.

        The phase-balanced option prevents a long held phase and its global
        worst sample from numerically diluting a short stationary acquisition.
        It changes objective reduction only, never timing or the contact gate.
        """
        start, end = event["start_frame"], event["end_frame"]
        hold = event.get("hold_until_frame", end)
        if self.reduction == "legacy_event":
            return [("event", start, hold)]
        windows = [("acquisition", start, end)]
        if hold > end:
            windows.append(("hold", end + 1, hold))
        return windows

    def _validate_geometry(self, human, objects):
        if tuple(human.shape) != (
            self.review.frame_count,
            self.review.human_vertex_count,
            3,
        ) or tuple(objects.shape) != (self.review.frame_count, self.review.object_vertex_count, 3):
            raise ValueError("Acquisition geometry disagrees with reviewed topology/timeline")

    def loss(
        self, human, objects, normal_weight_m2=0.0, opposition_cos=0.2, phase_loss_transform=None
    ):
        import torch

        if not torch.is_tensor(human) or not torch.is_tensor(objects):
            raise ValueError("Acquisition loss requires tensors")
        self._validate_geometry(human, objects)
        if (
            human.device != objects.device
            or human.dtype != objects.dtype
            or not human.is_floating_point()
        ):
            raise ValueError("Acquisition geometry needs matching floating dtype/device")
        validate_normal_options(normal_weight_m2, opposition_cos)
        if normal_weight_m2 > 0 and self.human_faces is None:
            raise ValueError("Normal preference requires original human faces")
        losses = []
        for event, phase in zip(self.events, self.phases):
            ids = [p.human_vertex_id for p in phase.pairs]
            faces = torch.tensor(
                [self.review.object_faces[p.object_face_id] for p in phase.pairs],
                device=human.device,
            )
            bary = human.new_tensor([p.object_barycentric for p in phase.pairs])
            scale = max(p.weight for p in phase.pairs)
            weights = human.new_tensor([p.weight / scale for p in phase.pairs])
            window_values = []
            for _, start, end in self.reduction_windows(event):
                sl = slice(start, end + 1)
                targets = (objects[sl][:, faces] * bary[None, :, :, None]).sum(-2)
                squared = (human[sl][:, ids] - targets).square().sum(-1)
                value = 0.5 * (
                    (squared * weights).sum() / (squared.shape[0] * weights.sum()) + squared.max()
                )
                if normal_weight_m2 > 0:
                    from .hoi_contact_normals import material_normal_loss

                    value = value + normal_weight_m2 * material_normal_loss(
                        human[sl],
                        objects[sl],
                        self.human_faces,
                        self.review.object_faces,
                        phase.pairs,
                        opposition_cos,
                    )
                window_values.append(value)
            # Preserve the exact previous arithmetic/gradient when there is one
            # window, including phase-balanced events without a later hold.
            value = (
                window_values[0] if len(window_values) == 1 else torch.stack(window_values).mean()
            )
            if phase_loss_transform is not None:
                value = phase_loss_transform(value, phase.side)
            losses.append(value)
        return torch.stack(losses).mean() if losses else human.sum() * 0 + objects.sum() * 0

    def measure(self, human, objects, opposition_cos=None):
        if opposition_cos is not None:
            validate_normal_options(1.0, opposition_cos)
            if self.human_faces is None:
                raise ValueError("Normal guard requires original human faces")
        human, objects = np.asarray(human), np.asarray(objects)
        self._validate_geometry(human, objects)
        if not np.isfinite(human).all() or not np.isfinite(objects).all():
            raise ValueError("Acquisition geometry must be finite")
        records = []
        for event, phase in zip(self.events, self.phases):
            sl = slice(event["start_frame"], event.get("hold_until_frame", event["end_frame"]) + 1)
            ids = [p.human_vertex_id for p in phase.pairs]
            faces = np.asarray([self.review.object_faces[p.object_face_id] for p in phase.pairs])
            bary = np.asarray([p.object_barycentric for p in phase.pairs])
            targets = (objects[sl][:, faces] * bary[None, :, :, None]).sum(-2)
            distances = np.linalg.norm(human[sl][:, ids] - targets, axis=-1)
            maximum = float(distances.max())
            record = {
                **event,
                "human_vertex_ids": ids,
                "distances_m": distances.tolist(),
                "max_observed_gap_m": maximum,
                "satisfied": maximum <= event["max_gap_m"],
            }
            record["optimization_reduction_windows"] = [
                {"phase": name, "start_frame": start, "end_frame": end}
                for name, start, end in self.reduction_windows(event)
            ]
            if opposition_cos is not None:
                import torch

                from .hoi_contact_normals import material_normal_dots

                dots = material_normal_dots(
                    torch.as_tensor(human[sl]),
                    torch.as_tensor(objects[sl]),
                    self.human_faces,
                    self.review.object_faces,
                    phase.pairs,
                ).numpy()
                normal_pass = bool((dots <= -opposition_cos).all())
                record.update(
                    normal_dots=dots.tolist(),
                    required_opposition_cos=opposition_cos,
                    normals_satisfied=normal_pass,
                    distance_satisfied=record["satisfied"],
                )
                record["satisfied"] = record["satisfied"] and normal_pass
            records.append(record)
        return {
            "all_satisfied": all(e["satisfied"] for e in records),
            "event_count": len(records),
            "events": records,
            "reduction": self.reduction,
            "measurement_policy": "all authored pairs in every acquisition and held frame; unchanged by reduction",
        }


def validate_normal_options(normal_weight_m2=0.0, opposition_cos=0.2):
    for name, value in (("normal_weight_m2", normal_weight_m2), ("opposition_cos", opposition_cos)):
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value < 0
        ):
            raise ValueError(f"{name} must be finite and nonnegative")
    if opposition_cos > 1:
        raise ValueError("opposition_cos must lie within [0,1]")
