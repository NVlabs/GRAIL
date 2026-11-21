"""Differentiable latent contact between source-predicted surface regions.

Only canonical IDs, canonical triangle barycentrics, anatomical areas and source
fields are read. Saved world coordinates and nearest-geometry diagnostics are
never targets. Distances use metres; returned losses use square metres.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class DenseContactConfig:
    max_object_samples: int = 192
    field_threshold: float = 0.01
    gap_tolerance_m: float = 0.001
    robust_delta_m: float = 0.010
    softmin_temperature_m2: float = 0.000016
    normal_distance_scale_m: float = 0.005
    min_opposing_cosine: float = 0.5
    left_weight: float = 0.5
    # Optional reproducibility control. Indices address the ORIGINAL side's
    # released surface-sample array before confidence/eligibility filtering.
    # Omission retains the legacy field-mass FPS start exactly.
    fps_start_surface_index_by_interface: dict[str, int] | None = None
    # Optional shared quadrature universe for matched prediction comparisons.
    # Sorted original side-sample indices may retain zero-mass candidates;
    # confidence is never added. Each supplied interface needs an explicit FPS
    # start so changing fields cannot change geometric representatives.
    candidate_surface_indices_by_interface: dict[str, list[int]] | None = None


def _array(x):
    return x.detach().cpu().numpy() if torch.is_tensor(x) else np.asarray(x)


def canonical_object_hashes(vertices, faces):
    """Stable hashes include canonical ordering, independent of file dtype."""
    return {
        "vertices_sha256": hashlib.sha256(np.asarray(vertices, dtype="<f8").tobytes()).hexdigest(),
        "faces_sha256": hashlib.sha256(np.asarray(faces, dtype="<i8").tobytes()).hexdigest(),
    }


def _interface_specs(meta):
    """An explicit schema cannot silently omit patches or include hand support."""
    patches = meta["patches"]
    by_id = {p["id"]: p for p in patches}
    if len(by_id) != len(patches):
        raise ValueError("Prediction patch IDs must be unique")
    if "optimization_interfaces" not in meta:
        # Preserve the reviewed video03 objective and numerical grouping. Other
        # sequences must declare their own interfaces, not inherit its clasp.
        if meta.get("sequence") in ("rand00001", "rand00002"):
            raise ValueError(
                "Sequence-specific optimization_interfaces are required for videos 01/02"
            )
        names = [p["id"] for p in patches if p["relationship"] == "hand_object"]
        groups = [
            (f"right_{digit}", "right") for digit in ["index", "middle", "ring", "little", "thumb"]
        ]
        groups += [(f"left_{digit}", "left") for digit in ["index", "little"]]
        return (
            [
                dict(name=name, side=side, patches=[n for n in names if n.startswith(name + "_")])
                for name, side in groups
            ],
            "legacy_video03",
            [],
        )
    specs = meta["optimization_interfaces"]
    if not isinstance(specs, list) or not specs:
        raise ValueError("optimization_interfaces must be a nonempty list")
    assigned, names = set(), set()
    for spec in specs:
        if not isinstance(spec, dict) or set(spec) - {
            "name",
            "side",
            "patches",
            "surface_mask_key",
            "surface_dataset",
        }:
            raise ValueError("Invalid optimization interface schema")
        if "surface_dataset" in spec:
            from grail.optimization.dense_contact_surface_dataset import validate_descriptor

            validate_descriptor(spec["surface_dataset"])
            if "surface_mask_key" in spec:
                raise ValueError(
                    "Dedicated face-owned datasets cannot reuse a shared-side surface mask"
                )
        name, side, ids = spec.get("name"), spec.get("side"), spec.get("patches")
        if not isinstance(name, str) or not name or name in names or side not in ("right", "left"):
            raise ValueError("Interface names must be unique and sides must be right or left")
        if not isinstance(ids, list) or not ids:
            raise ValueError("Each interface must name at least one source patch")
        names.add(name)
        for pid in ids:
            if pid not in by_id or pid in assigned:
                raise ValueError(f"Unknown or multiply assigned interface patch: {pid}")
            if by_id[pid]["relationship"] != "hand_object" or by_id[pid]["side"] != side:
                raise ValueError(
                    f"Interface {name} must contain same-side hand_object patches only"
                )
            assigned.add(pid)
    excluded = meta.get("optimization_excluded_patches", [])
    if not isinstance(excluded, list):
        raise ValueError("optimization_excluded_patches must be a list")
    for spec in excluded:
        if not isinstance(spec, dict) or set(spec) != {"patch", "reason"} or not spec["reason"]:
            raise ValueError("Excluded patches require patch and a nonempty reason")
        pid = spec["patch"]
        if pid not in by_id or pid in assigned or by_id[pid]["relationship"] != "hand_object":
            raise ValueError(f"Invalid or multiply assigned excluded patch: {pid}")
        assigned.add(pid)
    omitted = {p["id"] for p in patches if p["relationship"] == "hand_object"} - assigned
    if omitted:
        raise ValueError(f"Explicitly assign or exclude every hand_object patch: {sorted(omitted)}")
    return specs, "explicit", excluded


def _validate_field(name, value, shape):
    if (
        value.shape != shape
        or not np.isfinite(value).all()
        or np.any(value < 0)
        or np.any(value > 1 + 1e-6)
    ):
        raise ValueError(f"{name} must have shape {shape} with finite values in [0,1]")


def _fps(points, weight, count, start_index=None):
    """Deterministic geometric coverage; quadrature mass is reassigned afterwards."""
    if len(points) <= count:
        return np.arange(len(points))
    chosen = [int(np.argmax(weight)) if start_index is None else int(start_index)]
    distance = np.sum((points - points[chosen[0]]) ** 2, axis=1)
    for _ in range(count - 1):
        j = int(np.argmax(distance))
        chosen.append(j)
        distance = np.minimum(distance, np.sum((points - points[j]) ** 2, axis=1))
    return np.asarray(chosen)


def _surface_quadrature(ov, of, points, max_edge=0.002):
    """Real subtriangle nodal areas, evaluated independently of label support.

    A bounding box only removes original faces that cannot touch any sample;
    every retained original triangle is subdivided completely. Summing area/3
    on the auxiliary mesh avoids assigning an entire coarse face to a small
    colored portion. Original geometry/topology is never changed.
    """
    import trimesh
    from scipy.spatial import cKDTree

    triangles = ov[of]
    lo, hi = points.min(0) - max_edge, points.max(0) + max_edge
    eligible = ((triangles.max(1) >= lo) & (triangles.min(1) <= hi)).all(1)
    v, faces = trimesh.remesh.subdivide_to_size(ov, of[eligible], max_edge=max_edge)
    _, first, inverse = np.unique(np.round(v, 7), axis=0, return_index=True, return_inverse=True)
    v, faces = v[first], inverse[faces]
    tri = v[faces]
    area = np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1) / 2
    nodal = np.zeros(len(v), np.float64)
    for k in range(3):
        np.add.at(nodal, faces[:, k], area / 3)
    error, node = cKDTree(v).query(points)
    if error.max() > 2e-6:
        raise ValueError(
            f"Released barycentric samples do not match original auxiliary topology: {error.max()}m"
        )
    return nodal[node], {
        "max_edge_m": max_edge,
        "original_faces": int(eligible.sum()),
        "auxiliary_nodes": len(v),
        "auxiliary_faces": len(faces),
        "auxiliary_surface_area_m2": float(area.sum()),
        "nodal_area_sum_m2": float(nodal.sum()),
        "barycentric_lookup_error_m": float(error.max()),
    }


class _Interface(nn.Module):
    def __init__(
        self, name, side, human_ids, object_triples, barycentric, hand_mass, object_mass, evidence
    ):
        super().__init__()
        self.name, self.side = name, side
        for key, value in [("human_ids", human_ids), ("object_triples", object_triples)]:
            self.register_buffer(key, torch.as_tensor(value, dtype=torch.long))
        for key, value in [
            ("barycentric", barycentric),
            ("hand_mass", hand_mass),
            ("object_mass", object_mass),
            ("evidence", evidence),
        ]:
            self.register_buffer(key, torch.as_tensor(value, dtype=torch.float32))


class DenseRegionContactLoss(nn.Module):
    """Original-topology surface loss with sequence-specific named interfaces.

    ``forward`` accepts dynamic original full mesh vertices or subsets with
    explicit original IDs. Subsets must contain ``required_*_vertex_ids``.
    Object vertices must already contain the fixed object scale. Normals are
    optional; when human normals are supplied, object face normals are computed
    from the current original triangles unless object vertex normals are given.
    """

    def __init__(
        self, interfaces, config=None, provenance=None, object_local_vertices=None, human_faces=None
    ):
        super().__init__()
        self.interfaces = nn.ModuleList(interfaces)
        self.config = config or DenseContactConfig()
        self.provenance = provenance or {}
        self.register_buffer(
            "required_human_vertex_ids", torch.unique(torch.cat([p.human_ids for p in interfaces]))
        )
        self.register_buffer(
            "required_object_vertex_ids",
            torch.unique(torch.cat([p.object_triples.reshape(-1) for p in interfaces])),
        )
        self.register_buffer(
            "object_local_vertices",
            (
                None
                if object_local_vertices is None
                else torch.as_tensor(object_local_vertices, dtype=torch.float32)
            ),
        )
        if human_faces is not None:
            hf = np.asarray(human_faces)
            hf = hf[np.isin(hf, self.required_human_vertex_ids.cpu().numpy()).any(1)]
            self.register_buffer("normal_human_faces", torch.as_tensor(hf, dtype=torch.long))
            self.required_human_vertex_ids = torch.unique(self.normal_human_faces)
        else:
            self.register_buffer("normal_human_faces", None)

    @classmethod
    def from_prediction(
        cls,
        prediction_dir,
        object_local_vertices,
        object_faces,
        *,
        config=None,
        device=None,
        anatomy_path=None,
    ):
        config = config or DenseContactConfig()
        if isinstance(config, dict):
            config = DenseContactConfig(**config)
        if (
            config.max_object_samples < 1
            or config.softmin_temperature_m2 <= 0
            or config.robust_delta_m <= 0
        ):
            raise ValueError("Positive sample count, temperature and robust delta are required")
        if (
            not 0 <= config.field_threshold < 1
            or config.gap_tolerance_m < 0
            or config.left_weight < 0
        ):
            raise ValueError("Invalid field threshold, gap tolerance or left weight")
        if config.normal_distance_scale_m < 0 or not -1 <= config.min_opposing_cosine <= 1:
            raise ValueError("Invalid surface orientation settings")
        d = Path(prediction_dir)
        files = {
            "semantic": d / "prediction/semantic_patch_fields.npz",
            "anatomy": (
                Path(anatomy_path)
                if anatomy_path is not None
                else d / "geometry_audit/dense_anatomical_candidates.npz"
            ),
            "surface": d / "prediction/dense_object_surface.npz",
            "metadata": d / "prediction/prediction_metadata.json",
        }
        meta = json.loads(files["metadata"].read_text())
        groups, interface_mode, exclusions = _interface_specs(meta)
        starts = config.fps_start_surface_index_by_interface
        if starts is None:
            starts = {}
        if not isinstance(starts, dict) or any(
            not isinstance(k, str) or not isinstance(v, int) or isinstance(v, bool) or v < 0
            for k, v in starts.items()
        ):
            raise ValueError(
                "FPS starts must map interface names to nonnegative original surface-sample indices"
            )
        unknown_starts = set(starts) - {spec["name"] for spec in groups}
        if unknown_starts:
            raise ValueError(f"FPS starts name unknown interfaces: {sorted(unknown_starts)}")
        pools = config.candidate_surface_indices_by_interface
        if pools is None:
            pools = {}
        if not isinstance(pools, dict) or any(not isinstance(k, str) for k in pools):
            raise ValueError(
                "Candidate pools must map interface names to sorted original surface-sample indices"
            )
        unknown_pools = set(pools) - {spec["name"] for spec in groups}
        if unknown_pools:
            raise ValueError(f"Candidate pools name unknown interfaces: {sorted(unknown_pools)}")
        for name, indices in pools.items():
            if (
                not isinstance(indices, list)
                or not indices
                or any(not isinstance(v, int) or isinstance(v, bool) or v < 0 for v in indices)
                or any(a >= b for a, b in zip(indices, indices[1:]))
            ):
                raise ValueError(
                    f"Candidate pool for {name} must be a nonempty sorted unique list of nonnegative integer indices"
                )
            if name not in starts:
                raise ValueError(f"Candidate pool for {name} requires an explicit FPS start")
        a = np.load(files["anatomy"], allow_pickle=False)
        s = np.load(files["semantic"], allow_pickle=False)
        surf = np.load(files["surface"], allow_pickle=False)
        ov, of = _array(object_local_vertices), _array(object_faces).astype(np.int64)
        if ov.ndim != 2 or ov.shape[1] != 3 or of.ndim != 2 or of.shape[1] != 3:
            raise ValueError("Expected original object vertices [O,3] and faces [T,3]")
        if not np.isfinite(ov).all() or not len(of) or np.any(of < 0) or np.any(of >= len(ov)):
            raise ValueError(
                "Canonical object geometry contains invalid vertices or triangle indices"
            )
        geometry_hashes = canonical_object_hashes(ov, of)
        declared_geometry = meta.get("canonical_object", {})
        for key, actual in {
            **geometry_hashes,
            "vertex_count": len(ov),
            "face_count": len(of),
        }.items():
            if key in declared_geometry and declared_geometry[key] != actual:
                raise ValueError(f"Canonical object {key} mismatch")
        # Reconstruct canonical sample points from face barycentrics. Auxiliary
        # quadrature is built once over both sides before any field threshold.
        sample_side, sample_points, canonical_error = {}, [], {}
        for side in ["right", "left"]:
            prefix = side + "_object_surface"
            fi, bary = surf[prefix + "_face_ids"], surf[prefix + "_barycentric"]
            if fi.ndim != 1 or fi.dtype.kind not in "iu" or np.any(fi < 0) or np.any(fi >= len(of)):
                raise ValueError(f"{prefix}: invalid original triangle IDs")
            if (
                bary.shape != (len(fi), 3)
                or not np.isfinite(bary).all()
                or np.any(bary < -1e-5)
                or not np.allclose(bary.sum(1), 1, atol=1e-5, rtol=0)
            ):
                raise ValueError(f"{prefix}: invalid triangle barycentrics")
            _validate_field(
                prefix + "_confidence", surf[prefix + "_confidence"], (meta["frame_count"], len(fi))
            )
            points = (ov[of[fi]] * bary[..., None]).sum(1)
            local_key = prefix + "_local_scaled_m"
            if local_key in surf:
                saved_local = surf[local_key]
                if saved_local.shape != points.shape or not np.isfinite(saved_local).all():
                    raise ValueError(f"{prefix}: invalid saved canonical geometry")
                error = float(np.max(np.abs(saved_local - points), initial=0))
                canonical_error[side] = error
                if error > 2e-6:
                    raise ValueError(f"{prefix}: canonical triangle/scale mismatch ({error:g}m)")
            elif interface_mode == "explicit" and not all(
                k in declared_geometry for k in geometry_hashes
            ):
                raise ValueError(
                    "Explicit schema requires canonical sample coordinates or canonical object hashes"
                )
            sample_side[side] = (len(sample_points), len(points))
            sample_points.extend(points)
        if not sample_points:
            raise ValueError("Prediction contains no object surface samples")
        area_all, quadrature_info = _surface_quadrature(ov, of, np.asarray(sample_points))
        interfaces, descriptions, read_keys = [], [], {"anatomy": [], "semantic": [], "surface": []}
        for spec in groups:
            name, side, patches = spec["name"], spec["side"], spec["patches"]
            if not patches:
                continue
            human_key, area_key = side + "_human_vertex_ids", side + "_vertex_area_mm2"
            human_ids = a[human_key]
            hand_area = a[area_key] * 1e-6
            if (
                human_ids.ndim != 1
                or human_ids.dtype.kind not in "iu"
                or len(np.unique(human_ids)) != len(human_ids)
                or np.any(human_ids < 0)
            ):
                raise ValueError(f"{human_key}: expected unique nonnegative original vertex IDs")
            if (
                hand_area.shape != human_ids.shape
                or not np.isfinite(hand_area).all()
                or np.any(hand_area < 0)
            ):
                raise ValueError(f"{area_key}: invalid anatomical surface areas")
            read_keys["anatomy"] += [human_key, area_key]
            hand_field = np.zeros((meta["frame_count"], len(human_ids)), np.float32)
            object_field = np.zeros((meta["frame_count"], len(ov)), np.float32)
            strength = np.zeros(meta["frame_count"], np.float32)
            for p in patches:
                keys = [p + "_hand_field", p + "_strength", p + "_object_field"]
                read_keys["semantic"] += keys
                ph, ps, po = (s[k] for k in keys)
                _validate_field(keys[0], ph, (len(human_ids),))
                _validate_field(keys[1], ps, (meta["frame_count"],))
                _validate_field(keys[2], po, object_field.shape)
                hand_field = np.maximum(hand_field, ph[None] * ps[:, None])
                object_field = np.maximum(object_field, po)
                strength = np.maximum(strength, ps)
            hk = hand_field.max(0) > config.field_threshold
            prefix = side + "_object_surface"
            skeys = [prefix + "_face_ids", prefix + "_barycentric", prefix + "_confidence"]
            dedicated = None
            if "surface_dataset" in spec:
                from grail.optimization.dense_contact_surface_dataset import load_surface_dataset

                dataset_spec = spec["surface_dataset"]
                dedicated = load_surface_dataset(d, dataset_spec, ov, of, declared_geometry)
                files["dataset:" + name] = dedicated["path"]
                fi, bary = dedicated["face_ids"], dedicated["barycentric"]
                tri = of[fi]
                spatial_key, evidence_key = (
                    dataset_spec["spatial_weight_key"],
                    dataset_spec["evidence_key"],
                )
                read_keys["semantic"] += [spatial_key, evidence_key]
                sampled, dedicated_evidence = s[spatial_key], s[evidence_key]
                _validate_field(spatial_key, sampled, (meta["frame_count"], len(fi)))
                _validate_field(evidence_key, dedicated_evidence, (meta["frame_count"],))
                if np.any(dedicated_evidence > strength):
                    raise ValueError("Dedicated evidence exceeds unchanged source patch strength")
                if np.any(sampled[dedicated_evidence == 0] != 0) or np.any(
                    (dedicated_evidence > 0) & (sampled.sum(1) == 0)
                ):
                    raise ValueError(
                        "Dedicated spatial mass and evidence have inconsistent active frames"
                    )
                retained_human_mass = (hand_field[:, hk] * hand_area[None, hk]).sum(1)
                if np.any((dedicated_evidence > 0) & (retained_human_mass <= 0)):
                    raise ValueError(
                        "Positive dedicated evidence requires retained positive human mass after thresholding"
                    )
            else:
                read_keys["surface"] += skeys
                fi, bary, supported = (surf[k] for k in skeys)
                tri = of[fi]
                # Named fields exist on canonical vertices. Linear interpolation onto
                # the released exterior samples preserves names without inventing
                # exact point correspondence. Released merged fields are only a
                # conservative support cap, never world-space targets.
                sampled = (object_field[:, tri] * bary[None]).sum(-1)
                sampled = np.minimum(sampled, supported)
            mask = None
            if "surface_mask_key" in spec:
                mask_key = spec["surface_mask_key"]
                mask = surf[mask_key]
                if mask.dtype.kind != "b" or mask.shape != (len(fi),):
                    raise ValueError(
                        f"{mask_key}: expected a boolean mask matching the side surface samples"
                    )
                read_keys["surface"].append(mask_key)
                sampled = sampled * mask[None]
            ok = (
                np.ones(len(fi), dtype=bool)
                if dedicated is not None
                else sampled.max(0) > config.field_threshold
            )
            if name in pools:
                indices = pools[name]
                if indices[-1] >= len(fi):
                    raise ValueError(
                        f"Candidate pool for {name} contains an out-of-range original side-surface index"
                    )
                if dedicated is not None and indices != list(range(len(fi))):
                    raise ValueError(
                        "Dedicated dataset requires its complete fixed candidate domain"
                    )
                fixed = np.zeros(len(fi), dtype=bool)
                fixed[indices] = True
                if mask is not None and np.any(fixed & ~mask):
                    raise ValueError(
                        f"Candidate pool for {name} violates its semantic surface mask"
                    )
                if np.any(ok & ~fixed):
                    raise ValueError(
                        f"Candidate pool for {name} omits above-threshold positive surface evidence"
                    )
                ok = fixed
            original_indices = np.flatnonzero(ok)
            start_index = 0 if dedicated is not None else None
            if name in starts:
                source_start = starts[name]
                if source_start >= len(ok) or not ok[source_start]:
                    raise ValueError(
                        f"FPS start for {name} is not an eligible original side-surface sample"
                    )
                start_index = int(np.searchsorted(original_indices, source_start))
                if not hk.any():
                    raise ValueError(f"FPS start for {name} has no supported human region")
            fi, bary, tri, sampled = fi[ok], bary[ok], tri[ok], sampled[:, ok]
            if not hk.any() or not len(fi):
                continue
            start, length = sample_side[side]
            area = (
                dedicated["area"][ok]
                if dedicated is not None
                else area_all[start : start + length][ok]
            )
            points = (
                dedicated["points"][ok]
                if dedicated is not None
                else (ov[tri] * bary[..., None]).sum(1)
            )
            max_samples = (
                dataset_spec["max_samples"] if dedicated is not None else config.max_object_samples
            )
            chosen = _fps(points, sampled.max(0) * area, max_samples, start_index)
            # Preserve every original quadrature point's field-weighted mass
            # while limiting GPU cost; duplicate points cannot increase force.
            if dedicated is not None:
                from grail.optimization.dense_contact_surface_dataset import assign_representatives

                assign, dedicated_nearest_distance = assign_representatives(points, points[chosen])
            else:
                assign = np.argmin(
                    np.sum((points[:, None] - points[chosen][None]) ** 2, axis=-1), axis=1
                )
            mass = np.zeros((meta["frame_count"], len(chosen)), np.float64)
            full_mass = sampled * area[None]
            for f in range(meta["frame_count"]):
                np.add.at(mass[f], assign, full_mass[f])
            hmass = hand_field[:, hk] * hand_area[None, hk]
            evidence = (
                dedicated_evidence.copy()
                if dedicated is not None
                else np.minimum(strength, sampled.max(1))
            )
            evidence[(hmass.sum(1) == 0) | (mass.sum(1) == 0)] = 0
            interfaces.append(
                _Interface(
                    name, side, human_ids[hk], tri[chosen], bary[chosen], hmass, mass, evidence
                )
            )
            descriptions.append(
                {
                    "name": name,
                    "side": side,
                    "source_patches": patches,
                    "human_candidates": int(hk.sum()),
                    "object_samples_before_cap": len(fi),
                    "object_samples": len(chosen),
                    "active_frames": np.flatnonzero(evidence > 0).tolist(),
                    "max_evidence": float(evidence.max()),
                    "surface_mask_key": spec.get("surface_mask_key"),
                    "candidate_pool_policy": (
                        "explicit_original_side_surface_indices"
                        if name in pools
                        else "field_threshold"
                    ),
                    "candidate_original_surface_indices": pools.get(name),
                    "fps_start_policy": (
                        "explicit_original_side_surface_index"
                        if name in starts
                        else "legacy_field_mass"
                    ),
                    "fps_subsampling_performed": len(points) > config.max_object_samples,
                    "fps_start_requested_original_surface_index": starts.get(name),
                    "fps_start_original_surface_index": int(original_indices[chosen[0]]),
                    "eligible_original_surface_indices_sha256": hashlib.sha256(
                        np.asarray(original_indices, dtype="<i8").tobytes()
                    ).hexdigest(),
                    "eligible_sample_areas_sha256": hashlib.sha256(
                        np.asarray(area, dtype="<f8").tobytes()
                    ).hexdigest(),
                    "selected_original_surface_indices": original_indices[chosen].tolist(),
                    "sample_area_assignment_sha256": hashlib.sha256(
                        np.asarray(assign, dtype="<i8").tobytes()
                    ).hexdigest(),
                    "sampling_cover_radius_m": (
                        float(dedicated_nearest_distance.max())
                        if dedicated is not None
                        else float(
                            np.sqrt(
                                np.min(
                                    np.sum((points[:, None] - points[chosen][None]) ** 2, axis=-1),
                                    axis=1,
                                )
                            ).max()
                        )
                    ),
                }
            )
            if dedicated is not None:
                radius = descriptions[-1]["sampling_cover_radius_m"]
                if radius > 0.002 + 1e-12:
                    raise ValueError(
                        f"Dedicated dataset FPS exceeds 2mm centroid coverage: {radius:g}m"
                    )
                continuous_bound = np.linalg.norm(
                    dedicated["cell_corners"] - points[chosen][assign, None], axis=-1
                ).max()
                descriptions[-1].update(
                    candidate_pool_policy="complete_dedicated_original_face_owned_dataset",
                    fps_start_policy=(
                        "explicit_original_dataset_sample_index"
                        if name in starts
                        else "dedicated_geometry_start_zero"
                    ),
                    fps_subsampling_performed=len(points) > max_samples,
                    max_samples=max_samples,
                    dedicated_surface_dataset=dedicated["provenance"],
                    spatial_weight_key=spatial_key,
                    evidence_key=evidence_key,
                    continuous_cell_cover_upper_bound_m=float(continuous_bound),
                    evidence_policy="Dedicated E used once outside normalized Q times represented area; unchanged patch strength is an upper bound, not another multiplier.",
                )
        if not interfaces:
            raise ValueError("No supported bilateral interfaces were loaded")
        read_keys["anatomy"].append("human_faces")
        for side in sample_side:
            if side in canonical_error:
                read_keys["surface"].append(side + "_object_surface_local_scaled_m")
        provenance = {
            "schema_version": 2,
            "prediction_dir": str(d),
            "config": asdict(config),
            "input_sha256": {
                str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in files.values()
            },
            "read_array_keys": {k: sorted(set(v)) for k, v in read_keys.items()},
            "interfaces": descriptions,
            "interface_mode": interface_mode,
            "explicit_exclusions": exclusions,
            "canonical_object": {**geometry_hashes, "vertex_count": len(ov), "face_count": len(of)},
            "canonical_sample_max_error_m": canonical_error,
            "old_contact_labels_read": False,
            "saved_world_targets_read": False,
            "fixed_geometry_nearest_diagnostics_read": False,
            "surface_mass_note": "Original raw faces subdivided to <=2mm; subtriangle area/3 accumulated at welded nodes before field masking; then mass-conserving nearest representative aggregation.",
            "surface_quadrature": quadrature_info,
            "confidence_note": "Minimum of strongest temporal evidence and named object field peak remains outside normalized spatial priors. Source confidence already present in semantic object fields is not multiplied again. Localization uncertainty changes the normalized spatial field; its existing mapping quality cap attenuates the single evidence factor.",
        }
        if any("surface_dataset" in spec for spec in groups):
            provenance["dedicated_surface_note"] = (
                "Dedicated interfaces use their explicitly separated E once and normalized Q times face-owned area. Legacy confidence_note and shared welded quadrature apply only to interfaces without surface_dataset."
            )
        result = cls(interfaces, config, provenance, ov, a["human_faces"])
        return result.to(device) if device is not None else result

    @staticmethod
    def _remap(required, supplied):
        if supplied is None:
            return required
        supplied = torch.as_tensor(supplied, device=required.device, dtype=torch.long)
        if torch.unique(supplied).numel() != supplied.numel():
            raise ValueError("Subset canonical IDs must be unique")
        order = torch.argsort(supplied)
        sorted_ids = supplied[order]
        ix = torch.searchsorted(sorted_ids, required)
        if torch.any(ix >= len(sorted_ids)) or not torch.equal(sorted_ids[ix], required):
            raise ValueError("A required original vertex ID is missing from the provided subset")
        return order[ix]

    def forward(
        self,
        human_vertices,
        object_vertices=None,
        frame_indices=None,
        *,
        object_R=None,
        object_t=None,
        human_vertex_ids=None,
        object_vertex_ids=None,
        human_normals=None,
        object_normals=None,
        return_details=False,
    ):
        if object_vertices is None:
            if self.object_local_vertices is None or object_R is None or object_t is None:
                raise ValueError(
                    "Provide object_vertices or object_R/object_t with canonical local geometry"
                )
            object_vertices = (
                self.object_local_vertices.to(human_vertices.dtype)[None]
                @ object_R.transpose(-1, -2)
                + object_t[:, None]
            )
            if object_vertex_ids is not None:
                raise ValueError(
                    "object_R/object_t produce full canonical vertices; omit object_vertex_ids"
                )
        if human_vertices.ndim != 3 or object_vertices.ndim != 3:
            raise ValueError("Vertices must be [frames, vertices, 3] in metres")
        if len(human_vertices) != len(object_vertices):
            raise ValueError("Human and object batches need the same frames")
        if human_vertices.shape[-1] != 3 or object_vertices.shape[-1] != 3:
            raise ValueError("Vertex coordinates need three components")
        if (
            human_vertex_ids is None
            and int(self.required_human_vertex_ids.max()) >= human_vertices.shape[1]
        ):
            raise ValueError(
                "Human mesh lacks original canonical vertex IDs; provide an explicitly mapped subset"
            )
        if object_vertex_ids is None:
            if self.object_local_vertices is not None and object_vertices.shape[1] != len(
                self.object_local_vertices
            ):
                raise ValueError(
                    "Object mesh topology differs; provide original canonical vertices or an explicitly mapped subset"
                )
            if int(self.required_object_vertex_ids.max()) >= object_vertices.shape[1]:
                raise ValueError("Object mesh lacks original canonical vertex IDs")
        cfg, dtype, device = self.config, human_vertices.dtype, human_vertices.device
        frames = (
            torch.arange(len(human_vertices), device=device)
            if frame_indices is None
            else torch.as_tensor(frame_indices, device=device, dtype=torch.long)
        )
        if len(frames) != len(human_vertices):
            raise ValueError("frame_indices must identify each input source frame")
        if torch.any(frames < 0) or torch.any(frames >= self.interfaces[0].evidence.shape[0]):
            raise ValueError("frame_indices are outside the original source timeline")
        if (
            human_normals is None
            and self.normal_human_faces is not None
            and cfg.normal_distance_scale_m > 0
        ):
            nf = self._remap(self.normal_human_faces.flatten(), human_vertex_ids).reshape_as(
                self.normal_human_faces
            )
            tri = human_vertices[:, nf]
            face_normal = torch.cross(
                tri[:, :, 1] - tri[:, :, 0], tri[:, :, 2] - tri[:, :, 0], dim=-1
            )
            accum = torch.zeros_like(human_vertices)
            for corner in range(3):
                accum = accum.index_add(1, nf[:, corner], face_normal)
            human_normals = F.normalize(accum, dim=-1)
        zero = human_vertices.sum() * 0 + object_vertices.sum() * 0
        sides = {"right": zero, "left": zero}
        counts = {side: sum(p.side == side for p in self.interfaces) for side in sides}
        details = {}
        for p in self.interfaces:
            evidence = p.evidence[frames].to(dtype)
            active = evidence > 0
            if not bool(active.any()):
                if return_details:
                    details[p.name] = {"active": active.detach(), "evidence": evidence.detach()}
                continue
            hi = self._remap(p.human_ids, human_vertex_ids)
            oi = self._remap(p.object_triples.flatten(), object_vertex_ids).reshape_as(
                p.object_triples
            )
            h = human_vertices[:, hi]
            tri = object_vertices[:, oi]
            bary = p.barycentric.to(dtype)
            o = (tri * bary[None, ..., None]).sum(2)
            # Inactive rows receive a normalized harmless prior; evidence makes
            # their cost and gradient exactly zero without log(0)/NaN branches.
            hm, om = p.hand_mass[frames].to(dtype), p.object_mass[frames].to(dtype)
            hm = torch.where(active[:, None], hm, torch.ones_like(hm))
            om = torch.where(active[:, None], om, torch.ones_like(om))
            hp = hm / hm.sum(-1, keepdim=True).clamp_min(torch.finfo(dtype).tiny)
            op = om / om.sum(-1, keepdim=True).clamp_min(torch.finfo(dtype).tiny)
            d = torch.linalg.vector_norm(h[:, :, None] - o[:, None], dim=-1)
            residual = F.relu(d - cfg.gap_tolerance_m)
            cost = cfg.robust_delta_m**2 * (
                torch.sqrt(1 + (residual / cfg.robust_delta_m) ** 2) - 1
            )
            opposition = None
            if human_normals is not None:
                hn = F.normalize(human_normals[:, hi], dim=-1)
                if object_normals is None:
                    on = F.normalize(
                        torch.cross(
                            tri[:, :, 1] - tri[:, :, 0], tri[:, :, 2] - tri[:, :, 0], dim=-1
                        ),
                        dim=-1,
                    )
                else:
                    on = F.normalize((object_normals[:, oi] * bary[None, ..., None]).sum(2), dim=-1)
                opposition = -torch.einsum("fhi,foi->fho", hn, on)
                cost = (
                    cost
                    + cfg.normal_distance_scale_m**2
                    * F.relu(cfg.min_opposing_cosine - opposition).square()
                )
            logprior = torch.where(
                hp[:, :, None] * op[:, None] > 0,
                hp[:, :, None].clamp_min(torch.finfo(dtype).tiny).log()
                + op[:, None].clamp_min(torch.finfo(dtype).tiny).log(),
                torch.full_like(cost, -torch.inf),
            )
            logits = logprior - cost / cfg.softmin_temperature_m2
            region_cost = -cfg.softmin_temperature_m2 * torch.logsumexp(logits.flatten(1), dim=-1)
            weighted = evidence * region_cost
            sides[p.side] = sides[p.side] + weighted.mean() / counts[p.side]
            if return_details:
                with torch.no_grad():
                    posterior = torch.softmax(logits.flatten(1), dim=-1).reshape_as(cost)
                    hmarginal, omarginal = posterior.sum(2), posterior.sum(1)
                    rec = {
                        "active": active.detach(),
                        "evidence": evidence.detach(),
                        "robust_latent_cost_m2": region_cost.detach(),
                        "posterior_mean_gap_m": (posterior * d).sum((1, 2)),
                        "minimum_gap_m": torch.where(
                            hp[:, :, None] * op[:, None] > 0, d, torch.full_like(d, torch.inf)
                        )
                        .flatten(1)
                        .min(1)
                        .values,
                        "human_effective_sample_count": 1 / hmarginal.square().sum(1),
                        "object_effective_sample_count": 1 / omarginal.square().sum(1),
                        "max_pair_posterior": posterior.flatten(1).max(1).values,
                        "prior_mass_gap_le_3mm": ((d <= 0.003) * hp[:, :, None] * op[:, None]).sum(
                            (1, 2)
                        ),
                    }
                    if opposition is not None:
                        rec["posterior_opposing_cosine"] = (posterior * opposition).sum((1, 2))
                        rec["posterior_gap_normal_pass_mass"] = (
                            posterior * (d <= 0.003) * (opposition >= cfg.min_opposing_cosine)
                        ).sum((1, 2))
                    details[p.name] = rec
        return {
            "total": sides["right"] + cfg.left_weight * sides["left"],
            "right": sides["right"],
            "left": sides["left"],
            "interfaces": details,
        }
