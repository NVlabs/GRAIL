"""Optional source-predicted contact regions for the native HOI optimizer."""

import json
from pathlib import Path

import numpy as np
import torch


class DenseContactAdapter:
    """Keep annotation topology separate from the optimizer's simplified mesh.

    Canonical contact surfaces follow the current object rigid transform. The
    saved reconstruction's world-space points are never attraction targets.
    """

    def __init__(self, data, settings, device, *, human_model=None, cameras=None):
        from grail.optimization.dense_region_contact import DenseRegionContactLoss

        if isinstance(settings, str):
            settings = {"prediction_dir": settings}
        self.settings = dict(settings)
        self.native_input_binding = None
        binding = settings.get("native_input_binding")
        if binding is not None and binding is not False:
            from grail.optimization.dense_native_input_binding import NativeDenseInputBinding

            self.native_input_binding = NativeDenseInputBinding(settings)
        prediction_dir = Path(settings["prediction_dir"])
        geometry_path = Path(
            settings.get(
                "canonical_geometry", prediction_dir / "geometry_audit/fixed014_dense_geometry.npz"
            )
        )
        anatomy_path = Path(
            settings.get(
                "anatomy_path", prediction_dir / "geometry_audit/dense_anatomical_candidates.npz"
            )
        )
        with np.load(geometry_path, allow_pickle=False) as geometry:
            object_vertices = geometry["object_vertices_scaled_m"].copy()
            object_faces = geometry["object_faces"].copy()
        with np.load(anatomy_path, allow_pickle=False) as anatomy:
            human_faces = anatomy["human_faces"]
            if self.native_input_binding is not None:
                anatomy_rest = anatomy["human_rest_vertices_m"].copy()
        actual_faces = data.human.faces.detach().cpu().numpy()
        if not np.array_equal(actual_faces, human_faces):
            raise ValueError("Dense contact map does not match the original human topology")
        metadata = json.loads((prediction_dir / "prediction/prediction_metadata.json").read_text())
        if data.frame_num != metadata["frame_count"]:
            raise ValueError("Dense contact timeline must match the untrimmed input sequence")
        if "expected_object_scale" in settings:
            actual_scale = torch.as_tensor(data.obj.scale).detach().cpu().numpy().reshape(3)
            expected_scale = np.asarray(settings["expected_object_scale"]).reshape(3)
            if not np.allclose(actual_scale, expected_scale, atol=1e-7, rtol=0):
                raise ValueError("Dense contact map and current fixed object scale differ")
        if self.native_input_binding is not None:
            self.native_input_binding.initialize(
                data, human_model, cameras, object_vertices, object_faces, human_faces, anatomy_rest
            )
        self.loss = DenseRegionContactLoss.from_prediction(
            prediction_dir,
            object_vertices,
            object_faces,
            config=settings.get("options", {}),
            device=device,
            anatomy_path=anatomy_path,
        )
        self.last_report = None

    def __call__(self, data, pred, *, human_model=None, cameras=None):
        if self.native_input_binding is not None:
            self.native_input_binding.validate_live(data, human_model, cameras)
        self.last_report = self.loss(
            pred.human.verts_seq,
            object_R=pred.obj.R,
            object_t=pred.obj.trans,
            frame_indices=torch.arange(data.frame_num, device=pred.obj.R.device),
        )
        return self.last_report["total"]
