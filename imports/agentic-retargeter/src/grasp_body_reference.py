"""Fixed, explicitly mapped body targets for mounted grasp refinement.

Targets are supplied in native world coordinates; this module does not infer
human/robot correspondences or substitute the changing optimization seed.
"""
from pathlib import Path
import copy
import re

import numpy as np
from scipy.spatial.transform import Rotation

from .contact_edit import native_io
from .contact_edit.pose import _finite, _keys, _sha
from .grasp_io import digest


class BodyReference:
    def __init__(self, context, config, directory="."):
        _keys(config, {"npz", "sha256", "evidence_kind", "source"}, set(), "body_reference")
        if config["evidence_kind"] not in {"mapped_human", "native_reference"}:
            raise ValueError("body_reference evidence_kind must distinguish mapped_human/native_reference")
        source = config["source"]
        _keys(source, {"path", "sha256"}, set(), "body reference source")
        self.files = []
        for path, expected in ((config["npz"], config["sha256"]), (source["path"], source["sha256"])):
            if not isinstance(expected, str) or not re.fullmatch(r"[0-9a-f]{64}", expected):
                raise ValueError("Body reference requires lowercase SHA256 bindings")
            resolved = (Path(directory) / path).resolve(strict=True)
            if _sha(resolved) != expected:
                raise ValueError("Body reference hash mismatch")
            self.files.append((resolved, expected))
        data = native_io.load_npz(self.files[0][0])
        for key, expected in (("units", "m"), ("world_frame", "native_world"),
                              ("source_sha256", source["sha256"]),
                              ("model_binding_sha256", digest(context.binding["model_dependencies"]))):
            if key not in data or data[key].shape != () or str(data[key].item()) != expected:
                raise ValueError("Invalid body reference " + key)
        n = len(context.q)
        fps = np.asarray(data.get("fps"))
        if fps.shape != () or float(fps) != float(context.native["fps"]):
            raise ValueError("Body reference FPS differs from native input")
        times = context.native.get("timestamps", np.arange(n) / float(context.native["fps"]))
        if not np.array_equal(data.get("timestamps"), times):
            raise ValueError("Body reference timestamps differ; align targets explicitly upstream")
        names = data.get("body_names")
        if (names is None or names.ndim != 1 or names.dtype.kind not in "US"
                or not len(names) or len(set(names.tolist())) != len(names)):
            raise ValueError("Body reference requires unique named robot bodies")
        self.model = context.model
        self.names = names.tolist()
        import mujoco
        self.bodies = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name) for name in self.names]
        if any(body <= 0 for body in self.bodies):
            raise ValueError("Body reference must name existing non-world robot bodies")
        size = len(names)
        self.positions = _finite(data.get("positions"), (n, size, 3), "body reference positions").copy()
        self.rotations = _finite(data.get("rotations"), (n, size, 3, 3), "body reference rotations").copy()
        if (not np.allclose(self.rotations.swapaxes(-1, -2) @ self.rotations, np.eye(3), atol=1e-6)
                or not np.allclose(np.linalg.det(self.rotations), 1, atol=1e-6)):
            raise ValueError("Body reference rotations must be proper rotations")
        self.position_weights = _finite(data.get("position_weights"), (size,), "position_weights").copy()
        self.rotation_weights = _finite(data.get("rotation_weights"), (size,), "rotation_weights").copy()
        if (np.any(self.position_weights < 0) or np.any(self.rotation_weights < 0)
                or not np.any(self.position_weights + self.rotation_weights > 0)):
            raise ValueError("Body reference requires nonnegative weights with an active objective")
        for array in (self.positions, self.rotations, self.position_weights, self.rotation_weights):
            array.setflags(write=False)
        self.binding = dict(config=copy.deepcopy(config), npz=dict(path=str(self.files[0][0]), sha256=self.files[0][1]),
                            source=dict(path=str(self.files[1][0]), sha256=self.files[1][1]),
                            evidence_kind=config["evidence_kind"], body_names=self.names,
                            position_weights=self.position_weights.tolist(),
                            rotation_weights=self.rotation_weights.tolist(),
                            world_frame="native_world", units="m", motion_accepted=False,
                            limitations=["Mapped targets must be prepared using the bound human/robot calibration; no mapping is inferred.",
                                         "Soft body tracking is not a source-fidelity, support or collision acceptance gate."])
        self.check_unchanged()

    def evaluate(self, data, frame, dofs, columns, jacobian=False):
        import mujoco
        rows, residuals = [], []
        for index, body in enumerate(self.bodies):
            rotation = data.xmat[body].reshape(3, 3)
            pw, rw = self.position_weights[index], self.rotation_weights[index]
            if not jacobian:
                residuals.extend(pw * (data.xpos[body] - self.positions[frame, index]))
                residuals.extend((rw * (rotation - self.rotations[frame, index])).ravel())
                continue
            jp, jr = np.zeros((3, self.model.nv)), np.zeros((3, self.model.nv))
            mujoco.mj_jac(self.model, data, jp, jr, data.xpos[body], body)
            position = np.zeros((3, columns)); position[:, :len(dofs)] = pw * jp[:, dofs]
            orientation = np.zeros((9, columns))
            for column, dof in enumerate(dofs):
                # Angular Jacobian is world-space: dR/dq = [omega]_x R.
                orientation[:, column] = (rw * np.cross(jr[:, dof], rotation.T).T).ravel()
            rows.extend((position, orientation))
        return np.vstack(rows) if jacobian else np.asarray(residuals)

    def diagnostics(self, data, frame):
        rows = []
        for index, body in enumerate(self.bodies):
            rotation = data.xmat[body].reshape(3, 3)
            rows.append(dict(body=self.names[index],
                             position_error_m=float(np.linalg.norm(data.xpos[body] - self.positions[frame, index])),
                             rotation_error_rad=float(Rotation.from_matrix(rotation @ self.rotations[frame, index].T).magnitude())))
        return dict(evidence_kind=self.binding["evidence_kind"], frame=frame, bodies=rows,
                    motion_accepted=False)

    def check_unchanged(self):
        if any(_sha(path) != expected for path, expected in self.files):
            raise ValueError("Body reference changed during refinement")
