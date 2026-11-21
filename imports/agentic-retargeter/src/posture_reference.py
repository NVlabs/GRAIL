"""Bound native posture guidance and a separately scoped deviation gate.

This compares native hinge angles, not human/robot perceptual similarity. The
reference supplies neither the edit corridor nor object/scene transforms.
"""
import copy
import re
from pathlib import Path

import numpy as np

from .contact_edit import native_io
from .contact_edit.pose import _keys, _sha
from .motion_diagnostics import timestamps


class PostureReference:
    def __init__(self, context, config, directory="."):
        _keys(config, {"native_npz", "sha256", "window", "joint_deviation_limits_rad"},
              set(), "posture reference")
        self.config = copy.deepcopy(config)
        if (not isinstance(config["native_npz"], str) or not config["native_npz"]
                or not isinstance(config["sha256"], str)
                or not re.fullmatch(r"[0-9a-f]{64}", config["sha256"])):
            raise ValueError("Posture reference requires a native path and lowercase SHA256")
        self.path = (Path(directory) / config["native_npz"]).resolve(strict=True)
        if _sha(self.path) != config["sha256"]:
            raise ValueError("Posture reference hash mismatch")
        native = native_io.load_npz(self.path)
        self.qpos = native_io._native_qpos(native).copy()
        if self.qpos.shape != context.parent["qpos"].shape:
            raise ValueError("Posture reference native dimensions differ")
        if not ("dof_names" in native or "joint_names" in native):
            raise ValueError("Posture reference requires native joint-order metadata")
        native_io._validate_native_binding(native, native_io._model_joints(context.model_snapshot))
        compiled = np.asarray(native.get("compiled_model_sha256", ""))
        if compiled.shape != () or str(compiled.item()) != context.compiled_digest:
            raise ValueError("Posture reference compiled model binding differs or is missing")
        if (float(native["fps"]) != context.fps
                or not np.array_equal(timestamps(native), timestamps(context.parent))):
            raise ValueError("Posture reference timing differs; align explicitly upstream")
        window = config["window"]
        if (not isinstance(window, list) or len(window) != 2
                or any(type(f) is not int for f in window)
                or not 0 <= window[0] <= window[1] < context.T):
            raise ValueError("Posture reference window must be inclusive source frame indices")
        self.frames = np.arange(window[0], window[1] + 1)
        limits = config["joint_deviation_limits_rad"]
        if not isinstance(limits, dict) or not limits:
            raise ValueError("Posture reference requires explicit named joint deviation limits")
        joints = {j["joint"]: j for j in native_io._model_joints(context.model_snapshot)}
        self.joints = []
        for name, limit in limits.items():
            if name not in joints or joints[name]["joint_type"] != 3:
                raise ValueError("Posture reference must name native hinges: " + str(name))
            if type(limit) not in (int, float) or not np.isfinite(limit) or limit < 0:
                raise ValueError("Posture deviation limits must be finite nonnegative radians")
            self.joints.append(dict(joint=name, address=joints[name]["address"], limit=float(limit)))
        self.qpos.setflags(write=False)
        self.binding = dict(path=str(self.path), sha256=config["sha256"])
        context.bindings["posture_reference"] = self.binding.copy()
        context.code_bindings[str(Path(__file__))] = _sha(Path(__file__))

    def target(self, parent_qpos):
        """Change only explicitly declared reference joints/frames in the prior."""
        result = parent_qpos.copy()
        addresses = [j["address"] for j in self.joints]
        index = np.ix_(self.frames, addresses)
        result[index] = self.qpos[index]
        return result

    def evaluate(self, qpos):
        qpos = np.asarray(qpos)
        if qpos.shape != self.qpos.shape or not np.isfinite(qpos).all():
            raise ValueError("Invalid candidate for posture reference gate")
        rows = []
        for joint in self.joints:
            difference = np.abs(qpos[self.frames, joint["address"]]
                                - self.qpos[self.frames, joint["address"]])
            roundoff = 1e-8 * max(1., joint["limit"])
            failures = self.frames[difference > joint["limit"] + roundoff]
            rows.append(dict(joint=joint["joint"], qpos_address=joint["address"],
                             max_deviation_rad=float(difference.max()),
                             rms_deviation_rad=float(np.sqrt(np.mean(difference**2))),
                             peak_frame=int(self.frames[int(difference.argmax())]),
                             limit_rad=joint["limit"], roundoff_rad=roundoff,
                             violating_frames=failures.tolist(), passed=not len(failures)))
        passed = all(row["passed"] for row in rows)
        return dict(configured=True, status="PASS" if passed else "FAIL", passed=passed,
                    motion_accepted=False, selected=False, configuration=self.config,
                    binding=self.binding, scope="all declared native hinges and integer frames, including frozen samples",
                    source_frames=self.frames.tolist(), per_joint=rows,
                    violation_count=sum(len(row["violating_frames"]) for row in rows),
                    limitations=["Native joint-angle deviation without angle unwrapping; not human similarity or source visual acceptance.",
                                 "No object trajectory, contact, collision or continuous subframe certificate."])
