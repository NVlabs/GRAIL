"""Bounded external local-repair worker shipped with the Blender add-on."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
import traceback
import types

# Source checkout and built add-on use the same relative package imports without
# mutating sys.path or importing an unrelated installed agentic_retargeter.
if not __package__:
    folder = Path(__file__).resolve().parent
    package_name = "_grail_repair_worker_" + __import__("hashlib").sha256(str(folder).encode()).hexdigest()[:16]
    package = types.ModuleType(package_name)
    package.__path__ = [str(folder)]
    package.__file__ = str(folder / "__init__.py")
    sys.modules[package_name] = package
    __package__ = package_name

import numpy as np

from . import repair_bridge as bridge


def _check_model(request, context):
    """Compare all compiled numerical arrays advertised by the visual model."""
    snapshot = bridge.native_io.load_npz(request["model_path"])
    model = context.model
    for key, values in context.model_snapshot.items():
        if key not in snapshot or not np.array_equal(snapshot[key], values):
            raise ValueError("Scene/XML native model mismatch: " + key)
    for key, values in snapshot.items():
        if key in context.model_snapshot:
            continue
        actual = getattr(model, key, None)
        if isinstance(actual, np.ndarray) and not np.array_equal(actual, values):
            raise ValueError("Scene/XML compiled model mismatch: " + key)
        if key in {"nq", "nv", "njnt", "nbody", "ngeom", "nmesh"} and not np.array_equal(values, actual):
            raise ValueError("Scene/XML model dimension mismatch: " + key)
        if key == "compiled_model_sha256" and str(values.item()) != context.compiled_digest:
            raise ValueError("Scene/XML compiled model hash mismatch")
    for key, count, getter in (("body_names", model.nbody, model.body),
                                ("geom_names", model.ngeom, model.geom),
                                ("mesh_names", model.nmesh, model.mesh)):
        if key in snapshot:
            names = np.asarray([getter(i).name for i in range(count)])
            if not np.array_equal(snapshot[key], names):
                raise ValueError("Scene/XML model name mismatch: " + key)


def execute(request_path, *, audit_candidate=None, audit_samples=None, output=None):
    started = time.perf_counter()
    request_path = Path(request_path).resolve(strict=True)
    request = json.loads(request_path.read_text())
    if request_path != Path(request["job_dir"]) / "request.json":
        raise ValueError("Request must be read from its original job directory")
    config = bridge.validate_request(request)
    from .motion_audit import AuditContext
    context = AuditContext(config, Path(request["config_path"]).parent)
    _check_model(request, context)
    root = Path(request["job_dir"])
    if audit_candidate is not None:
        if output is None:
            raise ValueError("Auditing the Blender replay requires a new --output directory")
        target = Path(output).resolve()
        target.mkdir(parents=True, exist_ok=False)
        candidate_path = Path(audit_candidate).resolve(strict=True)
        candidate_sha = bridge.file_hash(candidate_path)
        candidate = bridge.validate_candidate(request, bridge.native_io.load_npz(candidate_path))
        extra = {}
        if audit_samples is not None:
            if "samples_path" not in request:
                raise ValueError("Actual-pose audit requires captured input samples")
            sample_path = Path(audit_samples).resolve(strict=True)
            sample_sha = bridge.file_hash(sample_path)
            provided = bridge.native_io.load_npz(sample_path)
            before = bridge.native_io.load_npz(request["samples_path"])
            times = np.asarray(context.times())
            if (set(provided) != {"source_frames", "qpos"}
                    or not np.array_equal(provided["source_frames"], times)
                    or not np.array_equal(before["source_frames"], times)):
                raise ValueError("Replay samples do not cover every audit source frame")
            poses, original = provided["qpos"], before["qpos"]
            if (poses.dtype.kind != "f" or poses.shape != (len(times), context.model.nq)
                    or not np.isfinite(poses).all()):
                raise ValueError("Invalid actual replay poses")
            joints = bridge.native_io._model_joints(bridge.native_io.load_npz(request["model_path"]))
            max_difference = 0.
            for index, frame in enumerate(times):
                low, high = int(np.floor(frame)), min(int(np.ceil(frame)), context.T - 1)
                editable = context.mask[low] | context.mask[high]
                if not np.array_equal(poses[index, ~editable], original[index, ~editable]):
                    raise ValueError("Replay changed frozen sampled coordinates")
                linear = context.sample(candidate["qpos"], frame)[0]
                parent_linear = context.sample(context.parent["qpos"], frame)[0]
                difference = float(np.max(np.abs(poses[index] - linear)))
                max_difference = max(max_difference, difference)
                if difference > 1e-6:
                    raise ValueError("Replay differs from repaired linear native motion; bake/review curves first")
                if float(frame).is_integer() and not np.array_equal(poses[index], candidate["qpos"][int(frame)]):
                    raise ValueError("Replay integer sample differs from stored candidate")
                alpha = float(frame) - low
                for name, address in zip(context.names, context.qids):
                    support = ((1 - alpha) * context.mask[low, address]
                               + alpha * context.mask[high, address])
                    limits = np.asarray(config["editable_joints"][name]["delta_bounds_rad"]) * support
                    correction = poses[index, address] - parent_linear[address]
                    if not limits[0] - 1e-7 <= correction <= limits[1] + 1e-7:
                        raise ValueError("Replay sample exceeds correction corridor")
                for j in joints:
                    if j["joint_type"] in (2, 3) and j["limited"]:
                        value = poses[index, j["address"]]
                        if not j["limits"][0] - 1e-7 <= value <= j["limits"][1] + 1e-7:
                            raise ValueError("Replay sample exceeds native joint limit")
            class SuppliedPoseContext(AuditContext):
                def sample(self, qpos, frame):
                    _, rotation, translation = super().sample(qpos, frame)
                    matches = np.flatnonzero(times == float(frame))
                    if len(matches) != 1:
                        raise ValueError("Audit requested an uncovered actual replay pose")
                    return poses[int(matches[0])].copy(), rotation, translation
            # Compiled model and original geometry are reused; only its pose
            # sampler changes, and the cache is empty for this fresh context.
            context.__class__ = SuppliedPoseContext
            extra = dict(supplied_poses_audited=True, samples_sha256=sample_sha,
                         original_samples_sha256=request["samples_sha256"],
                         max_native_linear_difference_rad=max_difference)
        report = context.audit(candidate)
        bridge.validate_iteration_report(request, report, context.bindings)
        if context.iteration is not None and audit_samples is not None:
            # Every guard stencil touches an editable integer pose and has at
            # most three neighboring frames. The iteration halo captures all
            # those integers; the loop above bound them exactly to candidate.
            needed = set()
            for frame in context.frames:
                needed.update(range(max(0, frame - 3), min(context.T, frame + 4)))
            if not needed <= set(times):
                raise ValueError("Actual replay omits integer smoothness stencil poses")
            extra.update(smoothness_pose_source="actual_blender_integer_samples",
                         smoothness_integer_sample_count=len(needed))
        context.check_inputs_unchanged()
        bridge.validate_request(request)
        if candidate_sha != bridge.file_hash(candidate_path):
            raise ValueError("Replay candidate changed during audit")
        if audit_samples is not None and sample_sha != bridge.file_hash(sample_path):
            raise ValueError("Replay samples changed during audit")
        result = dict(schema_version=1, status="AUDITED", selected=False,
                      request_sha256=bridge.digest(request), candidate_sha256=candidate_sha,
                      config_sha256=request["config_sha256"], bindings=context.bindings,
                      audit_after=report, elapsed_seconds=time.perf_counter() - started, **extra)
        bridge._json(target / "report.json", result)
        return result
    if output is not None or audit_samples is not None:
        raise ValueError("--output is only used with --audit-candidate")
    from .local_repair import run
    bridge._json(root / "progress.json", {"status": "RUNNING", "phase": "repair"})
    result = run(request["config_path"], root / "result", progress=lambda record:
                 bridge._json(root / "progress.json", {"status": "RUNNING", "round": record}))
    bridge.validate_request(request)
    bridge.validate_candidate(request, result.trajectory)
    receipt = dict(schema_version=1, status="COMPLETE", request_sha256=bridge.digest(request),
                   candidate_sha256=bridge.file_hash(root / "result/candidate_UNREVIEWED.npz"),
                   report_sha256=bridge.file_hash(root / "result/report.json"),
                   elapsed_seconds=time.perf_counter() - started)
    bridge._json(root / "completion.json", receipt)
    bridge._json(root / "progress.json", receipt)
    return receipt


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", required=True)
    parser.add_argument("--audit-candidate")
    parser.add_argument("--audit-samples")
    parser.add_argument("--output")
    args = parser.parse_args(argv)
    try:
        result = execute(args.request, audit_candidate=args.audit_candidate,
                         audit_samples=args.audit_samples, output=args.output)
    except Exception as error:
        traceback.print_exc()
        # Failure evidence is limited to the caller's already-created job.
        path = Path(args.request).resolve().parent / "worker_error.json"
        bridge._json(path, dict(status="FAILED", type=type(error).__name__, message=str(error)))
        return 1
    print(json.dumps(result, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
