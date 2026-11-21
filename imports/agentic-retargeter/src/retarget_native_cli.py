"""Convert a completed, bound SOMA run to a native robot NPZ."""
import argparse
import hashlib
import json
from pathlib import Path

from .retarget_native import convert_soma_csv


def run(report_path, output_path, *, object_sidecar=None):
    from . import soma

    report_path = Path(report_path).expanduser().resolve(strict=True)
    report_bytes = report_path.read_bytes()
    report = json.loads(report_bytes)
    if (not isinstance(report, dict) or report.get("schema_version") != 1
            or report.get("status") != "CONVERTED_UNREVIEWED"):
        raise ValueError("Expected a completed SOMA adapter run.json")
    handoff = report["native_handoff"]
    expected = {
        "csv_path": report["outputs"]["raw/source.csv"]["path"],
        "expected_csv_sha256": report["outputs"]["raw/source.csv"]["sha256"],
        "source_path": report["source"]["path"],
        "expected_source_sha256": report["source"]["sha256"],
        "model_xml": report["model_xml"]["path"],
        "expected_model_sha256": report["model_xml"]["sha256"],
        "expected_model_assets": {entry["path"]: entry["sha256"] for entry in report["robot_assets"].values()},
        "fps": report["source_info"]["fps"], "world_frame": "soma_retargeter_z_up",
    }
    if handoff != expected:
        raise ValueError("Native descriptor disagrees with the completed source/model/output bindings")
    soma._assert_unchanged(report)
    output = Path(output_path).expanduser().resolve()
    summary = output.with_suffix(output.suffix + ".json")
    if summary.exists():
        raise FileExistsError(f"Use a new native report path: {summary}")
    optimized = None
    if "object_optimization" in report:
        import numpy as np
        from .soma_object import resolve_object_policy, validate_object_config
        configured = json.loads(Path(report["config"]["path"]).read_text()).get("object_optimization")
        if validate_object_config(configured) != report["object_optimization"]:
            raise ValueError("Object optimization settings differ from the bound original config")
        entry = report.get("optimized_object")
        if not isinstance(entry, dict) or entry != report["outputs"].get("optimized_object.npz"):
            raise ValueError("Missing or inconsistent optimized object output binding")
        reference = report["contact_inputs"]["object_trajectory"]
        with np.load(reference["path"], allow_pickle=False) as original:
            policy = resolve_object_policy(original["obj_R"], original["obj_t"], report["object_optimization"])
        if report.get("object_motion_policy") != policy:
            raise ValueError("Object motion policy differs from the bound original trajectory")
        optimized = dict(path=entry["path"], sha256=entry["sha256"],
            reference_path=reference["path"], reference_sha256=reference["sha256"],
            config=report["object_optimization"], original_sidecar_sha256=report["bindings"]["object"]["sha256"])
    result = convert_soma_csv(**handoff, output_path=output, object_sidecar=object_sidecar,
                             optimized_object=optimized)
    result["backend_report"] = {"path": str(report_path), "sha256": hashlib.sha256(report_bytes).hexdigest()}
    with summary.open("x", encoding="utf8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    return {"native": result["path"], "report": str(summary), "frames": result["frames"],
            "status": "UNREVIEWED", "blender_scene_created": False,
            "missing_scene_inputs": result["missing_scene_inputs"]}


def add_cli(subparsers):
    parser = subparsers.add_parser("retarget-native", help="Bound SOMA run.json → original-model native NPZ")
    parser.add_argument("--report", required=True, help="Completed SOMA adapter run.json")
    parser.add_argument("--output", required=True, help="New .npz file")
    parser.add_argument("--object-sidecar", help="Optional original object motion and mesh bindings in the robot world frame")
    return parser


def run_cli(args):
    return run(args.report, args.output, object_sidecar=args.object_sidecar)


def main(argv=None):
    import sys

    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    add_cli(sub)
    args = parser.parse_args(["retarget-native", *(sys.argv[1:] if argv is None else argv)])
    try:
        result = run_cli(args)
    except (ValueError, KeyError, OSError, RuntimeError) as error:
        parser.exit(2, f"Native handoff error: {error}\n")
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
