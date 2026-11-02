"""Public native SOMA03 to Unitree G1+DEx3 retargeting entry point.

This wrapper exports identity-preserving SOMA FK to an audited BVH and invokes
the pinned public NVIDIA SOMA Retargeter. It does not replace GRAIL's original
reconstruction contact-label optimization.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

from grail.retargeting.soma_export import export_soma03_bvh


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _binding(path):
    path = Path(path).expanduser().resolve(strict=True)
    return {"path": str(path), "sha256": _sha256(path), "bytes": path.stat().st_size}


def _retargeter_revision(root):
    result = subprocess.run(
        [
            "git",
            "-c",
            f"safe.directory={root}",
            "-C",
            str(root),
            "rev-parse",
            "HEAD",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _validate_csv(path, target, target_bundle=None):
    """Verify that the retargeter emitted the native target coordinate layout."""
    expected_root = [
        "Frame",
        "root_translateX",
        "root_translateY",
        "root_translateZ",
        "root_rotateX",
        "root_rotateY",
        "root_rotateZ",
    ]
    if target == "unitree_g1_dex3":
        report_path = Path(target_bundle) / "target_creation.json"
        report = json.loads(report_path.read_text())
        expected_joints = report["native_joint_names"]
        if len(expected_joints) != 43 or report["native_coordinate_count"] != 50:
            raise RuntimeError("G1+DEx3 target did not preserve 43 hinges/50 coordinates")
    elif target == "unitree_g1":
        expected_joints = None
    else:
        raise ValueError(f"Unsupported public target: {target}")
    with Path(path).open(newline="") as stream:
        reader = csv.reader(stream)
        header = next(reader)
        rows = list(reader)
    if header[:7] != expected_root:
        raise RuntimeError(f"Unexpected CSV root coordinate header: {header[:7]}")
    expected_columns = 50 if target == "unitree_g1_dex3" else 36
    if len(header) != expected_columns:
        raise RuntimeError(f"Expected {expected_columns} {target} CSV columns, found {len(header)}")
    if expected_joints is not None and header[7:] != expected_joints:
        raise RuntimeError("G1+DEx3 CSV joint order differs from its bound target model")
    if not rows or any(len(row) != len(header) for row in rows):
        raise RuntimeError("Retargeted CSV is empty or ragged")
    if not all(math.isfinite(float(value)) for row in rows for value in row):
        raise RuntimeError("Retargeted CSV contains non-finite values")
    return {"frames": len(rows), "columns": len(header), "joint_names": header[7:]}


def _create_dex3_target(
    source_bvh,
    source_rest,
    output,
    *,
    soma_root,
    model_xml,
    python,
):
    """Create the source-bound registry with the official SOMA environment."""
    executable = Path(os.path.abspath(Path(python).expanduser()))
    generator = Path(__file__).with_name("soma_target.py").resolve(strict=True)
    destination = Path(output).expanduser().resolve()
    completed = subprocess.run(
        [
            str(executable),
            str(generator),
            "--soma-root",
            str(Path(soma_root).resolve(strict=True)),
            "--model-xml",
            str(Path(model_xml).resolve(strict=True)),
            "--source-bvh",
            str(Path(source_bvh).resolve(strict=True)),
            "--source-rest",
            str(Path(source_rest).resolve(strict=True)),
            "--output",
            str(destination),
        ],
        capture_output=True,
        text=True,
        timeout=3600,
    )
    if destination.exists():
        (destination / "target_stdout.log").write_text(completed.stdout)
        (destination / "target_stderr.log").write_text(completed.stderr)
    if completed.returncode:
        raise RuntimeError(
            f"G1+DEx3 target creation failed with exit code {completed.returncode}; "
            f"see {destination / 'target_stderr.log'}"
        )
    report = json.loads((destination / "target_creation.json").read_text())
    if report["target"] != "unitree_g1_dex3":
        raise RuntimeError("Generated target has the wrong registry name")
    return report


def retarget_bvh(
    bvh_path,
    output_dir,
    *,
    soma_root="imports/soma-retargeter",
    python=None,
    target="unitree_g1_dex3",
    batch_size=1,
    extra_robot_paths=(),
):
    """Run the pinned public retargeter for one exported SOMA BVH.

    G1+DEx3 is the public default. The bundled body-only G1 target remains an
    explicit fallback. Generated CSVs remain unreviewed until scene replay.
    """
    if target not in {"unitree_g1_dex3", "unitree_g1"}:
        raise ValueError("Public GRAIL supports unitree_g1_dex3 and unitree_g1")
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError("batch_size must be a positive integer")
    source = Path(bvh_path).expanduser().resolve(strict=True)
    root = Path(soma_root).expanduser().resolve(strict=True)
    extra_paths = [Path(path).expanduser().resolve(strict=True) for path in extra_robot_paths]
    if target == "unitree_g1_dex3" and len(extra_paths) != 1:
        raise ValueError("unitree_g1_dex3 requires exactly one source-bound target path")
    destination = Path(output_dir).expanduser().resolve()
    if destination.exists():
        raise FileExistsError(f"Use a new SOMA retarget output directory: {destination}")
    executable = Path(python).expanduser() if python else root / ".venv/bin/python"
    executable = Path(os.path.abspath(executable))
    app = root / "app/bvh_to_csv_converter.py"
    if not executable.is_file() or not os.access(executable, os.X_OK):
        raise FileNotFoundError(f"SOMA Retargeter Python is not executable: {executable}")
    if not app.is_file():
        raise FileNotFoundError(f"SOMA Retargeter app is missing: {app}")

    destination.mkdir(parents=True)
    receipt_path = destination / "retarget_receipt.json"
    try:
        with tempfile.TemporaryDirectory(prefix="grail-soma-retarget-") as temporary:
            temporary = Path(temporary)
            input_dir = temporary / "input"
            input_dir.mkdir()
            staged = input_dir / source.name
            shutil.copyfile(source, staged)
            config = {
                "import_folder": str(input_dir),
                "export_folder": str(destination / "csv"),
                "batch_size": batch_size,
                "retargeter": "Newton",
                "retarget_source": "soma",
                "retarget_target": target,
                "retarget_source_facing_direction": "Mujoco",
                "extra_robot_paths": [str(path) for path in extra_paths],
            }
            config_path = temporary / "config.json"
            config_path.write_text(json.dumps(config, indent=2) + "\n")
            completed = subprocess.run(
                [str(executable), str(app), "--config", str(config_path)],
                cwd=root,
                capture_output=True,
                text=True,
                timeout=3600,
            )
        (destination / "stdout.log").write_text(completed.stdout)
        (destination / "stderr.log").write_text(completed.stderr)
        if completed.returncode:
            raise RuntimeError(
                f"SOMA Retargeter failed with exit code {completed.returncode}; "
                f"see {destination / 'stderr.log'}"
            )
        outputs = sorted((destination / "csv").rglob("*.csv"))
        if len(outputs) != 1:
            raise RuntimeError(f"Expected one G1 CSV, found {len(outputs)}")
        validation = _validate_csv(outputs[0], target, extra_paths[0] if extra_paths else None)
        receipt = {
            "schema": "grail.soma_retargeter.v1",
            "status": "RETARGETED_UNREVIEWED",
            "source": _binding(source),
            "output": _binding(outputs[0]),
            "target": target,
            "retargeter_root": str(root),
            "retargeter_revision": _retargeter_revision(root),
            "python": str(executable),
            "csv_validation": validation,
            "extra_robot_paths": [str(path) for path in extra_paths],
            "target_manifest": (
                _binding(extra_paths[0] / "manifest.json") if extra_paths else None
            ),
            "contact_policy": (
                "retargeting output only; GRAIL reconstruction uses " "legacy contact_labels"
            ),
            "limitations": [
                (
                    "Finger IK is a source-guided initialization and is not a "
                    "contact-aware grasp acceptance test."
                    if target == "unitree_g1_dex3"
                    else "The fallback unitree_g1 target has no actuated fingers."
                ),
                "The CSV is kinematic and requires simulation replay before policy "
                "training or hardware use.",
            ],
        }
        receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")
        return receipt
    except BaseException as error:
        receipt_path.write_text(
            json.dumps(
                {
                    "schema": "grail.soma_retargeter.v1",
                    "status": "FAILED_UNREVIEWED",
                    "source": _binding(source),
                    "target": target,
                    "extra_robot_paths": [str(path) for path in extra_paths],
                    "error": str(error),
                },
                indent=2,
            )
            + "\n"
        )
        raise


def run_pipeline(
    source,
    output_dir,
    *,
    model_config,
    soma_root="imports/soma-retargeter",
    python=None,
    batch_size=32,
    target="unitree_g1_dex3",
    robot_model_xml="imports/GMR/assets/unitree_g1/g1_mocap_29dof_with_hands.xml",
):
    """Export native SOMA03 and retarget it to G1+DEx3 by default."""
    output = Path(output_dir).expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"Use a new pipeline output directory: {output}")
    output.mkdir(parents=True)
    export = export_soma03_bvh(
        source,
        output / "soma_export",
        model_config=model_config,
        soma_root=soma_root,
        batch_size=batch_size,
        upstream_python=python,
    )
    root = Path(soma_root).expanduser().resolve(strict=True)
    executable = Path(python).expanduser() if python else root / ".venv/bin/python"
    executable = Path(os.path.abspath(executable))
    extra_robot_paths = []
    target_report = None
    if target == "unitree_g1_dex3":
        target_dir = output / "g1_dex3_target"
        target_report = _create_dex3_target(
            output / "soma_export/source.bvh",
            output / "soma_export/native_rest.npz",
            target_dir,
            soma_root=root,
            model_xml=robot_model_xml,
            python=executable,
        )
        extra_robot_paths = [target_dir]
    retarget = retarget_bvh(
        output / "soma_export/source.bvh",
        output / "g1_retarget",
        soma_root=soma_root,
        python=executable,
        target=target,
        extra_robot_paths=extra_robot_paths,
    )
    result = {
        "status": "RETARGETED_UNREVIEWED",
        "export": export,
        "target_creation": target_report,
        "retarget": retarget,
    }
    (output / "pipeline_receipt.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="Native SOMA03 HOI PKL/joblib/NPZ")
    parser.add_argument("output", type=Path, help="New output directory")
    parser.add_argument("--model-config", required=True, type=Path, help="SOMA03 model JSON")
    parser.add_argument("--soma-root", type=Path, default=Path("imports/soma-retargeter"))
    parser.add_argument("--python", type=Path)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--target",
        choices=("unitree_g1_dex3", "unitree_g1"),
        help="Defaults to retarget_target in the model config, or unitree_g1_dex3",
    )
    parser.add_argument("--robot-model-xml", type=Path)
    args = parser.parse_args(argv)
    config = json.loads(args.model_config.read_text())
    result = run_pipeline(
        args.source,
        args.output,
        model_config=config,
        soma_root=args.soma_root,
        python=args.python,
        batch_size=args.batch_size,
        target=args.target or config.get("retarget_target", "unitree_g1_dex3"),
        robot_model_xml=(
            args.robot_model_xml
            or config.get(
                "robot_model_xml",
                "imports/GMR/assets/unitree_g1/g1_mocap_29dof_with_hands.xml",
            )
        ),
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
