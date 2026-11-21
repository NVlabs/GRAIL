"""Apply the repository's metric-source scale policy in the official process.

The official checkout, robot meshes, joint offsets and solver stay unchanged.
The scalar is set before effectors and foot-processing scales are constructed.
"""
import argparse
import json
import math
from pathlib import Path
import runpy
import sys


def validate_scale(value):
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(value) or value <= 0):
        raise ValueError("human_scale_ratio must be a positive finite number")
    return float(value)


def install_scale_override(value, report_path, robot):
    """Override scaler construction only within this conversion subprocess."""
    from soma_retargeter.robotics.human_to_robot_scaler import HumanToRobotScaler

    scale = validate_scale(value)
    original_init = HumanToRobotScaler.__init__
    records = []

    def initialize(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        if self.robot_type != robot:
            raise ValueError(f"Scale policy expected {robot}, got {self.robot_type}")
        original = float(self.human_scale_ratio)
        self.human_scale_ratio = scale
        records.append({"robot": robot, "upstream_human_scale_ratio": original,
                        "human_scale_ratio": self.human_scale_ratio})
        Path(report_path).write_text(json.dumps({"scalers": records}, indent=2) + "\n")

    HumanToRobotScaler.__init__ = initialize
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--viewer", choices=["null"], default="null")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    root = Path(config["agentic_upstream_root"])
    sys.path.insert(0, str(root))
    records = install_scale_override(config.get("human_scale_ratio", 1.0),
                                     config["agentic_scale_report"], config["retarget_target"])
    script = root / "app/bvh_to_csv_converter.py"
    sys.argv = [str(script), "--config", args.config, "--viewer", args.viewer, "--device", args.device]
    runpy.run_path(str(script), run_name="__main__")
    if not records:
        raise RuntimeError("Official converter did not apply the configured SOMA scale")


if __name__ == "__main__":
    main()
