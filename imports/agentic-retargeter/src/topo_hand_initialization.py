"""Portable Topo-inspired multistart bank with optional original-pad refinement.

This generates free-hand key poses, not a mounted or accepted robot sequence.
An explicitly requested pad stage refines the same candidates and original inputs.
"""

import argparse
import copy
import hashlib
import json
from pathlib import Path
import time

from .contact_edit.pose import _integer, _keys
from .grasp_io import GraspInputs, code_binding, write_report
from .hand_sequence import resolve_static_object_policy
from .surface_grasp import propose


def stage_configs(config, directory):
    """Resolve input paths once; leave the original config unmodified."""
    _keys(config, {"bank", "source_arrays"},
          {"topo_steps", "pad_steps", "topo_objective", "is_static_obj"}, "topo hand initialization")
    if not isinstance(config["bank"], dict):
        raise ValueError("bank must be a surface-grasp configuration object")
    bank = copy.deepcopy(config["bank"])
    reserved = {"steps", "topo_objective", "contact_surface", "warm_start_bank"}
    if reserved.intersection(bank):
        raise ValueError("The staged runner owns steps, objective, contact surface and warm start")
    if "contact_patch" not in bank or "multistart" not in bank:
        raise ValueError("Topo initialization requires explicit contact_patch and multistart")
    base = Path(directory)
    for key in ("native_npz", "model_xml", "object_mesh_npz"):
        bank["inputs"][key] = str((base / bank["inputs"][key]).resolve(strict=True))
    objective = copy.deepcopy(config.get("topo_objective", {}))
    _keys(objective, set(), {"graph_weight", "direction_weight", "graph_decay",
          "object_sample_count", "fallback_neighbors", "node_set"}, "topo_objective")
    objective["source_arrays"] = str((base / config["source_arrays"]).resolve(strict=True))
    objective.setdefault("direction_weight", .01)
    counts = {"topo_steps": _integer(config.get("topo_steps", 200), "topo_steps", 1),
              "pad_steps": _integer(config.get("pad_steps", 0), "pad_steps", 0)}
    if any(n > 2000 for n in counts.values()):
        raise ValueError("Each stage is bounded to at most 2000 steps")
    topo = dict(copy.deepcopy(bank), steps=counts["topo_steps"], topo_objective=objective)
    pad = dict(copy.deepcopy(bank), steps=counts["pad_steps"], contact_surface="dex3_distal_pad")
    return topo, pad


def run(config_path, output):
    """Execute requested stages, saving complete rejected/unreviewed banks."""
    config_path = Path(config_path).resolve(strict=True)
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError("Topo initialization output must be a new directory")
    raw = config_path.read_bytes()
    config = json.loads(raw)
    topo, pad = stage_configs(config, config_path.parent)
    inputs = GraspInputs(topo["inputs"], config_path.parent)
    static_policy = resolve_static_object_policy(config, inputs.R, inputs.t)
    if static_policy["is_static_obj"]:
        raise ValueError("Static objects skip contact-key search; use hand_sequence/SOMA's no-contact path")
    output.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    stages = []
    for name, config in (("topo", topo), ("pad", pad)):
        if config["steps"] == 0:
            continue
        directory = output / name
        directory.mkdir()
        if name == "pad":
            parent = output / "topo" / "bank.json"
            config["warm_start_bank"] = dict(path=str(parent),
                                            sha256=hashlib.sha256(parent.read_bytes()).hexdigest())
        write_report(directory / "executed_config.json", config)
        tick = time.monotonic()
        bank = propose(config, config_path.parent)
        write_report(directory / "bank.json", bank)
        summary = dict(stage=name, wall_seconds=time.monotonic() - tick,
                       bank=str(directory / "bank.json"),
                       bank_sha256=hashlib.sha256((directory / "bank.json").read_bytes()).hexdigest(),
                       candidates={side: dict(count=len(rows), rejected=sum(c["rejected"] for c in rows))
                                   for side, rows in bank["candidates"].items()},
                       steps_per_candidate=config["steps"], motion_accepted=False)
        write_report(directory / "summary.json", summary)
        stages.append(summary)
    pad_enabled = pad["steps"] > 0
    report = dict(schema_version=1, status="UNREVIEWED_FREE_HAND_INITIALIZER",
                  method="topo_inspired_structure_then_distal_pad" if pad_enabled else "topo_inspired_structure",
                  official_reproduction=False,
                  motion_accepted=False, config_path=str(config_path),
                  static_object_policy=static_policy,
                  config_sha256=hashlib.sha256(raw).hexdigest(), stages=stages,
                  skipped_stages=[] if pad_enabled else [dict(stage="pad", reason="pad_steps=0")],
                  final_stage=stages[-1]["stage"], final_bank=stages[-1]["bank"],
                  final_bank_sha256=stages[-1]["bank_sha256"],
                  elapsed_seconds=time.monotonic() - start,
                  implementation_sha256=code_binding("topo_hand_initialization.py", "surface_grasp.py",
                                                      "grasp_bank_seed.py", "hand_sequence.py"),
                  limitations=["Free wrists are not mounted whole-body motion.",
                               "An optional pad stage refines the same multistart rows; counts are not doubled.",
                               "All parent rejection evidence is retained; no candidate is selected.",
                               "Surface samples and first-witness screens are not complete collision clearance."])
    write_report(output / "report.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    report = run(args.config, args.out)
    print(json.dumps(dict(output=str(args.out), status=report["status"],
                          elapsed_seconds=report["elapsed_seconds"])))


if __name__ == "__main__":
    main()
