#!/usr/bin/env python3
"""Build tracking packs and per-object training cohorts from tracking handoffs.

Run in the ``sonic`` environment (USD, joblib and the vendored SONIC tree are needed; no GPU)::

    python -m grail.retargeting.manipulation.pack \\
        --handoff runs/table_046_s02/tracking_handoff [--handoff ...] --output runs/table_046/pack

For every handoff (``python -m grail.retargeting.manipulation.handoff``) this converts the native
robot/object motion into a training pack (``conversion/convert_native.py``, then
``validate_pack.py``), authors the small table-support block for ``table-block`` motions
(``table_block.py``), writes the object physics override (0.1 kg) and finally groups all motions
by object mesh into frozen 2048-slot training rosters plus one single-environment evaluation
roster per motion (``conversion/build_group_cohorts.py``). The resulting ``tracking_pack.json``
lists each object group with the paths that ``train`` and ``evaluate`` consume.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

from grail.retargeting.manipulation import paths, table_block


def ref(path) -> dict:
    path = Path(path).resolve(strict=True)
    return {"path": str(path), "sha256": paths.sha256(path), "bytes": path.stat().st_size}


def write(path, obj) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(obj, stream, indent=2)
        stream.write("\n")


def pack_key(motion_key: str) -> str:
    return "p_" + hashlib.sha256(motion_key.encode()).hexdigest()[:24]


def load_handoff(directory: Path) -> dict:
    record = json.loads((directory / "handoff.json").read_text())
    if record.get("schema") != "grail.tracking.manipulation.handoff.v1":
        raise ValueError(f"Not a tracking handoff: {directory}")
    for name, digest in record["files"].items():
        if paths.sha256(directory / name) != digest:
            raise ValueError(f"Handoff file changed since it was written: {directory / name}")
    return record


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--handoff", type=Path, action="append", required=True, help="Tracking handoff directory (repeatable)")
    p.add_argument("--output", type=Path, required=True, help="New pack directory")
    p.add_argument("--num-envs", type=int, default=2048, help="Training slots per object cohort (production: 2048)")
    p.add_argument("--mass-kg", type=float, default=paths.OBJECT_MASS_KG, help="Object mass written to physics.json")
    p.add_argument("--basis", type=Path, default=paths.BPS_BASIS, help="Shared 10-point BPS basis")
    p.add_argument("--sonic-root", type=Path, default=paths.SONIC_MANIP_ROOT)
    args = p.parse_args(argv)
    output, sonic = args.output.resolve(), args.sonic_root.resolve()
    if output.exists():
        raise FileExistsError("Use a new pack directory: " + str(output))
    handoffs = [(h.resolve(), load_handoff(h.resolve())) for h in args.handoff]
    keys = [record["motion_key"] for _, record in handoffs]
    if len(keys) != len(set(keys)):
        raise ValueError("Motion keys must be unique across handoffs")
    rows = []
    for index, (directory, record) in enumerate(handoffs):
        key = pack_key(record["motion_key"])
        rows.append(dict(original_index=index, case_id=key, source_case=key,
                         native={"local_path": str(directory / "final_native.npz"), "sha256": record["files"]["final_native.npz"]},
                         inputs={"object_mesh.npz": {"local_path": str(directory / "object_mesh.npz"), "sha256": record["files"]["object_mesh.npz"]},
                                 "contact_map.json": {"local_path": str(directory / "contact_map.json"), "sha256": record["files"]["contact_map.json"]}},
                         physics={"mass_kg": args.mass_kg, "provenance": f"GRAIL manipulation tracking object mass {args.mass_kg} kg",
                                  "friction_restitution": "inherit_pinned"}))
    output.mkdir(parents=True)
    write(output / "conversion_plan.json", rows)
    env = dict(os.environ, OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", PXR_WORK_THREAD_LIMIT="1",
               PYTHONNOUSERSITE="1", CONTACTFIRST_SONIC_ROOT=str(sonic))
    conversion = paths.CONVERSION_DIR
    motions = []
    sys.path[:0] = [str(sonic), str(conversion)]
    for row, (directory, record) in zip(rows, handoffs):
        key = row["source_case"]
        for name, command in [
            ("convert", [sys.executable, "-B", str(conversion / "convert_native.py"), "--manifest", str(output / "conversion_plan.json"),
                         "--upstream", str(sonic), "--output-root", str(output), "--basis", str(args.basis.resolve()),
                         "--cases", str(row["original_index"])]),
            ("validate", [sys.executable, "-B", str(conversion / "validate_pack.py"), "--upstream", str(sonic),
                          "--pack", str(output / "packs" / key)]),
        ]:
            with (output / f"{name}_{key}.log").open("w") as stream:
                subprocess.run(command, env=env, stdout=stream, stderr=subprocess.STDOUT, check=True)
        import numpy as np
        pack = output / "packs" / key
        native = np.load(pack / "source_native/final_native_UNREVIEWED.npz", allow_pickle=False)
        manifest = json.loads((pack / "cohorts_env256/cohort_0000/manifest.json").read_text())
        context = pack / "static_context_usd_grail_paired" / (key + ".usd")
        block = None
        if record["support"] == "table-block":
            block = table_block.block_geometry(directory / "final_native.npz", directory / "object_mesh.npz")
            context = output / "support" / key / "static_context.usd"
            table_block.write_block_usd(context, block)
            block["usd_validation"] = table_block.validate_context(context, block)
        files = dict(mesh=pack / "source_native/object_mesh.npz", native=pack / "source_native/final_native_UNREVIEWED.npz",
                     robot=pack / "robot" / (key + ".pkl"), objects=pack / "objects_with_masks" / (key + ".pkl"),
                     bps=pack / "bps" / (key + ".npy"), basis=pack / "bps/_basis.npy",
                     active_usd=pack / "active_articulation_usd" / (key + ".usd"), context_usd=context,
                     active_surface=Path(manifest["slots"][0]["active_surface_path"]),
                     physics=pack / "physics/object_physics_overrides.json", validation=pack / "independent_validation.json")
        motions.append(dict(motion_key=record["motion_key"], pack_motion_key=key,
                            source_reference={"frames": len(native["qpos"]), "fps": float(native["fps"])},
                            scale_xyz=native["object_scale"].tolist(), physics_object_key="objectgroup/" + key,
                            files={name: ref(path) for name, path in files.items()},
                            mesh_identity=ref(directory / "object_mesh.obj"), mesh_identity_format="original_file",
                            input_binding=ref(directory / "handoff.json"), support=record["support"], table_block=block))
    write(output / "validated_handoff.json", {"schema": "contactfirst.objectgroup.handoff.v1", "dataset_id": "grail-tracking-pack",
                                              "purpose": "preparation", "handoff_complete": True,
                                              "expected_motion_keys": keys, "motions": motions})
    from build_group_cohorts import build
    from validate_group_cohorts import validate
    catalog = build(output / "validated_handoff.json", output / "grouped", args.num_envs)
    validation = validate(output / "grouped")
    groups = []
    for group in catalog["groups"]:
        binding = json.loads(Path(group["binding_path"]).read_text())
        groups.append({"group_id": group["group_id"], "object_mesh_sha256": group["object_mesh_sha256"],
                       "roster_path": group["roster_path"], "physics_path": binding["training_pack"]["physics_path"],
                       "num_envs": group["num_envs"],
                       "motions": [{k: m[k] for k in ("motion_id", "source_motion_key", "object_mesh_sha256",
                                                       "single_env_roster_path", "source_reference")}
                                   for m in binding["motions"]]})
    write(output / "tracking_pack.json", {"schema": "grail.tracking.manipulation.pack.v1",
                                          "upstream_sonic_commit": paths.UPSTREAM_SONIC_COMMIT,
                                          "cpu_validation": validation, "object_mass_kg": args.mass_kg,
                                          "motions": [{k: m[k] for k in ("motion_key", "pack_motion_key", "support", "table_block")} for m in motions],
                                          "groups": groups})
    print(json.dumps([{k: g[k] for k in ("group_id", "roster_path", "physics_path")} | {"motions": [m["source_motion_key"] for m in g["motions"]]} for g in groups], indent=2))


if __name__ == "__main__":
    main()
