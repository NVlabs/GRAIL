"""Inspect one selected G1 trajectory and compare its original object binding."""
import argparse
import hashlib
from pathlib import Path

import numpy as np


def inspect(archive, mesh):
    with np.load(archive, allow_pickle=False) as motion:
        poses = motion["qpos"]
        times = motion["timestamps"]
        fps = float(motion["fps"])
        binding = str(motion["object_mesh_sha256"])
        print(f"Robot: {archive.resolve(strict=True)}")
        print(f"Bound mesh: {motion['object_mesh_path']} (SHA-256 {binding})")
        print(f"{poses.shape}, {fps:g} fps, frame: {motion['world_frame']}")
        if poses.ndim != 2 or poses.shape[1] not in (36, 50) or not np.isfinite(poses).all():
            raise ValueError("Expected finite native G1 poses with 36 or 50 columns")
        frames, columns = poses.shape
        if frames < 2 or times.shape != (frames,) or fps <= 0 or not np.allclose(np.diff(times), 1 / fps):
            raise ValueError("Frame clock does not match the poses")
        if columns != 7 + len(motion["dof_names"]) or not np.allclose(
            np.linalg.norm(poses[:, 3:7], axis=1), 1, atol=1e-5
        ):
            raise ValueError("Joint count or root quaternion is invalid")
        if motion["obj_R"].shape != (frames, 3, 3) or motion["obj_t"].shape != (frames, 3):
            raise ValueError("Object and robot frame counts differ")
        if mesh is not None and hashlib.sha256(mesh.read_bytes()).hexdigest() != binding:
            raise ValueError(f"Original object SHA-256 mismatch: {mesh}")
        if mesh is not None:
            print(f"Matched original object: {mesh.resolve()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("--object-mesh", type=Path)
    args = parser.parse_args()
    inspect(args.archive, args.object_mesh)
