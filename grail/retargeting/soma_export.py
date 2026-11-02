"""Export native SOMA03 FK as official SOMA-base BVH, without a body-model conversion.

The official importer consumes centimetres, intrinsic ordered Euler channels and
animated translations that *replace* each joint's OFFSET. Six channels per joint
retain identity-dependent translations as well as the actual joint-orient basis.
The export is for the declared GRAIL Z-up metre world; inverse X+90 conversion
is undone by the retargeter's ``source_facing_direction=Mujoco`` setting.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import subprocess
import sys
import warnings
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from grail.retargeting.native_motion import load_native, validate_motion

BVH_FROM_GRAIL = np.array([[1.0, 0, 0], [0, 0, 1], [0, -1, 0]])


def _binding(path):
    path = Path(path).expanduser().resolve()
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return {"path": str(path), "sha256": digest.hexdigest()}


def _hierarchy(reference):
    """Read the bound reference's exact public ordering, including virtual Root."""
    names, parents, stack = [], [], []
    pending = None
    for line in Path(reference).read_text().splitlines():
        words = line.strip().split()
        if not words:
            continue
        if words[0] == "MOTION":
            break
        if words[0] in {"ROOT", "JOINT"}:
            if len(words) != 2 or words[1] in names:
                raise ValueError("Invalid SOMA reference hierarchy")
            pending = len(names)
            names.append(words[1])
            parents.append(stack[-1] if stack else -1)
        elif words == ["End", "Site"]:
            pending = -2
        elif words == ["{"]:
            if pending is None:
                raise ValueError("Invalid SOMA reference braces")
            stack.append(pending)
            pending = None
        elif words == ["}"]:
            if not stack:
                raise ValueError("Invalid SOMA reference braces")
            stack.pop()
    if stack or len(names) != 78 or names[:2] != ["Root", "Hips"]:
        raise ValueError("Expected resolved official SOMA-base 78-joint reference BVH")
    if parents[0] != -1 or any(p < 0 or p >= i for i, p in enumerate(parents[1:], 1)):
        raise ValueError("Invalid parent-before-child SOMA hierarchy")
    return names, np.asarray(parents, dtype=np.int64)


def _local_transforms(world, parents):
    local = world.copy()
    for joint, parent in enumerate(parents):
        if parent >= 0:
            rotation = np.swapaxes(world[:, parent, :3, :3], -1, -2)
            local[:, joint, :3, :3] = rotation @ world[:, joint, :3, :3]
            local[:, joint, :3, 3] = np.einsum(
                "tij,tj->ti", rotation, world[:, joint, :3, 3] - world[:, parent, :3, 3]
            )
    return local


def _world_transforms(local, parents):
    world = local.copy()
    for joint, parent in enumerate(parents):
        if parent >= 0:
            world[:, joint] = world[:, parent] @ local[:, joint]
    return world


def _check_transforms(world):
    if world.ndim != 4 or world.shape[2:] != (4, 4) or not np.isfinite(world).all():
        raise ValueError("Native transforms must be finite [T,J,4,4]")
    rotations = world[..., :3, :3]
    if not np.allclose(world[..., 3, :], [0, 0, 0, 1], atol=1e-6, rtol=0):
        raise ValueError("Invalid homogeneous native transforms")
    if not np.allclose(
        rotations @ np.swapaxes(rotations, -1, -2), np.eye(3), atol=1e-5, rtol=0
    ) or not np.allclose(np.linalg.det(rotations), 1, atol=1e-5, rtol=0):
        raise ValueError("Native transforms contain non-rigid rotation/scale")


def _write_bvh(path, world, rest_world, names, parents, fps):
    """Write and mathematically round-trip the same six-channel dialect as upstream."""
    world = np.asarray(world, dtype=np.float64)
    rest_world = np.asarray(rest_world, dtype=np.float64)
    _check_transforms(world)
    _check_transforms(rest_world)
    if world.shape[1] != len(names) or rest_world.shape != (1, len(names), 4, 4):
        raise ValueError("Transform count differs from bound hierarchy")
    if not np.isfinite(fps) or fps <= 0 or not np.isclose(fps, round(fps, 2), atol=1e-8, rtol=0):
        raise ValueError("FPS must survive the official importer's two-decimal rounding unchanged")
    world_bvh = world.copy()
    rest_bvh = rest_world.copy()
    for transforms in (world_bvh, rest_bvh):
        transforms[..., :3, :] = BVH_FROM_GRAIL @ transforms[..., :3, :]
    local = _local_transforms(world_bvh, parents)
    rest_local = _local_transforms(rest_bvh, parents)
    # Gimbal lock changes Euler representation, not the recovered rotation.
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Gimbal lock detected", category=UserWarning)
        euler = (
            Rotation.from_matrix(local[..., :3, :3].reshape(-1, 3, 3))
            .as_euler("ZYX", degrees=True)
            .reshape(len(world), len(names), 3)
        )
    channels = np.concatenate([local[..., :3, 3] * 100, euler], axis=-1)
    lines = ["HIERARCHY"]

    def visit(joint, depth):
        indent = "\t" * depth
        lines.append(f"{indent}{'ROOT' if joint == 0 else 'JOINT'} {names[joint]}")
        lines.append(indent + "{")
        offset = rest_local[0, joint, :3, 3] * 100
        lines.append(indent + "\tOFFSET " + " ".join(f"{v:.12g}" for v in offset))
        lines.append(
            indent + "\tCHANNELS 6 Xposition Yposition Zposition Zrotation Yrotation Xrotation"
        )
        for child in np.flatnonzero(parents == joint):
            visit(int(child), depth + 1)
        lines.append(indent + "}")

    visit(0, 0)
    lines.extend(["MOTION", f"Frames: {len(world)}", f"Frame Time: {1 / fps:.17g}"])
    lines.extend(
        " ".join(f"{value:.12g}" for value in row) for row in channels.reshape(len(world), -1)
    )
    Path(path).write_text("\n".join(lines) + "\n")
    # Read the serialized values, so truncation/format precision is included.
    serialized = np.asarray([[float(v) for v in line.split()] for line in lines[-len(world) :]])
    serialized = serialized.reshape(channels.shape)
    recovered = np.broadcast_to(np.eye(4), world.shape).copy()
    recovered[..., :3, 3] = serialized[..., :3] / 100
    recovered[..., :3, :3] = (
        Rotation.from_euler("ZYX", serialized[..., 3:].reshape(-1, 3), degrees=True)
        .as_matrix()
        .reshape(world.shape[:2] + (3, 3))
    )
    recovered = _world_transforms(recovered, parents)
    recovered[..., :3, :] = BVH_FROM_GRAIL.T @ recovered[..., :3, :]
    return _errors(world, recovered)


def _errors(expected, actual):
    if expected.shape != actual.shape or not np.isfinite(actual).all():
        raise ValueError("BVH round trip changed transform shape or produced nonfinite values")
    position = float(np.linalg.norm(expected[..., :3, 3] - actual[..., :3, 3], axis=-1).max())
    rotation = float(np.abs(expected[..., :3, :3] - actual[..., :3, :3]).max())
    if position > 2e-5 or rotation > 2e-5:
        raise ValueError(f"BVH FK round trip failed: position={position} m, rotation={rotation}")
    return {"max_joint_position_error_m": position, "max_rotation_matrix_error": rotation}


def _native_transforms(motion, model_config, batch_size):
    import torch

    from grail.models.soma03_model import Soma03HumanModel

    if model_config.get("model_type", model_config.get("body_model")) != "soma03":
        raise ValueError("Exporter requires explicit native soma03 model configuration")
    if model_config.get("backend", "native") != "native":
        raise ValueError("Exporter requires the native SOMA03 backend")
    device = model_config.get("device", "cpu")
    model = Soma03HumanModel(model_config, device)
    if model_config.get("soma_identity_params_path"):
        identity, scales = model.load_shape_params(model_config, None, device)
        for key, bound in (("identity_coeffs", identity), ("scale_params", scales)):
            actual = np.asarray(motion[key], dtype=np.float32)
            expected = bound.detach().cpu().numpy()
            if not np.array_equal(
                actual.reshape(-1, actual.shape[-1]),
                np.broadcast_to(expected, actual.reshape(-1, actual.shape[-1]).shape),
            ):
                raise ValueError(
                    f"Motion {key} differs from its bound identity; no shape replacement is allowed"
                )
    count = len(motion["poses"])
    outputs = []
    names = list(model.model.soma.public_joint_names)
    parents = model.model.soma.output_joint_parent_ids.detach().cpu().numpy()
    # Native virtual Root points to itself; BVH represents the same root by -1.
    if names[0] == "Root" and parents[0] == 0:
        parents = parents.copy()
        parents[0] = -1
    with torch.no_grad():
        for start in range(0, count, batch_size):
            stop = min(start + batch_size, count)
            chunk = {"poses": motion["poses"][start:stop], "trans": motion["trans"][start:stop]}
            for key in ("identity_coeffs", "scale_params"):
                value = np.asarray(motion[key])
                chunk[key] = value[start:stop] if value.ndim == 2 and len(value) == count else value
            result = model._forward(chunk, joints_only=True)
            transforms = result["transforms"].detach().cpu().numpy()
            if not np.allclose(
                transforms[:, 1:, :3, 3], result["joints"].cpu().numpy(), atol=1e-7, rtol=0
            ):
                raise ValueError("Native public transforms do not reproduce native joints")
            outputs.append(transforms)
        zero = {
            "poses": np.zeros((1, 77, 3), dtype=np.float32),
            "trans": np.zeros((1, 3), dtype=np.float32),
            "identity_coeffs": np.asarray(motion["identity_coeffs"]).reshape(-1, 128)[:1],
            "scale_params": np.asarray(motion["scale_params"]).reshape(-1, 57)[:1],
        }
        rest = model._forward(zero, joints_only=True)["transforms"].cpu().numpy()
    return np.concatenate(outputs), rest, names, parents


def _verify_official(source_bvh, expected, soma_root, python, output):
    """Exercise the actual bound importer/FK on CPU, in its own environment."""
    script = """
import pathlib,sys,numpy as np,warp as wp
root=pathlib.Path(sys.argv[1]).resolve()
sys.path.insert(0,str(root))
import soma_retargeter.io.bvh as bvh
from soma_retargeter.utils.space_conversion_utils import SpaceConverter,FacingDirectionType
assert pathlib.Path(bvh.__file__).resolve().is_relative_to(root)
with wp.ScopedDevice('cpu'):
    skeleton,tree=bvh.BVHImporter().create_skeleton(sys.argv[2])
    animation=bvh.BVHImporter.create_animation(tree,skeleton)
    converter=SpaceConverter(FacingDirectionType.MUJOCO)
    rows=[]
    for frame in range(animation.num_frames):
        transforms=animation.compute_global_transforms(frame,wp.transform(wp.vec3(0,0,0),converter.converter))
        rows.append([np.asarray(wp.transform_to_matrix(wp.transform(*value))).reshape(4,4) for value in transforms])
    np.savez(sys.argv[3],transforms=np.asarray(rows),fps=animation.sample_rate,names=np.asarray(skeleton.joint_names))
"""
    target = Path(output) / "official_import.npz"
    with (Path(output) / "official_import.log").open("w") as log:
        subprocess.run(
            [str(python), "-c", script, str(soma_root), str(source_bvh), str(target)],
            stdout=log,
            stderr=subprocess.STDOUT,
            check=True,
            timeout=120,
            env={**os.environ, "CUDA_VISIBLE_DEVICES": "", "PYTHONDONTWRITEBYTECODE": "1"},
        )
    with np.load(target, allow_pickle=False) as data:
        result = _errors(expected, data["transforms"])
        result.update(fps=float(data["fps"]), joint_names=data["names"].tolist())
    return result


def _native_runtime_inputs(model_config):
    """Bind the selected body runtime, independently of the BVH import runtime."""
    if model_config.get("soma_runtime", "gem_x") != "gem_x" or "gep_gem_bundle" in model_config:
        raise ValueError("Legacy SOMA exports require archived code and explicit model migration")
    from grail.models.soma_x_runtime import official_runtime_inputs

    return official_runtime_inputs(model_config["gem_x_soma_assets"])


def export_soma03_bvh(
    source,
    output_dir,
    *,
    model_config,
    soma_root,
    source_world_frame="grail_z_up_m",
    batch_size=32,
    upstream_python=None,
):
    """Export a trusted GRAIL HOI pickle/canonical NPZ and its immutable bindings.

    ``model_config`` uses GRAIL's native SOMA03 keys. Its identity file, when
    supplied, authorizes a fixed nonunit scale but must match the motion exactly.
    The official runtime Python defaults to ``soma_root/.venv/bin/python``.
    Existing outputs are refused; failures retain an explicit failed manifest.
    """
    if source_world_frame != "grail_z_up_m":
        raise ValueError(
            "Declare source_world_frame='grail_z_up_m'; other bases need an explicit adapter"
        )
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError("batch_size must be a positive integer")
    source, output, root = (Path(p).expanduser().resolve() for p in (source, output_dir, soma_root))
    if output.exists():
        raise FileExistsError(f"Export output already exists: {output}")
    reference = root / "soma_retargeter/assets/soma/soma_zero_frame0.bvh"
    names, parents = _hierarchy(reference)
    source_binding = _binding(source)
    motion = load_native(source, "soma03")
    if _binding(source) != source_binding:
        raise ValueError("Source changed while loading native SOMA motion")
    info = validate_motion(motion)
    native_paths, native_runtime = _native_runtime_inputs(model_config)
    python = Path(upstream_python) if upstream_python else root / ".venv/bin/python"
    # Keep venv symlinks intact; resolving the executable can select base Python.
    python = Path(os.path.abspath(python.expanduser()))
    if not python.is_file() or not os.access(python, os.X_OK):
        raise FileNotFoundError(
            f"Official SOMA runtime Python is required for FK verification: {python}"
        )
    paths = [source, reference, Path(__file__)]
    paths += sorted((root / "soma_retargeter").rglob("*.py"))
    paths += native_paths
    paths += [
        Path(__file__).parents[1] / name
        for name in (
            "models/soma03_model.py",
            "models/soma_model.py",
            "models/human_model.py",
            "retargeting/native_motion.py",
        )
    ]
    identity = model_config.get("soma_identity_params_path")
    if identity:
        paths.extend([Path(identity), Path(identity).with_suffix(".json")])
    bindings = [_binding(path) for path in paths]
    if bindings[0] != source_binding:
        raise ValueError("Source changed before export bindings were recorded")
    report = {
        "schema": "grail.soma03.bvh_export.v1",
        "status": "EXPORTING_UNREVIEWED",
        "source": _binding(source),
        "source_model": "soma03",
        "source_bvh": str(output / "source.bvh"),
        "source_units": "cm",
        "source_facing_direction": "Mujoco",
        "source_world_frame": source_world_frame,
        "world_from_grail": np.eye(3).tolist(),
        "bvh_from_grail": BVH_FROM_GRAIL.tolist(),
        "fps": info["fps"],
        "frame_count": info["frame_count"],
        "model_config": model_config,
        "joint_names": names,
        "parents": parents.tolist(),
        "input_bindings": bindings,
        "runtime": {
            "body_model": native_runtime,
            "native_python": sys.executable,
            "official_python": str(python),
            "python_version": sys.version,
            "versions": {
                name: importlib.metadata.version(name) for name in ("numpy", "scipy", "torch")
            },
        },
        "identity_policy": "preserve every source identity/scale frame; no averaging or replacement",
        "limitations": [
            "Source skeleton export only; robot retargeting and contact review are separate.",
            "The export does not choose a robot target; G1+DEx3 binding is a separate step.",
        ],
    }
    output.mkdir(parents=True)

    def save():
        (output / "manifest.json").write_text(json.dumps(report, indent=2) + "\n")

    save()
    try:
        transforms, rest, actual_names, actual_parents = _native_transforms(
            motion, model_config, batch_size
        )
        if actual_names != names or not np.array_equal(actual_parents, parents):
            raise ValueError(
                "Native SOMA03 public hierarchy differs from bound official SOMA-base reference"
            )
        report["serialized_fk"] = _write_bvh(
            output / "source.bvh", transforms, rest, names, parents, info["fps"]
        )
        report["official_import_fk"] = _verify_official(
            output / "source.bvh", transforms, root, python, output
        )
        if (
            report["official_import_fk"]["fps"] != info["fps"]
            or report["official_import_fk"]["joint_names"] != names
        ):
            raise ValueError("Official importer changed source FPS or skeleton ordering")
        for entry in bindings:
            if _binding(entry["path"]) != entry:
                raise ValueError(f"Bound source/model changed during export: {entry['path']}")
        np.savez_compressed(output / "native_source.npz", **motion)
        # BVH OFFSET records retain translation, while its animation channels
        # carry the complete rotations. A custom robot's neutral hand-frame
        # calibration must therefore use actual native zero-pose FK, not the
        # importer skeleton's rotation-free OFFSET pose.
        rest_bvh = rest.copy()
        rest_bvh[..., :3, :] = BVH_FROM_GRAIL @ rest_bvh[..., :3, :]
        np.savez_compressed(
            output / "native_rest.npz",
            transforms=rest,
            bvh_transforms=rest_bvh,
            names=np.asarray(names),
            parents=parents,
            world_frame=np.asarray(source_world_frame),
            units=np.asarray("m"),
            bvh_from_grail=BVH_FROM_GRAIL,
            source_sha256=np.asarray(source_binding["sha256"]),
            identity_coeffs=np.asarray(motion["identity_coeffs"]).reshape(-1, 128)[:1],
            scale_params=np.asarray(motion["scale_params"]).reshape(-1, 57)[:1],
        )
        report["bvh"] = _binding(output / "source.bvh")
        report["native_source"] = _binding(output / "native_source.npz")
        report["native_rest"] = _binding(output / "native_rest.npz")
        report["native_rest_policy"] = (
            "Zero poses and translation with the first source identity/scale; "
            "full native FK in grail_z_up_m and pre-facing BVH axes, both in metres. "
            "This calibration is not an animated source frame."
        )
        report["status"] = "EXPORTED_UNREVIEWED"
        save()
        return report
    except BaseException as exc:
        report.update(status="FAILED_UNREVIEWED", error=str(exc))
        save()
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--model-config", required=True, type=Path, help="Native SOMA03 model JSON")
    parser.add_argument("--soma-root", required=True, type=Path)
    parser.add_argument("--upstream-python", type=Path)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args(argv)
    config = json.loads(args.model_config.read_text())
    print(
        json.dumps(
            export_soma03_bvh(
                args.source,
                args.output,
                model_config=config,
                soma_root=args.soma_root,
                batch_size=args.batch_size,
                upstream_python=args.upstream_python,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
