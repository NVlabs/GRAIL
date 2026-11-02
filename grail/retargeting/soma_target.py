"""Build a source-bound G1+DEx3 target for NVIDIA's SOMA Retargeter.

The target uses the original G1+DEx3 MJCF and joint limits.  It adds explicit
SOMA hand-segment targets calibrated from the exported source's native rest
pose.  A new registry bundle is written for each source; the upstream robot,
source animation, and SOMA Retargeter installation are never modified.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import xml.etree.ElementTree as ET
from pathlib import Path


def finger_matches():
    """Return the explicit anatomical mapping from five human to three robot digits."""
    result = {}
    for human, robot in (("Left", "left"), ("Right", "right")):
        for suffix, segment in (
            ("1", "0_link"),
            ("2", "1_link"),
            ("3", "2_link"),
            ("End", "finger_tip"),
        ):
            result[human + "HandThumb" + suffix] = robot + "_hand_thumb_" + segment
        for digit in ("Index", "Middle"):
            for suffix, segment in (
                ("2", "0_link"),
                ("3", "1_link"),
                ("End", "finger_tip"),
            ):
                result[human + "Hand" + digit + suffix] = (
                    robot + "_hand_" + digit.lower() + "_" + segment
                )
    return result


def _binding(path):
    path = Path(path).resolve(strict=True)
    return {
        "path": str(path),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "bytes": path.stat().st_size,
    }


def calibrated_offset(human_joint, human_wrist, robot_joint, robot_wrist, wrist_offset):
    """Calibrate a target body in the native wrist frame.

    Transforms use xyz+xyzw. SOMA applies ``q=human.q*offset.q`` and
    ``p=human.p+q.rotate(offset.p)``, so the stored translation is expressed in
    the target rotation frame rather than the source joint frame.
    """
    import numpy as np
    from scipy.spatial.transform import Rotation as R

    hp, hr = np.asarray(human_joint[:3]), R.from_quat(human_joint[3:])
    wp, wr = np.asarray(human_wrist[:3]), R.from_quat(human_wrist[3:])
    rp, rr = np.asarray(robot_joint[:3]), R.from_quat(robot_joint[3:])
    rwp, rwr = np.asarray(robot_wrist[:3]), R.from_quat(robot_wrist[3:])
    aligned_wr = wr * R.from_quat(wrist_offset[1])
    desired_r = aligned_wr * rwr.inv() * rr
    desired_p = wp + aligned_wr.apply(np.asarray(wrist_offset[0]) + rwr.inv().apply(rp - rwp))
    translation = desired_r.inv().apply(desired_p - hp)
    quaternion = (hr.inv() * desired_r).as_quat()
    return [translation.tolist(), quaternion.tolist()]


def create_target(
    soma_root,
    model_xml,
    source_bvh,
    source_rest,
    output,
    *,
    name="unitree_g1_dex3",
    human_scale=1.0,
):
    """Create a model- and source-bound SOMA registry bundle for G1+DEx3."""
    soma_root, model_xml, source_bvh, source_rest, output = map(
        lambda path: Path(path).resolve(),
        (soma_root, model_xml, source_bvh, source_rest, output),
    )
    if output.exists():
        raise ValueError(f"Output must be new: {output}")
    if human_scale != 1.0:
        raise ValueError("This wrist-calibrated target requires metric source scale 1.0")
    source_manifest_path = source_bvh.parent / "manifest.json"
    if not source_manifest_path.is_file():
        raise ValueError("Native SOMA export manifest is required beside source BVH")
    source_manifest = json.loads(source_manifest_path.read_text())
    for key, path in (("bvh", source_bvh), ("native_rest", source_rest)):
        if source_manifest[key]["sha256"] != _binding(path)["sha256"]:
            raise ValueError(f"Source export manifest hash mismatch: {key}")

    sys.path.insert(0, str(soma_root))
    import newton
    import numpy as np
    import warp as wp
    from soma_retargeter.io.bvh import BVHImporter
    from soma_retargeter.utils.newton_utils import (
        get_filtered_joint_names,
        get_name_from_label,
    )

    xml = ET.parse(model_xml).getroot()
    if xml.findall("include"):
        raise ValueError("Expand and bind MJCF includes explicitly before target creation")
    compiler = xml.find("compiler")
    meshdir = Path(compiler.get("meshdir", ".")) if compiler is not None else Path(".")
    if meshdir.is_absolute() or ".." in meshdir.parts:
        raise ValueError("Target requires a relative in-tree meshdir")
    output.mkdir(parents=True)
    model_dir = output / "model"
    model_dir.mkdir()
    copied_model = model_dir / model_xml.name
    shutil.copyfile(model_xml, copied_model)
    assets = [_binding(model_xml)]
    for mesh in xml.findall("./asset/mesh"):
        filename = Path(mesh.attrib["file"])
        if filename.is_absolute() or ".." in filename.parts:
            raise ValueError("Target mesh reference escapes model tree")
        original = model_xml.parent / meshdir / filename
        target = model_dir / meshdir / filename
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(original, target)
        assets.append(_binding(original))
    for texture in xml.findall("./asset/texture"):
        if "file" not in texture.attrib:
            continue
        filename = Path(texture.attrib["file"])
        texturedir = Path(compiler.get("texturedir", ".")) if compiler is not None else Path(".")
        if (
            filename.is_absolute()
            or texturedir.is_absolute()
            or ".." in (texturedir / filename).parts
        ):
            raise ValueError("Texture reference escapes original model tree")
        original = model_xml.parent / texturedir / filename
        target = model_dir / texturedir / filename
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(original, target)
        assets.append(_binding(original))

    with wp.ScopedDevice("cpu"):
        builder = newton.ModelBuilder()
        builder.add_mjcf(str(copied_model))
        model = builder.finalize()
        state = model.state()
        newton.eval_fk(model, model.joint_q, model.joint_qd, state)
        native = state.body_q.numpy()
        body_names = [get_name_from_label(label) for label in model.body_label]
        reference_q = model.joint_q.numpy()
        joints = get_filtered_joint_names(builder, [newton.JointType.REVOLUTE])
        skeleton, _ = BVHImporter().create_skeleton(str(source_bvh))
    if len(joints) != 43 or len(reference_q) != 50:
        raise ValueError(
            "Expected G1+DEx3 43 hinges/50 coordinates, " f"got {len(joints)}/{len(reference_q)}"
        )

    matches = finger_matches()
    if any(body not in body_names for body in matches.values()):
        raise ValueError("Newton did not retain an original finger segment or fingertip body")
    with np.load(source_rest, allow_pickle=False) as rest:
        from scipy.spatial.transform import Rotation as R

        names = rest["names"].tolist()
        matrices = np.asarray(rest["bvh_transforms"], dtype=float)
        if str(rest["source_sha256"].item()) != source_manifest["source"]["sha256"]:
            raise ValueError("Rest and animation have different source checkpoint lineage")
        if str(rest["units"].item()) != "m" or str(rest["world_frame"].item()) != "grail_z_up_m":
            raise ValueError("Native source rest requires declared GRAIL metre coordinates")
        bvh_from_grail = np.asarray(rest["bvh_from_grail"], dtype=float)
        expected_axis = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]])
        if not np.allclose(bvh_from_grail, expected_axis, atol=1e-8):
            raise ValueError("Native rest uses an unsupported BVH axis convention")
        if not np.array_equal(rest["parents"], skeleton.parent_indices):
            raise ValueError("Native rest parents differ from the source BVH hierarchy")
        if matrices.shape != (1, len(names), 4, 4):
            raise ValueError("Native rest must contain one model-bound zero-pose FK frame")
        rotations = matrices[0, :, :3, :3]
        if (
            not np.isfinite(matrices).all()
            or not np.allclose(matrices[0, :, 3], [0, 0, 0, 1], atol=1e-7)
            or not np.allclose(rotations @ rotations.transpose(0, 2, 1), np.eye(3), atol=1e-5)
            or not np.allclose(np.linalg.det(rotations), 1.0, atol=1e-5)
        ):
            raise ValueError("Native rest contains non-rigid transforms")
        global_rest = np.c_[
            matrices[0, :, :3, 3],
            R.from_matrix(matrices[0, :, :3, :3]).as_quat(),
        ]
    if names != skeleton.joint_names or global_rest.shape != (len(names), 7):
        raise ValueError("Rest FK must match the exact exported SOMA skeleton and xyz+xyzw layout")
    if not np.isfinite(global_rest).all() or not np.allclose(
        np.linalg.norm(global_rest[:, 3:], axis=1), 1, atol=1e-5
    ):
        raise ValueError("Rest FK has nonfinite values or non-unit quaternions")

    default = soma_root / "soma_retargeter/assets/robotics/unitree/g1/configs"
    scaler = json.loads((default / "soma_to_g1_scaler_config.json").read_text())
    config = json.loads((default / "soma_to_g1_retargeter_config.json").read_text())
    scaler.update(
        robot_type=name,
        human_scale_ratio=human_scale,
        reference_joint_q=reference_q.tolist(),
    )
    for source, body in matches.items():
        side = "Left" if source.startswith("Left") else "Right"
        wrist = side + "Hand"
        robot_wrist = side.lower() + "_wrist_yaw_link"
        if skeleton.joint_index(source) < 0:
            raise ValueError(f"Source lacks explicit SOMA anatomical joint {source}")
        scaler["human_joint_offsets"][source] = calibrated_offset(
            global_rest[skeleton.joint_index(source)],
            global_rest[skeleton.joint_index(wrist)],
            native[body_names.index(body)],
            native[body_names.index(robot_wrist)],
            scaler["human_joint_offsets"][wrist],
        )
        config["ik_match_table"][source] = {
            "t_body": body,
            "r_body": body,
            "t_weight": 1.0 if source.endswith("End") else 0.5,
            "r_weight": 0.0 if source.endswith("End") else 0.15,
        }
    config["human_robot_scaler_config"] = "configs/soma_to_g1_dex3_scaler.json"
    config["post_processing"]["robot_config"] = "configs/g1_post_processing_config.json"
    config_dir = output / "configs"
    config_dir.mkdir()
    (config_dir / "soma_to_g1_dex3_scaler.json").write_text(json.dumps(scaler, indent=2) + "\n")
    (config_dir / "soma_to_g1_dex3_retarget.json").write_text(json.dumps(config, indent=2) + "\n")
    shutil.copyfile(
        default / "g1_post_processing_config.json",
        config_dir / "g1_post_processing_config.json",
    )
    manifest = {
        "name": name,
        "desc": {"xml_path": str(copied_model.relative_to(output))},
        "retarget_configs": {"soma": "configs/soma_to_g1_dex3_retarget.json"},
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    report = {
        "schema_version": 1,
        "status": "TARGET_UNREVIEWED",
        "target": name,
        "generator": _binding(__file__),
        "source_bvh": _binding(source_bvh),
        "source_rest": _binding(source_rest),
        "source_export_manifest": _binding(source_manifest_path),
        "original_model_assets": assets,
        "target_model": _binding(copied_model),
        "finger_matches": matches,
        "native_joint_names": joints,
        "native_coordinate_count": len(reference_q),
        "root_quaternion_order": "xyzw (Newton)",
        "human_scale_ratio": human_scale,
        "hand_target_policy": (
            "wrist-relative rest-offset calibration; original joint limits; "
            "same official Newton IK"
        ),
        "omitted_human_digits": ["Ring", "Pinky"],
        "long_digit_reduction": (
            "Index/Middle2 MCP -> robot0; 3 PIP -> robot1; End -> tip; " "segment4 omitted"
        ),
        "limitations": [
            "Initialization hypothesis, not contact-aware grasp acceptance.",
            "Body mapping and foot postprocessing retained from official G1; "
            "finger targets added.",
            "Human Index and Middle represent the two long robot digits; "
            "Ring/Pinky are not independent robot DOFs.",
        ],
    }
    (output / "target_creation.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ("soma-root", "model-xml", "source-bvh", "source-rest", "output"):
        parser.add_argument("--" + flag, required=True)
    parser.add_argument("--name", default="unitree_g1_dex3")
    args = parser.parse_args(argv)
    print(json.dumps(create_target(**vars(args)), indent=2))


if __name__ == "__main__":
    main()
