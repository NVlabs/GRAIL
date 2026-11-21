"""Author an explicitly permitted static support replacement without editing motion.

The block is axis-aligned in the native z-up world, centered under the original
frame-zero object bounds. This geometry recipe establishes neither support nor
collision clearance. Numerical and visual checks remain independent.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path

import numpy as np

from .contact_edit.native_io import CANDIDATE_STATE_FIELDS, _native_qpos, load_npz
from .static_scene import MANIFEST_KEY, static_scene_payload


BLOCK_NAME = "AuthoredFloatingSupportBlock"
SCHEMA = "agentic_retargeter.support_block.v1"
REVIEW_FIELDS = CANDIDATE_STATE_FIELDS | {
    "motion_accepted", "full_motion_accepted", "approved", "approval", "review",
    "collision_checked", "support_checked", "collision_status", "support_status",
    "geometry_status", "passed", "validated",
}


def _review_fields(values):
    """Review metadata belongs to the old scene and must not approve a new one."""
    return sorted(key for key in values if key in REVIEW_FIELDS
                  or key.startswith(("review_", "approval_", "acceptance_", "selection_")))


def _encoded(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _array_hash(value):
    value = np.asarray(value)
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode())
    digest.update(str(value.shape).encode())
    digest.update(value.tobytes())
    return digest.hexdigest()


def _file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _positive(value, name, *, zero=False):
    if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.integer, np.floating))
            or not np.isfinite(value) or value < 0 or (not zero and value == 0)):
        raise ValueError(name + " must be finite and " + ("nonnegative" if zero else "positive"))
    return float(value)


def build_replacement(parent, object_mesh, table_name, authorization, *,
                      margin_m=.03, thickness_m=.08, min_xy_m=(.60, .50)):
    """Return a new native payload and unreviewed receipt, preserving input arrays.

    ``object_mesh`` contains the original object-local ``vertices`` and ``faces``.
    Its optional world_frame must match the parent. The parent must identify a
    z-up frame and an explicit packed static scene. Only the selected support's
    vertices/faces and static_scene_json may change geometrically. Prior review
    metadata is archived separately and replaced by explicit unreviewed flags;
    every motion field and every other static geometry array stays byte-exact.
    """
    if not isinstance(authorization, str) or not authorization.strip():
        raise ValueError("An explicit nonempty authorization string is required")
    if not isinstance(table_name, str) or not table_name.strip():
        raise ValueError("An explicit table_name is required")
    original, _ = static_scene_payload(parent)
    if original["status"] != "PRESENT":
        raise ValueError("A PRESENT original static scene is required")
    world_frame = original["world_frame"]
    if not world_frame.endswith("z_up"):
        raise ValueError("Support placement requires an explicitly named z_up world frame")
    qpos = _native_qpos(parent)
    if "world_frame" in object_mesh:
        declared = np.asarray(object_mesh["world_frame"])
        if declared.shape != () or declared.item() != world_frame:
            raise ValueError("Object mesh world_frame differs from native world_frame")
    margin = _positive(margin_m, "margin_m", zero=True)
    thickness = _positive(thickness_m, "thickness_m")
    minimum = np.asarray(min_xy_m)
    if (minimum.shape != (2,) or minimum.dtype.kind not in "fiu"
            or not np.isfinite(minimum).all() or np.any(minimum <= 0)):
        raise ValueError("min_xy_m must contain two positive finite lengths")
    minimum = minimum.astype(float)
    matches = [(i, item) for i, item in enumerate(original["meshes"])
               if item["name"] == table_name]
    if len(matches) != 1:
        raise ValueError("Select exactly one original table by its declared name")
    index, old = matches[0]
    if any(i != index and item["name"] == BLOCK_NAME for i, item in enumerate(original["meshes"])):
        raise ValueError("Authored support name collides with another scene mesh")
    vertices = np.asarray(object_mesh["vertices"])
    faces = np.asarray(object_mesh["faces"])
    if (vertices.ndim != 2 or vertices.shape[1:] != (3,) or len(vertices) < 3
            or vertices.dtype.kind != "f" or not np.isfinite(vertices).all()):
        raise ValueError("Original object vertices must be finite floating Nx3")
    if (faces.ndim != 2 or faces.shape[1:] != (3,) or not len(faces)
            or faces.dtype.kind not in "iu" or np.any(faces < 0) or np.any(faces >= len(vertices))):
        raise ValueError("Original object faces must be integer Mx3 indices in range")
    rotation, translation = np.asarray(parent["obj_R"]), np.asarray(parent["obj_t"])
    scale = np.asarray(parent["object_scale"])
    if (rotation.shape != (len(qpos), 3, 3) or translation.shape != (len(qpos), 3)
            or not np.isfinite(rotation).all() or not np.isfinite(translation).all()
            or scale.shape not in ((), (1,), (3,)) or not np.isfinite(scale).all()
            or np.any(scale <= 0)):
        raise ValueError("Original object trajectory and positive constant scalar/XYZ scale required")
    if (not np.allclose(rotation @ rotation.transpose(0, 2, 1), np.eye(3), atol=1e-5, rtol=0)
            or not np.allclose(np.linalg.det(rotation), 1., atol=1e-5, rtol=0)):
        raise ValueError("Original object rotations must be proper SO(3) matrices")
    world = (vertices.astype(float) * scale.reshape(-1)) @ rotation[0].T + translation[0]
    if not np.isfinite(world).all():
        raise ValueError("Nonfinite frame-zero world object geometry")
    lower_object, upper_object = world.min(0), world.max(0)
    center = .5 * (lower_object[:2] + upper_object[:2])
    object_margin_xy = upper_object[:2] - lower_object[:2] + 2 * margin
    footprint = np.maximum(object_margin_xy, minimum)
    lower = np.r_[center - .5 * footprint, lower_object[2] - thickness]
    upper = np.r_[center + .5 * footprint, lower_object[2]]
    x0, y0, z0 = lower
    x1, y1, z1 = upper
    vk, fk = old["vertices_key"], old["faces_key"]
    block_vertices = np.array([[x0, y0, z0], [x1, y0, z0], [x1, y1, z0], [x0, y1, z0],
                               [x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1]],
                              dtype=parent[vk].dtype)
    # Same outward-wound twelve original recipe triangles as the approved trial.
    block_faces = np.array([[0, 2, 1], [0, 3, 2], [4, 5, 6], [4, 6, 7],
                           [0, 1, 5], [0, 5, 4], [1, 2, 6], [1, 6, 5],
                           [2, 3, 7], [2, 7, 6], [3, 0, 4], [3, 4, 7]], dtype=parent[fk].dtype)
    recipe = dict(schema=SCHEMA, original_table_name=table_name, block_name=BLOCK_NAME,
                  world_frame=world_frame, placement="world_axis_aligned_under_original_frame_zero_object",
                  object_initial_world_bounds=[lower_object.tolist(), upper_object.tolist()],
                  block_world_bounds=[lower.tolist(), upper.tolist()], margin_xy_m=margin,
                  minimum_footprint_xy_m=minimum.tolist(), thickness_m=thickness,
                  block_dimensions_m=[*footprint.tolist(), thickness], static_all_frames=True,
                  object_vertices_sha256=_array_hash(vertices), object_faces_sha256=_array_hash(faces),
                  vertices_sha256=_array_hash(block_vertices), faces_sha256=_array_hash(block_faces))
    recipe_hash = hashlib.sha256(_encoded(recipe).encode()).hexdigest()
    manifest = copy.deepcopy(original)
    manifest["meshes"][index] = dict(name=BLOCK_NAME, role="support", vertices_key=vk,
        faces_key=fk, source=dict(path="procedural://agentic_retargeter/support_block.v1", sha256=recipe_hash),
        source_object=BLOCK_NAME, source_matrix_world=np.eye(4).tolist(),
        source_vertex_count=8, source_polygon_count=12, triangle_count=12,
        world_bounds=[block_vertices.min(0).tolist(), block_vertices.max(0).tolist()],
        authored_geometry=True, original_table_name=table_name, original_table_archived_not_active=True)
    manifest["provenance"]["authored_support_replacement"] = dict(
        schema=SCHEMA, authorization=authorization, recipe_sha256=recipe_hash,
        original_static_scene_json_sha256=_array_hash(parent[MANIFEST_KEY]),
        original_table_source=copy.deepcopy(old["source"]), original_table_name=table_name,
        baseline_status="UNREVIEWED", soma_with_new_scene="NOT_RUN")
    candidate = {key: np.array(value, copy=True) for key, value in parent.items()}
    candidate[vk], candidate[fk] = block_vertices, block_faces
    candidate[MANIFEST_KEY] = np.asarray(_encoded(manifest))
    prior_review = _review_fields(parent)
    review_archive = {key: dict(archive_key=f"original_review_{index:03d}",
                              dtype=str(np.asarray(parent[key]).dtype),
                              shape=list(np.asarray(parent[key]).shape),
                              sha256=_array_hash(parent[key]))
                      for index, key in enumerate(prior_review)}
    for key in prior_review:
        candidate.pop(key)
    reset = dict(status=np.asarray("UNREVIEWED"), review_status=np.asarray("UNREVIEWED"),
                 accepted=np.asarray(False), approved=np.asarray(False), reviewed=np.asarray(False),
                 selected=np.asarray(False), motion_accepted=np.asarray(False),
                 full_motion_accepted=np.asarray(False))
    candidate.update(reset)
    static_scene_payload(candidate, world_frame=world_frame)
    receipt = dict(**recipe, authorization=authorization, recipe_sha256=recipe_hash,
        original_table_metadata=copy.deepcopy(old), vertices_key=vk, faces_key=fk,
        changed_native_fields=sorted({MANIFEST_KEY, vk, fk, *prior_review, *reset}), status="UNREVIEWED",
        whole_motion_byte_exact=True, original_first_frame_byte_exact=True,
        other_scene_arrays_byte_exact=True, motion_accepted=False,
        geometry_status="UNREVIEWED", support_status="UNREVIEWED", soma_with_new_scene="NOT_RUN",
        prior_review_metadata=review_archive,
        review_metadata_policy="Archived separately; the new candidate clears approval and is UNREVIEWED",
        limitations=["Frame-zero object bounds only; later support and collisions are untested.",
                     "The block is a new authored scene; no retargeting or pose repair was run."])
    return candidate, receipt


def validate_replacement(parent, candidate, object_mesh, table_name, authorization, **options):
    """Regenerate the recipe and reject any extra or altered native field."""
    expected, receipt = build_replacement(parent, object_mesh, table_name, authorization, **options)
    if set(expected) != set(candidate):
        raise ValueError("Native field set changed beyond the authorized scene replacement")
    for key, wanted in expected.items():
        actual = np.asarray(candidate[key])
        if actual.dtype != wanted.dtype or actual.shape != wanted.shape or actual.tobytes() != wanted.tobytes():
            raise ValueError("Unauthorized native change or incorrect support block: " + key)
    receipt["replacement_contract_passed"] = True
    return receipt


def run(native_path, object_mesh_path, table_name, authorization, output, **options):
    """Write a new scene candidate and exact selected-support archive; no solve."""
    native_path, object_mesh_path = Path(native_path).resolve(strict=True), Path(object_mesh_path).resolve(strict=True)
    output = Path(output)
    if output.exists():
        raise FileExistsError("Use a new output directory: " + str(output))
    bindings = {"native": dict(path=str(native_path), sha256=_file_hash(native_path)),
                "object_mesh": dict(path=str(object_mesh_path), sha256=_file_hash(object_mesh_path))}
    parent, mesh = load_npz(native_path), load_npz(object_mesh_path)
    candidate, receipt = build_replacement(parent, mesh, table_name, authorization, **options)
    validate_replacement(parent, candidate, mesh, table_name, authorization, **options)
    output.mkdir(parents=True, exist_ok=False)
    candidate_path = output / "candidate_UNREVIEWED.npz"
    np.savez_compressed(candidate_path, **candidate)
    validate_replacement(parent, load_npz(candidate_path), mesh, table_name, authorization, **options)
    archived = dict(vertices=parent[receipt["vertices_key"]], faces=parent[receipt["faces_key"]],
                    metadata_json=np.asarray(_encoded(receipt["original_table_metadata"])),
                    original_static_scene_json=parent[MANIFEST_KEY], world_frame=parent["world_frame"])
    for key, binding in receipt["prior_review_metadata"].items():
        archived[binding["archive_key"]] = parent[key]
    archived["original_review_metadata_json"] = np.asarray(_encoded(receipt["prior_review_metadata"]))
    archive_path = output / "original_support.npz"
    np.savez_compressed(archive_path, **archived)
    reloaded = load_npz(archive_path)
    if set(reloaded) != set(archived) or any(_array_hash(reloaded[k]) != _array_hash(v) for k, v in archived.items()):
        raise RuntimeError("Original support archive round trip changed bytes")
    if any(_file_hash(value["path"]) != value["sha256"] for value in bindings.values()):
        raise RuntimeError("Support replacement inputs changed during execution")
    receipt.update(inputs=bindings, candidate=dict(path=candidate_path.name, sha256=_file_hash(candidate_path)),
                   archived_original_support=dict(path=archive_path.name, sha256=_file_hash(archive_path)),
                   replacement_contract_passed=True)
    (output / "report.json").write_text(json.dumps(receipt, indent=2, allow_nan=False) + "\n")
    handoff = dict(schema=SCHEMA, status="NOT_RUN", motion_accepted=False,
                   native=receipt["candidate"], archived_original_support=receipt["archived_original_support"],
                   required_next_step="Bind this exact scene in SOMA inputs, collision audit and render configs before a new run",
                   solver_config_rewritten=False, object_sidecar_rewritten=False)
    (output / "soma_scene_handoff.json").write_text(json.dumps(handoff, indent=2) + "\n")
    return receipt


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native", required=True)
    parser.add_argument("--object-mesh", required=True)
    parser.add_argument("--table-name", required=True)
    parser.add_argument("--authorization", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--margin-m", type=float, default=.03)
    parser.add_argument("--thickness-m", type=float, default=.08)
    parser.add_argument("--min-xy-m", type=float, nargs=2, default=(.60, .50))
    args = parser.parse_args(argv)
    result = run(args.native, args.object_mesh, args.table_name, args.authorization, args.output,
                 margin_m=args.margin_m, thickness_m=args.thickness_m, min_xy_m=args.min_xy_m)
    print(json.dumps(dict(status=result["status"], dimensions_m=result["block_dimensions_m"], output=args.output)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
