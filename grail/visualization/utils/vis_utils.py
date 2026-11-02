import os

import numpy as np
import torch
import trimesh
from PIL import ImageColor

from grail.core.io import load_mesh
from grail.core.torch_utils import tensor_to, tensor_to_numpy


def embedded_static_mesh(obj_data, device="cuda"):
    """Read a self-contained support mesh and apply its world transform once."""
    vertices = torch.as_tensor(obj_data["vertices"], dtype=torch.float32, device=device)
    faces = torch.as_tensor(obj_data["faces"], dtype=torch.int64, device=device)
    scale = torch.as_tensor(obj_data["scale"], dtype=torch.float32, device=device)
    rotation = torch.as_tensor(obj_data["rot"], dtype=torch.float32, device=device)
    position = torch.as_tensor(obj_data["pos"], dtype=torch.float32, device=device)
    return (vertices * scale.reshape(3)) @ rotation.reshape(3, 3).T + position.reshape(3), faces


def make_checker_board_texture(
    color1="black", color2="white", width=10, height=10, n_tile=15, to_bgr=False
):
    c1 = np.asarray(ImageColor.getcolor(color1, "RGB")).astype(np.uint8)
    c2 = np.asarray(ImageColor.getcolor(color2, "RGB")).astype(np.uint8)
    if to_bgr:
        c1 = c1[[2, 1, 0]]
        c2 = c2[[2, 1, 0]]
    hw = width // 2
    hh = height // 2
    c1_block = np.tile(c1, (hh, hw, 1))
    c2_block = np.tile(c2, (hh, hw, 1))
    tex = np.block([[[c1_block], [c2_block]], [[c2_block], [c1_block]]])
    tex = np.tile(tex, (n_tile, n_tile, 1))
    return tex


def prep_visualizer_input(
    hoi_data,
    human_model=None,
    normalize_trans=True,
    to_numpy=True,
    device="cuda",
    obj_mesh_data=None,
    static_meshes_data=None,
    simplify_mesh=True,
):
    """Prepare HOI data for visualization.

    Args:
        hoi_data: Dictionary containing human and object data
        human_model: HumanModel instance. If provided, uses its generate_mesh
                     which handles G1/SOMA/SMPLX correctly. If None, falls back
                     to creating a default SMPLX or SOMA model.
        normalize_trans: Whether to normalize translation to XY origin
        to_numpy: Whether to convert output to numpy arrays
        device: Device for computation
        obj_mesh_data: Optional (vertices, faces) tuple of pre-loaded object mesh
                       to skip redundant load_mesh calls across visualize invocations.
        static_meshes_data: Optional dict {key: {"vertices": Tensor, "faces": Tensor}}
                            of pre-loaded static meshes (vertices already rot+pos
                            transformed) to skip redundant loading.
        simplify_mesh: Whether to simplify mesh.
    Returns:
        dict: Motion sequence data for visualization
    """
    hoi_data = tensor_to(hoi_data, device=device)

    human_data = hoi_data["human_data"]
    model_type = human_data.get("model_type", "smplx")

    if "trans" in human_data:
        trans = human_data["trans"][:, :3]
    elif "transl" in human_data:
        trans = human_data["transl"][:, :3]
    else:
        raise KeyError("human_data must contain either 'trans' or 'transl' key")

    trans_init = trans[:1, :].clone()
    if normalize_trans:
        # move to XY origin
        trans_init[0, 2] = 0
    else:
        trans_init = 0 * trans_init

    frame_num = human_data["poses"].shape[0]

    if human_model is not None:
        human_vertices, human_faces, human_joints = human_model.generate_mesh(
            human_data, output_joints=True, require_grad=False
        )
    elif model_type == "soma03":
        raise ValueError(
            "SOMA03 visualization requires an explicit human_model with bound "
            "GEM-X/SOMA-X assets and identity; estimator provenance alone is insufficient"
        )
    elif model_type == "soma":
        from grail.models.soma_model import generate_soma_mesh, setup_soma_model

        soma_model = setup_soma_model(device=device)
        human_vertices, human_faces, human_joints = generate_soma_mesh(
            soma_model, human_data, output_joints=True, require_grad=False, device=device
        )
    else:
        from grail.models.smplx_model import generate_smplx_mesh, setup_smplx_model

        smplx_model = setup_smplx_model(flat_hand_mean=True, device=device)
        human_vertices, human_faces, human_joints = generate_smplx_mesh(
            smplx_model, human_data, output_joints=True, require_grad=False, device=device
        )

    human_vertices = human_vertices - trans_init
    human_joints = human_joints - trans_init

    human_seq = {
        "joints_pos": human_joints,
        "vertices": human_vertices,
        "triangles": human_faces,
        "rigid": False,
    }

    # Load object mesh (with decimation to match optimizer mesh resolution)
    if obj_mesh_data is not None:
        obj_vertices, obj_faces = obj_mesh_data
        obj_vertices = obj_vertices.to(device)
        obj_faces = obj_faces.to(device)
    else:
        obj_scale = hoi_data["obj_data"]["obj_scale"]
        obj_vertices, obj_faces, _ = load_mesh(
            hoi_data["object_path"],
            mesh_scale=obj_scale,
            target_num_verts=1000 if simplify_mesh else None,
            device=device,
        )

    obj_transform = torch.tile(torch.eye(4), (frame_num, 1, 1)).to(device)
    obj_transform[:, :3, :3] = hoi_data["obj_data"]["obj_R"]
    obj_transform[:, :3, 3] = hoi_data["obj_data"]["obj_t"] - trans_init  # move to XY origin

    obj_R = hoi_data["obj_data"]["obj_R"].float().to(device)
    obj_t = hoi_data["obj_data"]["obj_t"].float().to(device)
    obj_verts_transformed = torch.bmm(
        obj_vertices.float().unsqueeze(0).expand(frame_num, -1, -1), obj_R.transpose(1, 2)
    ) + (obj_t - trans_init.float()).unsqueeze(1)

    obj_seq = {
        "vertices": obj_vertices,
        "faces": obj_faces,
        "vertices_transformed": obj_verts_transformed,
        "triangles": obj_faces,
        "transforms": obj_transform,
        "rigid": True,
    }

    motion_seq = {
        "human_seq": human_seq,
        "obj_seq": obj_seq,
    }

    # Add static objects (e.g., table from OBJ) if static_objects exists
    if hoi_data.get("scene_data") is not None:
        static_objects = hoi_data["scene_data"]
        for obj_key, obj_data in static_objects.items():
            if "vertices" in obj_data or "faces" in obj_data:
                static_vertices, static_faces = embedded_static_mesh(obj_data, device=device)
                static_vertices = static_vertices - trans_init.float()
            elif static_meshes_data is not None and obj_key in static_meshes_data:
                cached = static_meshes_data[obj_key]
                static_vertices = cached["vertices"].to(device).float() - trans_init.float()
                static_faces = cached["faces"].to(device)
            elif obj_key != "table":
                obj_path = obj_data.get("path", "data/Scene/long_table.obj")
                if not os.path.exists(obj_path):
                    continue
                static_vertices, static_faces, _ = load_mesh(
                    obj_path,
                    mesh_scale=obj_data["scale"],
                    target_num_verts=1000 if simplify_mesh else None,
                    device=device,
                )
                static_vertices = static_vertices.float()
                static_vertices = torch.matmul(
                    static_vertices, obj_data["rot"].float().to(device).T
                ) + obj_data["pos"].float().to(device)
                static_vertices = static_vertices - trans_init.float()
            else:
                # fake table created with box mesh
                # this format will be obsoleted in the future
                static_pos = torch.tensor(obj_data["pos"], device=device, dtype=torch.float32)
                static_mesh = trimesh.creation.box(extents=obj_data["size"].cpu().numpy())
                static_vertices = torch.tensor(
                    static_mesh.vertices, device=device, dtype=torch.float32
                )
                static_faces = torch.tensor(static_mesh.faces, device=device)
                static_vertices = static_vertices + static_pos - trans_init

            # Identity transform for all frames (static object)
            static_transform = torch.tile(torch.eye(4), (frame_num, 1, 1)).to(device)

            static_seq = {
                "vertices": static_vertices,
                "triangles": static_faces,
                "transforms": static_transform,
                "rigid": True,
            }
            motion_seq[f"static_{obj_key}_seq"] = static_seq

    return motion_seq if not to_numpy else tensor_to_numpy(motion_seq)


def motion_seq_to_scenepic(motion_seq, hoi_data, normalize_trans=True):
    """Derive ScenePic-ready numpy input from an unnormalized motion sequence."""
    trans = hoi_data["human_data"]["trans"][:, :3]
    trans_init = trans[:1, :].clone()
    if normalize_trans:
        trans_init[0, 2] = 0
    else:
        trans_init.zero_()

    sp_seq = {}

    human = motion_seq["human_seq"]
    sp_seq["human_seq"] = {
        "vertices": (human["vertices"] - trans_init).cpu().numpy(),
        "joints_pos": (human["joints_pos"] - trans_init).cpu().numpy(),
        "triangles": (
            human["triangles"].cpu().numpy()
            if isinstance(human["triangles"], torch.Tensor)
            else human["triangles"]
        ),
        "rigid": human["rigid"],
    }

    obj = motion_seq["obj_seq"]
    obj_transforms = obj["transforms"].clone()
    obj_transforms[:, :3, 3] -= trans_init.squeeze(0)
    sp_seq["obj_seq"] = {
        "vertices": (
            obj["vertices"].cpu().numpy()
            if isinstance(obj["vertices"], torch.Tensor)
            else obj["vertices"]
        ),
        "triangles": (
            obj["triangles"].cpu().numpy()
            if isinstance(obj["triangles"], torch.Tensor)
            else obj["triangles"]
        ),
        "transforms": obj_transforms.cpu().numpy(),
        "rigid": obj["rigid"],
    }

    for key, val in motion_seq.items():
        if key.startswith("static_") and key.endswith("_seq"):
            sp_seq[key] = {
                "vertices": (val["vertices"].float() - trans_init.float().squeeze(0)).cpu().numpy(),
                "triangles": (
                    val["triangles"].cpu().numpy()
                    if isinstance(val["triangles"], torch.Tensor)
                    else val["triangles"]
                ),
                "transforms": (
                    val["transforms"].cpu().numpy()
                    if isinstance(val["transforms"], torch.Tensor)
                    else val["transforms"]
                ),
                "rigid": val["rigid"],
            }

    return sp_seq
