"""HOI visualization, extracted from HOIOptimizer."""

import os

import cv2
import imageio
import numpy as np
import torch
from pytorch3d.structures import join_meshes_as_scene
from tqdm import tqdm

from grail.core.io import load_mesh, save_mesh
from grail.core.torch_utils import tensor_to
from grail.rendering.camera import get_camera
from grail.rendering.renderer import create_renderer, render_frame
from grail.rendering.textures import create_colored_meshes, create_mesh_with_vertex_colors
from grail.visualization.scenepic import ScenepicVisualizer
from grail.visualization.utils.vis_utils import motion_seq_to_scenepic, prep_visualizer_input


class HOIVisualizer:
    """Renders HOI optimization results as videos and interactive HTML."""

    def __init__(self, device, human_model, cameras, image_list, video_fps, log_dir, obj_path):
        self.device = device
        self.human_model = human_model
        self.cameras = cameras
        self.image_list = image_list
        self.video_fps = video_fps
        self.log_dir = log_dir
        self.obj_path = obj_path
        self._vis_obj_mesh_data = None
        self._vis_static_meshes_data = {}
        self.sp_visualizer = None

    def init_vis_meshes(self, data):
        """Load simplified meshes once for visualization."""
        obj_scale = data.obj.scale
        vis_obj_verts, vis_obj_faces, _ = load_mesh(
            self.obj_path, mesh_scale=obj_scale, target_num_verts=1000, device=self.device
        )
        self._vis_obj_mesh_data = (vis_obj_verts, vis_obj_faces)

        self._vis_static_meshes_data = {}
        static_objects = data.static_objects
        if static_objects is not None:
            for obj_key, obj_data in static_objects.items():
                if obj_key != "table":
                    obj_path = obj_data.get("path", "data/Scene/long_table.obj")
                    if not os.path.exists(obj_path):
                        continue
                    static_vertices, static_faces, _ = load_mesh(
                        obj_path,
                        mesh_scale=obj_data["scale"],
                        target_num_verts=1000,
                        device=self.device,
                    )
                    static_vertices = static_vertices.float()
                    static_vertices = torch.matmul(
                        static_vertices, obj_data["rot"].float().to(self.device).T
                    ) + obj_data["pos"].float().to(self.device)
                    self._vis_static_meshes_data[obj_key] = {
                        "vertices": static_vertices,
                        "faces": static_faces,
                    }

    def visualize(self, data, pred, hoi_data, vis_name, vis_cfg):
        """Render and save visualization videos and HTML."""
        hoi_data["object_path"] = self.obj_path
        motion_seq = prep_visualizer_input(
            hoi_data,
            human_model=self.human_model,
            normalize_trans=False,
            to_numpy=False,
            device=self.device,
            obj_mesh_data=self._vis_obj_mesh_data,
            static_meshes_data=self._vis_static_meshes_data,
        )

        human_verts_seq = motion_seq["human_seq"]["vertices"]
        human_faces = data.human.faces
        obj_verts_seq = pred.obj.verts_seq
        obj_faces = data.obj.faces
        frame_num = human_verts_seq.shape[0]

        human_color = torch.tensor([0.8, 0.6, 0.4], device=self.device)
        obj_color = torch.tensor([0.4, 0.6, 0.8], device=self.device)

        vis_dir = os.path.join(self.log_dir, vis_name)
        os.makedirs(vis_dir, exist_ok=True)

        export_mesh = vis_cfg.get("export_mesh", False)
        if export_mesh:
            export_dir = os.path.join(vis_dir, "meshes")
            os.makedirs(export_dir, exist_ok=True)

        render_video = vis_cfg.get("render_video", False)
        vis_contact = vis_cfg.get("vis_contact", True)

        renderer = create_renderer(
            self.cameras, (data.camera.frame_height, data.camera.frame_width), device=self.device
        )

        # Get hand vertex indices for contact visualization
        left_hand_indices = right_hand_indices = None
        if vis_contact:
            left_hand_indices = self.human_model.get_segment_indices(["L_Hand"])
            right_hand_indices = self.human_model.get_segment_indices(["R_Hand"])

        inter_start_idx = data.inter_start_idx
        inter_end_idx = data.inter_end_idx

        if render_video:
            # Main camera view
            self._render_video(
                os.path.join(vis_dir, f"{vis_name}.mp4"),
                frame_num,
                human_verts_seq,
                human_faces,
                obj_verts_seq,
                obj_faces,
                human_color,
                obj_color,
                self.cameras,
                renderer,
                vis_contact,
                left_hand_indices,
                right_hand_indices,
                inter_start_idx,
                inter_end_idx,
                vis_name,
                background_images=self.image_list,
                export_mesh=export_mesh,
                export_dir=export_dir if export_mesh else None,
            )

            # Extra views
            extra_views = self._setup_extra_views(
                vis_cfg.get("extra_views", ["top"]),
                human_verts_seq,
                obj_verts_seq,
                data,
            )

            for view_name, view_cam, view_renderer in extra_views:
                # A table obscures the human-object interaction from above and
                # adds no useful context there. Keep every other static scene
                # object, and retain tables in side/front views.
                static_object_meshes = self._build_static_meshes(
                    motion_seq, exclude_tables=view_name == "top_view"
                )
                self._render_video(
                    os.path.join(vis_dir, f"{vis_name}_{view_name}.mp4"),
                    frame_num,
                    human_verts_seq,
                    human_faces,
                    obj_verts_seq,
                    obj_faces,
                    human_color,
                    obj_color,
                    view_cam,
                    view_renderer,
                    vis_contact,
                    left_hand_indices,
                    right_hand_indices,
                    inter_start_idx,
                    inter_end_idx,
                    f"{vis_name} ({view_name})",
                    background_images=None,
                    static_meshes=static_object_meshes,
                )

            # Create symlinks
            self._create_result_symlinks(vis_name, extra_views)

        else:
            # No video — just export meshes
            for i in tqdm(range(frame_num), desc=f"Visualizing {vis_name}"):
                hoi_mesh = self._build_frame_mesh(
                    i,
                    human_verts_seq,
                    human_faces,
                    obj_verts_seq,
                    obj_faces,
                    human_color,
                    obj_color,
                    vis_contact,
                    left_hand_indices,
                    right_hand_indices,
                )
                if export_mesh:
                    save_mesh(
                        hoi_mesh.verts_packed(),
                        hoi_mesh.faces_packed(),
                        f"{export_dir}/{i:06d}.obj",
                    )

        if vis_cfg.get("vis_html", False) and self.sp_visualizer is not None:
            vis_input = motion_seq_to_scenepic(motion_seq, hoi_data)
            self.sp_visualizer.vis_scene(
                vis_input,
                os.path.join(vis_dir, f"{vis_name}.html"),
                window_size=(400, 400),
                fps=self.video_fps,
            )

    # ── Private helpers ──────────────────────────────────────────────────────

    def _render_video(
        self,
        output_path,
        frame_num,
        human_verts_seq,
        human_faces,
        obj_verts_seq,
        obj_faces,
        human_color,
        obj_color,
        camera,
        renderer,
        vis_contact,
        left_hand_indices,
        right_hand_indices,
        inter_start_idx,
        inter_end_idx,
        desc,
        background_images=None,
        static_meshes=None,
        export_mesh=False,
        export_dir=None,
    ):
        """Render a video from a single camera viewpoint."""
        with imageio.get_writer(output_path, fps=self.video_fps) as writer:
            for i in tqdm(range(frame_num), desc=f"Visualizing {desc}"):
                hoi_mesh = self._build_frame_mesh(
                    i,
                    human_verts_seq,
                    human_faces,
                    obj_verts_seq,
                    obj_faces,
                    human_color,
                    obj_color,
                    vis_contact,
                    left_hand_indices,
                    right_hand_indices,
                    static_meshes=static_meshes,
                )

                if export_mesh and export_dir:
                    save_mesh(
                        hoi_mesh.verts_packed(),
                        hoi_mesh.faces_packed(),
                        f"{export_dir}/{i:06d}.obj",
                    )

                image = self._render_composited_frame(
                    hoi_mesh,
                    camera,
                    renderer,
                    bg_image_path=background_images[i] if background_images else None,
                )
                image = self._annotate_frame(image, i, inter_start_idx, inter_end_idx)
                writer.append_data(image)

    def _build_frame_mesh(
        self,
        frame_idx,
        human_verts_seq,
        human_faces,
        obj_verts_seq,
        obj_faces,
        human_color,
        obj_color,
        vis_contact,
        left_hand_indices,
        right_hand_indices,
        static_meshes=None,
    ):
        """Build a single-frame HOI mesh with optional contact coloring."""
        if vis_contact and left_hand_indices is not None:
            human_verts = human_verts_seq[frame_idx]
            obj_verts = obj_verts_seq[frame_idx]
            human_colors, obj_colors = self._compute_contact_colors(
                human_verts,
                obj_verts,
                left_hand_indices,
                right_hand_indices,
                human_color,
                obj_color,
            )
            human_mesh = create_mesh_with_vertex_colors(
                human_verts, human_faces, human_colors, device=self.device
            )
            obj_mesh = create_mesh_with_vertex_colors(
                obj_verts, obj_faces, obj_colors, device=self.device
            )
        else:
            human_mesh = create_colored_meshes(human_verts_seq[frame_idx], human_faces, human_color)
            obj_mesh = create_colored_meshes(obj_verts_seq[frame_idx], obj_faces, obj_color)

        scene_meshes = [human_mesh, obj_mesh]
        if static_meshes:
            scene_meshes.extend(static_meshes)
        return join_meshes_as_scene(scene_meshes)

    def _compute_contact_colors(
        self,
        human_verts,
        obj_verts,
        left_hand_indices,
        right_hand_indices,
        human_color,
        obj_color,
        contact_threshold=0.05,
    ):
        """Compute per-vertex colors based on hand-object contact proximity."""
        red_color = torch.tensor([1.0, 0.0, 0.0], device=self.device)
        human_colors = human_color.unsqueeze(0).repeat(human_verts.shape[0], 1)
        obj_colors = obj_color.unsqueeze(0).repeat(obj_verts.shape[0], 1)

        for hand_indices in [left_hand_indices, right_hand_indices]:
            hand_indices_tensor = torch.tensor(hand_indices, device=self.device, dtype=torch.long)
            hand_verts = human_verts[hand_indices_tensor]

            diff = obj_verts.unsqueeze(1) - hand_verts.unsqueeze(0)
            distances = torch.norm(diff, dim=-1)

            min_dist_per_obj, _ = distances.min(dim=1)
            obj_colors[min_dist_per_obj < contact_threshold] = red_color

            min_dist_per_hand, _ = distances.min(dim=0)
            hand_contact_mask = min_dist_per_hand < contact_threshold
            human_colors[hand_indices_tensor[hand_contact_mask]] = red_color

        return human_colors, obj_colors

    def _render_composited_frame(self, hoi_mesh, camera, renderer, bg_image_path=None):
        """Render a mesh and composite onto background (original image or white)."""
        image, alpha = render_frame(hoi_mesh, camera, renderer)
        image = image.cpu().numpy() * 255
        alpha = (alpha.cpu().numpy() > 0.5).astype(np.float32)[..., None]

        if bg_image_path is not None:
            bg = cv2.imread(bg_image_path)
            bg = cv2.cvtColor(bg, cv2.COLOR_BGR2RGB).astype(np.float32)
        else:
            bg = np.ones_like(image) * 255

        return (image * alpha + bg * (1 - alpha)).astype(np.uint8)

    @staticmethod
    def _annotate_frame(image, frame_idx, inter_start_idx, inter_end_idx, border_thickness=8):
        """Add frame index text and red border at interaction boundaries."""
        image = image.copy()
        cv2.putText(
            image,
            f"{frame_idx:03d}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (255, 0, 0),
            2,
            cv2.LINE_AA,
        )
        if frame_idx == inter_start_idx or frame_idx == inter_end_idx - 1:
            h, w = image.shape[:2]
            cv2.rectangle(image, (0, 0), (w - 1, h - 1), (255, 0, 0), border_thickness)
        return image

    def _setup_extra_views(self, enabled_views, human_verts_seq, obj_verts_seq, data):
        """Create extra camera views based on scene bounds."""
        if not enabled_views:
            return []

        all_verts_first = torch.cat([human_verts_seq[0], obj_verts_seq[0]], dim=0)
        scene_center = (all_verts_first.min(dim=0)[0] + all_verts_first.max(dim=0)[0]) / 2
        cam_distance = (
            all_verts_first.max(dim=0)[0] - all_verts_first.min(dim=0)[0]
        ).max().item() * 2.5

        image_size = (data.camera.frame_height, data.camera.frame_width)
        focal = data.camera.focal_length

        view_configs = {
            "top": {
                "R": torch.tensor(
                    [[-1, 0, 0], [0, 1, 0], [0, 0, -1]], dtype=torch.float32, device=self.device
                ),
                "T_offset": torch.tensor([0, 0, cam_distance], device=self.device),
            },
            "side": {
                "R": torch.tensor(
                    [[0, 0, 1], [1, 0, 0], [0, 1, 0]], dtype=torch.float32, device=self.device
                ),
                "T_offset": torch.tensor([-cam_distance, 0, 0], device=self.device),
            },
            "front": {
                "R": torch.tensor(
                    [[-1, 0, 0], [0, 0, 1], [0, 1, 0]], dtype=torch.float32, device=self.device
                ),
                "T_offset": torch.tensor([0, -cam_distance, 0], device=self.device),
            },
        }

        extra_views = []
        for vname in enabled_views:
            if vname not in view_configs:
                print(f"Warning: unknown extra view '{vname}', skipping")
                continue
            vcfg = view_configs[vname]
            cam_T = scene_center + vcfg["T_offset"]
            cam = get_camera(vcfg["R"], cam_T, focal, image_size, device=self.device)
            view_renderer = create_renderer(cam, image_size, device=self.device)
            extra_views.append((f"{vname}_view", cam, view_renderer))
        return extra_views

    def _build_static_meshes(self, motion_seq, exclude_tables=False):
        """Build static object meshes (e.g. table) from motion_seq data."""
        static_meshes = []
        table_color = [0.7, 0.7, 0.7]
        for seq_key in motion_seq.keys():
            if seq_key.startswith("static_") and seq_key.endswith("_seq"):
                object_key = seq_key[len("static_") : -len("_seq")]
                normalized_key = object_key.lower()
                is_table = (
                    normalized_key == "table"
                    or normalized_key.startswith("table")
                    or normalized_key.endswith("_table")
                )
                if exclude_tables and is_table:
                    continue
                static_seq = motion_seq[seq_key]
                static_verts = tensor_to(static_seq["vertices"], device=self.device)
                static_faces = tensor_to(static_seq["triangles"], device=self.device)
                static_meshes.append(create_colored_meshes(static_verts, static_faces, table_color))
        return static_meshes

    def _create_result_symlinks(self, vis_name, extra_views):
        """Create symlinks from result files to the log directory."""
        os.system(f"rm -f {os.path.join(self.log_dir, 'result.mp4')}")
        for view_name, _, _ in extra_views:
            os.system(f"rm -f {os.path.join(self.log_dir, f'result_{view_name}.mp4')}")
            os.system(
                f"ln -sf {os.path.join(vis_name, f'{vis_name}_{view_name}.mp4')} "
                f"{os.path.join(self.log_dir, f'result_{view_name}.mp4')}"
            )
        os.system(
            f"ln -sf {os.path.join(vis_name, f'{vis_name}.mp4')} "
            f"{os.path.join(self.log_dir, 'result.mp4')}"
        )
        os.system(
            f"ln -sf {os.path.join(vis_name, f'{vis_name}.html')} "
            f"{os.path.join(self.log_dir, 'result.html')}"
        )
