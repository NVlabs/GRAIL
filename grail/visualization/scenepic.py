import numpy as np
import scenepic as sp
import torch

from grail.core.video import DEFAULT_MOTION_FPS
from grail.visualization.utils.rotations import quat_between_two_vec, quat_to_exp_map
from grail.visualization.utils.vis_utils import make_checker_board_texture


class ObjectActor:
    def __init__(self, scene, name, vertices, triangles, color="Orange"):
        self.scene = scene
        self.name = name
        self.color = getattr(sp.Colors, color, sp.Colors.Green)
        self.vertices = vertices
        self.triangles = triangles

        self.object_mesh = scene.create_mesh(f"{name}_object", layer_id=f"{self.name}")
        self.color = np.ones((vertices.shape[0], 3)) * self.color[None]
        self.object_mesh.add_mesh_without_normals(vertices, triangles, self.color)

    def add_mesh_to_frames(self, sp_frame, transform=None):
        sp_frame.add_mesh(self.object_mesh, transform=transform)


class SkinActor:
    def __init__(self, scene, name, init_vertices, triangles, color="Lavender"):  # "Mint" Lavender
        self.scene = scene
        self.name = name
        self.color = getattr(sp.Colors, color, sp.Colors.Green)
        self.triangles = triangles

        self.mesh_id = f"{name}_skin"
        self.skin_mesh = scene.create_mesh(self.mesh_id, layer_id=f"{self.name}")
        self.color = np.ones((init_vertices.shape[0], 3)) * self.color[None]
        self.skin_mesh.add_mesh_without_normals(init_vertices, triangles, self.color)

        self.floor_img = scene.create_image(image_id="floor")
        self.floor_img.from_numpy(make_checker_board_texture("#81C6EB", "#D4F1F7"))
        self.floor_mesh = scene.create_mesh(texture_id="floor", layer_id="floor")
        self.floor_mesh.add_image(transform=sp.Transforms.Scale(20))

    def add_mesh_to_frames(self, sp_frame, vertices):
        sp_frame.add_mesh(self.floor_mesh)

        mesh_update = self.scene.update_mesh_positions(self.mesh_id, vertices)
        sp_frame.add_mesh(mesh_update)


class SkeletonActor:
    def __init__(
        self,
        scene,
        name,
        joint_parents,
        joint_color="Yellow",
        bone_color="Green",
        root_color="Yellow",
        joint_constr_color="Cyan",
        joint_radius=0.02,  # 0.06,
        bone_radius=0.04,
        joint_constr_radius=0.1,
        opacity=1.0,
    ):
        self.scene = scene
        self.name = name
        self.joint_parents = joint_parents
        self.joint_radius = joint_radius
        self.joint_color = getattr(sp.Colors, joint_color, sp.Colors.Green)
        self.bone_color = getattr(sp.Colors, bone_color, sp.Colors.Green)
        self.root_color = getattr(sp.Colors, root_color, sp.Colors.Green)
        self.joint_constr_color = getattr(sp.Colors, joint_constr_color, sp.Colors.Green)
        self.bone_radius = bone_radius
        self.joint_meshes = []
        self.joint_constr_meshes = []
        self.bone_meshes = []
        self.bone_pairs = []

        self.floor_img = scene.create_image(image_id="floor")
        self.floor_img.from_numpy(make_checker_board_texture("#81C6EB", "#D4F1F7"))
        self.floor_mesh = scene.create_mesh(texture_id="floor", layer_id="floor")
        self.floor_mesh.add_image(transform=sp.Transforms.Scale(20))

        for j, pa in enumerate(self.joint_parents):
            # joint
            joint_mesh = scene.create_mesh(f"{name}_joint{j}", layer_id=f"{self.name}")
            # joint_mesh.add_sphere(color=self.root_color if j==0 else self.joint_color,
            #                       transform=sp.Transforms.scale(joint_radius))
            joint_mesh.add_sphere(
                color=self.root_color if j == 0 else self.joint_color,
                transform=sp.Transforms.scale(joint_radius),
            )

            self.joint_meshes.append(joint_mesh)
            # joint constraints
            joint_constr_mesh = scene.create_mesh(
                f"{name}_joint_constr{j}", layer_id=f"{self.name}"
            )
            joint_constr_mesh.add_sphere(
                color=self.joint_constr_color,
                transform=sp.Transforms.scale(joint_constr_radius),
            )
            self.joint_constr_meshes.append(joint_constr_mesh)
            # bone
            if pa >= 0:
                bone_mesh = scene.create_mesh(f"{name}_bone{j}", layer_id=f"{self.name}")
                bone_mesh.add_cone(
                    color=self.bone_color,
                    transform=sp.Transforms.scale(np.array([1, joint_radius, joint_radius])),
                )
                self.bone_meshes.append(bone_mesh)
                self.bone_pairs.append((j, pa, bone_mesh))

    def add_mesh_to_frames(self, sp_frame, jpos):
        sp_frame.add_mesh(self.floor_mesh)
        # joint
        for j, pos in enumerate(jpos):
            sp_frame.add_mesh(self.joint_meshes[j], transform=sp.Transforms.translate(pos))

        # bone
        vec = []
        for j, pa, _ in self.bone_pairs:
            vec.append(jpos[j] - jpos[pa])
        vec = np.stack(vec)
        dist = np.linalg.norm(vec, axis=-1)
        vec = torch.tensor(vec / dist[..., None])
        aa = quat_to_exp_map(
            quat_between_two_vec(torch.tensor([-1.0, 0.0, 0.0]).expand_as(vec), vec)
        ).numpy()
        angle = np.linalg.norm(aa, axis=-1, keepdims=True)
        axis = aa / (angle + 1e-6)

        for (j, pa, bone_mesh), angle_i, axis_i, dist_i in zip(self.bone_pairs, angle, axis, dist):
            transform = sp.Transforms.translate((jpos[pa] + jpos[j]) * 0.5)
            transform = transform @ sp.Transforms.RotationMatrixFromAxisAngle(axis_i, angle_i)
            transform = transform @ sp.Transforms.Scale(np.array([dist_i, 1, 1]))
            sp_frame.add_mesh(bone_mesh, transform=transform)

    def add_root_mesh_to_frames(self, sp_frame, jpos, path_only=False):
        sp_frame.add_mesh(self.floor_mesh)
        pos = jpos[0]
        if path_only:
            pos[2] = 0.0
        sp_frame.add_mesh(self.joint_meshes[0], transform=sp.Transforms.translate(pos))

    def add_joint_constr_meshes_to_frames(self, sp_frame, jpos, joint_mask):
        # joint constraints
        for j, pos in enumerate(jpos):
            if joint_mask[j].any():
                sp_frame.add_mesh(
                    self.joint_constr_meshes[j], transform=sp.Transforms.translate(pos)
                )

    def add_joints_mesh_to_frames(
        self, sp_frame, jpos, joint_inds=[21, 20, 7, 8], proj_to_floor=False
    ):
        sp_frame.add_mesh(self.floor_mesh)
        for idx in joint_inds:
            cur_pos = jpos[idx]
            if (
                proj_to_floor and idx == 0
            ):  # root TODO: this is hacky, shouldn't assume only the root is projected
                cur_pos = np.copy(cur_pos)
                cur_pos[2] = 0.0
            sp_frame.add_mesh(self.joint_meshes[idx], transform=sp.Transforms.translate(cur_pos))


class ScenepicVisualizer:
    def __init__(self):
        super().__init__()

        self.color_sequences = [
            ["Yellow", "Green", "Teal"],
            ["Yellow", "Red", "Teal"],
            ["Yellow", "Blue", "Teal"],
            ["Yellow", "Purple", "Teal"],
            ["Yellow", "Orange", "Teal"],
        ]
        self.object_colors = [
            "Maroon",
            "Orange",
            "Navy",
            "Pink",
            "Brown",
            "Coral",
            "Olive",
            "Beige",
            "Lime",
            "Mint",
            "Purple",
            "Magenta",
        ]

    def load_default_camera(self):
        return sp.Camera(
            center=(5, 0, 1.5),
            look_at=(0, 0, 0.8),
            up_dir=(0, 0, 1),
            fov_y_degrees=45.0,
            aspect_ratio=1.0,
        )

    def vis_scene(self, seq=None, html_path=None, window_size=(400, 400), fps=DEFAULT_MOTION_FPS):
        scene = self.generate_scene(seq, window_size=window_size, fps=fps)
        scene.save_as_html(html_path)

    def vis_scenes(self, seqs=None, html_path=None, window_size=(400, 400), fps=DEFAULT_MOTION_FPS):
        # Create a ScenePic instance
        sp_scene = sp.Scene()
        for idx, seq in enumerate(seqs):
            # Create multiple scenes
            scene = sp_scene.create_scene(f"scene_{idx}")
            self.generate_scene(seq, window_size=window_size, fps=fps, scene=scene)
        # Saving the scene
        sp_scene.save_as_html(html_path)

    def generate_scene(
        self,
        seq=None,
        window_size=None,
        return_canvas=False,
        fps=DEFAULT_MOTION_FPS,
        scene=None,
    ):
        if scene is None:
            scene = sp.Scene()
        scene.framerate = fps
        canvas = scene.create_canvas_3d(width=window_size[0], height=window_size[1])

        if "joints_pos" in seq:  # single person
            seq = {"skel0": seq}
        seq = {k: v.copy() for k, v in seq.items()}  # copy to avoid inplace modification

        num_fr = -1
        for i, (skel_name, pose_dict) in enumerate(seq.items()):
            colors = self.color_sequences[i % len(self.color_sequences)]

            if "vertices" in pose_dict:
                if pose_dict.get("rigid", False):
                    pose_dict["object_actor"] = ObjectActor(
                        scene,
                        f"{skel_name}_object",
                        pose_dict["vertices"],
                        pose_dict["triangles"],
                        color=self.object_colors[i % len(self.object_colors)],
                    )
                else:
                    pose_dict["skin_actor"] = SkinActor(
                        scene,
                        f"{skel_name}_skin",
                        pose_dict["vertices"][0],
                        pose_dict["triangles"],
                    )
                    num_fr = max(num_fr, pose_dict["vertices"].shape[0])

            # if "offset" in pose_dict:
            #     pose_dict["joints_pos"] = pose_dict["joints_pos"] + pose_dict["offset"]
            # the predicted motion
            # pose_dict["skeleton_jpos"] = SkeletonActor(
            #     scene,
            #     f"{skel_name}_jpos",
            #     pose_dict["joint_parents"],
            #     joint_color=colors[0],
            #     bone_color=colors[1],
            #     joint_constr_color="Brown" if skel_name == "gt" else "Cyan",
            # )
            # # dummy joints on the feet to show contacts
            # if "foot_contacts" in pose_dict:
            #     pose_dict["foot_contacts_jpos"] = SkeletonActor(
            #         scene,
            #         f"{skel_name}_foot_contacts_jpos",
            #         self.joint_parents,
            #         joint_color="Red",
            #         bone_color="Yellow",
            #         root_color="Red",
            #         joint_radius=0.07,
            #     )

        for fr in range(num_fr):
            main_frame = canvas.create_frame()
            main_frame.camera = self.load_default_camera()

            for skel_name, pose_dict in seq.items():
                if "object_actor" in pose_dict:
                    ind = min(fr, pose_dict["transforms"].shape[0] - 1)
                    pose_dict["object_actor"].add_mesh_to_frames(
                        main_frame, pose_dict["transforms"][ind]
                    )
                    continue
                if "skin_actor" in pose_dict:
                    ind = min(fr, pose_dict["vertices"].shape[0] - 1)
                    pose_dict["skin_actor"].add_mesh_to_frames(
                        main_frame, pose_dict["vertices"][ind]
                    )
                    continue

                # ind = min(fr, pose_dict["joints_pos"].shape[0] - 1)

                # always add current pose
                # pose_dict["skeleton_jpos"].add_mesh_to_frames(
                #     main_frame, pose_dict["joints_pos"][ind].cpu().numpy()
                # )

                # if "foot_contacts" in pose_dict:
                #     cur_contacts = pose_dict["foot_contacts"][ind]
                #     if cur_contacts.sum() > 0:
                #         foot_inds = np.array(self.skeleton.foot_joint_idx)
                #         contact_inds = foot_inds[cur_contacts]
                #         pose_dict["foot_contacts_jpos"].add_joints_mesh_to_frames(
                #             main_frame,
                #             pose_dict["joints_pos"][ind].cpu().numpy(),
                #             contact_inds,
                #         )

            # if 'text' in seq:
            #     label = scene.create_label(text=seq['text'], color=sp.Colors.White, size_in_pixels=60, offset_distance=0.0, horizontal_align='center', camera_space=True)
            #     main_frame.add_label(label=label, position=[0.0, 1.5, -5.0])
        if return_canvas:
            return scene, canvas
        else:
            return scene
