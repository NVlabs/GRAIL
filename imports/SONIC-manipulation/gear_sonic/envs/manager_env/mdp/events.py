"""Event functions for domain randomization and environment resets in RL training."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs.mdp.events import _randomize_prop_by_op
from isaaclab.managers import SceneEntityCfg
import isaaclab.sim.utils as sim_utils_find
from isaaclab.sim import get_current_stage
from isaaclab.utils import configclass
import isaaclab.utils.math as math_utils
from pxr import Gf, Sdf, UsdGeom, Vt
import torch

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


def randomize_rigid_body_scale(
    env: "ManagerBasedEnv",
    env_ids: torch.Tensor | None,
    scale_range: tuple[float, float] | dict[str, tuple[float, float]],
    asset_cfg: SceneEntityCfg,
    relative_child_path: str | None = None,
):
    """Fixed version of isaaclab.envs.mdp.events.randomize_rigid_body_scale.

    The upstream version passes the full prim path as the attribute *name*
    to ``Sdf.AttributeSpec`` instead of just ``"xformOp:scale"`` — the
    resulting AttributeSpec is malformed and the scaling silently no-ops,
    so tables/objects spawn at their authored (unit) scale regardless of
    scale_range. This variant passes just the attribute name and also
    respects any pre-existing ``xformOp:scale`` default as the base scale
    (multiplied by the sample) so that authored per-asset scales are
    preserved when randomizing. Ports the patch used by the gr00t-internal
    HOI sweeps (bp2/bp3/grab).
    """
    if env.sim.is_playing():
        raise RuntimeError(
            "Randomizing scale while simulation is running leads to unpredictable behaviors."
            " Please ensure that the event term is called before the simulation starts."
        )

    asset_name = asset_cfg.name
    if asset_name == "object" and asset_name not in env.scene.rigid_objects:
        if "object_0" in env.scene.rigid_objects:
            asset_name = "object_0"

    asset_names_to_scale = [asset_name]
    if asset_name == "object_0":
        i = 1
        while f"object_{i}" in env.scene.rigid_objects:
            asset_names_to_scale.append(f"object_{i}")
            i += 1

    for cur_asset_name in asset_names_to_scale:
        asset: RigidObject = env.scene[cur_asset_name]

        if isinstance(asset, Articulation):
            raise ValueError(
                "Scaling an articulation randomly is not supported, as it affects joint attributes."
            )

        if env_ids is None:
            cur_env_ids = torch.arange(env.scene.num_envs, device="cpu")
        else:
            cur_env_ids = env_ids.cpu()

        stage = get_current_stage()
        prim_paths = sim_utils_find.find_matching_prim_paths(asset.cfg.prim_path)

        if isinstance(scale_range, dict):
            range_list = [scale_range.get(key, (1.0, 1.0)) for key in ["x", "y", "z"]]
            ranges = torch.tensor(range_list, device="cpu")
            rand_samples = math_utils.sample_uniform(
                ranges[:, 0], ranges[:, 1], (len(cur_env_ids), 3), device="cpu"
            )
        else:
            rand_samples = math_utils.sample_uniform(
                *scale_range, (len(cur_env_ids), 1), device="cpu"
            )
            rand_samples = rand_samples.repeat(1, 3)
        rand_samples = rand_samples.tolist()

        if relative_child_path is None:
            cur_relative_child_path = ""
        elif not relative_child_path.startswith("/"):
            cur_relative_child_path = "/" + relative_child_path
        else:
            cur_relative_child_path = relative_child_path

        with Sdf.ChangeBlock():
            for i, env_id in enumerate(cur_env_ids):
                prim_path = prim_paths[env_id] + cur_relative_child_path
                prim_spec = Sdf.CreatePrimInLayer(stage.GetRootLayer(), prim_path)

                scale_spec = prim_spec.GetAttributeAtPath(prim_path + ".xformOp:scale")
                has_scale_attr = scale_spec is not None
                if not has_scale_attr:
                    scale_spec = Sdf.AttributeSpec(
                        prim_spec, "xformOp:scale", Sdf.ValueTypeNames.Double3
                    )

                base_scale = (1.0, 1.0, 1.0)
                if has_scale_attr and scale_spec.default is not None:
                    sv = scale_spec.default
                    base_scale = (float(sv[0]), float(sv[1]), float(sv[2]))

                final_scale = Gf.Vec3f(
                    base_scale[0] * rand_samples[i][0],
                    base_scale[1] * rand_samples[i][1],
                    base_scale[2] * rand_samples[i][2],
                )
                scale_spec.default = final_scale

                if not has_scale_attr:
                    op_order_spec = prim_spec.GetAttributeAtPath(prim_path + ".xformOpOrder")
                    if op_order_spec is None:
                        op_order_spec = Sdf.AttributeSpec(
                            prim_spec, UsdGeom.Tokens.xformOpOrder, Sdf.ValueTypeNames.TokenArray
                        )
                    op_order_spec.default = Vt.TokenArray(
                        ["xformOp:translate", "xformOp:orient", "xformOp:scale"]
                    )


@configclass
class EventCfg:
    """Configuration for events."""

    # startup
    physics_material = None
    add_joint_default_pos = None
    add_hand_joint_default_pos = None
    base_com = None

    # interval - balance training
    push_robot = None
    force_push_robot = None

    randomize_rigid_body_mass = None
    randomize_table_size = None
    randomize_object_size = None
    randomize_object_mass = None
    randomize_robot_material = None
    randomize_dome_light = None
    randomize_floor_material = None
    randomize_table_material = None
    randomize_object_material = None


def randomize_joint_default_pos(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg,
    pos_distribution_params: tuple[float, float] | None = None,
    operation: Literal["add", "scale", "abs"] = "abs",
    distribution: Literal["uniform", "log_uniform", "gaussian"] = "uniform",
):
    """Randomize joint default positions to simulate calibration errors.

    Applies random offsets to the default joint positions of the robot, modeling
    real-world joint encoder calibration inaccuracies. Also updates the action
    manager offset to keep action space aligned with the new defaults.

    Args:
        env: The environment instance.
        env_ids: Environment indices to randomize. If None, randomizes all.
        asset_cfg: Scene entity config with joint IDs to randomize.
        pos_distribution_params: Min/max range for the position offset distribution.
        operation: How to combine the random value with the original ("add", "scale", "abs").
        distribution: Sampling distribution type.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]

    # save nominal value for export
    asset.data.default_joint_pos_nominal = torch.clone(asset.data.default_joint_pos[0])

    # resolve environment ids
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device=asset.device)

    # resolve joint indices
    if asset_cfg.joint_ids == slice(None):
        joint_ids = slice(None)  # for optimization purposes
    else:
        joint_ids = torch.tensor(asset_cfg.joint_ids, dtype=torch.int, device=asset.device)

    if pos_distribution_params is not None:
        pos = asset.data.default_joint_pos.to(asset.device).clone()
        pos = _randomize_prop_by_op(
            pos,
            pos_distribution_params,
            env_ids,
            joint_ids,
            operation=operation,
            distribution=distribution,
        )[env_ids][:, joint_ids]

        if env_ids != slice(None) and joint_ids != slice(None):
            env_ids = env_ids[:, None]
        asset.data.default_joint_pos[env_ids, joint_ids] = pos
        # update the offset in action since it is not updated automatically

        action_joint_names = env.action_manager.get_term("joint_pos")._joint_names
        asset_joint_names = asset.joint_names
        shared_joint_names = list(set(action_joint_names).intersection(set(asset_joint_names)))
        shared_joint_indices_action = [
            action_joint_names.index(name) for name in shared_joint_names
        ]
        shared_joint_indices_asset = [asset_joint_names.index(name) for name in shared_joint_names]

        shared_offset = asset.data.default_joint_pos[env_ids, shared_joint_indices_asset]
        env.action_manager.get_term("joint_pos")._offset[
            env_ids, shared_joint_indices_action
        ] = shared_offset


def randomize_rigid_body_com(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    com_range: dict[str, tuple[float, float]],
    asset_cfg: SceneEntityCfg,
):
    """Randomize the center of mass (CoM) of rigid bodies by adding a random value sampled from the given ranges.

    .. note::
        This function uses CPU tensors to assign the CoM. It is recommended to use this function
        only during the initialization of the environment.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # resolve environment ids
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device="cpu")
    else:
        env_ids = env_ids.cpu()

    # resolve body indices
    if asset_cfg.body_ids == slice(None):
        body_ids = torch.arange(asset.num_bodies, dtype=torch.int, device="cpu")
    else:
        body_ids = torch.tensor(asset_cfg.body_ids, dtype=torch.int, device="cpu")

    # sample random CoM values
    range_list = [com_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z"]]
    ranges = torch.tensor(range_list, device="cpu")
    rand_samples = math_utils.sample_uniform(
        ranges[:, 0], ranges[:, 1], (len(env_ids), 3), device="cpu"
    ).unsqueeze(1)

    # get the current com of the bodies (num_assets, num_bodies)
    coms = asset.root_physx_view.get_coms().clone()

    # Randomize the com in range
    coms[:, body_ids, :3] += rand_samples

    # Set the new coms
    asset.root_physx_view.set_coms(coms, env_ids)
