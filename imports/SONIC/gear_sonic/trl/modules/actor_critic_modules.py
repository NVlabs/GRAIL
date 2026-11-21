"""Actor and Critic modules for PPO-based reinforcement learning."""

from __future__ import annotations

from copy import deepcopy

from tensordict import TensorDict
import torch
from torch.distributions import Normal
import torch.nn as nn

from gear_sonic.trl.utils.common import Timer, custom_instantiate
from gear_sonic.trl.utils.rl import compute_episode_attnmask
from gear_sonic.utils.batch_normalizer import BatchNormNormalizer
from gear_sonic.utils.running_mean_std import RunningMeanStd



def backbone_consumes_episode_attnmask(module) -> bool:
    """False for the plain MLP BaseModule forward, which accepts and ignores every keyword argument; True for any backbone
    that overrides forward (sequence / transformer / aux modules), which may consume ``episode_attnmask``."""
    from gear_sonic.trl.modules.base_module import BaseModule

    return not (isinstance(module, BaseModule) and type(module).forward is BaseModule.forward)


class Actor(nn.Module):
    """Policy network that maps observations to action distributions.

    Wraps an arbitrary backbone network and adds a diagonal Gaussian action
    distribution on top. Supports both direct ``std`` and ``log_std``
    parameterizations for exploration noise, with optional clamping.

    The actor maintains an observation buffer for temporal models (e.g.
    transformers) that require a history of past observations. During rollout,
    observations are appended to this buffer up to ``max_rollout_history``
    steps, and an episode attention mask is computed from done signals so the
    backbone can attend only within episode boundaries.
    """

    def __init__(
        self,
        env_config,
        algo_config,
        backbone,
        obs_dim_dict=None,
        module_dim_dict={},
        running_mean_std=False,
        use_batch_norm=False,
        max_rollout_history=1,
        input_key="actor_obs",
        input_obs_dict=False,
        has_aux_loss=False,
        output_original_obs_dict=False,
        backbone_kwargs={},
    ):
        """Initialize the Actor.

        Args:
            env_config: Environment configuration containing robot specs.
            algo_config: Algorithm configuration (noise std, clamping, etc.).
            backbone: Hydra-style config for the backbone network to instantiate.
            obs_dim_dict: Mapping from observation keys to their dimensions.
                Defaults to ``env_config.robot.algo_obs_dim_dict``.
            module_dim_dict: Additional dimension info passed to the backbone.
            running_mean_std: Whether to normalize inputs with running statistics.
            use_batch_norm: Whether to normalize inputs with batch normalization.
            max_rollout_history: Number of past timesteps to keep in the
                observation buffer for temporal models.
            input_key: Key in ``obs_dict`` used as network input.
            input_obs_dict: If True, pass the entire obs dict to the backbone
                instead of a single tensor.
            has_aux_loss: Whether the backbone produces auxiliary losses
                (e.g. commitment loss from VQ).
            output_original_obs_dict: If True, include the observation dict
                in the output TensorDict.
            backbone_kwargs: Extra keyword arguments forwarded to the backbone
                constructor.
        """
        super().__init__()

        self.algo_config = algo_config
        self.env_config = env_config
        if obs_dim_dict is None:
            obs_dim_dict = env_config.robot.algo_obs_dim_dict
        self.input_key = input_key
        self.input_obs_dict = input_obs_dict
        self.has_aux_loss = has_aux_loss
        self.aux_losses = None
        self.aux_loss_coef = None
        self.output_original_obs_dict = output_original_obs_dict
        self.max_rollout_history = max_rollout_history
        self.actor_module = custom_instantiate(
            backbone,
            env_config=env_config,
            algo_config=algo_config,
            obs_dim_dict=obs_dim_dict,
            module_dim_dict=module_dim_dict,
            _resolve=False,
            **backbone_kwargs,
        )
        self.use_batch_norm = use_batch_norm

        self.use_running_mean_std = running_mean_std

        self.running_mean_std = None
        if running_mean_std:
            self.running_mean_std = RunningMeanStd(
                (obs_dim_dict[self.input_key],), per_channel=True
            )

        if use_batch_norm:
            self.running_mean_std = BatchNormNormalizer((obs_dim_dict[self.input_key],))

        assert not (
            running_mean_std and use_batch_norm
        ), "running_mean_std and use_batch_norm cannot be both True"

        # Action noise
        self.num_actions = self.env_config.robot.actions_dim
        init_noise_std = algo_config.init_noise_std
        self.init_noise_std = float(init_noise_std)

        # Support both std and log_std parameterization
        # Using log_std ensures exp(log_std) > 0, which is numerically more stable
        self.use_log_std = algo_config.get("use_log_std", False)
        if self.use_log_std:
            self.log_std = nn.Parameter(torch.log(init_noise_std * torch.ones(self.num_actions)))
        else:
            self.std = nn.Parameter(init_noise_std * torch.ones(self.num_actions))

        if algo_config.get("freeze_noise_std", False):
            if self.use_log_std:
                self.log_std.requires_grad = False
            else:
                self.std.requires_grad = False

        self.clamp_noise_std = algo_config.get("clamp_noise_std", False)
        if self.clamp_noise_std:
            self.max_noise_std = algo_config.get("max_noise_std", 1.0)

        # Optional per-slice noise configuration for heterogeneous action spaces.
        # This is used by RP2 experiments whose policy output contains latent,
        # joint-space residual, and hand-primitive coordinates with different units.
        self.action_std_group_slices = {}
        self._has_action_std_groups = False
        self._configure_action_std_groups(algo_config.get("action_std_groups", None))
        self._zero_action_mean_slices(algo_config.get("zero_action_mean_slices", None))

        contact_std_cfg = algo_config.get("contact_phase_std", {}) or {}
        self.contact_phase_std_enabled = bool(contact_std_cfg.get("enabled", False))
        self.contact_phase_std_obs_key = contact_std_cfg.get("obs_key", "contact_phase")
        self.contact_phase_std_far_scale = float(contact_std_cfg.get("far_scale", 1.0))
        self.contact_phase_std_contact_scale = float(contact_std_cfg.get("contact_scale", 0.25))
        self.contact_phase_std_min = float(contact_std_cfg.get("min_std", 1e-6))
        if self.contact_phase_std_enabled:
            print(  # noqa: T201
                "[Actor] contact_phase_std enabled: "
                f"obs_key={self.contact_phase_std_obs_key}, "
                f"far_scale={self.contact_phase_std_far_scale}, "
                f"contact_scale={self.contact_phase_std_contact_scale}"
            )
        self.safe_nan = False  # Propagated from env wrapper via ppo_trainer

        self.distribution = None
        # disable args validation for speedup
        Normal.set_default_validate_args(False)

        # Initialize observation buffer for rollout
        self.obs_dict_buffer = TensorDict()
        self.dones_buffer = None
        self.steps = 0
        self.is_eval_mode = False

    def reset(self, dones=None):
        pass

    def _configure_action_std_groups(self, groups_cfg):
        """Initialize optional action slices with independent std constraints."""
        if not groups_cfg:
            return

        if hasattr(groups_cfg, "items"):
            groups = list(groups_cfg.items())
        else:
            groups = [(str(index), cfg) for index, cfg in enumerate(groups_cfg)]

        if self.algo_config.get("use_clampped_std", False):
            default_min = float(self.algo_config.std_clamp_min)
            default_max = float(self.algo_config.std_clamp_max)
        elif self.clamp_noise_std:
            default_min = 1.0e-6
            default_max = float(self.max_noise_std)
        else:
            default_min = 1.0e-6
            default_max = float("inf")

        std_min = torch.full((self.num_actions,), default_min, dtype=torch.float32)
        std_max = torch.full((self.num_actions,), default_max, dtype=torch.float32)
        fixed_mask = torch.zeros(self.num_actions, dtype=torch.bool)
        fixed_value = torch.zeros(self.num_actions, dtype=torch.float32)

        covered = torch.zeros(self.num_actions, dtype=torch.bool)
        for name, cfg in groups:
            start = int(cfg.get("start", 0))
            end = int(cfg.get("end", 0))
            if start < 0 or end <= start or end > self.num_actions:
                raise ValueError(
                    f"Invalid action std group '{name}' range [{start}, {end}) "
                    f"for {self.num_actions} actions"
                )
            if covered[start:end].any():
                raise ValueError(f"Action std group '{name}' overlaps an earlier group")

            init_std = float(cfg.get("init_std", self.init_noise_std))
            group_min = float(cfg.get("min_std", default_min))
            group_max = float(cfg.get("max_std", default_max))
            is_fixed = bool(cfg.get("fixed", False))
            if init_std <= 0.0 or group_min <= 0.0 or group_max < group_min:
                raise ValueError(
                    f"Invalid std values for action group '{name}': "
                    f"init={init_std}, min={group_min}, max={group_max}"
                )
            if not group_min <= init_std <= group_max:
                raise ValueError(
                    f"Initial std {init_std} for action group '{name}' is outside "
                    f"[{group_min}, {group_max}]"
                )

            with torch.no_grad():
                if self.use_log_std:
                    self.log_std[start:end].fill_(float(torch.log(torch.tensor(init_std))))
                else:
                    self.std[start:end].fill_(init_std)
            std_min[start:end] = group_min
            std_max[start:end] = group_max
            if is_fixed:
                fixed_mask[start:end] = True
                fixed_value[start:end] = init_std
            covered[start:end] = True
            self.action_std_group_slices[str(name)] = (start, end)

        self.register_buffer("_action_std_min", std_min, persistent=False)
        self.register_buffer("_action_std_max", std_max, persistent=False)
        self.register_buffer("_action_std_fixed_mask", fixed_mask, persistent=False)
        self.register_buffer("_action_std_fixed_value", fixed_value, persistent=False)
        self._has_action_std_groups = True
        print(  # noqa: T201
            "[Actor] action std groups: "
            + ", ".join(
                f"{name}=[{start}:{end}]"
                for name, (start, end) in self.action_std_group_slices.items()
            )
        )

    def _zero_action_mean_slices(self, slices_cfg):
        """Zero selected rows of the actor's final action-mean projection."""
        if not slices_cfg:
            return

        candidates = [
            module
            for module in self.actor_module.modules()
            if isinstance(module, nn.Linear) and module.out_features == self.num_actions
        ]
        if not candidates:
            raise ValueError(
                "zero_action_mean_slices was configured, but no final Linear layer "
                f"with {self.num_actions} outputs was found"
            )
        output_layer = candidates[-1]
        normalized_slices = []
        with torch.no_grad():
            for bounds in slices_cfg:
                start, end = int(bounds[0]), int(bounds[1])
                if start < 0 or end <= start or end > self.num_actions:
                    raise ValueError(
                        f"Invalid zero-action-mean range [{start}, {end}) "
                        f"for {self.num_actions} actions"
                    )
                output_layer.weight[start:end].zero_()
                if output_layer.bias is not None:
                    output_layer.bias[start:end].zero_()
                normalized_slices.append((start, end))
        print(f"[Actor] zero-initialized action mean slices: {normalized_slices}")  # noqa: T201

    def action_std_group_means(self):
        """Return detached mean std values for configured action groups."""
        if not self._has_action_std_groups:
            return {}
        std = self.get_std.detach()
        return {
            name: std[start:end].mean().item()
            for name, (start, end) in self.action_std_group_slices.items()
        }

    @property
    def get_std(self):
        """Get the standard deviation, handling both std and log_std parameterizations."""
        if self._has_action_std_groups:
            if self.use_log_std:
                raw_std = torch.exp(torch.clamp(self.log_std, min=-20.0, max=20.0))
            else:
                raw_std = self.std
            std = torch.maximum(
                torch.minimum(raw_std, self._action_std_max), self._action_std_min
            )
            std = torch.where(
                self._action_std_fixed_mask,
                self._action_std_fixed_value,
                std,
            )
            return std

        if self.use_log_std:
            # First, handle NaN or inf in log_std
            if torch.any(torch.isnan(self.log_std)) or torch.any(torch.isinf(self.log_std)):
                print("[ERROR] log_std contains NaN or Inf! Resetting to safe values.")
                with torch.no_grad():
                    self.log_std.data = torch.log(torch.ones_like(self.log_std) * 0.5)

            # Apply clamping if configured before computing std
            if self.algo_config.get("use_clampped_std", False):
                std_min = self.algo_config.std_clamp_min
                std_max = self.algo_config.std_clamp_max
                log_std_clamped = torch.clamp(
                    self.log_std,
                    min=torch.log(
                        torch.tensor(std_min, dtype=self.log_std.dtype, device=self.log_std.device)
                    ),
                    max=torch.log(
                        torch.tensor(std_max, dtype=self.log_std.dtype, device=self.log_std.device)
                    ),
                )
                std = torch.exp(log_std_clamped)
                std = torch.clamp(std, min=std_min, max=std_max)
                return std

            if self.clamp_noise_std:
                log_std_clamped = torch.clamp(
                    self.log_std,
                    max=torch.log(
                        torch.tensor(
                            self.max_noise_std, dtype=self.log_std.dtype, device=self.log_std.device
                        )
                    ),
                )
                std = torch.exp(log_std_clamped)
                std = torch.clamp(std, min=1e-6)
                return std

            # Default case: clamp log_std to prevent extreme values
            log_std_clamped = torch.clamp(self.log_std, min=-20, max=2)
            std = torch.exp(log_std_clamped)
            std = torch.clamp(std, min=1e-6)
            return std
        else:
            # Original std parameterization with in-place clamping
            if self.algo_config.get("use_clampped_std", False):
                with torch.no_grad():
                    self.std.clamp_(
                        min=self.algo_config.std_clamp_min, max=self.algo_config.std_clamp_max
                    )
            if self.clamp_noise_std:
                with torch.no_grad():
                    self.std.clamp_(max=self.max_noise_std)
            return self.std

    def _apply_contact_phase_std(self, std: torch.Tensor, obs_dict: dict, mean: torch.Tensor):
        """Scale action std per sample from a motion-reference contact phase."""
        if not self.contact_phase_std_enabled or self.contact_phase_std_obs_key not in obs_dict:
            return std

        phase = obs_dict[self.contact_phase_std_obs_key]
        if not torch.is_tensor(phase):
            return std
        phase = phase.to(device=mean.device, dtype=mean.dtype)
        if phase.shape[-1] > 1:
            phase = phase.max(dim=-1, keepdim=True).values
        if mean.ndim == 2 and phase.ndim == 3:
            phase = phase[:, -1]
        elif mean.ndim == 3 and phase.ndim == 2:
            phase = phase.unsqueeze(1).expand(mean.shape[0], mean.shape[1], phase.shape[-1])
        while phase.ndim < mean.ndim:
            phase = phase.unsqueeze(-1)
        phase = torch.clamp(phase, min=0.0, max=1.0)

        scale = self.contact_phase_std_far_scale + (
            self.contact_phase_std_contact_scale - self.contact_phase_std_far_scale
        ) * phase
        std_view = std.view(*([1] * (mean.ndim - 1)), -1)
        return torch.clamp((mean * 0.0 + std_view) * scale, min=self.contact_phase_std_min)

    def forward(self, obs_dict, is_training=False, **kwargs):
        """Compute action means from observations.

        Optionally normalizes input observations and collects auxiliary losses
        from the backbone when training with VQ or similar modules.

        Args:
            obs_dict: Dictionary mapping observation keys to tensors.
            is_training: If True and ``has_aux_loss``, request auxiliary losses
                from the backbone.
            **kwargs: Forwarded to the backbone (e.g. ``episode_attnmask``).

        Returns:
            Action mean tensor of shape ``(batch, act_dim)`` or
            ``(batch, seq, act_dim)`` for temporal models.
        """
        obs_dict = obs_dict.copy()
        if self.running_mean_std is not None:
            if self.use_batch_norm:
                obs_dict[self.input_key] = self.running_mean_std(obs_dict[self.input_key])
            else:
                with torch.no_grad():
                    obs_dict[self.input_key] = self.running_mean_std(obs_dict[self.input_key])

        with Timer("actor_module", instance_enabled=self.training):
            if self.input_obs_dict:
                net_input = obs_dict
            else:
                net_input = obs_dict[self.input_key]
            net_kwargs = kwargs.copy()
            if self.has_aux_loss and is_training:
                net_kwargs["compute_aux_loss"] = True
            output = self.actor_module(net_input, **net_kwargs)
        if self.has_aux_loss and is_training:
            # output needs to be a dict
            action_mean = output["action_mean"]
            self.aux_losses = output["aux_losses"]
            self.aux_loss_coef = output["aux_loss_coef"]
        else:
            action_mean = output
        return action_mean

    @property
    def has_normalized_actions(self):
        return False

    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev

    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)

    def update_distribution(
        self, obs_dict, episode_attnmask=None, last_step_only=False, is_training=False, **kwargs
    ):
        """Compute forward pass and update the internal Gaussian distribution.

        Args:
            obs_dict: Observation dictionary. For temporal models this has shape
                ``{key: (batch, seq, dim)}``.
            episode_attnmask: Optional causal attention mask of shape
                ``(batch, seq, seq)`` for transformer backbones.
            last_step_only: If True, use only the last timestep's mean for
                the distribution (used during rollout with temporal models).
            is_training: Forwarded to ``forward`` to enable aux loss collection.
            **kwargs: Forwarded to ``forward``.
        """
        mean = self.forward(
            obs_dict, episode_attnmask=episode_attnmask, is_training=is_training, **kwargs
        )
        if last_step_only:
            mean = mean[:, -1]

        # Get std using the property that handles both parameterizations
        std = self.get_std
        std = self._apply_contact_phase_std(std, obs_dict, mean)
        if self.safe_nan:
            if torch.any(torch.isnan(mean)) or torch.any(torch.isinf(mean)):
                print("[WARNING] Invalid action mean detected! Replacing invalid entries with 0.0.")  # noqa: T201
                mean = torch.nan_to_num(mean, nan=0.0, posinf=0.0, neginf=0.0)
            if torch.any(torch.isnan(std)) or torch.any(torch.isinf(std)) or torch.any(std <= 0):
                print(f"[WARNING] Invalid std detected! std: {std}")  # noqa: T201
                std = torch.nan_to_num(std, nan=self.init_noise_std, posinf=self.init_noise_std, neginf=self.init_noise_std)
                std = torch.clamp(std, min=1e-6)
            scale = (torch.zeros_like(mean) + std).clamp(min=1e-6)
        else:
            # Safety check for NaN or negative values
            if torch.any(torch.isnan(std)) or torch.any(std <= 0):
                print(f"[WARNING] Invalid std detected! std: {std}")
                std = torch.clamp(std, min=1e-6)
            scale = (mean * 0.0 + std).clamp(min=1e-6)
        self.distribution = Normal(mean, scale)

    def act(self, obs_dict, episode_attnmask=None, **kwargs):
        """Sample actions from the current policy for a single timestep.

        Update the action distribution and sample from it. Used during the
        PPO learning phase (not rollout) where the full observation sequence
        is available.

        Args:
            obs_dict: Observation dictionary with shape
                ``{key: (batch, seq, dim)}``.
            episode_attnmask: Optional attention mask of shape
                ``(batch, seq, seq)``.
            **kwargs: Forwarded to ``update_distribution``.

        Returns:
            TensorDict with keys ``actions`` ``(batch, act_dim)``,
            ``action_mean``, ``action_sigma``, and optionally ``obs_dict``.
        """
        # try:
        self.update_distribution(
            obs_dict, episode_attnmask=episode_attnmask, is_training=True, **kwargs
        )
        # except Exception as e:
        #     import ipdb; ipdb.set_trace()
        #     raise e
        actions = self.distribution.sample()
        return TensorDict(
            {
                "actions": actions,
                "action_mean": self.action_mean,
                "action_sigma": self.action_std,
                "obs_dict": obs_dict if self.output_original_obs_dict else None,
            }
        )

    def update_dones_buffer_and_compute_episode_attnmask(self, cur_dones):
        """Update the done-signal buffer and derive an episode attention mask.

        Maintains a sliding window of done flags (length
        ``max_rollout_history - 1``) and converts them into a causal attention
        mask that prevents attending across episode boundaries.

        Args:
            cur_dones: Done flags for the current step, shape ``(batch,)``.

        Returns:
            Episode attention mask of shape
            ``(batch, history_len, history_len)``.
        """
        if self.steps > 0 and self.max_rollout_history > 1:
            if self.dones_buffer is None:
                self.dones_buffer = cur_dones.clone().unsqueeze(1)
            else:
                self.dones_buffer = torch.cat(
                    [self.dones_buffer, cur_dones.clone().unsqueeze(1)], dim=1
                )
            if self.dones_buffer.shape[1] > self.max_rollout_history - 1:
                self.dones_buffer = self.dones_buffer[:, -self.max_rollout_history + 1 :]
            dones = torch.cat(
                [self.dones_buffer, torch.zeros_like(self.dones_buffer[:, :1])], dim=1
            )
        else:
            dones = torch.zeros_like(cur_dones.unsqueeze(1))
        episode_attnmask_from_dones = compute_episode_attnmask(dones)
        return episode_attnmask_from_dones

    def _update_obs_buffer(self, obs_dict, episode_attnmask=None, cur_dones=None):
        """Append new observations to the rolling history buffer.

        Grows the buffer up to ``max_rollout_history`` timesteps, then slides
        the window. When ``cur_dones`` is provided, also updates the done
        buffer and computes the episode attention mask.

        Args:
            obs_dict: Single-step observations ``{key: (batch, dim)}``.
            episode_attnmask: Optional externally provided attention mask.
                If both this and ``cur_dones`` are given, consistency is
                asserted.
            cur_dones: Optional done flags for the current step,
                shape ``(batch,)``.

        Returns:
            Episode attention mask of shape
            ``(batch, history_len, history_len)`` or None.
        """
        update_episode_attnmask = False

        for key in obs_dict.keys():
            if key not in self.obs_dict_buffer:
                self.obs_dict_buffer[key] = obs_dict[key].unsqueeze(1)
            else:
                self.obs_dict_buffer[key] = torch.cat(
                    [self.obs_dict_buffer[key], obs_dict[key].unsqueeze(1)], dim=1
                )
            if self.obs_dict_buffer[key].shape[1] > self.max_rollout_history:
                update_episode_attnmask = True
                self.obs_dict_buffer[key] = self.obs_dict_buffer[key][
                    :, -self.max_rollout_history :
                ]

        if episode_attnmask is not None and update_episode_attnmask:
            episode_attnmask = episode_attnmask[
                :, -self.max_rollout_history :, -self.max_rollout_history :
            ]

        if cur_dones is not None:
            episode_attnmask_from_dones = self.update_dones_buffer_and_compute_episode_attnmask(
                cur_dones
            )
            if episode_attnmask is not None:
                assert (episode_attnmask == episode_attnmask_from_dones).all()
            if episode_attnmask is None:
                episode_attnmask = episode_attnmask_from_dones

        return episode_attnmask

    def rollout(self, obs_dict, episode_attnmask=None, cur_dones=None, **kwargs):
        """Execute one rollout step: buffer observations, sample actions.

        Appends the current observations to the history buffer, runs the
        forward pass over the full buffer, and samples from the resulting
        distribution using only the last timestep's output.

        Args:
            obs_dict: Single-step observations ``{key: (batch, dim)}``.
            episode_attnmask: Optional attention mask.
            cur_dones: Done flags from the previous step, shape ``(batch,)``.
            **kwargs: Forwarded to ``update_distribution``.

        Returns:
            TensorDict with keys ``actions`` ``(batch, act_dim)``,
            ``action_mean``, ``action_sigma``, and optionally ``obs_dict``.
        """
        episode_attnmask = self._update_obs_buffer(obs_dict, episode_attnmask, cur_dones)
        self.update_distribution(
            obs_dict=self.obs_dict_buffer,
            episode_attnmask=episode_attnmask,
            last_step_only=True,
            **kwargs,
        )
        self.steps += 1
        return TensorDict(
            {
                "actions": self.distribution.sample(),
                "action_mean": self.action_mean,
                "action_sigma": self.action_std,
                "obs_dict": self.obs_dict_buffer if self.output_original_obs_dict else None,
            }
        )

    def rollout_with_tokens(
        self, obs_dict, external_tokens, episode_attnmask=None, cur_dones=None, **kwargs
    ):
        """
        Rollout with externally provided tokens (bypasses encoder).

        This is used when an external model (e.g., kinematic diffusion) provides
        pre-computed FSQ tokens, allowing the encoder to be bypassed while still
        using the decoder for action generation.

        Args:
            obs_dict: Observation dict (for proprioception)
            external_tokens: Pre-computed FSQ tokens, shape (B, 2, 32)
            episode_attnmask: Optional attention mask
            cur_dones: Optional done flags

        Returns:
            TensorDict with actions, action_mean, action_sigma
        """
        # Update observation buffer (for proprioception history)
        episode_attnmask = self._update_obs_buffer(obs_dict, episode_attnmask, cur_dones)

        # Get action mean using external tokens (bypasses encoder)
        # actor_module should have forward_with_external_tokens method
        if not hasattr(self.actor_module, "forward_with_external_tokens"):
            raise NotImplementedError(
                "actor_module does not have forward_with_external_tokens method. "
                "This is required for token bypass mode."
            )

        # Use the last step of obs buffer for proprioception
        obs_dict_last = {k: v[:, -1:] for k, v in self.obs_dict_buffer.items()}

        action_mean = self.actor_module.forward_with_external_tokens(
            input_data=obs_dict_last, external_tokens=external_tokens, **kwargs
        )

        # Update distribution
        if self.safe_nan:
            std = self.get_std
            std = self._apply_contact_phase_std(std, obs_dict, action_mean)
            if torch.any(torch.isnan(action_mean)) or torch.any(torch.isinf(action_mean)):
                print("[WARNING] Invalid action mean detected in rollout_with_tokens! Replacing invalid entries with 0.0.")  # noqa: T201
                action_mean = torch.nan_to_num(action_mean, nan=0.0, posinf=0.0, neginf=0.0)
            if torch.any(torch.isnan(std)) or torch.any(torch.isinf(std)) or torch.any(std <= 0):
                print(f"[WARNING] Invalid std detected in rollout_with_tokens! std: {std}")  # noqa: T201
                std = torch.nan_to_num(std, nan=self.init_noise_std, posinf=self.init_noise_std, neginf=self.init_noise_std)
                std = torch.clamp(std, min=1e-6)
            scale = (torch.zeros_like(action_mean) + std).clamp(min=1e-6)
        else:
            std = self.get_std if self.use_log_std else self.std
            std = self._apply_contact_phase_std(std, obs_dict, action_mean)
            scale = (action_mean * 0.0 + std).clamp(min=1e-6)
        self.distribution = Normal(action_mean, scale)

        self.steps += 1
        actions = self.distribution.sample()
        return TensorDict(
            {
                "actions": actions,
                "action_mean": self.action_mean,
                "action_sigma": self.action_std,
            }
        )

    def get_actions_log_prob(self, actions):
        """Compute log-probability of actions under the current distribution.

        Args:
            actions: Action tensor of shape ``(batch, act_dim)``.

        Returns:
            Log-probability scalar per batch element, shape ``(batch,)``.
        """
        return self.distribution.log_prob(actions).sum(dim=-1)

    def act_inference(self, obs_dict, episode_attnmask=None, cur_dones=None, **kwargs):
        """Compute deterministic actions for inference during rollout.

        Similar to ``rollout`` but returns only the action mean (no sampling),
        suitable for evaluation where stochastic exploration is not desired.

        Args:
            obs_dict: Single-step observations ``{key: (batch, dim)}``.
            episode_attnmask: Optional attention mask.
            cur_dones: Done flags from the previous step, shape ``(batch,)``.
            **kwargs: Forwarded to ``forward``.

        Returns:
            Deterministic action tensor of shape ``(batch, act_dim)``.
        """
        episode_attnmask = self._update_obs_buffer(obs_dict, episode_attnmask, cur_dones)
        actions_mean = self.forward(
            obs_dict=self.obs_dict_buffer, episode_attnmask=episode_attnmask, **kwargs
        )
        # last step only
        actions_mean = actions_mean[:, -1]
        self.steps += 1
        return actions_mean

    def act_pure_inference(self, obs_dict, episode_attnmask=None, **kwargs):
        """
        Pure inference mode for temporal models like transformer.
        Need to construct obs_dict and episode_attnmask outside the model.
        """
        actions_mean = self.forward(obs_dict=obs_dict, episode_attnmask=episode_attnmask, **kwargs)
        # last step only
        actions_mean = actions_mean[:, -1]
        return actions_mean

    def to_cpu(self):
        """Move the actor and its normalizers to CPU."""
        if self.running_mean_std is not None:
            self.running_mean_std.to("cpu")
        self.actor = deepcopy(self.actor).to("cpu")
        if self.use_log_std:
            self.log_std.to("cpu")
        else:
            self.std.to("cpu")

    def init_rollout(self):
        """Initialize the observation buffer for rollout phase."""
        self.obs_dict_buffer = TensorDict()
        self.dones_buffer = None
        self.steps = 0

    def clear_rollout(self):
        """Clear the observation buffer after rollout phase."""
        self.obs_dict_buffer = TensorDict()
        self.dones_buffer = None
        self.steps = 0
        del self.distribution
        if self.has_aux_loss:
            del self.aux_losses
            del self.aux_loss_coef

    def eval_mode(self):
        self.is_eval_mode = True

    def train_mode(self):
        self.is_eval_mode = False


    @property
    def needs_episode_attnmask(self) -> bool:
        """Whether this actor's backbone can consume an episode attention mask (the MLP path cannot)."""
        return backbone_consumes_episode_attnmask(self.actor_module)


class Critic(nn.Module):
    """Value function network that estimates state values for PPO.

    Wraps a backbone network and optionally normalizes critic observations
    via running mean/std or batch normalization. The backbone receives the
    full observation dict (keyed by ``critic_obs``) and outputs a scalar
    value estimate per environment.
    """

    def __init__(
        self,
        env_config,
        algo_config,
        backbone,
        obs_dim_dict=None,
        module_dim_dict={},
        running_mean_std=False,
        use_batch_norm=False,
        backbone_kwargs={},
    ):
        """Initialize the Critic.

        Args:
            env_config: Environment configuration containing robot specs.
            algo_config: Algorithm configuration.
            backbone: Hydra-style config for the backbone network to instantiate.
            obs_dim_dict: Mapping from observation keys to their dimensions.
                Defaults to ``env_config.robot.algo_obs_dim_dict``.
            module_dim_dict: Additional dimension info passed to the backbone.
            running_mean_std: Whether to normalize inputs with running statistics.
            use_batch_norm: Whether to normalize inputs with batch normalization.
            backbone_kwargs: Extra keyword arguments forwarded to the backbone
                constructor.
        """
        super().__init__()

        if obs_dim_dict is None:
            obs_dim_dict = env_config.robot.algo_obs_dim_dict
        self.critic_module = custom_instantiate(
            backbone,
            env_config=env_config,
            algo_config=algo_config,
            obs_dim_dict=obs_dim_dict,
            module_dim_dict=module_dim_dict,
            _resolve=False,
            **backbone_kwargs,
        )
        self.use_batch_norm = use_batch_norm
        self.use_running_mean_std = running_mean_std

        self.running_mean_std = None
        if running_mean_std:
            self.running_mean_std = RunningMeanStd((obs_dim_dict["critic_obs"],), per_channel=True)

        if use_batch_norm:
            self.running_mean_std = BatchNormNormalizer((obs_dim_dict["critic_obs"],))

        assert not (
            running_mean_std and use_batch_norm
        ), "running_mean_std and use_batch_norm cannot be both True"

    @property
    def critic(self):
        return self.critic_module

    def reset(self, dones=None):
        pass

    def evaluate(self, obs_dict, **kwargs):
        """Compute the value estimate for given observations.

        Normalizes the ``critic_obs`` entry if a normalizer is configured,
        then forwards through the critic backbone.

        Args:
            obs_dict: Observation dictionary containing at least
                ``critic_obs`` of shape ``(batch, critic_obs_dim)``.
            **kwargs: Forwarded to the critic backbone.

        Returns:
            Value estimate tensor of shape ``(batch, 1)``.
        """
        obs_dict = obs_dict.copy()
        if self.running_mean_std is not None:
            if self.use_batch_norm:
                obs_dict["critic_obs"] = self.running_mean_std(obs_dict["critic_obs"])
            else:
                with torch.no_grad():
                    obs_dict["critic_obs"] = self.running_mean_std(obs_dict["critic_obs"])
        value = self.critic(obs_dict, **kwargs)
        return value

    @property
    def needs_episode_attnmask(self) -> bool:
        """Whether this critic's backbone can consume an episode attention mask (the MLP path cannot)."""
        return backbone_consumes_episode_attnmask(self.critic_module)
