"""LoRA finetuning of the SONIC ``g1_dyn`` decoder inside the PPO likelihood.

In the RP2 residual setup the policy action is the 66-D latent residual + hand
primitives; SONIC (encoder -> FSQ -> decoder) runs inside ``env.step`` under
``torch.no_grad`` and is not in the PPO loss, so decoder adapters placed there
get no gradient.  ``SonicDecoderLoRAActor`` therefore appends a Gaussian factor
over the decoder's body-joint targets, whose mean is the LoRA-adapted decoder
evaluated at the *sampled* latent residual:

    (r, h) ~ N(mu_actor(o), sigma)            # unchanged 66-D factor
    b      ~ N(D_lora(FSQ(e(o) + s * r), p), sigma_body)
    log pi(r, h, b | o) = log N(r, h) + log N(b | r)

The body mean depends on the stored sample r (not on mu), so the factorization is
exact and old/new log-probs are evaluated on identical inputs.  The environment
applies b as the body joint targets.  With ``stochastic_body=False`` the adapters
are attached and optimized but stay outside the likelihood (the invalid
"naive LoRA" setup, kept only as a measured negative control).
"""

import math

import torch
import torch.nn as nn
from loguru import logger
from torch.distributions import Normal

from gear_sonic.trl.modules.actor_critic_modules import Actor


class LoRAAdapter(nn.Module):
    """Low-rank update ``x -> scaling * B(A(x))`` with B zero-initialized."""

    def __init__(self, in_features, out_features, rank, alpha, dropout=0.0):
        super().__init__()
        self.lora_A = nn.Parameter(torch.empty(rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        self.scaling = float(alpha) / float(rank)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x):
        return (self.dropout(x) @ self.lora_A.t() @ self.lora_B.t()) * self.scaling


class LoRALinear(nn.Module):
    """Frozen ``nn.Linear`` plus an adapter that is owned (registered) elsewhere."""

    def __init__(self, base: nn.Linear, adapter: LoRAAdapter):
        super().__init__()
        self.base = base
        # Not registered here: the policy owns the adapter so that its weights are
        # in the optimizer and the policy checkpoint, while the SONIC weights are not.
        object.__setattr__(self, "adapter", adapter)
        self.enabled = True

    def forward(self, x):
        y = self.base(x)
        if self.enabled:
            y = y + self.adapter(x)
        return y


def inject_lora(module: nn.Module, rank, alpha, dropout):
    """Replace every ``nn.Linear`` under ``module``; return ``{name: LoRALinear}``."""
    targets = [(name, m) for name, m in module.named_modules() if isinstance(m, nn.Linear)]
    if not targets:
        raise ValueError("No nn.Linear layers found to adapt")
    wrapped = {}
    for name, linear in targets:
        adapter = LoRAAdapter(linear.in_features, linear.out_features, rank, alpha, dropout).to(
            device=linear.weight.device, dtype=linear.weight.dtype
        )
        parent_name, _, child = name.rpartition(".")
        parent = module.get_submodule(parent_name) if parent_name else module
        lora_linear = LoRALinear(linear, adapter)
        setattr(parent, child, lora_linear)
        wrapped[name] = lora_linear
    return wrapped


class SonicDecoderLoRAActor(Actor):
    """B+HiPHI actor plus LoRA adapters on the SONIC ``g1_dyn`` decoder."""

    def __init__(self, *args, sonic_decoder_lora=None, **kwargs):
        super().__init__(*args, **kwargs)
        cfg = dict(sonic_decoder_lora or {})
        self.lora_rank = int(cfg.get("rank", 8))
        self.lora_alpha = float(cfg.get("alpha", 16))
        self.lora_dropout = float(cfg.get("dropout", 0.0))
        self.target_decoder = str(cfg.get("target_decoder", "g1_dyn"))
        self.latent_dim = int(cfg.get("latent_dim", 64))
        self.stochastic_body = bool(cfg.get("stochastic_body", True))
        # trainable=False: noise-only control.  Adapters are created but disabled and
        # frozen (B stays 0, not in the optimizer), so the decoder is frozen SONIC while the
        # 95-D stochastic body action and factored PPO are unchanged.
        self.lora_trainable = bool(cfg.get("trainable", True))
        self.policy_action_dim = self.num_actions
        self.body_action_dim = int(cfg.get("body_action_dim", 29)) if self.stochastic_body else 0
        self.num_actions = self.policy_action_dim + self.body_action_dim
        self.register_buffer(
            "body_action_std",
            torch.full((self.body_action_dim,), float(cfg.get("body_action_std", 0.05))),
        )
        self.sonic_lora = nn.ModuleDict()
        # Adapter learning rate (own adaptive-KL state under factored PPO); a persistent
        # buffer so that it is checkpointed and restored with the adapters.
        self.register_buffer(
            "sonic_lora_lr",
            torch.tensor(float(cfg.get("learning_rate", self.algo_config.get("actor_learning_rate", 1e-4)))),
        )
        self._sonic_wrapper = None
        self._pending_obs = None

    # ------------------------------------------------------------------ wiring
    def attach_sonic(self, env_wrapper):
        """Adapt the env wrapper's SONIC decoder in place; adapters live in this module."""
        atm = env_wrapper.action_transform_module
        if atm is None:
            raise ValueError("SonicDecoderLoRAActor requires an action_transform_module")
        if self._sonic_wrapper is not None:
            raise RuntimeError("attach_sonic called twice")
        if int(getattr(env_wrapper, "_raw_body_residual_dim", 0)) != 0:
            raise ValueError("raw_body_residual_dim must be 0 for the SONIC LoRA arm")
        if int(env_wrapper._tokenizer_action_dim) != self.latent_dim:  # noqa: SLF001
            raise ValueError("latent_dim does not match the wrapper's tokenizer_action_dim")
        env_body_dim = int(env_wrapper.config.get("stochastic_decoder_body_action_dim", 0) or 0)
        if env_body_dim != self.body_action_dim:
            raise ValueError(
                f"env stochastic_decoder_body_action_dim={env_body_dim} != actor {self.body_action_dim}"
            )
        for parameter in atm.parameters():
            parameter.requires_grad_(False)
        decoder = atm.actor_module.decoders[self.target_decoder]
        wrapped = inject_lora(decoder, self.lora_rank, self.lora_alpha, self.lora_dropout)
        for name, lora_linear in wrapped.items():
            self.sonic_lora[name.replace(".", "__")] = lora_linear.adapter
            lora_linear.enabled = self.lora_trainable
        if not self.lora_trainable:
            for parameter in self.sonic_lora.parameters():
                parameter.requires_grad_(False)
        object.__setattr__(self, "_sonic_wrapper", env_wrapper)
        object.__setattr__(self, "_lora_linears", wrapped)
        self.latent_residual_scale = float(env_wrapper._latent_residual_scale)  # noqa: SLF001
        self.latent_residual_mode = env_wrapper._latent_residual_mode  # noqa: SLF001
        n_lora = sum(p.numel() for p in self.sonic_lora.parameters())
        n_decoder = sum(m.base.weight.numel() + (m.base.bias.numel() if m.base.bias is not None else 0)
                        for m in wrapped.values())
        n_atm = sum(p.numel() for p in atm.parameters())
        n_actor = sum(p.numel() for n, p in self.named_parameters() if not n.startswith("sonic_lora"))
        self.lora_summary = {
            "targets": {n: (m.base.in_features, m.base.out_features) for n, m in wrapped.items()},
            "rank": self.lora_rank, "alpha": self.lora_alpha, "dropout": self.lora_dropout,
            "lora_params": n_lora, "decoder_params": n_decoder, "sonic_params": n_atm,
            "actor_params_excl_lora": n_actor,
            "stochastic_body": self.stochastic_body, "body_action_dim": self.body_action_dim,
            "lora_trainable": self.lora_trainable,
            "body_action_std": self.body_action_std.tolist()[:1],
        }
        logger.info(f"[SONIC-LoRA] attached: {self.lora_summary}")
        self._register_grad_probes()
        # Snapshot of the frozen decoder base weights; diagnostics report any change.
        self._decoder_base_init = [m.base.weight.detach().clone() for m in wrapped.values()]
        return self.lora_summary

    def load_state_dict(self, state_dict, strict=True, **kwargs):
        result = super().load_state_dict(state_dict, strict=strict, **kwargs)
        if getattr(self, "_lora_linears", None):
            with torch.no_grad():
                b_norm = sum(float(p.pow(2).sum()) for n, p in self.sonic_lora.named_parameters()
                             if n.endswith("lora_B")) ** 0.5
            enabled = sorted({m.enabled for m in self._lora_linears.values()})
            logger.info(f"[SONIC-LoRA] loaded policy state: adapter B norm={b_norm:.6g}, "
                        f"adapters enabled={enabled}, trainable={self.lora_trainable}, "
                        f"lora_lr={float(self.sonic_lora_lr):.3g}")
        return result

    def set_lora_enabled(self, enabled: bool):
        for lora_linear in self._lora_linears.values():
            lora_linear.enabled = bool(enabled)

    # ------------------------------------------------------------ diagnostics
    def _register_grad_probes(self):
        self._lora_grad_sq = 0.0
        self._lora_grad_events = 0
        self._lora_init = {n: p.detach().clone() for n, p in self.sonic_lora.named_parameters()}
        # D perf migration: with algo.config.host_probe_batching the squared-gradient
        # sum is accumulated on device in fp64 (same fp32 addends, same order) and
        # converted once per update in sonic_lora_diagnostics, instead of one
        # blocking float() per adapter parameter per micro-batch.
        self._lora_grad_sq_dev = None
        algo_config = getattr(self, "algo_config", None) or {}
        device_accumulate = bool(algo_config.get("host_probe_batching", False))

        def hook(param):
            if param.grad is not None:
                grad_sq = param.grad.detach().float().pow(2).sum()
                if device_accumulate:
                    if self._lora_grad_sq_dev is None:
                        self._lora_grad_sq_dev = torch.zeros(
                            (), dtype=torch.float64, device=grad_sq.device
                        )
                    self._lora_grad_sq_dev += grad_sq.double()
                else:
                    self._lora_grad_sq += float(grad_sq)
                self._lora_grad_events += 1

        for p in self.sonic_lora.parameters():
            if p.requires_grad:
                p.register_post_accumulate_grad_hook(hook)

    def sonic_lora_diagnostics(self, reset=True):
        with torch.no_grad():
            delta = sum(float((p - self._lora_init[n]).pow(2).sum())
                        for n, p in self.sonic_lora.named_parameters()) ** 0.5
            b_norm = sum(float(p.pow(2).sum()) for n, p in self.sonic_lora.named_parameters()
                         if n.endswith("lora_B")) ** 0.5
        grad_sq = self._lora_grad_sq
        if getattr(self, "_lora_grad_sq_dev", None) is not None:
            grad_sq = grad_sq + float(self._lora_grad_sq_dev)
        with torch.no_grad():
            # Rank-comparable checksums of the live adapter parameters (fp64
            # sums); identical values across ranks after an update indicate
            # the adapters are synchronized. Note: _lora_init is captured in
            # attach_sonic BEFORE accelerator.prepare broadcasts rank 0's
            # parameters, so lora_delta_from_init on ranks > 0 also contains
            # that broadcast difference and is not comparable across ranks.
            a_sum = sum(float(p.double().sum()) for n, p in self.sonic_lora.named_parameters() if n.endswith("lora_A"))
            b_sum = sum(float(p.double().sum()) for n, p in self.sonic_lora.named_parameters() if n.endswith("lora_B"))
            a_norm = sum(float(p.pow(2).sum()) for n, p in self.sonic_lora.named_parameters() if n.endswith("lora_A")) ** 0.5
        out = {"lora_grad_norm_sum": grad_sq ** 0.5, "lora_grad_events": self._lora_grad_events,
               "lora_delta_from_init": delta, "lora_B_norm": b_norm, "lora_A_norm": a_norm,
               "lora_A_param_sum": a_sum, "lora_B_param_sum": b_sum,
               "sonic_decoder_base_max_abs_change": max(
                   float((m.base.weight - w0).abs().max())
                   for m, w0 in zip(self._lora_linears.values(), self._decoder_base_init)),
               "body_std": float(self.body_action_std.mean()) if self.body_action_dim else 0.0,
               "policy_std_min": float(self.get_std.min()), "policy_std_max": float(self.get_std.max())}
        if hasattr(self, "_latent_induced_body_spread"):
            out["latent_induced_body_spread_std"] = self._latent_induced_body_spread
            out["latent_induced_body_abs_p90"] = self._latent_induced_body_abs_p90
        if reset:
            self._lora_grad_sq = 0.0
            self._lora_grad_events = 0
            if getattr(self, "_lora_grad_sq_dev", None) is not None:
                self._lora_grad_sq_dev.zero_()
        return out

    # ----------------------------------------------------------- body factor
    CODE_KEY = "sonic_code"

    @property
    def extra_rollout_keys(self):
        """Per-step tensors the trainer stores and returns in the minibatch obs dict."""
        return {self.CODE_KEY: (self.latent_dim,)} if self.body_action_dim else {}

    @property
    def storage_required_obs_keys(self):
        """Obs groups this actor reads back from RolloutStorage during the PPO update.

        ``_proprio`` prefers ``policy_atm`` (falling back to ``actor_obs`` only when
        the group does not exist), so ``policy_atm`` must stay in storage whenever
        the body factor is active. Consulted by ``storage_skip_unused_obs``.
        """
        return {"policy_atm"} if self.body_action_dim else set()

    def sonic_codes(self, obs_dict, latent_residual):
        """Post-FSQ decoder token for ``latent_residual`` exactly as env.step computes it.

        FSQ rounding is discontinuous, so the code is computed once at rollout and
        stored; the PPO update decodes the stored code instead of re-quantizing.
        """
        n = latent_residual.shape[0]
        atm_obs = self._sonic_wrapper._prepare_obs_for_action_transform_module(  # noqa: SLF001
            {k: v for k, v in obs_dict.items() if isinstance(v, torch.Tensor) and v.shape[0] == n}
        )
        atm = self._sonic_wrapper.action_transform_module
        atm(atm_obs, latent_residual=latent_residual * self.latent_residual_scale,
            latent_residual_mode=self.latent_residual_mode)
        code = atm.actor_module._last_full_latent_flat  # noqa: SLF001 (batch, 1, 64), detached
        return code[:, -1].clone()

    def decode_codes(self, codes, proprioception):
        """``g1_dyn`` (with adapters) on stored codes, in strict fp32 (no autocast/TF32).

        The PPO update runs under the accelerator's autocast and TF32 matmuls depend on
        the batch shape; strict fp32 keeps rollout and update body means identical up to
        float rounding.  Any leading dims.
        """
        lead = codes.shape[:-1]
        n = int(math.prod(lead))
        decode_input = {
            "token_flattened": codes.reshape(n, 1, -1).float(),
            "proprioception": proprioception.reshape(n, 1, -1).float(),
        }
        um = self._sonic_wrapper.action_transform_module.actor_module
        prev = torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        try:
            with torch.autocast(device_type=codes.device.type, enabled=False):
                body = um.decode(self.target_decoder, decode_input)["action"][:, -1]
        finally:
            torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = prev
        if self.safe_nan:
            body = torch.nan_to_num(body, nan=0.0, posinf=0.0, neginf=0.0)
        return body.reshape(*lead, body.shape[-1])

    def _proprio(self, obs_dict):
        return obs_dict["policy_atm"] if "policy_atm" in obs_dict else obs_dict["actor_obs"]

    def _set_full_distribution(self, body_mean):
        base = self.distribution
        body_std = (body_mean * 0.0 + self.body_action_std).clamp(min=1e-6)
        self.distribution = Normal(
            torch.cat([base.mean, body_mean], dim=-1), torch.cat([base.stddev, body_std], dim=-1)
        )

    def rollout(self, obs_dict, episode_attnmask=None, cur_dones=None, **kwargs):
        out = super().rollout(obs_dict, episode_attnmask=episode_attnmask, cur_dones=cur_dones, **kwargs)
        if not self.body_action_dim:
            return out
        policy_actions = out["actions"]
        last_obs = {k: v[:, -1] for k, v in self.obs_dict_buffer.items()}
        codes = self.sonic_codes(last_obs, policy_actions[..., : self.latent_dim])
        self._set_full_distribution(self.decode_codes(codes, self._proprio(last_obs)))
        body_dist = Normal(self.distribution.mean[..., self.policy_action_dim:],
                           self.distribution.stddev[..., self.policy_action_dim:])
        out["actions"] = torch.cat([policy_actions, body_dist.sample()], dim=-1)
        out[self.CODE_KEY] = codes
        self._rollout_calls = getattr(self, "_rollout_calls", 0) + 1
        if self._rollout_calls % 200 == 1:
            # Joint-action spread that the 0.3 latent noise induces through SONIC
            # (reference scale for the fixed body-factor std).
            at_mean = self.decode_codes(
                self.sonic_codes(last_obs, out["action_mean"][..., : self.latent_dim]),
                self._proprio(last_obs),
            )
            spread = body_dist.mean - at_mean
            self._latent_induced_body_spread = float(spread.std())
            self._latent_induced_body_abs_p90 = float(spread.abs().flatten().quantile(0.9))
        out["action_mean"] = self.action_mean
        out["action_sigma"] = self.action_std
        return out

    def act(self, obs_dict, episode_attnmask=None, **kwargs):
        out = super().act(obs_dict, episode_attnmask=episode_attnmask, **kwargs)
        self._pending_obs = obs_dict if self.body_action_dim else None
        return out

    def get_actions_log_prob(self, actions):
        if self.body_action_dim and self.distribution.mean.shape[-1] == self.policy_action_dim:
            obs = self._pending_obs
            if obs is None or self.CODE_KEY not in obs:
                raise RuntimeError("body log-prob needs act() observations with stored SONIC codes")
            self._set_full_distribution(self.decode_codes(obs[self.CODE_KEY], self._proprio(obs)))
            self._pending_obs = None
        return self.distribution.log_prob(actions).sum(dim=-1)

    def act_inference(self, obs_dict, episode_attnmask=None, cur_dones=None, **kwargs):
        mean = super().act_inference(obs_dict, episode_attnmask=episode_attnmask, cur_dones=cur_dones, **kwargs)
        if not self.body_action_dim:
            return mean
        last_obs = {k: v[:, -1] for k, v in self.obs_dict_buffer.items()}
        codes = self.sonic_codes(last_obs, mean[..., : self.latent_dim])
        return torch.cat([mean, self.decode_codes(codes, self._proprio(last_obs))], dim=-1)
