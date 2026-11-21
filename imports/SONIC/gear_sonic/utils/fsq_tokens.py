"""Strict execution helpers for already-quantized FSQ token codes.

This module is deliberately independent of Isaac Sim so the token contract can
be tested without constructing an environment.  ``direct_latent`` inputs are
continuous pre-quantization values and must still pass through FSQ.  In
contrast, the GPC prior emits the post-quantization grid values captured from
the teacher; applying FSQ to those values a second time is not idempotent.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch

DIRECT_FSQ_ACTION_MODE = "direct_fsq_codes"
DIRECT_FSQ_DECODER_PROPRIO_TERMS = (
    "base_ang_vel",
    "joint_pos_wo_hand",
    "joint_vel_wo_hand",
    "actions_wo_hand",
    "gravity_dir",
)
DIRECT_FSQ_RUNTIME_AUDIT_VERSION = "direct_fsq_runtime_guard_v3"
_DIRECT_FSQ_RUNTIME_AUDIT_ATTR = "_gpc_direct_fsq_runtime_audit"


def direct_fsq_executor_contract() -> dict:
    """Serializable description of the GPC action-execution layout."""
    return {
        "version": 3,
        "action_mode": DIRECT_FSQ_ACTION_MODE,
        "input_semantics": "complete_post_quantization_fsq_codes",
        "generator_output_layout": "64_full_fsq_code_scalars_then_left_right_hand",
        "tokenizer_action_dim": 64,
        "raw_body_residual_dim": 0,
        "hand_action_dim": 2,
        "meta_action_dim": 66,
        "hand_primitive_order": ["left", "right"],
        "hand_close_boundary": "raw_primitive_greater_than_or_equal_to_zero",
        "action_clip_value": 20.0,
        "sim_dt_s": 0.005,
        "decimation": 4,
        "control_dt_s": 0.02,
        "active_entity_runtime_clamp_static_roots": False,
        "environment_reward_terms": "disabled_controller_uses_external_physical_metrics",
        "invalid_recovery_inputs": "physical_sim_state_and_executed_action_only",
        "safe_nan_physical_recovery": True,
        "reference_success_lift_metric": "disabled",
        "decoder": "g1_dyn",
        "encoder_bypassed": True,
        "quantizer_bypassed": True,
        "grid_validation": "normalized_integer_level_grid_and_range_v1",
        "decoder_proprioception_group": "policy_atm",
        "decoder_proprioception_terms": list(DIRECT_FSQ_DECODER_PROPRIO_TERMS),
        "decoder_proprioception_source": "live_sim_state_and_previous_sim_action",
        "reference_tokenizer_fields_ignored": True,
    }


def validate_direct_fsq_profile_selectors(
    selectors: object, *, label: str = "teacher runtime selectors"
) -> dict:
    """Require the saved teacher wrapper layout used by the GPC executor."""
    if not isinstance(selectors, dict):
        raise ValueError(f"{label} must be a mapping")
    execution = direct_fsq_executor_contract()
    expected = {
        "sim_dt": execution["sim_dt_s"],
        "decimation": execution["decimation"],
        "meta_action_dim": execution["meta_action_dim"],
        "needs_policy_atm": True,
        "tokenizer_action_dim": execution["tokenizer_action_dim"],
        "raw_body_residual_dim": execution["raw_body_residual_dim"],
        "raw_body_residual_units": "joint_position_radians",
        "raw_body_residual_clip": None,
        "hand_action_dim": execution["hand_action_dim"],
        "use_finger_primitive": True,
        "action_clip_value": execution["action_clip_value"],
        "use_staged_wrapper": False,
    }
    mismatched = {
        key: (selectors.get(key), expected_value)
        for key, expected_value in expected.items()
        if selectors.get(key) != expected_value
    }
    if mismatched:
        raise ValueError(f"{label} action layout differs: {mismatched}")
    # The finger primitive is compared structurally (it must be a non-empty
    # mapping); no content fingerprint is pinned.
    if not isinstance(selectors.get("finger_primitive"), dict) or not selectors[
        "finger_primitive"
    ]:
        raise ValueError(f"{label} finger primitive must be a non-empty mapping")
    return selectors


def resolve_gpc_action_mode(
    explicit_action_mode: str | None,
    configured_gpc_action_mode: str | None,
) -> str | None:
    """Resolve an action mode without allowing callers to escape pinned GPC."""
    if configured_gpc_action_mode is None:
        return explicit_action_mode
    if configured_gpc_action_mode != DIRECT_FSQ_ACTION_MODE:
        raise ValueError(
            f"unsupported configured GPC action mode {configured_gpc_action_mode!r}"
        )
    if explicit_action_mode not in (None, configured_gpc_action_mode):
        raise ValueError(
            "explicit action_mode conflicts with pinned GPC execution: "
            f"{explicit_action_mode!r} != {configured_gpc_action_mode!r}"
        )
    return configured_gpc_action_mode


def require_policy_atm_proprioception(atm_obs_dict: dict) -> torch.Tensor:
    """Return only the reference-free live-sim decoder observation group."""
    if "policy_atm" not in atm_obs_dict:
        raise RuntimeError(
            "exact post-FSQ GPC execution requires the policy_atm observation group"
        )
    proprioception = atm_obs_dict["policy_atm"]
    if not isinstance(proprioception, torch.Tensor):
        raise TypeError("policy_atm observation must be a torch.Tensor")
    if proprioception.ndim == 2:
        proprioception = proprioception.unsqueeze(1)
    return proprioception


def install_direct_fsq_runtime_audit(atm) -> None:
    """Make encoder/FSQ calls fatal and audit exact ``g1_dyn`` token input.

    Optional runtime diagnostic: static call-graph inspection cannot show that
    a future ATM implementation never invokes its encoder or quantizer from
    ``decode``.  The installed hooks fail at the first such call and compare
    the token prefix seen by the real decoder with the validated post-FSQ
    tensor.  Nothing in train/load/resume requires this probe.
    """
    if hasattr(atm, _DIRECT_FSQ_RUNTIME_AUDIT_ATTR):
        raise RuntimeError("direct-FSQ runtime audit is already installed")
    quantizer = getattr(atm, "quantizer", None)
    decoders = getattr(atm, "decoders", None)
    if quantizer is None or not hasattr(quantizer, "register_forward_pre_hook"):
        raise RuntimeError("direct-FSQ runtime audit requires a hookable quantizer")
    if decoders is None or "g1_dyn" not in decoders:
        raise RuntimeError("direct-FSQ runtime audit requires the g1_dyn decoder")

    audit = {
        "version": DIRECT_FSQ_RUNTIME_AUDIT_VERSION,
        "encoder_calls": 0,
        "quantizer_calls": 0,
        "decoder_calls": 0,
        "decoded_rows": 0,
        "decoder_token_exact": True,
        "decoder_token_max_abs_error": 0.0,
        "decoder_proprio_exact": True,
        "decoder_proprio_max_abs_error": 0.0,
        "_expected_codes": None,
        "_expected_proprio": None,
        "_handles": [],
        "_original_encode": getattr(atm, "encode", None),
    }
    setattr(atm, _DIRECT_FSQ_RUNTIME_AUDIT_ATTR, audit)

    def reject_encoder(*_args, **_kwargs):
        audit["encoder_calls"] += 1
        raise RuntimeError("reference encoder executed during direct-FSQ GPC rollout")

    # Guard both the public encode route and individual encoder modules in case
    # a later implementation bypasses the public method.
    if audit["_original_encode"] is not None:
        atm.encode = reject_encoder
    encoders = getattr(atm, "encoders", None)
    if encoders is not None:
        for encoder in encoders.values():
            audit["_handles"].append(
                encoder.register_forward_pre_hook(
                    lambda _module, _inputs: reject_encoder()
                )
            )

    def reject_quantizer(_module, _inputs):
        audit["quantizer_calls"] += 1
        raise RuntimeError("FSQ requantization executed during direct-FSQ GPC rollout")

    audit["_handles"].append(quantizer.register_forward_pre_hook(reject_quantizer))

    def audit_decoder_input(_module, inputs):
        expected = audit["_expected_codes"]
        expected_proprio = audit["_expected_proprio"]
        audit["_expected_codes"] = None
        audit["_expected_proprio"] = None
        if expected is None or expected_proprio is None:
            audit["decoder_token_exact"] = False
            audit["decoder_proprio_exact"] = False
            raise RuntimeError(
                "g1_dyn executed without expected direct-FSQ token/proprioception"
            )
        if not inputs or not isinstance(inputs[0], torch.Tensor):
            audit["decoder_token_exact"] = False
            raise RuntimeError("g1_dyn decoder input is not a tensor")
        decoder_input = inputs[0]
        expected_width = expected.shape[-1] + expected_proprio.shape[-1]
        if (
            decoder_input.shape[:-1] != expected.shape[:-1]
            or expected_proprio.shape[:-1] != expected.shape[:-1]
            or decoder_input.shape[-1] != expected_width
        ):
            audit["decoder_token_exact"] = False
            audit["decoder_proprio_exact"] = False
            raise RuntimeError(
                "g1_dyn decoder input shape differs from the exact token/proprio input"
            )
        actual = decoder_input[..., : expected.shape[-1]]
        if not torch.equal(actual, expected):
            audit["decoder_token_exact"] = False
            if actual.shape == expected.shape:
                audit["decoder_token_max_abs_error"] = max(
                    float(audit["decoder_token_max_abs_error"]),
                    float((actual - expected).abs().amax()),
                )
            raise RuntimeError("g1_dyn received altered post-FSQ token values")
        actual_proprio = decoder_input[..., expected.shape[-1] :]
        if not torch.equal(actual_proprio, expected_proprio):
            audit["decoder_proprio_exact"] = False
            if actual_proprio.shape == expected_proprio.shape:
                audit["decoder_proprio_max_abs_error"] = max(
                    float(audit["decoder_proprio_max_abs_error"]),
                    float((actual_proprio - expected_proprio).abs().amax()),
                )
            raise RuntimeError("g1_dyn received altered live-sim proprioception")
        audit["decoder_calls"] += 1
        audit["decoded_rows"] += int(expected.shape[0])

    audit["_handles"].append(
        decoders["g1_dyn"].register_forward_pre_hook(audit_decoder_input)
    )


def direct_fsq_runtime_audit_report(atm, *, require_decoder_call: bool = True) -> dict:
    """Return a serializable report for the optional installed guards."""
    audit = getattr(atm, _DIRECT_FSQ_RUNTIME_AUDIT_ATTR, None)
    if not isinstance(audit, dict):
        raise RuntimeError("direct-FSQ runtime audit was not installed")
    if (
        audit["_expected_codes"] is not None
        or audit["_expected_proprio"] is not None
    ):
        raise RuntimeError(
            "a direct-FSQ decoder call did not consume its token/proprioception"
        )
    report = {
        key: audit[key]
        for key in (
            "version",
            "encoder_calls",
            "quantizer_calls",
            "decoder_calls",
            "decoded_rows",
            "decoder_token_exact",
            "decoder_token_max_abs_error",
            "decoder_proprio_exact",
            "decoder_proprio_max_abs_error",
        )
    }
    if report["encoder_calls"] or report["quantizer_calls"]:
        raise RuntimeError(f"direct-FSQ bypass contract was violated: {report}")
    if not report["decoder_token_exact"]:
        raise RuntimeError(f"direct-FSQ decoder token was altered: {report}")
    if not report["decoder_proprio_exact"]:
        raise RuntimeError(f"direct-FSQ decoder proprioception was altered: {report}")
    if require_decoder_call and report["decoder_calls"] <= 0:
        raise RuntimeError("direct-FSQ runtime audit observed no decoder calls")
    return report


def validate_post_fsq_codes(
    full_codes: torch.Tensor,
    *,
    levels: Sequence[int],
    max_num_tokens: int,
    token_dim: int,
    atol: float = 1.0e-5,
) -> torch.Tensor:
    """Validate and reshape flat post-FSQ values without changing them.

    The normalized grid used by ``vector_quantize_pytorch.FSQ`` is
    ``integer / floor(levels / 2)``.  For an even number of levels ``L``, valid
    integer coordinates are ``[-L/2, ..., L/2-1]``.
    """
    if not isinstance(full_codes, torch.Tensor):
        raise TypeError(
            f"post-FSQ codes must be a torch.Tensor, got {type(full_codes).__name__}"
        )
    if full_codes.ndim != 2:
        raise ValueError(
            "post-FSQ codes must have shape (batch, flat_token_dim), got "
            f"{tuple(full_codes.shape)}"
        )
    if not torch.is_floating_point(full_codes):
        raise TypeError(f"post-FSQ codes must be floating point, got {full_codes.dtype}")
    if not bool(torch.isfinite(full_codes).all()):
        raise ValueError("post-FSQ codes contain NaN or infinity")

    max_num_tokens = int(max_num_tokens)
    token_dim = int(token_dim)
    if max_num_tokens <= 0 or token_dim <= 0:
        raise ValueError(
            "FSQ token dimensions must be positive, got "
            f"max_num_tokens={max_num_tokens}, token_dim={token_dim}"
        )
    expected_flat_dim = max_num_tokens * token_dim
    if full_codes.shape[1] != expected_flat_dim:
        raise ValueError(
            f"post-FSQ codes have width {full_codes.shape[1]}, expected "
            f"{expected_flat_dim} = {max_num_tokens} tokens x {token_dim} dims"
        )

    level_values = [int(level) for level in levels]
    if len(level_values) != token_dim:
        raise ValueError(
            f"FSQ levels have width {len(level_values)}, expected token_dim={token_dim}"
        )
    if any(level < 2 for level in level_values):
        raise ValueError(f"FSQ levels must all be at least two, got {level_values}")

    reshaped = full_codes.reshape(full_codes.shape[0], max_num_tokens, token_dim)
    levels_tensor = torch.as_tensor(
        level_values, device=full_codes.device, dtype=full_codes.dtype
    )
    half_width = torch.div(levels_tensor, 2, rounding_mode="floor")
    scaled = reshaped * half_width
    rounded = scaled.round()
    max_grid_error = (scaled - rounded).abs().amax()
    if float(max_grid_error) > float(atol):
        raise ValueError(
            "post-FSQ codes are off the configured level grid: "
            f"max scaled error={float(max_grid_error):.3e}"
        )
    indices = rounded + half_width
    if not bool(((indices >= 0) & (indices < levels_tensor)).all()):
        raise ValueError("post-FSQ codes fall outside the configured level vocabulary")
    return reshaped.contiguous()


def decode_post_fsq_codes(
    atm,
    full_codes: torch.Tensor,
    proprioception: torch.Tensor,
) -> torch.Tensor:
    """Decode exact post-FSQ codes while bypassing both encoder and quantizer."""
    if getattr(atm, "quantizer", None) is None:
        raise RuntimeError("post-FSQ execution requires an ATM FSQ quantizer contract")
    if proprioception.ndim != 3:
        raise ValueError(
            "ATM decoder proprioception must have shape (batch, seq, dim), got "
            f"{tuple(proprioception.shape)}"
        )
    if proprioception.shape[0] != full_codes.shape[0]:
        raise ValueError("post-FSQ code/proprioception batch sizes differ")

    codes = validate_post_fsq_codes(
        full_codes,
        levels=atm.fsq_level_list,
        max_num_tokens=atm.max_num_tokens,
        token_dim=atm.token_dim,
    )
    tokens = codes.unsqueeze(1)
    decode_input_dict = {
        "token": tokens,
        "token_flattened": codes.reshape(codes.shape[0], 1, -1),
        "proprioception": proprioception,
    }
    audit = getattr(atm, _DIRECT_FSQ_RUNTIME_AUDIT_ATTR, None)
    if isinstance(audit, dict):
        if (
            audit["_expected_codes"] is not None
            or audit["_expected_proprio"] is not None
        ):
            raise RuntimeError("overlapping direct-FSQ decoder calls are not supported")
        # Independent snapshots are essential: a decoder implementation that
        # mutates either input in-place must not mutate the expected value too.
        audit["_expected_codes"] = (
            decode_input_dict["token_flattened"].detach().clone()
        )
        audit["_expected_proprio"] = proprioception.detach().clone()
    try:
        decoded_output = atm.decode("g1_dyn", decode_input_dict)
    finally:
        if isinstance(audit, dict) and (
            audit["_expected_codes"] is not None
            or audit["_expected_proprio"] is not None
        ):
            audit["_expected_codes"] = None
            audit["_expected_proprio"] = None
            raise RuntimeError("g1_dyn decoder hook did not observe the direct-FSQ token")
    body_actions = decoded_output.get(
        "body_action", decoded_output.get("meta_action", decoded_output.get("action"))
    )
    if body_actions is None:
        raise KeyError(
            "ATM g1_dyn decoder output is missing 'body_action', 'meta_action', "
            "or 'action'; "
            f"keys={list(decoded_output)}"
        )
    return body_actions
