from __future__ import annotations

import numpy as np
import torch
from torch import nn


class Conv2dProjector(nn.Module):
    """Conv2d-based projector for spatial inputs (e.g. depth images).

    Accepts either:
      - (..., C*H*W) flattened input (from tokenizer obs) — reshaped using input_shape
      - (..., C, H, W) spatial input

    Outputs shape (..., output_dim).
    """

    def __init__(self, conv_layers, mlp_head, input_shape):
        super().__init__()
        self.conv_layers = conv_layers
        self.mlp_head = mlp_head
        self.input_shape = input_shape  # (C, H, W)

    def forward(self, x):
        C, H, W = self.input_shape
        leading_shape = x.shape[:-1] if x.shape[-1] == C * H * W else x.shape[:-3]
        # Reshape from flat (..., C*H*W) or spatial (..., C, H, W) to (N, C, H, W)
        x = x.reshape(-1, C, H, W)
        x = self.conv_layers(x)
        x = x.reshape(x.shape[0], -1)  # flatten spatial dims
        x = self.mlp_head(x)
        return x.reshape(*leading_shape, -1)


def _build_conv2d_projector(
    input_shape,
    channels,
    kernel_sizes,
    strides,
    paddings,
    use_maxpool,
    hidden_dims,
    output_dim,
    activation_cls,
):
    """Build a Conv2d + MLP projector module.

    Args:
        input_shape: [C_in, H, W] of the spatial input.
        channels: list of output channels per conv layer.
        kernel_sizes: list of kernel sizes per conv layer.
        strides: list of strides per conv layer.
        paddings: list of paddings per conv layer.
        use_maxpool: whether to apply 2x2 max pooling after each conv layer.
        hidden_dims: list of MLP hidden layer sizes after flattening.
        output_dim: final output dimension.
        activation_cls: activation class (e.g. nn.ReLU).
    """
    in_channels = input_shape[0]
    H, W = input_shape[1], input_shape[2]

    conv_layers = []
    for ch_out, ks, stride, pad in zip(channels, kernel_sizes, strides, paddings):
        conv_layers.append(nn.Conv2d(in_channels, ch_out, ks, stride, pad))
        conv_layers.append(activation_cls())
        if use_maxpool:
            conv_layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        # track spatial size
        H = (H + 2 * pad - ks) // stride + 1
        if use_maxpool:
            H = H // 2
        W = (W + 2 * pad - ks) // stride + 1
        if use_maxpool:
            W = W // 2
        in_channels = ch_out

    flat_dim = in_channels * H * W

    mlp_layers = []
    in_d = flat_dim
    for h in hidden_dims:
        mlp_layers.extend([nn.Linear(in_d, h), activation_cls()])
        in_d = h
    mlp_layers.append(nn.Linear(in_d, output_dim))

    return Conv2dProjector(
        conv_layers=nn.Sequential(*conv_layers),
        mlp_head=nn.Sequential(*mlp_layers),
        input_shape=tuple(input_shape),
    )


class GRUConv2dProjector(nn.Module):
    """Spatial Conv2d encoder + GRU for temporal depth processing.

    Each forward call processes a **single frame** and updates the GRU hidden
    state.  The caller is responsible for carrying ``hidden`` across time steps
    and resetting it on episode boundaries.

    Accepts flattened ``(..., 1*H*W)`` or spatial ``(..., 1, H, W)`` input for
    a single depth frame.  Outputs ``(..., output_dim)``.
    """

    def __init__(self, conv_layers, gru_cell, mlp_head, input_shape, gru_hidden_dim):
        super().__init__()
        self.conv_layers = conv_layers
        self.gru_cell = gru_cell
        self.mlp_head = mlp_head
        self.input_shape = input_shape  # (1, H, W) — single frame
        self.gru_hidden_dim = gru_hidden_dim

    def forward(self, x, hidden=None):
        """Process depth input through spatial Conv2d then GRU.

        Supports two modes automatically based on input shape:

        **Single-step** (rollout): ``x`` shape ``(B, 1, C*H*W)`` — processes
        one frame, updates hidden state once.

        **Sequence** (training): ``x`` shape ``(B, T, C*H*W)`` where ``T > 1``
        — unrolls GRU along the T dimension using ``hidden`` as the initial
        state, returning outputs for every step.

        Args:
            x: Depth observations.  Last dim is ``C*H*W``.
            hidden: GRU hidden state ``(B, gru_hidden_dim)`` or ``None``.
                    For single-step: the current hidden carried from previous step.
                    For sequence: the **initial** hidden at ``t=0``.

        Returns:
            output: Same leading shape as ``x``, last dim = ``output_dim``.
            new_hidden: ``(B, gru_hidden_dim)`` — hidden state after last step.
        """
        C, H, W = self.input_shape
        flat_dim = C * H * W

        # Determine if input has a seq dimension: (..., T, flat_dim)
        if x.ndim >= 3 and x.shape[-1] == flat_dim and x.shape[-2] > 0:
            # Shape: (*, T, flat_dim)  — could be (B, T, flat_dim) or (B*seq, flat_dim) when T=1
            leading_shape = x.shape[:-1]  # (*, T)
            B = int(torch.tensor(x.shape[:-2]).prod().item()) if x.ndim > 2 else 1
            T = x.shape[-2]
        else:
            # Flat: (N, flat_dim)
            leading_shape = x.shape[:-1]
            B = x.shape[0] if x.ndim >= 2 else 1
            T = 1

        if hidden is None:
            hidden = torch.zeros(B, self.gru_hidden_dim, device=x.device, dtype=x.dtype)

        x_flat = x.reshape(B, T, flat_dim)

        # Spatial encode all frames at once: (B*T, C, H, W)
        x_4d = x_flat.reshape(B * T, C, H, W)
        x_4d = self.conv_layers(x_4d)
        spatial_feats = x_4d.reshape(B, T, -1)  # (B, T, conv_flat_dim)

        # Unroll GRU along time
        h = hidden
        outputs = []
        for t in range(T):
            h = self.gru_cell(spatial_feats[:, t], h)  # (B, gru_hidden_dim)
            outputs.append(self.mlp_head(h))  # (B, output_dim)
        out = torch.stack(outputs, dim=1)  # (B, T, output_dim)
        new_hidden = h

        return out.reshape(*leading_shape, -1), new_hidden


def _build_gru_conv2d_projector(
    input_shape,
    conv_channels,
    conv_kernels,
    gru_hidden_dim,
    output_dim,
    activation_cls,
    conv_strides=None,
    conv_paddings=None,
    use_maxpool=True,
):
    """Build a GRUConv2dProjector: Conv2d spatial encoder → GRU → MLP head.

    Args:
        input_shape: ``[1, H, W]`` single-frame depth.
        conv_channels: List of output channels per conv layer.
        conv_kernels: List of kernel sizes.
        gru_hidden_dim: GRU hidden state size.
        output_dim: Final embedding dimension.
        activation_cls: Activation class.
    """
    in_channels = input_shape[0]
    H, W = input_shape[1], input_shape[2]
    if conv_strides is None:
        conv_strides = [1] * len(conv_channels)
    if conv_paddings is None:
        conv_paddings = [k // 2 for k in conv_kernels]

    conv_layers = []
    for ch_out, ks, stride, pad in zip(conv_channels, conv_kernels, conv_strides, conv_paddings):
        conv_layers.append(nn.Conv2d(in_channels, ch_out, ks, stride, pad))
        conv_layers.append(activation_cls())
        if use_maxpool:
            conv_layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        H = (H + 2 * pad - ks) // stride + 1
        if use_maxpool:
            H = H // 2
        W = (W + 2 * pad - ks) // stride + 1
        if use_maxpool:
            W = W // 2
        in_channels = ch_out

    conv_flat_dim = in_channels * H * W
    gru_cell = nn.GRUCell(conv_flat_dim, gru_hidden_dim)

    mlp_head = nn.Sequential(
        nn.Linear(gru_hidden_dim, output_dim),
    )

    return GRUConv2dProjector(
        conv_layers=nn.Sequential(*conv_layers),
        gru_cell=gru_cell,
        mlp_head=mlp_head,
        input_shape=tuple(input_shape),
        gru_hidden_dim=gru_hidden_dim,
    )


def build_projector_from_config(proj_cfg, feat_dim=None):
    """Build a projector module from a config dict.

    Dispatches to conv2d, gru_conv2d, or MLP based on ``proj_cfg["type"]``.

    Args:
        proj_cfg: Dict with keys ``type`` (``"conv2d"``/``"gru_conv2d"``/``"mlp"``),
            ``output_dim``, and type-specific parameters.
        feat_dim: Flat input feature dimension, required for MLP type.

    Returns:
        ``(module, description_str)`` — the nn.Module and a human-readable
        summary string suitable for logging.
    """
    proj_type = proj_cfg.get("type", "mlp")
    out_dim = proj_cfg["output_dim"]
    activation_cls = getattr(nn, proj_cfg.get("activation", "SiLU"))

    if proj_type == "gru_conv2d":
        module = _build_gru_conv2d_projector(
            input_shape=proj_cfg["input_shape"],
            conv_channels=proj_cfg.get("conv_channels", [16, 32]),
            conv_kernels=proj_cfg.get("conv_kernels", [3, 3]),
            gru_hidden_dim=proj_cfg.get("gru_hidden_dim", 128),
            output_dim=out_dim,
            activation_cls=activation_cls,
            conv_strides=proj_cfg.get("conv_strides", None),
            conv_paddings=proj_cfg.get("conv_paddings", None),
            use_maxpool=proj_cfg.get("use_maxpool", True),
        )
        desc = (
            f"GRU+Conv2d: input_shape={proj_cfg['input_shape']} "
            f"conv_channels={proj_cfg.get('conv_channels', [16, 32])} "
            f"gru_hidden={proj_cfg.get('gru_hidden_dim', 128)} -> {out_dim}"
        )
    elif proj_type == "conv2d":
        channels = proj_cfg.get("channels", None) or proj_cfg.get("conv_channels", [16, 32])
        kernel_sizes = proj_cfg.get("kernel_sizes", None) or proj_cfg.get("conv_kernels", [3, 3])
        module = _build_conv2d_projector(
            input_shape=proj_cfg["input_shape"],
            channels=channels,
            kernel_sizes=kernel_sizes,
            strides=proj_cfg.get("strides", [1] * len(channels)),
            paddings=proj_cfg.get("paddings", [k // 2 for k in kernel_sizes]),
            use_maxpool=proj_cfg.get("use_maxpool", True),
            hidden_dims=proj_cfg.get("hidden_dims", [256]),
            output_dim=out_dim,
            activation_cls=activation_cls,
        )
        desc = f"Conv2d: input_shape={proj_cfg['input_shape']} channels={channels} -> {out_dim}"
    else:
        assert feat_dim is not None, "feat_dim required for MLP projector"
        hidden_dims = proj_cfg.get("hidden_dims", [256])
        layers = []
        in_d = feat_dim
        for h in hidden_dims:
            layers.extend([nn.Linear(in_d, h), activation_cls()])
            in_d = h
        layers.append(nn.Linear(in_d, out_dim))
        module = nn.Sequential(*layers)
        desc = f"MLP: {feat_dim} -> {hidden_dims} -> {out_dim}"

    return module, desc
