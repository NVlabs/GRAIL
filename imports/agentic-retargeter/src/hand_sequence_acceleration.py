"""Execution-only helpers for the hand trajectory objective.

The rotation formula is the SO(3) exponential in closed form. CUDA replay
captures the caller's unchanged scalar objective and its backward pass; it does
not include an optimizer, select checkpoints, or omit any loss/gradient checks.
"""
from __future__ import annotations

import math
import contextlib
import sys


def stable_rotation(vector):
    """Return Exp([vector]x), with finite first derivatives at zero.

    ``torch.sinc`` supplies the analytic zero limits without a small-angle
    approximation. Inputs may have any leading batch dimensions and must use
    float32 or float64. This is mathematically equivalent to ``matrix_exp``;
    floating-point operation order, and therefore final bits, may differ.
    """
    import torch

    if (not isinstance(vector, torch.Tensor) or vector.ndim < 1
            or vector.shape[-1] != 3
            or vector.dtype not in (torch.float32, torch.float64)):
        raise ValueError('Rotation vectors must be floating tensors shaped [..., 3]')
    x, y, z = vector.unbind(-1)
    zero = torch.zeros_like(x)
    skew = torch.stack((zero, -z, y, z, zero, -x, -y, x, zero), -1)
    skew = skew.reshape(*vector.shape[:-1], 3, 3)
    angle = torch.linalg.vector_norm(vector, dim=-1)
    first = torch.sinc(angle / math.pi)[..., None, None]
    second = (.5 * torch.sinc(angle / (2 * math.pi)).square())[..., None, None]
    identity = torch.eye(3, dtype=vector.dtype, device=vector.device)
    return identity + first * skew + second * (skew @ skew)


class CudaObjectiveReplay:
    """Replay a static forward/backward closure using persistent gradient buffers.

    ``evaluate`` must return a tuple whose first element is a scalar loss; the
    remaining nested outputs are retained and returned by :meth:`replay`. The
    closure must not mutate parameters, step optimizers, read CUDA values on the
    CPU, or change tensor shapes/control flow. All live objective inputs must
    retain their storage addresses. Retain this object while using its outputs.

    Construction warms up/captures without an optimizer step and restores the
    caller's original ``.grad`` attributes. Every replay zeros and fills this
    object's own gradient buffers, then exposes them through ``parameter.grad``.
    Thus the caller may use either mode of ``optimizer.zero_grad`` between
    replays, but must not perform another backward on the returned loss.

    Finiteness checks, bounds checks, checkpoint selection and optimizer steps
    remain the caller's responsibility and must run after replay. Capture errors
    are propagated; there is no silent fallback to a different objective.
    """

    def __init__(self, evaluate, trainables, *, warmup_steps=2):
        import torch

        if not callable(evaluate):
            raise ValueError('CUDA objective evaluation must be callable')
        if type(warmup_steps) is not int or not 1 <= warmup_steps <= 10:
            raise ValueError('CUDA graph warmup_steps must be an integer in [1, 10]')
        self.trainables = tuple(trainables)
        if not self.trainables or any(
            not isinstance(value, torch.Tensor) or not value.is_leaf
            or not value.requires_grad or value.device.type != 'cuda'
            or value.dtype not in (torch.float32, torch.float64)
            for value in self.trainables
        ):
            raise ValueError('CUDA replay requires nonempty CUDA floating leaf trainables with gradients')
        self.device = self.trainables[0].device
        if any(value.device != self.device for value in self.trainables):
            raise ValueError('CUDA graph trainables must use one CUDA device')
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA graph replay requires an available CUDA runtime')
        if len({id(value) for value in self.trainables}) != len(self.trainables):
            raise ValueError('CUDA graph trainables must be distinct tensors')
        self._evaluate = evaluate
        self._specs = tuple(self._spec(value) for value in self.trainables)
        self._grads = tuple(torch.zeros_like(value) for value in self.trainables)
        original_grads = tuple(value.grad for value in self.trainables)
        versions = tuple(value._version for value in self.trainables)
        current = torch.cuda.current_stream(self.device)
        capture_stream = torch.cuda.Stream(device=self.device)
        capture_stream.wait_stream(current)
        self.graph = torch.cuda.CUDAGraph()
        self.outputs = None
        try:
            self._install_grads()
            with torch.cuda.stream(capture_stream):
                for _ in range(warmup_steps):
                    self._forward_backward()
            current.wait_stream(capture_stream)
            # Warp's default ScopedStream synchronizes on entry. During Torch
            # capture that would wait for unrelated, uncaptured default-stream
            # work and invalidate the graph. Bind Warp to the capture stream
            # before capture starts, with ordering already established above.
            # Pure Torch callers do not need or load the optional Warp runtime.
            warp = sys.modules.get('warp')
            warp_scope = (warp.ScopedStream(warp.stream_from_torch(capture_stream), sync_enter=False)
                          if warp is not None else contextlib.nullcontext())
            with warp_scope:
                with torch.cuda.graph(self.graph, stream=capture_stream):
                    self.outputs = self._forward_backward()
            current.wait_stream(capture_stream)
            if versions != tuple(value._version for value in self.trainables):
                raise ValueError('CUDA objective closure mutated trainable parameters during capture')
        finally:
            for value, original in zip(self.trainables, original_grads):
                value.grad = original

    @staticmethod
    def _spec(value):
        return (value.data_ptr(), tuple(value.shape), tuple(value.stride()),
                value.dtype, value.device)

    def _install_grads(self):
        for value, gradient in zip(self.trainables, self._grads):
            value.grad = gradient

    def _forward_backward(self):
        import torch

        for gradient in self._grads:
            gradient.zero_()
        outputs = self._evaluate()
        if (not isinstance(outputs, tuple) or not outputs
                or not isinstance(outputs[0], torch.Tensor)
                or outputs[0].ndim != 0 or not outputs[0].requires_grad
                or outputs[0].device != self.device):
            raise ValueError('CUDA objective must return a tuple beginning with a scalar differentiable CUDA loss')
        outputs[0].backward()
        return outputs

    def replay(self):
        """Run captured forward/backward and return live output/gradient buffers."""
        if any(self._spec(value) != spec for value, spec in zip(self.trainables, self._specs)):
            raise ValueError('CUDA replay trainable storage, shape, stride, dtype or device changed')
        if any(not value.requires_grad or not value.is_leaf for value in self.trainables):
            raise ValueError('CUDA replay trainables must remain differentiable leaves')
        self._install_grads()
        self.graph.replay()
        return self.outputs
