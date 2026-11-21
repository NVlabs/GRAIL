"""Time-normalized scalar-joint objectives for SOMA sequence sweeps.

The conditional target contains every finite-difference stencil touching a
frame, including future neighbors. Solving a frame against this target is an
exact block-coordinate step of the global temporal quadratic, not a causal
filter or a target motion substituted for the original human observations.
"""
import math
from numbers import Real

import numpy as np


_STENCILS = {
    "velocity": np.array([-1., 1.]),
    "acceleration": np.array([1., -2., 1.]),
    "jerk": np.array([-1., 3., -3., 1.]),
}
_FIELDS = {"sweeps", "max_seconds", *(name + "_weight" for name in _STENCILS)}
_OPTIONAL_FIELDS = {"joint_weight_multipliers", "sequence_warm_start", "method", "synchronous_solver",
                    "evaluation_backend", "evaluation_device", "relative_hand_temporal", "world_hand_temporal", "hinge_speed", "prior_global_seed", "refinement_scope"}


def default_forward_device(temporal_config):
    """Match the synchronous evaluator while retaining explicit CPU controls.

    The converter/bridge resolves an explicitly supplied forward device before
    using this default. CUDA forward uses the bounded-memory LM implementation;
    unavailable devices or unsupported solver configurations must raise rather
    than silently choosing a different device or optimizer.
    """
    if (not isinstance(temporal_config, dict)
            or temporal_config.get("method") != "synchronous_sparse"):
        return "cuda:0"
    backend, device = synchronous_evaluation_settings(temporal_config)
    return "cpu" if backend == "serial" else device


def synchronous_evaluation_settings(config):
    """Resolve synchronous evaluation, including direct runtime callers.

    Forward initialization follows this evaluator's device unless explicitly
    overridden; the all-frame sparse linear solve retains its CPU path.
    Explicit ``serial`` keeps CPU controls and CPU-only deployments available.
    """
    backend = config.get("evaluation_backend", "cuda_graph")
    if backend not in ("serial", "cuda_graph"):
        raise ValueError("evaluation_backend must be serial or cuda_graph")
    if backend == "serial":
        if "evaluation_device" in config:
            raise ValueError("evaluation_device requires evaluation_backend=cuda_graph")
        return backend, None
    import re
    device = config.get("evaluation_device", "cuda:0")
    if not isinstance(device, str) or not re.fullmatch(r"cuda(?::\d+)?", device):
        raise ValueError("cuda_graph requires an explicit CUDA evaluation device")
    return backend, device


def validate_temporal_config(value):
    """Validate explicit settings; coefficients multiply squared SI derivatives."""
    if (not isinstance(value, dict) or not _FIELDS <= set(value)
            or set(value) - _FIELDS - _OPTIONAL_FIELDS):
        raise ValueError("temporal config requires " + ", ".join(sorted(_FIELDS))
                         + "; optional: " + ", ".join(sorted(_OPTIONAL_FIELDS)))
    sweeps = value["sweeps"]
    if isinstance(sweeps, bool) or not isinstance(sweeps, int) or sweeps < 1:
        raise ValueError("temporal sweeps must be a positive integer")
    result = {"sweeps": sweeps}
    for field in _FIELDS - {"sweeps"}:
        number = value[field]
        if (isinstance(number, bool) or not isinstance(number, Real)
                or not math.isfinite(number) or number < 0):
            raise ValueError(f"temporal {field} must be finite and nonnegative")
        result[field] = float(number)
    if result["max_seconds"] <= 0:
        raise ValueError("temporal max_seconds must be positive")
    if not any(result[name + "_weight"] > 0 for name in _STENCILS):
        raise ValueError("at least one temporal derivative weight must be positive")
    if "joint_weight_multipliers" in value:
        multipliers = value["joint_weight_multipliers"]
        if not isinstance(multipliers, dict) or not multipliers:
            raise ValueError("temporal joint_weight_multipliers must be a nonempty mapping")
        clean = {}
        for name, multiplier in multipliers.items():
            if not isinstance(name, str) or not name or name != name.strip():
                raise ValueError("temporal multiplier names must be nonempty exact joint names")
            if (isinstance(multiplier, bool) or not isinstance(multiplier, Real)
                    or not math.isfinite(multiplier) or multiplier <= 0):
                raise ValueError("temporal joint weight multipliers must be finite and positive")
            clean[name] = float(multiplier)
        result["joint_weight_multipliers"] = clean
    if "sequence_warm_start" in value:
        if __package__:
            from .soma_sequence_warm_start import validate_sequence_warm_start_config
        else:
            from soma_sequence_warm_start import validate_sequence_warm_start_config
        result["sequence_warm_start"] = validate_sequence_warm_start_config(value["sequence_warm_start"])
    if "prior_global_seed" in value:
        if value.get("method") != "synchronous_sparse":
            raise ValueError("prior_global_seed requires synchronous_sparse")
        try:
            from .soma_prior_global_seed import validate_config as validate_prior_seed
        except ImportError:
            from soma_prior_global_seed import validate_config as validate_prior_seed
        result["prior_global_seed"] = validate_prior_seed(value["prior_global_seed"])
    if "refinement_scope" in value:
        if value.get("method") != "synchronous_sparse" or "prior_global_seed" not in value:
            raise ValueError("refinement_scope requires synchronous_sparse and a bound prior_global_seed")
        try:
            from .soma_refinement_scope import validate_config as validate_refinement_scope
        except ImportError:
            from soma_refinement_scope import validate_config as validate_refinement_scope
        result["refinement_scope"] = validate_refinement_scope(value["refinement_scope"])
    if "method" in value:
        if value["method"] not in ("alternating_sweeps", "synchronous_sparse"):
            raise ValueError("Unknown temporal optimization method")
        result["method"] = value["method"]
    if "synchronous_solver" in value:
        if value.get("method") != "synchronous_sparse":
            raise ValueError("synchronous_solver requires method=synchronous_sparse")
        if __package__:
            from .soma_sparse_sequence import validate_sparse_sequence_config
        else:
            from soma_sparse_sequence import validate_sparse_sequence_config
        result["synchronous_solver"] = validate_sparse_sequence_config(value["synchronous_solver"])
    if value.get("method") == "synchronous_sparse":
        if "synchronous_solver" not in value:
            raise ValueError("Synchronous temporal optimization requires explicit solver controls")
        if "sequence_warm_start" in value:
            raise ValueError("Synchronous baseline does not implicitly combine the G20 seed experiment")
    if value.get("method") == "synchronous_sparse":
        backend, device = synchronous_evaluation_settings(value)
        result["evaluation_backend"] = backend
        if device is not None:
            result["evaluation_device"] = device
    elif "evaluation_backend" in value or "evaluation_device" in value:
        raise ValueError("evaluation_backend/device require method=synchronous_sparse")
    if "relative_hand_temporal" in value:
        if value.get("method") != "synchronous_sparse":
            raise ValueError("relative_hand_temporal requires method=synchronous_sparse")
        if __package__:
            from .soma_relative_temporal import validate_relative_temporal_config
        else:
            from soma_relative_temporal import validate_relative_temporal_config
        result["relative_hand_temporal"] = validate_relative_temporal_config(value["relative_hand_temporal"])
    if "world_hand_temporal" in value:
        if result.get("method") != "synchronous_sparse":
            raise ValueError("world_hand_temporal requires method=synchronous_sparse")
        try:
            from .soma_world_temporal import validate_world_temporal_config
        except ImportError:
            from soma_world_temporal import validate_world_temporal_config
        result["world_hand_temporal"] = validate_world_temporal_config(value["world_hand_temporal"])
        if ("relative_hand_temporal" in result and
                set(result["world_hand_temporal"]["roles"]) != set(result["relative_hand_temporal"]["roles"])):
            raise ValueError("World and relative wrist temporal roles must form the same stencil partition")
    if "hinge_speed" in value:
        if result.get("method") != "synchronous_sparse":
            raise ValueError("hinge_speed requires method=synchronous_sparse")
        try:
            from .soma_hinge_speed import validate_hinge_speed_config
        except ImportError:
            from soma_hinge_speed import validate_hinge_speed_config
        result["hinge_speed"] = validate_hinge_speed_config(value["hinge_speed"])
    return result


def joint_weight_vector(config, joint_names):
    """Resolve explicit full labels or unique leaf names; unspecified hinges use 1.

    These coefficients multiply derivative *energy*, so residuals use their
    square roots. No automatic anatomy selection or wildcard matching occurs.
    A typo, an ambiguous leaf, or two aliases for one hinge fails closed.
    """
    names = list(joint_names)
    if (not names or any(not isinstance(name, str) or not name for name in names)
            or len(set(names)) != len(names)):
        raise ValueError("temporal joint names must be nonempty and unique")
    weights = np.ones(len(names), dtype=np.float64)
    assigned = set()
    for name, value in config.get("joint_weight_multipliers", {}).items():
        matches = [i for i, label in enumerate(names) if label == name]
        if not matches:
            matches = [i for i, label in enumerate(names) if label.rsplit("/", 1)[-1] == name]
        if len(matches) != 1:
            raise ValueError(f"temporal multiplier joint {name!r} is absent or ambiguous")
        index = matches[0]
        if index in assigned:
            raise ValueError(f"temporal multiplier names refer to the same hinge: {name!r}")
        assigned.add(index)
        weights[index] = value
    return weights


def _inputs(q, fps, config, joint_names):
    q = np.asarray(q, dtype=np.float64)
    if q.ndim != 2 or min(q.shape) < 1 or not np.isfinite(q).all():
        raise ValueError("temporal joint sequence must be a nonempty finite [frames, joints] array")
    if isinstance(fps, bool) or not isinstance(fps, Real) or not math.isfinite(fps) or fps <= 0:
        raise ValueError("temporal fps must be positive and finite")
    config = validate_temporal_config(config)
    weights = None
    if "joint_weight_multipliers" in config:
        if joint_names is None or len(joint_names) != q.shape[1]:
            raise ValueError("named temporal multipliers require one bound name per hinge")
        weights = joint_weight_vector(config, joint_names)
    return q, float(fps), config, weights


def conditional_target(q, frame, fps, config, joint_names=None):
    """Return (target[J], gain) for residual gain * (q[frame] - target).

    Its squared norm equals the full sequence's weighted derivative energy
    as a function of this frame, up to a constant independent of this frame.
    No imaginary samples or repeated endpoints are introduced at boundaries.
    ``gain`` is the square root of the conditional quadratic coefficient. It
    stays scalar by default and is a per-hinge vector with named multipliers.
    """
    q, fps, config, weights = _inputs(q, fps, config, joint_names)
    if isinstance(frame, bool) or not isinstance(frame, (int, np.integer)) or not 0 <= frame < len(q):
        raise ValueError("temporal frame must index the source sequence")
    quadratic = 0.
    linear = np.zeros(q.shape[1], dtype=np.float64)
    for name, coefficients in _STENCILS.items():
        order = len(coefficients) - 1
        scale = config[name + "_weight"] * fps ** (2 * order)
        if scale == 0 or len(q) <= order:
            continue
        for start in range(max(0, frame-order), min(frame, len(q)-order-1)+1):
            local = frame-start
            coefficient = coefficients[local]
            other = sum((coefficients[k] * q[start+k] for k in range(order+1)
                         if k != local), start=np.zeros(q.shape[1]))
            quadratic += scale * coefficient**2
            linear += scale * coefficient * other
    if quadratic == 0:
        return q[frame].copy(), 0. if weights is None else np.zeros(q.shape[1])
    gain = math.sqrt(quadratic)
    return -linear / quadratic, gain if weights is None else gain * np.sqrt(weights)


def temporal_metrics(q, fps, config, joint_names=None):
    """Derivative energy and RMS/peak diagnostics over actual source frames."""
    q, fps, config, weights = _inputs(q, fps, config, joint_names)
    result = {"frame_count": len(q), "joint_count": q.shape[1], "fps": fps,
              "weighted_energy": 0., "derivatives": {}}
    for name, coefficients in _STENCILS.items():
        order = len(coefficients)-1
        derivative = np.diff(q, n=order, axis=0) * fps**order
        squared = np.square(derivative)
        energy = config[name + "_weight"] * float(
            squared.sum() if weights is None else (squared * weights).sum())
        result["weighted_energy"] += energy
        result["derivatives"][name] = {
            "sample_count": int(derivative.size),
            "unit": "rad/s" + (str(order) if order > 1 else ""),
            "rms": float(np.sqrt(np.square(derivative).mean())) if derivative.size else None,
            "maximum_abs": float(np.abs(derivative).max()) if derivative.size else None,
            "weighted_energy": energy,
        }
    if weights is not None:
        result["joint_weight_multipliers"] = dict(zip(joint_names, weights.tolist()))
    return result
