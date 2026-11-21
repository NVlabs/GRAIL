"""Optional visible-support observation records and transactional AdamW retries.

This is a feasibility guard, not a descent guarantee. Policies have no default
area floor or retry schedule. Observers own native geometry/query work; this
module never evaluates a model or objective. State dictionaries contain CPU
tensors and are intended for torch.save, not JSON.
"""

import copy
import hashlib
import json
import math
import random
from dataclasses import dataclass
from typing import Callable, Mapping

import numpy as np
import torch


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def _cpu(value):
    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {k: _cpu(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return type(value)(_cpu(v) for v in value)
    return copy.deepcopy(value)


@dataclass(frozen=True)
class VisibleRegionSupport:
    frame: int
    region: str
    pool_sha256: str
    status: str
    raw_area_m2: float
    total_samples: int
    visible_samples: int
    self_visible_samples: int
    object_clear_samples: int
    source_minima: tuple
    membership: dict | None = None

    @property
    def key(self):
        return f"{self.frame}:{self.region}"

    def summary(self):
        return {k: copy.deepcopy(v) for k, v in vars(self).items() if k != "membership"}


@dataclass(frozen=True)
class SupportObservation:
    source_sha256: str
    regions: tuple[VisibleRegionSupport, ...]

    @property
    def pool_sha256(self):
        return _digest([(r.frame, r.region, r.pool_sha256) for r in self.regions])

    @property
    def empty(self):
        return any(r.status == "empty" for r in self.regions)

    def summary(self):
        return dict(
            source_sha256=self.source_sha256,
            pool_sha256=self.pool_sha256,
            empty=self.empty,
            regions=[r.summary() for r in self.regions],
        )

    def state_dict(self):
        return dict(source_sha256=self.source_sha256, regions=[_cpu(vars(r)) for r in self.regions])

    @classmethod
    def from_state_dict(cls, state):
        return cls(
            state["source_sha256"],
            tuple(VisibleRegionSupport(**r) for r in copy.deepcopy(state["regions"])),
        )


@dataclass(frozen=True)
class SupportGuardPolicy:
    minimum_area_fraction: float
    lr_multipliers: tuple[float, ...]

    def __post_init__(self):
        if not math.isfinite(self.minimum_area_fraction) or not 0 < self.minimum_area_fraction <= 1:
            raise ValueError("Area fraction must be finite in (0, 1]")
        values = tuple(self.lr_multipliers)
        if not values or any(not math.isfinite(v) or not 0 < v <= 1 for v in values):
            raise ValueError("Explicit finite learning-rate multipliers in (0, 1] required")
        if any(b >= a for a, b in zip(values, values[1:])):
            raise ValueError("Learning-rate multipliers must strictly decrease")
        object.__setattr__(self, "lr_multipliers", values)

    @property
    def sha256(self):
        return _digest(vars(self))


@dataclass(frozen=True)
class GuardStepResult:
    accepted: bool
    accepted_multiplier: float | None
    attempts: int
    observation: SupportObservation | None
    reason: str | None


def capture_rng(generators=None):
    generators = {} if generators is None else generators
    if not isinstance(generators, Mapping) or any(
        not isinstance(k, str) or not isinstance(v, torch.Generator) for k, v in generators.items()
    ):
        raise TypeError("Explicit generators must be a named mapping")
    return dict(
        python=random.getstate(),
        numpy=copy.deepcopy(np.random.get_state()),
        torch_cpu=torch.get_rng_state().clone(),
        torch_cuda=(
            [s.clone() for s in torch.cuda.get_rng_state_all()]
            if torch.cuda.is_initialized()
            else None
        ),
        generators={
            k: dict(device=str(v.device), state=v.get_state().clone())
            for k, v in generators.items()
        },
    )


def restore_rng(state, generators=None):
    generators = {} if generators is None else generators
    if set(generators) != set(state["generators"]):
        raise ValueError("Explicit RNG generator names differ from saved state")
    if any(str(generators[k].device) != v["device"] for k, v in state["generators"].items()):
        raise ValueError("Explicit RNG generator devices differ from saved state")
    if state["torch_cuda"] is not None and (
        not torch.cuda.is_initialized() or len(state["torch_cuda"]) != torch.cuda.device_count()
    ):
        raise ValueError("CUDA RNG device count/context differs from saved state")
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    if state["torch_cuda"] is not None:
        torch.cuda.set_rng_state_all(state["torch_cuda"])
    for k, v in state["generators"].items():
        generators[k].set_state(v["state"])


class VisibleSupportGuard:
    """Bounded same-gradient AdamW retries against fixed initial raw-area anchors.

    Pass all named runtime tensors, including protected tensors outside AdamW.
    The caller's observe() must inspect freshly evaluated candidate geometry and
    enforce its frozen-parameter constraints. invalidate() clears candidate-owned
    caches/details after rejection or error. Existing Tensor objects, gradients
    (including None), optimizer groups/state, and RNG are restored on rejection.
    No callback may change model/reference data; those remain caller-audited.

    on_attempt receives durable event dictionaries. Counts include unsuccessful
    calls and are never rolled back. Only an accepted proposal increments the
    accepted step and retains Adam moments. Persist state only at committed steps.
    """

    VERSION = 1

    def __init__(self, policy, *, source_sha256, pool_sha256):
        if not isinstance(policy, SupportGuardPolicy):
            raise TypeError("SupportGuardPolicy required")
        for value in (source_sha256, pool_sha256):
            if not isinstance(value, str) or len(value) != 64:
                raise ValueError("Explicit source/pool SHA256 identities required")
        self.policy, self.source_sha256, self.pool_sha256 = policy, source_sha256, pool_sha256
        self.anchor = self.current = self.accepted_rng = None
        self.accepted_steps = 0
        self.counts = dict(
            attempts_started=0,
            optimizer_calls_attempted=0,
            optimizer_calls_returned=0,
            observations_returned=0,
            rejected_attempts=0,
            accepted_updates=0,
            errors=0,
        )

    def _validate(self, observation):
        if not isinstance(observation, SupportObservation):
            raise TypeError("Typed SupportObservation required")
        if (
            observation.source_sha256 != self.source_sha256
            or observation.pool_sha256 != self.pool_sha256
        ):
            raise ValueError("Support source or canonical pool identity changed")
        if not observation.regions or len({r.key for r in observation.regions}) != len(
            observation.regions
        ):
            raise ValueError("Nonempty unique frame/region support records required")
        for r in observation.regions:
            if (
                r.status not in ("available", "empty")
                or not math.isfinite(r.raw_area_m2)
                or r.raw_area_m2 < 0
                or not 0 <= r.visible_samples <= r.total_samples
                or r.total_samples <= 0
                or (r.visible_samples == 0) != (r.status == "empty")
                or (r.raw_area_m2 == 0) != (r.status == "empty")
            ):
                raise ValueError("Invalid typed support area/count/status")
            if r.membership is None or "visible" not in r.membership:
                raise ValueError("Full membership required for guard persistence")
            mask = r.membership["visible"]
            if (
                not torch.is_tensor(mask)
                or mask.dtype != torch.bool
                or mask.shape != (r.total_samples,)
            ):
                raise ValueError("Invalid support membership mask")
            if int(mask.sum()) != r.visible_samples:
                raise ValueError("Support membership/count mismatch")

    def initialize(self, observation, *, accepted_step=0, generators=None):
        if self.anchor is not None:
            raise ValueError("Guard already initialized; use load_state_dict for own-resume")
        self._validate(observation)
        if observation.empty or type(accepted_step) is not int or accepted_step < 0:
            raise ValueError("Initialization requires nonempty support and a nonnegative step")
        self.anchor = observation.state_dict()
        self.current = observation.state_dict()
        self.accepted_steps = accepted_step
        self.accepted_rng = capture_rng(generators)

    def validate_current(self, observation, *, area_atol_m2=0.0, distance_atol_px=0.0):
        """Validate a normal loss's support after resume, without another query.

        Pool, source and membership are exact. Optional scalar tolerances must be
        explicitly supplied and do not waive membership differences.
        """
        self._validate(observation)
        if self.current is None:
            raise ValueError("Guard is not initialized")
        if any(not math.isfinite(x) or x < 0 for x in (area_atol_m2, distance_atol_px)):
            raise ValueError("Invalid explicit current-support tolerance")
        saved = SupportObservation.from_state_dict(self.current)
        for old, new in zip(saved.regions, observation.regions):
            if (
                old.key != new.key
                or old.status != new.status
                or not torch.equal(old.membership["visible"].cpu(), new.membership["visible"].cpu())
                or abs(old.raw_area_m2 - new.raw_area_m2) > area_atol_m2
            ):
                raise ValueError("Current support differs from accepted checkpoint")
            if len(old.source_minima) != len(new.source_minima):
                raise ValueError("Current source-minimum schema changed")
            for a, b in zip(old.source_minima, new.source_minima):
                if set(a) != set(b):
                    raise ValueError("Current source-minimum fields changed")
                for key in a:
                    if (
                        key in ("minimum_visible_pixel_distance_px", "minimum_visible_excess_px")
                        and a[key] is not None
                        and b[key] is not None
                    ):
                        if abs(a[key] - b[key]) > distance_atol_px:
                            raise ValueError("Current source minimum differs from checkpoint")
                    elif a[key] != b[key]:
                        raise ValueError("Current source minimum identity changed")

    def state_dict(self):
        if self.anchor is None:
            raise ValueError("Guard is not initialized")
        return _cpu(
            dict(
                version=self.VERSION,
                policy=vars(self.policy),
                policy_sha256=self.policy.sha256,
                source_sha256=self.source_sha256,
                pool_sha256=self.pool_sha256,
                anchor=self.anchor,
                current=self.current,
                accepted_steps=self.accepted_steps,
                accepted_rng=self.accepted_rng,
                counts=self.counts,
            )
        )

    def load_state_dict(self, state):
        if (
            state["version"] != self.VERSION
            or state["policy_sha256"] != self.policy.sha256
            or state["source_sha256"] != self.source_sha256
            or state["pool_sha256"] != self.pool_sha256
            or SupportGuardPolicy(**state["policy"]) != self.policy
        ):
            raise ValueError("Guard checkpoint schema/policy/source/pool changed")
        anchor = SupportObservation.from_state_dict(state["anchor"])
        current = SupportObservation.from_state_dict(state["current"])
        self._validate(anchor)
        self._validate(current)
        if (
            anchor.empty
            or current.empty
            or type(state["accepted_steps"]) is not int
            or state["accepted_steps"] < 0
        ):
            raise ValueError("Invalid accepted guard checkpoint")
        for a, b in zip(anchor.regions, current.regions):
            if b.raw_area_m2 < a.raw_area_m2 * self.policy.minimum_area_fraction:
                raise ValueError("Checkpoint support violates its fixed anchor")
        self.anchor, self.current = anchor.state_dict(), current.state_dict()
        self.accepted_steps = state["accepted_steps"]
        self.accepted_rng, self.counts = _cpu(state["accepted_rng"]), copy.deepcopy(state["counts"])

    def restore_rng(self, generators=None):
        if self.accepted_rng is None:
            raise ValueError("No accepted RNG state")
        restore_rng(self.accepted_rng, generators)

    def step(
        self,
        optimizer,
        parameters,
        observe: Callable,
        *,
        generators=None,
        on_attempt=None,
        invalidate=None,
    ):
        if self.anchor is None or not isinstance(optimizer, torch.optim.AdamW):
            raise ValueError("Initialized guard and AdamW optimizer required")
        if (
            not isinstance(parameters, Mapping)
            or not parameters
            or any(not isinstance(k, str) or not torch.is_tensor(v) for k, v in parameters.items())
            or len({id(v) for v in parameters.values()}) != len(parameters)
        ):
            raise ValueError("Unique named runtime tensor mapping required")
        all_ids = {id(v) for v in parameters.values()}
        opt_params = [p for g in optimizer.param_groups for p in g["params"]]
        if len({id(p) for p in opt_params}) != len(opt_params) or any(
            id(p) not in all_ids for p in opt_params
        ):
            raise ValueError("All optimizer parameters must be uniquely covered")
        if any(
            not bool(torch.isfinite(p).all())
            or (p.grad is not None and not bool(torch.isfinite(p.grad).all()))
            for p in parameters.values()
        ):
            raise ValueError("Finite current parameters and gradients required")
        snapshot = {
            k: (
                p.detach().clone(),
                p.requires_grad,
                p.grad,
                None if p.grad is None else p.grad.detach().clone(),
            )
            for k, p in parameters.items()
        }
        groups = list(optimizer.param_groups)
        group_data = [
            {k: list(v) if k == "params" else copy.deepcopy(v) for k, v in g.items()}
            for g in groups
        ]
        for g in group_data:
            if not isinstance(g["lr"], (int, float)) or not math.isfinite(g["lr"]) or g["lr"] < 0:
                raise ValueError("Guard requires finite scalar group learning rates")
        initial_state = {p: copy.deepcopy(s) for p, s in optimizer.state.items()}
        defaults = copy.deepcopy(optimizer.defaults)
        rng = capture_rng(generators)
        current_before, accepted_rng_before = _cpu(self.current), _cpu(self.accepted_rng)
        step_before = self.accepted_steps

        def rollback():
            with torch.no_grad():
                for k, p in parameters.items():
                    value, required, grad_object, grad_value = snapshot[k]
                    p.copy_(value)
                    p.requires_grad_(required)
                    p.grad = grad_object
                    if grad_object is not None:
                        grad_object.copy_(grad_value)
            optimizer.state.clear()
            optimizer.state.update({p: copy.deepcopy(s) for p, s in initial_state.items()})
            optimizer.param_groups[:] = groups
            for g, old in zip(groups, group_data):
                g.clear()
                g.update(
                    {k: list(v) if k == "params" else copy.deepcopy(v) for k, v in old.items()}
                )
            optimizer.defaults.clear()
            optimizer.defaults.update(copy.deepcopy(defaults))
            self.current, self.accepted_rng = _cpu(current_before), _cpu(accepted_rng_before)
            self.accepted_steps = step_before
            restore_rng(rng, generators)

        def emit(event, *, event_counts=None, **values):
            if on_attempt is not None:
                log_rng = capture_rng(generators)
                try:
                    on_attempt(
                        dict(
                            event=event,
                            accepted_step_before=step_before,
                            counts=dict(self.counts) if event_counts is None else event_counts,
                            **values,
                        )
                    )
                finally:
                    restore_rng(log_rng, generators)

        anchor = SupportObservation.from_state_dict(self.anchor)
        last = None
        for local, multiplier in enumerate(self.policy.lr_multipliers, 1):
            try:
                rollback()
                self.counts["attempts_started"] += 1
                for g, old in zip(groups, group_data):
                    g["lr"] = old["lr"] * multiplier
                emit("started", attempt=local, multiplier=multiplier)
                self.counts["optimizer_calls_attempted"] += 1
                optimizer.step()
                self.counts["optimizer_calls_returned"] += 1
                emit("optimizer_returned", attempt=local, multiplier=multiplier)
                if any(not bool(torch.isfinite(p).all()) for p in parameters.values()):
                    raise ValueError("Nonfinite proposed runtime parameter")
                last = observe()
                self.counts["observations_returned"] += 1
                self._validate(last)
                emit("observed", attempt=local, multiplier=multiplier, support=last.summary())
                accepted = all(
                    b.status == "available"
                    and b.raw_area_m2 >= a.raw_area_m2 * self.policy.minimum_area_fraction
                    for a, b in zip(anchor.regions, last.regions)
                )
                if accepted:
                    for g, old in zip(groups, group_data):
                        g["lr"] = old["lr"]
                    self.current = last.state_dict()
                    self.accepted_steps += 1
                    # Observations/loggers are diagnostics, not stochastic training
                    # work. Accepted continuation uses the post-gradient RNG state.
                    restore_rng(rng, generators)
                    self.accepted_rng = _cpu(rng)
                    emit(
                        "accepted",
                        attempt=local,
                        multiplier=multiplier,
                        event_counts=dict(
                            self.counts, accepted_updates=self.counts["accepted_updates"] + 1
                        ),
                    )
                    self.counts["accepted_updates"] += 1
                    return GuardStepResult(True, multiplier, local, last, None)
                self.counts["rejected_attempts"] += 1
                rollback()
                if invalidate is not None:
                    invalidate()
                # Invalidating diagnostics must not advance proposal randomness.
                restore_rng(rng, generators)
                emit(
                    "rejected",
                    attempt=local,
                    multiplier=multiplier,
                    reason="empty_or_below_fixed_initial_area_anchor",
                )
            except BaseException as error:
                self.counts["errors"] += 1
                rollback()
                secondary_errors = []
                try:
                    if invalidate is not None:
                        invalidate()
                except BaseException as invalidation_error:
                    secondary_errors.append("invalidate: " + repr(invalidation_error))
                finally:
                    restore_rng(rng, generators)
                try:
                    emit(
                        "error",
                        attempt=local,
                        multiplier=multiplier,
                        error=repr(error),
                        cleanup_errors=secondary_errors,
                    )
                except BaseException as logging_error:
                    secondary_errors.append("error logging: " + repr(logging_error))
                if secondary_errors and hasattr(error, "add_note"):
                    error.add_note(
                        "State restored; secondary callback failures: "
                        + "; ".join(secondary_errors)
                    )
                raise
        return GuardStepResult(
            False, None, len(self.policy.lr_multipliers), last, "retry_schedule_exhausted_no_update"
        )
