"""Opt-in runtime parity probe for contact-sensor cadence changes.

When ``manager_env.config.contact_sensor_parity_shadow`` is true the scene
config duplicates every ``ContactSensorCfg`` under ``shadow__<name>`` with the
ORIGINAL cadence (history_length / update_period as constructed), while the
primary sensor may run with the overridden cadence
(``contact_sensor_history_length`` / ``contact_sensor_update_period``).

This probe is called by the env wrapper right after ``env.step`` returns (after
the IsaacLab reward/termination/reset/observation phases). No physics step
happens between the reward-phase read and this point, so for non-reset
environments the primary tensors compared here are exactly the tensors the
reward terms consumed; reset environments were zeroed and lazily recomputed on
their first post-reset read, which both the primary (observation read) and the
shadow (this probe's read) perform against the same physics state. Equality
here therefore implies equality at the actual consumption points, including
reset boundaries. Nothing in this module is active unless the probe is enabled.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import torch

SHADOW_PREFIX = "shadow__"


@dataclass
class _PairStats:
    steps: int = 0
    exact_steps: int = 0
    max_abs_diff: float = 0.0
    first_diff_step: int | None = None
    fields: dict[str, dict[str, Any]] = field(default_factory=dict)


def reward_phase_snapshot(sensors: Mapping[str, Any], step: int, *, family_force_fn=None, env=None,
                          family_names=("canonical_active_entity_contact_sensor",
                                        "canonical_ground_contact_sensor",
                                        "canonical_static_context_contact_sensor")) -> dict[str, Any]:
    """Compare primary vs ``shadow__`` sensors AT THE REWARD PHASE (called by the detector).

    The end-of-step probe cannot see the pre-reset values of environments that
    are reset in the same step (IsaacLab zeroes and lazily re-reads them before
    the probe). The required-contact reward reads the detector exactly once per
    step in the reward phase, so a snapshot taken there compares the tensors the
    reward terms (and the grasp rewards, which read the hand sensors in the same
    phase) actually consumed, for EVERY environment including those about to be
    reset. Returns a plain dict with per-pair / per-family exactness.
    """
    out = {"step": int(step), "pairs": {}, "families": {}}
    for name, shadow_sensor in sensors.items():
        name = str(name)
        if not name.startswith(SHADOW_PREFIX):
            continue
        primary = sensors.get(name[len(SHADOW_PREFIX):])
        if primary is None:
            continue
        rec = {}
        for field_name in ("force_matrix_w", "net_forces_w"):
            a = getattr(primary.data, field_name, None)
            b = getattr(shadow_sensor.data, field_name, None)
            if a is None and b is None:
                continue
            if a is None or b is None or tuple(a.shape) != tuple(b.shape):
                rec[field_name] = {"exact": False, "max_abs_diff": float("inf"), "shape_mismatch": True}
                continue
            exact = bool(torch.equal(a, b))
            rec[field_name] = {"exact": exact, "shape_mismatch": False,
                               "max_abs_diff": 0.0 if exact else float((a.float() - b.float()).abs().max().item())}
        out["pairs"][name[len(SHADOW_PREFIX):]] = rec
    if family_force_fn is not None and env is not None:
        for family in family_names:
            has_primary = any(str(n) == family or str(n).startswith(family + "__") for n in sensors)
            has_shadow = any(str(n) == SHADOW_PREFIX + family or str(n).startswith(SHADOW_PREFIX + family + "__") for n in sensors)
            if not (has_primary and has_shadow):
                continue
            a = family_force_fn(env, family)
            b = family_force_fn(env, SHADOW_PREFIX + family)
            exact = bool(torch.equal(a, b))
            out["families"][family] = {"exact": exact,
                                       "max_abs_diff": 0.0 if exact else float((a.float() - b.float()).abs().max().item())}
    return out


class ContactSensorParityProbe:
    """Compare each ``shadow__<name>`` ContactSensor against ``<name>``."""

    def __init__(self, *, enabled: bool = False, log_every_steps: int = 100,
                 output_path: str | None = None, logger=None) -> None:
        self.enabled = bool(enabled)
        self.log_every_steps = max(1, int(log_every_steps or 1))
        self.output_path = Path(output_path).expanduser() if output_path else None
        self._logger = logger
        self.stats: dict[str, _PairStats] = {}
        self.family_stats: dict[str, _PairStats] = {}
        self.sensor_cadence: dict[str, dict[str, Any]] = {}
        self.observed_steps = 0
        self.steps_with_resets = 0
        self.total_reset_envs = 0
        # reward-phase snapshots recorded by the detector (see reward_phase_snapshot)
        self.reward_phase_steps = 0
        self.reward_phase_pair_stats: dict[str, dict[str, Any]] = {}
        self.reward_phase_pair_fields: dict[str, dict[str, dict[str, Any]]] = {}
        self.reward_phase_family_stats: dict[str, dict[str, Any]] = {}
        self._last_reward_phase_step = None
        self.missing_primary: list[str] = []
        self.started_utc = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    @classmethod
    def from_config(cls, config: Mapping[str, Any] | None, logger=None):
        cfg = (config or {}).get("contact_sensor_parity_probe", None) if config else None
        if not cfg or not bool(cfg.get("enabled", False)):
            return None
        return cls(
            enabled=True,
            log_every_steps=int(cfg.get("log_every_steps", 100) or 100),
            output_path=cfg.get("output_path", None),
            logger=logger,
        )

    @staticmethod
    def _cadence(sensor) -> dict[str, Any]:
        cfg = getattr(sensor, "cfg", None)
        return {
            "history_length": getattr(cfg, "history_length", None),
            "update_period": getattr(cfg, "update_period", None),
            "track_air_time": getattr(cfg, "track_air_time", None),
            "prim_path": getattr(cfg, "prim_path", None),
            "num_filters": len(getattr(cfg, "filter_prim_paths_expr", ()) or ()),
        }

    def _compare_field(self, pair: _PairStats, field_name: str, primary: torch.Tensor | None,
                       shadow: torch.Tensor | None, step: int) -> bool:
        if primary is None and shadow is None:
            return True
        entry = pair.fields.setdefault(
            field_name, {"steps": 0, "exact_steps": 0, "max_abs_diff": 0.0, "shape_mismatch": 0}
        )
        entry["steps"] += 1
        if primary is None or shadow is None or tuple(primary.shape) != tuple(shadow.shape):
            entry["shape_mismatch"] += 1
            return False
        exact = bool(torch.equal(primary, shadow))
        if exact:
            entry["exact_steps"] += 1
            return True
        diff = float((primary.float() - shadow.float()).abs().max().item())
        if diff > entry["max_abs_diff"]:
            entry["max_abs_diff"] = diff
        return False

    def observe(self, sensors: Mapping[str, Any], step: int, *, env=None,
                family_force_fn=None, family_names=("canonical_active_entity_contact_sensor",
                                                     "canonical_ground_contact_sensor",
                                                     "canonical_static_context_contact_sensor")) -> None:
        """Read every shadow/primary pair once and accumulate exactness statistics.

        ``family_force_fn(env, family_name)`` (optional) must return the grouped
        canonical family force exactly as the reward detector consumes it
        (``_canonical_sensor_force``); it is evaluated for each primary family and
        its ``shadow__`` twin so the comparison happens on the consumed quantity,
        not only on raw sensor tensors. ``env`` additionally provides
        ``episode_length_buf`` so steps containing resets are counted.
        """
        if not self.enabled:
            return
        self.observed_steps += 1
        snapshot = getattr(env, "_contact_sensor_parity_reward_phase", None) if env is not None else None
        if isinstance(snapshot, dict) and snapshot.get("step") != self._last_reward_phase_step:
            self._last_reward_phase_step = snapshot.get("step")
            self.reward_phase_steps += 1
            for pname, rec in snapshot.get("pairs", {}).items():
                # Combined (all fields) per-pair view, kept for backward compatibility ...
                st = self.reward_phase_pair_stats.setdefault(pname, {"steps": 0, "exact_steps": 0, "max_abs_diff": 0.0, "first_diff_step": None})
                st["steps"] += 1
                if all(v["exact"] for v in rec.values()):
                    st["exact_steps"] += 1
                else:
                    st["max_abs_diff"] = max(st["max_abs_diff"], max(v["max_abs_diff"] for v in rec.values()))
                    if st["first_diff_step"] is None:
                        st["first_diff_step"] = snapshot.get("step")
                # ... and the PER-FIELD view required for consumption-point acceptance: each
                # field (force_matrix_w = consumed by the hand/canonical consumers; net_forces_w)
                # keeps its own steps / exact_steps / max_abs_diff / shape_mismatch / first_diff_step.
                fields = self.reward_phase_pair_fields.setdefault(pname, {})
                for field_name, v in rec.items():
                    fst = fields.setdefault(field_name, {"steps": 0, "exact_steps": 0, "max_abs_diff": 0.0,
                                                         "shape_mismatch": 0, "first_diff_step": None})
                    fst["steps"] += 1
                    if v.get("shape_mismatch", False):
                        fst["shape_mismatch"] += 1
                    if v["exact"]:
                        fst["exact_steps"] += 1
                    else:
                        if v["max_abs_diff"] != float("inf"):
                            fst["max_abs_diff"] = max(fst["max_abs_diff"], v["max_abs_diff"])
                        if fst["first_diff_step"] is None:
                            fst["first_diff_step"] = snapshot.get("step")
            for fname, rec in snapshot.get("families", {}).items():
                st = self.reward_phase_family_stats.setdefault(fname, {"steps": 0, "exact_steps": 0, "max_abs_diff": 0.0, "first_diff_step": None})
                st["steps"] += 1
                if rec["exact"]:
                    st["exact_steps"] += 1
                else:
                    st["max_abs_diff"] = max(st["max_abs_diff"], rec["max_abs_diff"])
                    if st["first_diff_step"] is None:
                        st["first_diff_step"] = snapshot.get("step")
        if env is not None and hasattr(env, "episode_length_buf"):
            reset_envs = int((env.episode_length_buf == 0).sum().item())
            self.total_reset_envs += reset_envs
            if reset_envs > 0:
                self.steps_with_resets += 1
        if family_force_fn is not None and env is not None:
            for family in family_names:
                has_primary = any(str(n) == family or str(n).startswith(family + "__") for n in sensors)
                has_shadow = any(str(n) == SHADOW_PREFIX + family or str(n).startswith(SHADOW_PREFIX + family + "__") for n in sensors)
                if not (has_primary and has_shadow):
                    continue
                pair = self.family_stats.setdefault(family, _PairStats())
                try:
                    primary_force = family_force_fn(env, family)
                    shadow_force = family_force_fn(env, SHADOW_PREFIX + family)
                except Exception as exc:  # noqa: BLE001
                    pair.fields.setdefault("error", {"count": 0, "last": ""})
                    pair.fields["error"]["count"] += 1
                    pair.fields["error"]["last"] = f"{type(exc).__name__}: {exc}"[:200]
                    continue
                exact = self._compare_field(pair, "grouped_force", primary_force, shadow_force, step)
                pair.steps += 1
                if exact:
                    pair.exact_steps += 1
                elif pair.first_diff_step is None:
                    pair.first_diff_step = int(step)
                    pair.max_abs_diff = pair.fields["grouped_force"]["max_abs_diff"]
                else:
                    pair.max_abs_diff = max(pair.max_abs_diff, pair.fields["grouped_force"]["max_abs_diff"])
        for name, shadow_sensor in sensors.items():
            name = str(name)
            if not name.startswith(SHADOW_PREFIX):
                continue
            primary_name = name[len(SHADOW_PREFIX):]
            primary_sensor = sensors.get(primary_name)
            if primary_sensor is None:
                if primary_name not in self.missing_primary:
                    self.missing_primary.append(primary_name)
                continue
            pair = self.stats.setdefault(primary_name, _PairStats())
            if primary_name not in self.sensor_cadence:
                self.sensor_cadence[primary_name] = {
                    "primary": self._cadence(primary_sensor),
                    "shadow": self._cadence(shadow_sensor),
                }
            # Reading .data lazily refreshes any outdated (reset) environments of
            # either sensor against the current physics state.
            primary_data = primary_sensor.data
            shadow_data = shadow_sensor.data
            all_exact = True
            for field_name in ("force_matrix_w", "net_forces_w"):
                all_exact &= self._compare_field(
                    pair, field_name,
                    getattr(primary_data, field_name, None),
                    getattr(shadow_data, field_name, None),
                    step,
                )
            pair.steps += 1
            if all_exact:
                pair.exact_steps += 1
            else:
                pair.max_abs_diff = max(
                    pair.max_abs_diff,
                    max(entry["max_abs_diff"] for entry in pair.fields.values()),
                )
                if pair.first_diff_step is None:
                    pair.first_diff_step = int(step)
        if self.observed_steps % self.log_every_steps == 0:
            self.flush(step)

    def summary(self) -> dict[str, Any]:
        pairs = {
            name: {
                "steps": s.steps,
                "exact_steps": s.exact_steps,
                "max_abs_diff": s.max_abs_diff,
                "first_diff_step": s.first_diff_step,
                "fields": s.fields,
            }
            for name, s in self.stats.items()
        }
        families = {
            name: {
                "steps": s.steps,
                "exact_steps": s.exact_steps,
                "max_abs_diff": s.max_abs_diff,
                "first_diff_step": s.first_diff_step,
                "fields": s.fields,
            }
            for name, s in self.family_stats.items()
        }
        all_exact = all(
            s.steps > 0 and s.exact_steps == s.steps for s in self.stats.values()
        ) and bool(self.stats)
        families_exact = all(
            s.steps > 0 and s.exact_steps == s.steps and "error" not in s.fields
            for s in self.family_stats.values()
        ) and bool(self.family_stats)
        rp_pairs_exact = bool(self.reward_phase_pair_stats) and all(
            v["steps"] > 0 and v["exact_steps"] == v["steps"] for v in self.reward_phase_pair_stats.values())
        rp_families_exact = bool(self.reward_phase_family_stats) and all(
            v["steps"] > 0 and v["exact_steps"] == v["steps"] for v in self.reward_phase_family_stats.values())
        return {
            "started_utc": self.started_utc,
            "observed_steps": self.observed_steps,
            "steps_with_resets": self.steps_with_resets,
            "total_reset_envs": self.total_reset_envs,
            "num_pairs": len(self.stats),
            "all_pairs_exact_every_step": all_exact,
            "num_families": len(self.family_stats),
            "all_families_exact_every_step": families_exact,
            "reward_phase_steps": self.reward_phase_steps,
            "reward_phase_all_pairs_exact": rp_pairs_exact,
            "reward_phase_all_families_exact": rp_families_exact,
            "reward_phase_pairs": self.reward_phase_pair_stats,
            "reward_phase_pair_fields": self.reward_phase_pair_fields,
            "reward_phase_field_exact": {
                field: all(
                    f[field]["steps"] > 0 and f[field]["exact_steps"] == f[field]["steps"]
                    for f in self.reward_phase_pair_fields.values() if field in f
                ) and any(field in f for f in self.reward_phase_pair_fields.values())
                for field in ("force_matrix_w", "net_forces_w")
            },
            "reward_phase_families": self.reward_phase_family_stats,
            "missing_primary": list(self.missing_primary),
            "sensor_cadence": self.sensor_cadence,
            "pairs": pairs,
            "families": families,
        }

    def flush(self, step: int) -> None:
        summary = self.summary()
        inexact = sorted(
            name for name, s in self.stats.items() if s.exact_steps != s.steps
        )
        message = (
            "[CONTACT SENSOR PARITY] "
            f"step={int(step)} observed_steps={self.observed_steps} pairs={len(self.stats)} "
            f"all_exact={summary['all_pairs_exact_every_step']} inexact={inexact[:8]} "
            f"families={len(self.family_stats)} families_exact={summary['all_families_exact_every_step']} "
            f"reward_phase_steps={self.reward_phase_steps} rp_pairs_exact(all fields)={summary['reward_phase_all_pairs_exact']} "
            f"rp_force_matrix_w_exact={summary['reward_phase_field_exact'].get('force_matrix_w')} "
            f"rp_net_forces_w_exact={summary['reward_phase_field_exact'].get('net_forces_w')} "
            f"rp_families_exact={summary['reward_phase_all_families_exact']} "
            f"steps_with_resets={self.steps_with_resets} reset_envs={self.total_reset_envs} "
            f"missing_primary={self.missing_primary[:4]}"
        )
        if self._logger is not None:
            self._logger.info(message)
        else:
            print(message, flush=True)  # noqa: T201
        if self.output_path is not None:
            self.output_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.output_path.with_name(self.output_path.name + f".tmp{os.getpid()}")
            with tmp.open("w", encoding="utf-8") as handle:
                json.dump(summary, handle, indent=1, sort_keys=True)
            tmp.replace(self.output_path)
