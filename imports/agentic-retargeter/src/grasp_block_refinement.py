"""Optional thumb-first coordinate refinement of independent hand candidates.

Inspired by DexFlow's sequential finger refinement, not a reproduction of its
unreleased implementation. No geometry, contact normals, or acceptance rules are
defined here. Callers own the objective, native limits, and independent review.
"""

from collections.abc import Mapping
import math
from numbers import Integral, Real


def _stages(finger_columns, steps_per_stage, width):
    if not isinstance(finger_columns, Mapping) or not finger_columns:
        raise ValueError("finger_columns must be a nonempty ordered mapping")
    if isinstance(steps_per_stage, Mapping):
        if set(steps_per_stage) != set(finger_columns):
            raise ValueError("steps_per_stage must cover exactly the finger stages")
        counts = steps_per_stage
    else:
        counts = dict.fromkeys(finger_columns, steps_per_stage)
    stages, used = [], set()
    for name, columns in finger_columns.items():
        if not isinstance(name, str) or not name.strip():
            raise ValueError("finger stage names must be nonempty strings")
        try:
            columns = list(columns)
        except TypeError as exc:
            raise ValueError("finger columns must be integer sequences") from exc
        if not columns or any(isinstance(c, bool) or not isinstance(c, Integral)
                              or c < 6 or c >= width for c in columns):
            raise ValueError("finger columns must be valid indices >= 6; wrist is frozen")
        if len(set(columns)) != len(columns) or used.intersection(columns):
            raise ValueError("finger stages must have disjoint unique columns")
        count = counts[name]
        if isinstance(count, bool) or not isinstance(count, Integral) or count < 1:
            raise ValueError("steps_per_stage must contain positive integer counts")
        used.update(columns)
        stages.append((name, [int(c) for c in columns], int(count)))
    return stages


def refine_finger_blocks(parameter, closure, finger_columns, *, steps_per_stage,
                         learning_rate=.008, project=None, ranking=None):
    """Refine ``parameter[B,D]`` in place and return a serializable trace.

    ``closure()`` returns one finite differentiable objective per candidate.
    Candidate rows must be independent: selection restores each row separately.
    ``finger_columns`` maps stage names to parameter columns, in desired order;
    columns 0:6 (free wrist) and every other column stay exactly frozen per stage.
    A fresh Adam is created at every stage, preventing stale optimizer momentum.

    ``project(parameter)`` optionally enforces native limits in-place after each
    step, under no_grad. Its edits to frozen columns are discarded. The caller
    must pass an already admissible seed: initial points are not projected.

    ``ranking(loss)`` optionally supplies [B,K] lower-is-better lexicographic
    scores, e.g. [clearance_violation, gap_violation, objective]. It is a search
    proxy, never an acceptance check. Include objective as its final column if
    desired. Without it, objective alone selects the best snapshot per row.
    Every iterate, including each stage's seed, is considered. Each stage starts
    from the preceding stage's selected rows, not its final unselected iterate.
    Exceptions restore the original parameter; no candidate is accepted here.
    """
    import torch

    if (not isinstance(parameter, torch.Tensor) or parameter.ndim != 2
            or parameter.shape[0] < 1 or parameter.shape[1] <= 6
            or not parameter.is_floating_point() or not parameter.is_leaf
            or not parameter.requires_grad):
        raise ValueError("parameter must be a nonempty floating leaf [B,D>6] with gradients")
    if not torch.isfinite(parameter).all():
        raise ValueError("parameter must be finite")
    if (isinstance(learning_rate, bool) or not isinstance(learning_rate, Real)
            or not math.isfinite(learning_rate) or learning_rate <= 0):
        raise ValueError("learning_rate must be finite and positive")
    for name, callback in [("closure", closure), ("project", project), ("ranking", ranking)]:
        if (name == "closure" or callback is not None) and not callable(callback):
            raise ValueError(f"{name} must be callable")
    stages = _stages(finger_columns, steps_per_stage, parameter.shape[1])
    original = parameter.detach().clone()
    initial_gradient = None if parameter.grad is None else parameter.grad.detach().clone()
    ranking_width = None

    def evaluate():
        nonlocal ranking_width
        loss = closure()
        if (not isinstance(loss, torch.Tensor) or loss.shape != (parameter.shape[0],)
                or loss.device != parameter.device or not loss.is_floating_point()
                or not torch.isfinite(loss).all()):
            raise ValueError("closure must return one finite floating loss per candidate")
        with torch.no_grad():
            score = loss[:, None] if ranking is None else ranking(loss.detach())
            if (not isinstance(score, torch.Tensor) or score.ndim != 2
                    or score.shape[0] != parameter.shape[0] or score.shape[1] < 1
                    or score.device != parameter.device or not score.is_floating_point()
                    or not torch.isfinite(score).all()):
                raise ValueError("ranking must return finite floating scores [B,K>=1]")
            if ranking_width is not None and ranking_width != score.shape[1]:
                raise ValueError("ranking score width must remain constant")
            ranking_width = score.shape[1]
        return loss, score.detach().clone()

    def better_rows(score, best):
        better = torch.zeros(parameter.shape[0], dtype=torch.bool, device=parameter.device)
        tied = torch.ones_like(better)
        for column in range(score.shape[1]):
            better |= tied & (score[:, column] < best[:, column])
            tied &= score[:, column] == best[:, column]
        return better

    report = dict(schema="agentic.grasp_block_refinement.v1", acceptance=False,
                  selection="objective" if ranking is None else "lexicographic_proxy",
                  source="DexFlow-inspired ordered finger refinement",
                  frozen_wrist_columns=list(range(6)), candidate_count=parameter.shape[0],
                  learning_rate=float(learning_rate), stages=[])
    try:
        for name, columns, count in stages:
            frozen = parameter.detach().clone()
            mask = torch.zeros_like(parameter, dtype=torch.bool)
            mask[:, columns] = True
            optimizer = torch.optim.Adam([parameter], lr=float(learning_rate))
            loss, score = evaluate()
            best_value = parameter.detach().clone()
            best_score, best_loss = score.clone(), loss.detach().clone()
            best_step = torch.zeros(parameter.shape[0], dtype=torch.long, device=parameter.device)
            losses, scores = [loss.detach().clone()], [score.clone()]
            for step in range(1, count + 1):
                optimizer.zero_grad(set_to_none=True)
                if not loss.requires_grad:
                    raise ValueError("closure loss must be differentiable in parameter")
                loss.sum().backward()
                if parameter.grad is None or not torch.isfinite(parameter.grad).all():
                    raise ValueError("closure must produce finite parameter gradients")
                parameter.grad.masked_fill_(~mask, 0)
                optimizer.step()
                with torch.no_grad():
                    if project is not None:
                        project(parameter)
                    parameter[~mask] = frozen[~mask]
                    if not torch.isfinite(parameter).all():
                        raise ValueError("step or projection produced nonfinite parameter")
                loss, score = evaluate()
                losses.append(loss.detach().clone())
                scores.append(score.clone())
                with torch.no_grad():
                    improved = better_rows(score, best_score)
                    best_value[improved] = parameter[improved]
                    best_score[improved] = score[improved]
                    best_loss[improved] = loss.detach()[improved]
                    best_step[improved] = step
            with torch.no_grad():
                parameter.copy_(best_value)
            report["stages"].append(dict(
                name=name, columns=columns, steps=count,
                selected_step_per_candidate=best_step.cpu().tolist(),
                selected_loss_per_candidate=best_loss.cpu().tolist(),
                selected_ranking_per_candidate=best_score.cpu().tolist(),
                loss_history=torch.stack(losses).cpu().tolist(),
                ranking_history=torch.stack(scores).cpu().tolist(),
                optimizer_state_reset=True))
        report["total_steps"] = sum(count for _, _, count in stages)
        return report
    except Exception:
        with torch.no_grad():
            parameter.copy_(original)
        raise
    finally:
        parameter.grad = initial_gradient
