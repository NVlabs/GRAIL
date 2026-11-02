"""Closed-loop FoundationPose tracking with independently supplied MoSE3 proposals.

Inputs and outputs use the original, scaled CAD frame in OpenCV camera coordinates.
Only the selected current-frame hypothesis becomes the next tracker state. This
module neither smooths trajectories nor fills invalid outputs. Native CUDA/model
imports are lazy; callers may share a runtime across sequences.
"""

import copy
import random
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np


@dataclass
class HybridResult:
    poses: np.ndarray
    valid: np.ndarray
    records: list
    continuation_poses: np.ndarray
    proposal_poses: np.ndarray
    scores: np.ndarray
    selected_centered: np.ndarray
    live_centered: np.ndarray


def valid_pose(pose):
    """Check finite rigid transforms; this is not a tracking-quality test."""
    pose = np.asarray(pose)
    return bool(
        pose.shape == (4, 4)
        and np.isfinite(pose).all()
        and np.max(np.abs(pose[3] - [0, 0, 0, 1])) < 1e-4
        and np.max(np.abs(pose[:3, :3].T @ pose[:3, :3] - np.eye(3))) < 1e-3
        and abs(np.linalg.det(pose[:3, :3]) - 1) < 1e-3
    )


def centered_pose(original, center):
    # Use the same complete homogeneous composition as the inverse bridge.
    # A float32 camera inverse can retain tiny nonzero bottom-row elements;
    # modifying only the top translation silently assumes those are exact zero.
    translation = np.eye(4, dtype=np.float64)
    translation[:3, 3] = np.asarray(center, dtype=np.float64)
    return np.asarray(original, dtype=np.float64) @ translation


def select_hypothesis(continuation_valid, proposal_valid, scores, margin=0.05):
    """Compare a jointly scored pair; native scores are not probabilities."""
    if not np.isfinite(margin) or margin < 0:
        raise ValueError("Selection margin must be finite and nonnegative")
    if continuation_valid and not proposal_valid:
        return "continuation", "proposal_unavailable_or_invalid"
    if proposal_valid and not continuation_valid:
        return "proposal", "sole_valid_proposal_rescue"
    if not continuation_valid and not proposal_valid:
        return "invalid", "neither_hypothesis_valid"
    if scores is None or np.shape(scores) != (2,) or not np.isfinite(scores).all():
        return "continuation", "scoring_failed_or_nonfinite"
    if float(scores[1]) - float(scores[0]) > margin:
        return "proposal", "strictly_higher_same_call_score"
    if scores[1] > scores[0]:
        return "continuation", "positive_score_margin_not_above_threshold"
    return "continuation", "exact_tie" if scores[1] == scores[0] else "continuation_higher_score"


def _fatal_cuda(error):
    message = (type(error).__name__ + " " + str(error)).lower()
    return any(
        text in message
        for text in (
            "cuda",
            "device-side",
            "outofmemory",
            "out of memory",
            "cublas",
            "cudnn",
            "illegal memory access",
        )
    )


@contextmanager
def _isolate_proposal(torch, refiner):
    """Optional proposal work must not consume the continuation's RNG stream."""
    states = (
        random.getstate(),
        copy.deepcopy(np.random.get_state()),
        torch.get_rng_state().clone(),
        [state.clone() for state in torch.cuda.get_rng_state_all()],
    )
    tensor_type, dtype = torch.tensor([]).type(), torch.get_default_dtype()
    diagnostics = {key: getattr(refiner, key) for key in ("last_trans_update", "last_rot_update")}
    training = [(module, module.training) for module in refiner.model.modules()]
    poisoned = False
    try:
        yield
    except BaseException as error:
        poisoned = _fatal_cuda(error)
        raise
    finally:
        random.setstate(states[0])
        np.random.set_state(states[1])
        torch.set_rng_state(states[2])
        if not poisoned:
            torch.cuda.set_rng_state_all(states[3])
            torch.set_default_tensor_type(tensor_type)
            torch.set_default_dtype(dtype)
        for key, value in diagnostics.items():
            setattr(refiner, key, value)
        if any(module.training != flag for module, flag in training):
            raise RuntimeError("Native refiner training mode changed during proposal evaluation")


def load_runtime(foundationpose_dir=None):
    """Load native FoundationPose weights and CUDA context, once per worker.

    Requires the initialized FoundationPose checkout (including weights/native
    extensions) and its Python dependencies. No model download is performed.
    """
    root = (
        Path(foundationpose_dir).resolve()
        if foundationpose_dir is not None
        else Path(__file__).resolve().parents[2] / "imports" / "FoundationPose"
    )
    if not (root / "estimater.py").is_file():
        raise FileNotFoundError(f"Initialize FoundationPose and its weights at {root}")
    existing = sys.modules.get("estimater")
    if existing is not None and Path(existing.__file__).resolve() != root / "estimater.py":
        raise RuntimeError("A different FoundationPose checkout is already imported")
    sys.path.insert(0, str(root))
    import nvdiffrast.torch as dr
    import torch
    from estimater import FoundationPose
    from learning.training.predict_pose_refine import PoseRefinePredictor
    from learning.training.predict_score import ScorePredictor
    from Utils import bilateral_filter_depth, depth2xyzmap_batch, erode_depth, set_seed

    set_seed(0)
    scorer, refiner = ScorePredictor(), PoseRefinePredictor()
    if scorer.model.training or refiner.model.training:
        raise RuntimeError("Native scorer and refiner must be in evaluation mode")
    return SimpleNamespace(
        torch=torch,
        FoundationPose=FoundationPose,
        set_seed=set_seed,
        erode_depth=erode_depth,
        bilateral_filter_depth=bilateral_filter_depth,
        depth2xyzmap_batch=depth2xyzmap_batch,
        scorer=scorer,
        refiner=refiner,
        ctx=dr.RasterizeCudaContext(),
    )


def _settings(settings):
    options = dict(
        track_iterations=2,
        proposal_refine_iterations=2,
        score_margin=0.05,
        observed_depth="zero",
        proposal_policy="quality_pass",
        tie_policy="continuation",
        nonfinite_score_policy="continuation",
        sole_valid_proposal_rescue=True,
        smoothing=False,
    )
    supplied = dict(settings or {})
    unknown = set(supplied) - set(options)
    if unknown:
        raise ValueError(f"Unknown hybrid settings: {sorted(unknown)}")
    options.update(supplied)
    for name in ("track_iterations", "proposal_refine_iterations"):
        if (
            not isinstance(options[name], int)
            or isinstance(options[name], bool)
            or options[name] < 1
        ):
            raise ValueError(f"{name} must be a positive integer")
    select_hypothesis(False, False, None, options["score_margin"])
    fixed = dict(
        observed_depth="zero",
        proposal_policy="quality_pass",
        tie_policy="continuation",
        nonfinite_score_policy="continuation",
        sole_valid_proposal_rescue=True,
        smoothing=False,
    )
    if any(options[name] != value for name, value in fixed.items()):
        raise ValueError(f"Supported hybrid policies are {fixed}; postprocess smoothing separately")
    return options


def track_hybrid(
    mesh,
    initial_pose,
    K,
    rgb_frames,
    proposal_poses,
    proposal_quality_pass,
    *,
    settings=None,
    foundationpose_dir=None,
    debug_dir=None,
    runtime=None,
):
    """Track one sequence from an explicitly supplied initial pose.

    ``mesh`` is a native trimesh already scaled in meters. ``rgb_frames`` includes
    frame zero and yields HxWx3 uint8 RGB images; proposals and their quality mask
    have the same source timeline. The mask gates proposals independently of
    geometric SE(3) validity. Failed updates remain NaN/invalid, while the last
    selected live state is retained for the next frame. Frame zero is unchanged.
    """
    options = _settings(settings)
    proposals = np.asarray(proposal_poses, dtype=np.float64)
    quality = np.asarray(proposal_quality_pass)
    T0, K = np.asarray(initial_pose, dtype=np.float64), np.asarray(K, dtype=np.float64)
    if proposals.ndim != 3 or proposals.shape[1:] != (4, 4) or len(proposals) < 1:
        raise ValueError("Proposals must have shape (T, 4, 4), T >= 1")
    if quality.shape != (len(proposals),) or quality.dtype != bool:
        raise ValueError("Proposal quality mask must be bool with shape (T,)")
    if not valid_pose(T0) or not np.array_equal(proposals[0], T0) or not quality[0]:
        raise ValueError("Proposal frame zero must exactly match the valid supplied initial pose")
    if K.shape != (3, 3) or not np.isfinite(K).all() or min(K[0, 0], K[1, 1]) <= 0:
        raise ValueError("A finite calibrated 3x3 camera matrix is required")
    if len(mesh.vertices) == 0 or len(mesh.faces) == 0 or not np.isfinite(mesh.vertices).all():
        raise ValueError("A finite nonempty native scaled CAD mesh is required")
    frames = iter(rgb_frames)
    shape = None

    def read_rgb(index):
        nonlocal shape
        try:
            rgb = np.asarray(next(frames))
        except StopIteration as error:
            raise ValueError(f"Video ended before source frame {index}") from error
        if rgb.ndim != 3 or rgb.shape[-1] != 3 or rgb.dtype != np.uint8:
            raise ValueError(f"Source frame {index} must be HxWx3 uint8 RGB")
        if shape is not None and rgb.shape != shape:
            raise ValueError("Source image dimensions changed")
        shape = rgb.shape
        return rgb

    read_rgb(0)
    api = runtime if runtime is not None else load_runtime(foundationpose_dir)
    torch, refiner, scorer, ctx = api.torch, api.refiner, api.scorer, api.ctx
    api.set_seed(0)
    est = api.FoundationPose(
        model_pts=mesh.vertices,
        model_normals=mesh.vertex_normals,
        mesh=mesh,
        scorer=scorer,
        refiner=refiner,
        glctx=ctx,
        debug=0,
        debug_dir=str(debug_dir or Path.cwd() / "foundationpose_debug"),
    )
    est.diameter = float(est.diameter)
    init = centered_pose(T0, est.model_center)
    minus = np.eye(4)
    minus[:3, 3] = -np.asarray(est.model_center)
    if not np.allclose(init @ minus, T0, rtol=0, atol=1e-10):
        raise ValueError("Original CAD to centered FoundationPose bridge failed")
    est.pose_last = torch.as_tensor(init, dtype=torch.float32, device="cuda")
    n = len(proposals)
    poses = np.full((n, 4, 4), np.nan)
    valid = np.zeros(n, bool)
    poses[0], valid[0] = T0, True
    continuation, refined = np.full_like(poses, np.nan), np.full_like(poses, np.nan)
    continuation[0] = T0
    scores = np.full((n, 2), np.nan)
    selected = np.full((n, 4, 4), np.nan, dtype=np.float32)
    live = np.full_like(selected, np.nan)
    selected[0] = live[0] = est.pose_last.detach().cpu().numpy().reshape(4, 4)
    zero, Kn = np.zeros(shape[:2], dtype=float), K.astype(np.float32)
    records = []
    for t in range(1, n):
        rgb = read_rgb(t)
        incoming = est.pose_last.detach().clone()
        Qcont = Pcont = Qprop = Pprop = score_values = None
        row = dict(
            frame=t,
            continuation_valid=False,
            proposal_input_valid=False,
            proposal_refined_valid=False,
            score_pair_called=False,
            finite_score_pair=False,
            scores=None,
            errors={},
        )
        try:
            Pcont = np.asarray(
                est.track_one(
                    rgb=rgb,
                    depth=zero,
                    K=Kn,
                    iteration=options["track_iterations"],
                )
            ).copy()
            Qcont = est.pose_last.detach().clone()
            row["continuation_valid"] = valid_pose(Pcont)
            if row["continuation_valid"]:
                continuation[t] = Pcont
        except Exception as error:
            if _fatal_cuda(error):
                raise
            row["errors"]["continuation"] = repr(error)
        est.pose_last = (Qcont if row["continuation_valid"] else incoming).detach().clone()
        row["proposal_input_valid"] = bool(quality[t] and valid_pose(proposals[t]))
        if row["proposal_input_valid"]:
            with _isolate_proposal(torch, refiner):
                try:
                    depth = torch.as_tensor(zero, device="cuda", dtype=torch.float)
                    depth = api.erode_depth(depth, radius=2, device="cuda")
                    depth = api.bilateral_filter_depth(depth, radius=2, device="cuda")
                    xyz = api.depth2xyzmap_batch(
                        depth[None],
                        torch.as_tensor(Kn, dtype=torch.float, device="cuda")[None],
                        zfar=np.inf,
                    )[0]
                    Qprop = (
                        refiner.predict(
                            rgb=rgb,
                            depth=depth,
                            K=Kn,
                            ob_in_cams=centered_pose(proposals[t], est.model_center)[None],
                            xyz_map=xyz,
                            normal_map=None,
                            get_vis=False,
                            mesh=est.mesh,
                            mesh_tensors=est.mesh_tensors,
                            glctx=ctx,
                            mesh_diameter=est.diameter,
                            iteration=options["proposal_refine_iterations"],
                        )[0]
                        .detach()
                        .clone()
                    )
                    Pprop = (
                        (Qprop @ est.get_tf_to_centered_mesh())
                        .detach()
                        .cpu()
                        .numpy()
                        .reshape(4, 4)
                        .copy()
                    )
                    row["proposal_refined_valid"] = valid_pose(Pprop)
                    if row["proposal_refined_valid"]:
                        refined[t] = Pprop
                except Exception as error:
                    if _fatal_cuda(error):
                        raise
                    row["errors"]["proposal_refinement"] = repr(error)
                if row["continuation_valid"] and row["proposal_refined_valid"]:
                    try:
                        pair = (
                            torch.cat([Qcont.reshape(1, 4, 4), Qprop.reshape(1, 4, 4)], dim=0)
                            .detach()
                            .clone()
                        )
                        row["score_pair_called"] = True
                        score_values = (
                            scorer.predict(
                                rgb=rgb,
                                depth=zero,
                                K=Kn,
                                ob_in_cams=pair,
                                normal_map=None,
                                get_vis=False,
                                mesh=est.mesh,
                                mesh_tensors=est.mesh_tensors,
                                glctx=ctx,
                                mesh_diameter=est.diameter,
                            )[0]
                            .detach()
                            .cpu()
                            .numpy()
                            .reshape(-1)
                            .copy()
                        )
                        if score_values.shape == (2,) and np.isfinite(score_values).all():
                            row["finite_score_pair"] = True
                            row["scores"] = score_values.tolist()
                            scores[t] = score_values
                        else:
                            score_values = None
                            row["errors"]["scorer"] = "Nonfinite or wrong-shape returned scores"
                    except Exception as error:
                        if _fatal_cuda(error):
                            raise
                        score_values = None
                        row["errors"]["scorer"] = repr(error)
        choice, reason = select_hypothesis(
            row["continuation_valid"],
            row["proposal_refined_valid"],
            score_values,
            margin=options["score_margin"],
        )
        if choice == "continuation":
            poses[t], valid[t], est.pose_last = Pcont, True, Qcont.detach().clone()
        elif choice == "proposal":
            poses[t], valid[t], est.pose_last = Pprop, True, Qprop.detach().clone()
        else:
            est.pose_last = incoming.detach().clone()
        live[t] = est.pose_last.detach().cpu().numpy().reshape(4, 4)
        if valid[t]:
            selected[t] = live[t]
        row.update(selected=choice, selection_reason=reason)
        records.append(row)
    if next(frames, None) is not None:
        raise ValueError("Source video has more frames than the supplied proposals")
    return HybridResult(poses, valid, records, continuation, refined, scores, selected, live)
