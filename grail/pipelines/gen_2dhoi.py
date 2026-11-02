#!/usr/bin/env python3
"""2D Human-Object Interaction Generation Pipeline.

Steps:
    1. Object initial state simulation (Blender physics)
    2. Object scale determination (iterative render + ChatGPT evaluation)
    3. Blender scene rendering
    4. Video generation (Kling AI or MiniMax-H3)
"""

import argparse
import hashlib
import json
import math
import os
import random
import re
import shutil
import sys
import tempfile
import time
from glob import glob

from grail.adapters.kling import generate_video as generate_video_kling
from grail.adapters.openai_api import chat_with_image
from grail.core.config import load_gen_config, load_object_config, load_pipeline_config
from grail.core.dataset import category2object
from grail.core.io import run_subprocess
from grail.core.video import concatenate_videos, extract_last_frame, is_valid_video

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_blender_cmd(script_path, script_args):
    """Build a ``blender --background --python <script> -- <args>`` command."""
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    blender_bin = os.path.join(project_root, "imports", "blender", "blender")
    if not os.path.isfile(blender_bin):
        raise FileNotFoundError(
            f"Blender not found at {blender_bin}. Run: bash scripts/setup/install_env_docker.sh"
        )
    # Blender can carry an editable installation pointing at another checkout.
    # Bind imports to the same checkout as this entrypoint before loading a script.
    return [
        blender_bin,
        "--background",
        "--python-expr",
        f"import sys; sys.path.insert(0, {project_root!r})",
        "--python",
        script_path,
        "--",
    ] + script_args


def _get_object_extents(object_path):
    """Return axis-aligned bounding box extents (dx, dy, dz) in metres."""
    import trimesh

    loaded = trimesh.load(object_path, force="mesh")
    if isinstance(loaded, trimesh.Scene):
        mn, mx = loaded.bounds
        return tuple(mx[i] - mn[i] for i in range(3))
    return tuple(loaded.extents)


def _compute_bbox_volume(extents, scale, min_dim_cap=0.025):
    """Compute bbox volume, capping thin dimensions at *min_dim_cap*."""
    capped = [max(extents[i] * scale, min_dim_cap) for i in range(3)]
    return capped[0] * capped[1] * capped[2]


# ---------------------------------------------------------------------------
# Prompt / ChatGPT helpers
# ---------------------------------------------------------------------------

_DEFAULT_PROMPT_REFINEMENT = """Create a simple video prompt for a person interacting with a {category}.

Base prompt: "{base_prompt}"

Requirements:
1. The person interacts with the object as they would in daily life.
2. Static camera - no camera movement
3. If picking up the object, continue holding/moving it - don't put it back down

IMPORTANT: Respond with ONLY the refined prompt text. Do not include any preambles, explanations, or conversational phrases like "Sure, here's..." or "Here is...". Just output the prompt directly."""

_REFUSAL_PATTERNS = (
    "i'm sorry",
    "i can't assist",
    "i cannot assist",
    "i'm unable to",
    "i cannot help",
    "i can't help",
    "as an ai",
    "i'm not able to",
    "against my guidelines",
    "content policy",
)

_VIDEO_MODEL_APIS = ("kling-ai", "minimax-h3")
_MINIMAX_H3_MODES = ("dense", "quality", "fast")
_DEFAULT_MINIMAX_H3_MODE = "quality"


def generate_video_minimax_h3(*args, **kwargs):
    """Import the optional MiniMax adapter only on its provider path."""
    from grail.adapters.minimax_h3 import generate_video

    return generate_video(*args, **kwargs)


def select_base_prompt(base_prompt):
    """If *base_prompt* is a list, randomly pick one; otherwise return as-is."""
    if isinstance(base_prompt, list):
        selected = random.choice(base_prompt)
        print(f"  Selected prompt variant: {selected}")
        return selected
    return base_prompt


def _stable_video_seed(base_seed, *parts):
    """Derive a deterministic provider seed from the output identity."""
    digest = hashlib.sha256("/".join(str(part) for part in parts).encode()).digest()
    return (int(base_seed) + int.from_bytes(digest[:4], "big")) % (2**31 - 1)


def _configured_scene_seeds(num_rand_scenes):
    """Expand the configured render-seed count or inclusive range."""
    values = [num_rand_scenes] if isinstance(num_rand_scenes, int) else list(num_rand_scenes)
    if len(values) == 1:
        seeds = list(range(1, values[0] + 1))
    elif len(values) == 2:
        seeds = list(range(values[0], values[1] + 1))
    else:
        raise ValueError(f"Invalid num_rand_scenes: {num_rand_scenes}")
    if not seeds or any(seed < 1 for seed in seeds):
        raise ValueError(f"num_rand_scenes must select positive seeds: {num_rand_scenes}")
    return seeds


def _worker_scene_seeds(num_rand_scenes, args):
    """Return exactly the render seeds assigned to the current worker."""
    seeds = _configured_scene_seeds(num_rand_scenes)
    if not args.split_by_category:
        if args.num_job_chunks < 1:
            raise ValueError("num_job_chunks must be positive")
        if not 0 <= args.job_chunk_idx < args.num_job_chunks:
            raise ValueError(
                f"job_chunk_idx must be in [0, {args.num_job_chunks}), got {args.job_chunk_idx}"
            )
        if args.shuffle:
            random.Random(args.seed).shuffle(seeds)
        seeds = sorted(seeds[args.job_chunk_idx :: args.num_job_chunks])
    return seeds


def _object_config_for_category(object_config, category):
    """Resolve the same exact-or-prefix object entry used by the renderer."""
    if category in object_config:
        return object_config[category]
    category_prefix = "_".join(category.split("_")[:-1])
    if category_prefix and category_prefix in object_config:
        return object_config[category_prefix]
    return None


def _resolve_render_scene(
    category,
    object_config,
    *,
    scene=None,
    use_table_scene=False,
):
    """Resolve a render scene without falling back to an unsafe wildcard."""
    if scene:
        return scene
    obj_config = _object_config_for_category(object_config, category)
    if obj_config and obj_config.get("scene"):
        return obj_config["scene"]
    default_key = "table-default" if use_table_scene else "floor-default"
    default_config = object_config.get(default_key, {})
    if default_config.get("scene"):
        return default_config["scene"]
    raise ValueError(
        f"No scene configured for {category}; add an object-config scene or provide --scene"
    )


def _create_minimax_h3_service_session(args):
    """Create the portable service manager on the first actual H3 request."""
    from grail.adapters.minimax_h3_process import MiniMaxH3ServiceProcess

    return MiniMaxH3ServiceProcess(
        request_mode=args.minimax_h3_mode,
        launcher_mode=args.minimax_h3_launcher_mode,
        service_file=args.minimax_h3_service_file,
        startup_timeout=args.minimax_h3_startup_timeout,
    )


def _ensure_minimax_h3_service(args):
    session = getattr(args, "_minimax_h3_service_session", None)
    if session is None:
        session = _create_minimax_h3_service_session(args)
        args._minimax_h3_service_session = session
    service_file = session.ensure_running()
    args.minimax_h3_service_file = service_file
    return service_file


def _close_minimax_h3_service(args):
    session = getattr(args, "_minimax_h3_service_session", None)
    if session is not None:
        session.close()


def _resolved_minimax_h3_launcher_mode(args):
    launcher_mode = getattr(args, "minimax_h3_launcher_mode", None)
    if launcher_mode is not None:
        return launcher_mode
    if not getattr(args, "minimax_h3_autostart", False):
        return None
    from grail.adapters.minimax_h3_process import resolve_minimax_h3_launcher_mode

    return resolve_minimax_h3_launcher_mode(args.minimax_h3_mode)


def _check_existing_minimax_h3_provenance(
    video_path, condition_image, dataset, category, name, args
):
    from grail.adapters.minimax_h3_provenance import check_video_provenance

    return check_video_provenance(
        video_path,
        condition_image=condition_image,
        mode=args.minimax_h3_mode,
        duration=args.duration,
        seed=_stable_video_seed(args.minimax_h3_seed, dataset, category, name),
        launcher_mode=_resolved_minimax_h3_launcher_mode(args),
    )


def refine_prompt_with_chatgpt(image_path, category, base_prompt, system_message_template=None):
    """Refine a video generation prompt using vision chat completions."""
    templates = system_message_template or _DEFAULT_PROMPT_REFINEMENT
    template = random.choice(templates) if isinstance(templates, list) else templates
    system_msg = template.strip().format(category=category, base_prompt=base_prompt)

    try:
        refined = chat_with_image(
            prompt_text=(
                f"Refine this prompt for a video of human interaction with this {category}. "
                "Output only the refined prompt, nothing else."
            ),
            image_path=image_path,
            max_tokens=500,
            temperature=0.7,
            system_prompt=system_msg,
        )
        if any(p in refined.lower() for p in _REFUSAL_PATTERNS):
            print(f"  Model refused for '{category}', using base prompt")
            return base_prompt
        print(f"  Refined prompt: {refined}")
        return refined
    except Exception as e:
        print(f"  Prompt refinement error: {e}, using base prompt")
        return base_prompt


_DEFAULT_SCALE_EVALUATION = """You are evaluating whether a {category} in a rendered indoor scene has the correct real-world size.

Look at the image carefully. There is a person and a {category} in the scene. Judge whether the {category} appears to be the correct real-world size relative to the person and the room.

You MUST respond with EXACTLY one word — one of the following three options:
- "small" if the object looks too small compared to its real-world size
- "big" if the object looks too big compared to its real-world size
- "correct" if the object appears to be approximately the correct real-world size

Do NOT include any other text, explanation, or punctuation. Just one word."""


def ask_chatgpt_about_scale(image_path, category, system_message_template=None):
    """Ask the vision model whether the object scale looks correct.

    Returns 'small', 'big', or 'correct'.
    """
    category = re.sub(r"\d+", "", category.replace("_", " ")).strip()
    template = (system_message_template or _DEFAULT_SCALE_EVALUATION).strip()
    system_msg = template.format(category=category)

    try:
        answer = chat_with_image(
            prompt_text=(
                f"Does this {category} in the middle of the table look like the correct size? "
                "Answer with exactly one word: small, big, or correct."
            ),
            image_path=image_path,
            max_tokens=10,
            temperature=0.0,
            system_prompt=system_msg,
        )
        answer = answer.strip().lower()
        if answer not in ("small", "big", "correct"):
            print(f"  Unexpected scale response: '{answer}', defaulting to 'correct'")
            answer = "correct"
        return answer
    except Exception as e:
        print(f"  Scale evaluation error: {e}, defaulting to 'correct'")
        return "correct"


# ---------------------------------------------------------------------------
# Pipeline steps
# ---------------------------------------------------------------------------


def step1_simulate_initial_state(dataset, category, args):
    """Step 1: Run physics simulation to get stable object orientation."""
    script_path = os.path.join(os.path.dirname(__file__), "blender", "sim_object.py")
    script_args = [
        "--dataset",
        dataset,
        "--category",
        category,
        "--output_dir",
        args.initial_state_dir,
        "--drop_height",
        str(args.drop_height),
        "--settling_time",
        str(args.settling_time),
        "--initial_rotation_perturbation",
        str(args.initial_rotation_perturbation),
        "--seed",
        str(args.seed),
    ]
    if args.skip_done:
        script_args.append("--skip_done")
    if args.save_usd:
        script_args.append("--save_usd")
    if args.verbose:
        script_args.append("--verbose")

    cmd = [sys.executable, script_path] + script_args
    return run_subprocess(
        cmd,
        f"Object simulation for {dataset}/{category}",
        cwd=os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
    )


def _character_render_args(args):
    """Keep the exact character asset and its units identical in both render stages."""
    result = []
    if args.character:
        result.extend(["--character_name", args.character])
    if getattr(args, "character_usd", None):
        result.extend(["--character_usd", args.character_usd])
    if getattr(args, "character_scale", None) is not None:
        result.extend(["--character_scale", *map(str, args.character_scale)])
    return result


def step2_determine_obj_scale(dataset, category, args):
    """Step 2: Determine object scale via exponential + binary search with ChatGPT evaluation."""
    output_dir = os.path.join(args.obj_scale_dir, dataset, category)
    output_file = os.path.join(output_dir, "obj_scale.json")
    if args.skip_done and os.path.exists(output_file):
        with open(output_file) as f:
            saved = json.load(f)
        print(f"  Scale already determined: {saved['obj_scale']}")
        return True

    print(f"  Determining object scale for {dataset}/{category}")
    script_path = os.path.join(os.path.dirname(__file__), "blender", "render_scene.py")

    scale = 0.1
    low = high = None
    max_iter = args.max_scale_iterations
    search_history = []

    project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    os.makedirs(args.results_dir, exist_ok=True)
    tmp_dir = tempfile.mkdtemp(prefix="scale_search_", dir=args.results_dir)
    tmp_rel = os.path.relpath(tmp_dir, project_root)

    try:
        for iteration in range(1, max_iter + 1):
            current_scale = [scale, scale, scale]
            print(f"  Scale iter {iteration}/{max_iter}: scale={scale:.4f}")

            scale_args = [
                "--dataset",
                dataset,
                "--category",
                category,
                "--scene",
                "indoor2-pickup",
                "--results_dir",
                ".",
                "--image_save_dir",
                os.path.join(tmp_rel, "renders"),
                "--camera_dir",
                os.path.join(tmp_rel, "cameras"),
                "--foundation_pose_input_dir",
                os.path.join(tmp_rel, "foundation_pose"),
                "--depth_save_dir",
                os.path.join(tmp_rel, "depth"),
                "--samples",
                "1",
                "--width",
                str(args.width),
                "--height",
                str(args.height),
                "--initial_state_dir",
                args.initial_state_dir,
                "--obj_scale_override",
                str(current_scale[0]),
                str(current_scale[1]),
                str(current_scale[2]),
                "--rand_scene_seed",
                "1",
                "--render_only",
                "--pipeline_config",
                args._config_path,
                "--object_config",
                args.object_config,
                "--character_dir",
                args.character_dir,
                "--texture_dir",
                args.texture_dir,
            ]
            scale_args.extend(_character_render_args(args))
            if args.use_initial_state:
                scale_args.append("--use_initial_state")
            if args.gpu:
                scale_args.append("--gpu")
            if args.verbose:
                scale_args.append("--verbose")

            cmd = _build_blender_cmd(script_path, scale_args)
            if not run_subprocess(
                cmd,
                f"Scale render (iter {iteration}, scale={scale:.4f})",
                cwd=project_root,
            ):
                return False

            pattern = os.path.join(tmp_dir, "renders", dataset, category, "*_rand00001.png")
            images = sorted(glob(pattern))
            if not images:
                print(f"  No render found: {pattern}")
                return False
            image_path = images[-1]

            answer = ask_chatgpt_about_scale(
                image_path,
                category,
                system_message_template=getattr(args, "scale_evaluation_system_message", None),
            )
            search_history.append({"iteration": iteration, "scale": scale, "response": answer})
            print(f"  ChatGPT: {answer} (low={low}, high={high})")

            if answer == "correct":
                break
            elif answer == "small":
                low = scale
                scale = scale * 2 if high is None else (low + high) / 2
            elif answer == "big":
                high = scale
                scale = scale * 0.5 if low is None else (low + high) / 2
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    final_scale = [scale, scale, scale]
    result = {
        "obj_scale": final_scale,
        "num_iterations": len(search_history),
        "search_history": search_history,
    }

    # Minimum volume check
    min_obj_volume = getattr(args, "min_obj_volume", None)
    if min_obj_volume and min_obj_volume > 0:
        try:
            extents = _get_object_extents(category2object(f"data/{dataset}", category))
            volume = _compute_bbox_volume(extents, scale)
            if volume < min_obj_volume:
                old = scale
                scale *= (min_obj_volume / volume) ** (1.0 / 3.0)
                final_scale = [scale, scale, scale]
                result.update(
                    obj_scale=final_scale,
                    scale_adjusted_for_min_volume=True,
                    original_scale_before_min_volume=[old] * 3,
                    mesh_extents_m=list(extents),
                    volume_before_adjust_m3=_compute_bbox_volume(extents, old),
                    min_obj_volume_m3=min_obj_volume,
                    min_dim_cap_m=0.025,
                )
                print(
                    f"  Volume {volume:.6f}m³ < {min_obj_volume:.6f}m³ → scale {old:.4f} → {scale:.4f}"
                )
        except Exception as e:
            print(f"  Warning: min_obj_volume check failed: {e}")

    os.makedirs(output_dir, exist_ok=True)
    with open(output_file, "w") as f:
        json.dump(result, f, indent=4)
    print(f"  Scale result: {final_scale} → {output_file}")
    return True


def step3_render_blender_scene(dataset, category, args):
    """Step 3: Render Blender scenes with standalone Blender."""
    script_path = os.path.join(os.path.dirname(__file__), "blender", "render_scene.py")
    scene_num_all = _worker_scene_seeds(args.num_rand_scenes, args)

    project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))

    for scene_num in scene_num_all:
        # Pre-check: skip if output exists
        if args.skip_done:
            scene_key = _resolve_render_scene(
                category,
                args._object_config_dict,
                scene=getattr(args, "scene", None),
                use_table_scene=getattr(args, "use_table_scene", False),
            )
            char_name = args.character or "wo_character"
            phases = ("start", "end") if args.render_start_end else ("rand",)
            scene_ids = [
                f"{dataset}/{category}/{char_name}_{scene_key}_{phase}{scene_num:05d}"
                for phase in phases
            ]
            expected_outputs = [
                os.path.join(args.render_dir, f"{scene_id}.png") for scene_id in scene_ids
            ]
            expected_cameras = [os.path.join(args.camera_dir, f"{scene_ids[0]}.pickle")]
            if all(os.path.exists(path) for path in expected_outputs + expected_cameras):
                print(f"  Already rendered: {', '.join(expected_outputs)}")
                continue

        script_args = [
            "--dataset",
            dataset,
            "--category",
            category,
            "--results_dir",
            ".",
            "--image_save_dir",
            args.render_dir,
            "--camera_dir",
            args.camera_dir,
            "--foundation_pose_input_dir",
            args.foundation_pose_input_dir,
            "--depth_save_dir",
            args.depth_save_dir,
            "--samples",
            str(args.samples),
            "--width",
            str(args.width),
            "--height",
            str(args.height),
            "--initial_state_dir",
            args.initial_state_dir,
            "--obj_scale_dir",
            args.obj_scale_dir,
            "--pipeline_config",
            args._config_path,
            "--object_config",
            args.object_config,
            "--base_seed",
            str(args.seed),
            "--character_dir",
            args.character_dir,
            "--texture_dir",
            args.texture_dir,
        ]

        if not args.skip_step2:
            script_args.extend(["--obj_scale_dir", args.obj_scale_dir])
            script_args.append("--use_obj_scale")
        script_args.extend(_character_render_args(args))
        if args.character_init_pose_file:
            script_args.extend(["--character_init_pose_file"] + args.character_init_pose_file)
        if args.scene:
            script_args.extend(["--scene", args.scene])
        if args.skip_done:
            script_args.append("--skip_done")
        if args.use_initial_state:
            script_args.append("--use_initial_state")
        if args.gpu:
            script_args.append("--gpu")
        if args.no_rand_seed:
            script_args.append("--no_rand_seed")
        if args.verbose:
            script_args.append("--verbose")

        script_args.extend(["--rand_scene_seed", str(scene_num)])
        cmd = _build_blender_cmd(script_path, script_args)

        if args.render_start_end:
            for phase in ("start", "end"):
                if not run_subprocess(
                    cmd + ["--phase", phase],
                    f"Render {phase} for {dataset}/{category} (seed={scene_num})",
                    cwd=project_root,
                ):
                    return False
        else:
            if not run_subprocess(
                cmd,
                f"Render {dataset}/{category} (seed={scene_num})",
                cwd=project_root,
            ):
                return False
    return True


def _render_seed(path):
    """Return the seed encoded by a canonical render filename."""
    match = re.search(
        r"_(?:rand|start|end)(\d+)\.png\Z",
        os.path.basename(path),
        flags=re.IGNORECASE,
    )
    return int(match.group(1)) if match else None


def _find_rendered_images(
    dataset,
    category,
    render_dir,
    character=None,
    object_config=None,
    scene=None,
    use_table_scene=False,
    render_start_end=None,
):
    """Find exact rendered images, returning either a list or (start, end) tuple.

    Render scene names may share prefixes (for example ``indoor1-syn-stairs``
    and ``indoor1-syn-stairs-down-top``), so a broad ``scene*.png`` glob is
    unsafe.  Only filenames emitted by ``render_scene.py`` are accepted.
    """
    if object_config is None:
        object_config = load_object_config()
    resolved_scene = _resolve_render_scene(
        category,
        object_config,
        scene=scene,
        use_table_scene=use_table_scene,
    )

    render_character = character or "wo_character"
    prefix = f"{render_character}_{resolved_scene}"
    render_category_dir = os.path.join(render_dir, dataset, category)
    filename_re = re.compile(
        rf"{re.escape(prefix)}_(?P<phase>rand|start|end)(?P<seed>\d+)\.png\Z",
        flags=re.IGNORECASE,
    )
    by_phase = {"rand": {}, "start": {}, "end": {}}

    for path in sorted(glob(os.path.join(render_category_dir, "*.png"))):
        match = filename_re.fullmatch(os.path.basename(path))
        if match is None:
            continue
        phase = match.group("phase").lower()
        seed = int(match.group("seed"))
        if seed in by_phase[phase]:
            raise ValueError(
                f"Multiple {phase} renders for {category} seed {seed}: "
                f"{by_phase[phase][seed]}, {path}"
            )
        by_phase[phase][seed] = path

    random_images = by_phase["rand"]
    starts = by_phase["start"]
    ends = by_phase["end"]

    if render_start_end is False:
        if not random_images:
            raise FileNotFoundError(
                f"No exact rand renders for {dataset}/{category}; expected "
                f"{prefix}_rand<seed>.png in {render_category_dir}"
            )
        images = [random_images[seed] for seed in sorted(random_images)]
        print(f"  Found {len(images)} rendered images")
        return images

    if render_start_end is True:
        if not starts and not ends:
            raise FileNotFoundError(
                f"No exact start/end renders for {dataset}/{category}; expected "
                f"{prefix}_(start|end)<seed>.png in {render_category_dir}"
            )
        print(f"  Found {len(starts)} start + {len(ends)} end images")
        return (
            [starts[seed] for seed in sorted(starts)],
            [ends[seed] for seed in sorted(ends)],
        )

    if starts or ends:
        if random_images:
            raise ValueError(
                f"Mixed rand and start/end renders for {dataset}/{category} with scene "
                f"{resolved_scene}"
            )
        if starts.keys() != ends.keys():
            missing_starts = sorted(ends.keys() - starts.keys())
            missing_ends = sorted(starts.keys() - ends.keys())
            raise ValueError(
                f"Unpaired start/end renders for {dataset}/{category}: "
                f"missing starts={missing_starts}, missing ends={missing_ends}"
            )
        ordered_seeds = sorted(starts)
        print(f"  Found {len(starts)} start + {len(ends)} end images")
        return ([starts[seed] for seed in ordered_seeds], [ends[seed] for seed in ordered_seeds])

    if not random_images:
        raise FileNotFoundError(
            f"No exact rendered images for {dataset}/{category}; expected "
            f"{prefix}_(rand|start|end)<seed>.png in {render_category_dir}"
        )

    images = [random_images[seed] for seed in sorted(random_images)]
    print(f"  Found {len(images)} rendered images")
    return images


def step4_generate_videos(dataset, category, args):
    """Step 4: Generate videos from rendered images using the selected provider."""
    print(f"  Video generation for {dataset}/{category}")

    try:
        requested_seeds = _worker_scene_seeds(args.num_rand_scenes, args)
        if not requested_seeds:
            print("  No render seeds assigned to this worker")
            return True

        image_paths = _find_rendered_images(
            dataset,
            category,
            args.render_dir,
            args.character,
            args._object_config_dict,
            scene=getattr(args, "scene", None),
            use_table_scene=getattr(args, "use_table_scene", False),
            render_start_end=getattr(args, "render_start_end", False),
        )
        image_tail_paths = None
        if isinstance(image_paths, tuple):
            image_paths, image_tail_paths = image_paths

        valid = set(requested_seeds)

        def _select_requested(paths, label):
            selected = {}
            for path in paths:
                seed = _render_seed(path)
                if seed not in valid:
                    continue
                if seed in selected:
                    raise ValueError(
                        f"Multiple {label} renders for {dataset}/{category} seed {seed}: "
                        f"{selected[seed]}, {path}"
                    )
                selected[seed] = path
            missing = sorted(valid - selected.keys())
            if missing:
                raise FileNotFoundError(
                    f"Missing {label} renders for {dataset}/{category} seeds: {missing}"
                )
            return [selected[seed] for seed in sorted(selected)]

        image_paths = _select_requested(image_paths, "input")
        if image_tail_paths is not None:
            image_tail_paths = _select_requested(image_tail_paths, "tail")
        tail_by_seed = {_render_seed(path): path for path in (image_tail_paths or [])}

        # The render-seed helper already applied the same worker sharding as Step 3.
        image_paths = sorted(image_paths)

        # Parse multi-segment prompts
        prompts = None
        if args.num_video_segments > 1:
            selected = select_base_prompt(args.base_prompt)
            prompts = [p.strip() for p in selected.split("#")]
            if len(prompts) != args.num_video_segments:
                raise ValueError(
                    f"Prompt count ({len(prompts)}) != num_video_segments ({args.num_video_segments})"
                )

        # Helpers (closures over args)
        def _generate_segment(input_image, prompt, out_dir, name, tail=None):
            for attempt in range(1, args.video_max_retries + 1):
                if args.video_model_api == "kling-ai":
                    path = generate_video_kling(
                        input_image,
                        prompt,
                        out_dir,
                        name,
                        duration=args.duration,
                        mode="pro" if tail else args.kling_mode,
                        model_name=args.kling_model_name,
                        image_tail_path=tail,
                    )
                elif args.video_model_api == "minimax-h3":
                    from grail.adapters.minimax_h3 import MiniMaxH3Error

                    if tail is not None:
                        raise ValueError(
                            "MiniMax-H3 Sol-Engine does not support an explicit tail image"
                        )
                    try:
                        minimax_duration = float(args.duration)
                    except (TypeError, ValueError) as error:
                        raise ValueError("MiniMax-H3 duration must be 5 or 10 seconds") from error
                    if minimax_duration not in {5.0, 10.0}:
                        raise ValueError("MiniMax-H3 duration must be 5 or 10 seconds")
                    service_file = args.minimax_h3_service_file
                    if getattr(args, "minimax_h3_autostart", False):
                        service_file = _ensure_minimax_h3_service(args)
                    try:
                        path = generate_video_minimax_h3(
                            input_image,
                            prompt,
                            out_dir,
                            name,
                            duration=args.duration,
                            mode=args.minimax_h3_mode,
                            seed=_stable_video_seed(args.minimax_h3_seed, dataset, category, name),
                            image_tail_path=tail,
                            service_file=service_file,
                            timeout=args.minimax_h3_timeout,
                            launcher_mode=_resolved_minimax_h3_launcher_mode(args),
                        )
                    except MiniMaxH3Error as error:
                        if not error.retryable:
                            print(f"  Permanent MiniMax-H3 failure; not retrying: {error}")
                            return None
                        print(f"  Transient MiniMax-H3 failure: {error}")
                        path = None
                else:
                    raise ValueError(f"Unsupported video_model_api: {args.video_model_api}")
                if path is not None:
                    return path
                if attempt < args.video_max_retries:
                    print(
                        f"  Retry {attempt}/{args.video_max_retries} in {args.video_retry_wait}s..."
                    )
                    time.sleep(args.video_retry_wait)
            print(f"  Failed after {args.video_max_retries} attempts")
            return None

        def _get_prompt(input_image, base_prompt):
            if args.skip_prompt_refinement:
                return base_prompt
            if not os.getenv("OPENAI_API_KEY"):
                print("  Warning: OPENAI_API_KEY not set, use --skip_prompt_refinement")
                return None
            return refine_prompt_with_chatgpt(
                input_image,
                category,
                base_prompt,
                system_message_template=getattr(args, "prompt_refinement_system_message", None),
            )

        # Generate videos
        for image_path in image_paths:
            print(f"  Processing: {os.path.basename(image_path)}")

            for i in range(args.num_videos):
                image_basename = os.path.splitext(os.path.basename(image_path))[0]
                base_name = f"{image_basename}_{i:03d}" if args.num_videos > 1 else image_basename
                output_dir = os.path.join(args.video_output_dir, dataset, category)

                # Determine tail images
                if image_tail_paths is None:
                    tails = [None]
                elif args.use_all_tail_images:
                    tails = image_tail_paths
                else:
                    tails = [tail_by_seed.get(_render_seed(image_path))]

                for tail_image in tails:
                    cur_name = base_name
                    if tail_image is not None:
                        suffix = os.path.splitext(os.path.basename(tail_image))[0].split("_")[-1]
                        cur_name = f"{base_name}-{suffix}"

                    final_path = os.path.join(output_dir, f"{cur_name}.mp4")
                    if args.skip_done and os.path.exists(final_path):
                        if is_valid_video(final_path):
                            if args.video_model_api == "minimax-h3":
                                status, detail = _check_existing_minimax_h3_provenance(
                                    final_path,
                                    image_path,
                                    dataset,
                                    category,
                                    cur_name,
                                    args,
                                )
                                if status == "match":
                                    print(f"  Exists with matching provenance: {final_path}")
                                    continue
                                if status == "missing":
                                    print(
                                        "  Exists without provenance (legacy output; keeping): "
                                        f"{final_path}"
                                    )
                                    continue
                                print(
                                    "  Existing MiniMax-H3 provenance is "
                                    f"{status}; regenerating: {final_path} ({detail})"
                                )
                            else:
                                print(f"  Exists: {final_path}")
                                continue
                        else:
                            print(f"  Invalid existing video, regenerating: {final_path}")

                    if prompts is not None:
                        # Autoregressive multi-segment
                        seg_videos, temp_frames = [], []
                        cur_input = image_path
                        all_prompts = []
                        tmp_dir = os.path.join(output_dir, "tmp")
                        os.makedirs(tmp_dir, exist_ok=True)

                        for si, seg_prompt in enumerate(prompts):
                            is_last = si == len(prompts) - 1
                            seg_name = f"{cur_name}_seg{si:02d}"
                            seg_path = os.path.join(tmp_dir, f"{seg_name}.mp4")

                            # Resume from existing segment
                            if args.skip_done and os.path.exists(seg_path):
                                if not is_valid_video(seg_path):
                                    os.remove(seg_path)
                                else:
                                    seg_videos.append(seg_path)
                                    all_prompts.append(seg_prompt)
                                    if not is_last:
                                        try:
                                            tf = extract_last_frame(seg_path)
                                            temp_frames.append(tf)
                                            cur_input = tf
                                            continue
                                        except Exception:
                                            seg_videos.pop()
                                            all_prompts.pop()
                                            os.remove(seg_path)
                                    else:
                                        continue

                            final_prompt = _get_prompt(cur_input, seg_prompt)
                            if final_prompt is None:
                                return False
                            all_prompts.append(final_prompt)

                            target_si = (
                                args.seg_idx_for_tail_image
                                if args.seg_idx_for_tail_image >= 0
                                else len(prompts) - 1
                            )
                            seg_tail = tail_image if si == target_si else None

                            vid = _generate_segment(
                                cur_input, final_prompt, tmp_dir, seg_name, seg_tail
                            )
                            if vid is None:
                                for tf in temp_frames:
                                    if os.path.exists(tf):
                                        os.remove(tf)
                                return False

                            seg_videos.append(vid)
                            if not is_last:
                                try:
                                    tf = extract_last_frame(vid)
                                    temp_frames.append(tf)
                                    cur_input = tf
                                except Exception as e:
                                    print(f"  Last frame extraction failed: {e}")
                                    for tf in temp_frames:
                                        if os.path.exists(tf):
                                            os.remove(tf)
                                    return False
                        else:
                            # All segments complete
                            concatenate_videos(seg_videos, final_path)
                            prompt_dir = os.path.join(args.prompts_dir, dataset, category)
                            os.makedirs(prompt_dir, exist_ok=True)
                            with open(os.path.join(prompt_dir, f"{cur_name}.txt"), "w") as f:
                                for idx, p in enumerate(all_prompts):
                                    f.write(f"# Segment {idx+1}\n{p}\n\n")
                            for tf in temp_frames:
                                if os.path.exists(tf):
                                    os.remove(tf)
                            print(f"  Final video: {final_path}")
                    else:
                        # Single segment
                        final_prompt = _get_prompt(image_path, select_base_prompt(args.base_prompt))
                        if final_prompt is None:
                            return False

                        prompt_dir = os.path.join(args.prompts_dir, dataset, category)
                        os.makedirs(prompt_dir, exist_ok=True)
                        with open(os.path.join(prompt_dir, f"{cur_name}.txt"), "w") as f:
                            f.write(final_prompt)

                        vid = _generate_segment(
                            image_path, final_prompt, output_dir, cur_name, tail_image
                        )
                        if vid is None:
                            return False
                        print(f"  Generated: {vid}")

        return True

    except Exception as e:
        import traceback

        print(traceback.format_exc())
        print(f"  Video generation error: {e}")
        return False


# ---------------------------------------------------------------------------
# CLI / main
# ---------------------------------------------------------------------------


def main():
    """Entry point for the 2D HOI generation pipeline."""
    # Pre-parse to find config file + optional object_config override
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", type=str, default="configs/gen_2dhoi/manipulation.yaml")
    pre.add_argument("--object_config", type=str, default=None)
    pre_args, _ = pre.parse_known_args()

    yaml_flat = load_gen_config(pre_args.config)
    pipeline_cfg = load_pipeline_config(pre_args.config)

    # CLI --object_config overrides the pipeline yaml's object_config path so a single
    # base manipulation.yaml can drive multiple per-dataset object yamls without cloning.
    if pre_args.object_config:
        from grail.core.config import load_object_config_full

        full = load_object_config_full(pre_args.object_config)
        objects = full.get("objects", full)
        pipeline_cfg["objects"] = objects
        pipeline_cfg["dataset"] = full.get("dataset")
        pipeline_cfg["categories"] = [k for k in objects if k != "default"]
        pipeline_cfg["object_config_path"] = pre_args.object_config

    yaml_flat["dataset"] = pipeline_cfg.get("dataset")
    yaml_flat["object_config"] = pipeline_cfg.get("object_config_path")

    parser = argparse.ArgumentParser(description="2D HOI Generation Pipeline")
    parser.add_argument("--config", type=str, default="configs/gen_2dhoi/manipulation.yaml")
    parser.add_argument(
        "--object_config",
        type=str,
        default=None,
        help="Override the pipeline yaml's object_config path.",
    )
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--category", type=str, default=None)
    parser.add_argument("--scene", type=str, default=None)
    parser.add_argument("--num_job_chunks", type=int, default=1)
    parser.add_argument("--job_chunk_idx", type=int, default=0)
    parser.add_argument("--split_by_category", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--results_dir", type=str, default="results")
    parser.add_argument("--video_output_dir", type=str, default="generation/videos_kling")
    parser.add_argument("--character", type=str, default=None)
    parser.add_argument(
        "--character_usd",
        type=str,
        default=None,
        help="Exact native character USD; --character remains its output identifier.",
    )
    parser.add_argument(
        "--character_scale",
        type=float,
        nargs=3,
        default=None,
        help="External character scale; use 1 1 1 for meter-valued USD assets.",
    )
    parser.add_argument("--character_init_pose_file", type=str, default=None)
    parser.add_argument("--base_prompt", type=str, default=None)
    parser.add_argument("--num_rand_scenes", type=int, nargs="+", default=[1])
    parser.add_argument(
        "--video_model_api", type=str, default="kling-ai", choices=_VIDEO_MODEL_APIS
    )
    parser.add_argument("--kling_model_name", type=str, default="kling-v3")
    parser.add_argument("--kling_mode", type=str, default="std", choices=["std", "pro"])
    parser.add_argument(
        "--minimax_h3_mode",
        type=str,
        default=_DEFAULT_MINIMAX_H3_MODE,
        choices=_MINIMAX_H3_MODES,
    )
    parser.add_argument("--minimax_h3_seed", type=int, default=42)
    parser.add_argument("--minimax_h3_service_file", type=str, default=None)
    parser.add_argument("--minimax_h3_timeout", type=int, default=1800)
    parser.add_argument("--minimax_h3_autostart", action="store_true")
    parser.add_argument(
        "--minimax_h3_launcher_mode",
        type=str,
        default=None,
    )
    parser.add_argument("--minimax_h3_startup_timeout", type=int, default=1800)
    parser.add_argument("--video_max_retries", type=int, default=100)
    parser.add_argument("--video_retry_wait", type=int, default=30)
    parser.add_argument("--skip_step1", action=argparse.BooleanOptionalAction)
    parser.add_argument("--skip_step2", action=argparse.BooleanOptionalAction)
    parser.add_argument("--skip_step3", action=argparse.BooleanOptionalAction)
    parser.add_argument("--skip_step4", action=argparse.BooleanOptionalAction)
    parser.add_argument("--skip_done", action=argparse.BooleanOptionalAction)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--seed", type=int, default=None)

    parser.set_defaults(**{k: v for k, v in yaml_flat.items() if v is not None})
    args = parser.parse_args()

    if args.dataset is None:
        parser.error("--dataset is required")
    if args.character_usd:
        if not args.character or args.character == "G1":
            parser.error("--character_usd requires a non-G1 --character output identifier")
        args.character_usd = os.path.abspath(os.path.expanduser(args.character_usd))
        if not os.path.isfile(args.character_usd):
            parser.error(f"Character USD does not exist: {args.character_usd}")
    if args.character_scale is not None:
        # YAML defaults bypass argparse's type/nargs conversion. Normalize them
        # here as well so malformed scales fail before a render is launched.
        try:
            if not isinstance(args.character_scale, (list, tuple)) or any(
                isinstance(value, bool) for value in args.character_scale
            ):
                raise ValueError
            scale = [float(value) for value in args.character_scale]
            if len(scale) != 3 or any(not math.isfinite(value) or value <= 0 for value in scale):
                raise ValueError
        except (TypeError, ValueError):
            parser.error("--character_scale must contain three finite positive numbers")
        args.character_scale = scale

    args._pipeline_cfg = pipeline_cfg
    args._object_config_dict = pipeline_cfg.get("objects", {})
    args._config_path = pre_args.config

    categories = pipeline_cfg.get("categories", [])
    if args.category:
        categories = [args.category]
    elif not categories:
        parser.error("No categories found")

    for attr in (
        "initial_state_dir",
        "obj_scale_dir",
        "render_dir",
        "camera_dir",
        "foundation_pose_input_dir",
        "depth_save_dir",
        "video_output_dir",
        "prompts_dir",
    ):
        val = getattr(args, attr)
        if not os.path.isabs(val):
            setattr(args, attr, os.path.join(args.results_dir, val))

    categories = sorted(categories)
    if args.shuffle:
        random.seed(args.seed)
        random.shuffle(categories)
    if args.split_by_category:
        categories = sorted(categories[args.job_chunk_idx :: args.num_job_chunks])

    print(
        f"Config: {args._config_path} | Dataset: {args.dataset} | "
        f"Worker {args.job_chunk_idx}/{args.num_job_chunks} | {len(categories)} categories"
    )

    t0 = time.time()
    all_ok = True

    steps = [
        (1, "skip_step1", step1_simulate_initial_state),
        (2, "skip_step2", step2_determine_obj_scale),
        (3, "skip_step3", step3_render_blender_scene),
        (4, "skip_step4", step4_generate_videos),
    ]

    try:
        for category in categories:
            print(f"\n--- {args.dataset}/{category} ---")
            cat_t0 = time.time()

            try:
                for step_num, skip_attr, step_fn in steps:
                    if getattr(args, skip_attr):
                        continue
                    if not step_fn(args.dataset, category, args):
                        print(f"  Step {step_num} failed, skipping category")
                        all_ok = False
                        break
                else:
                    print(f"  Done in {time.time() - cat_t0:.1f}s")
                    continue
            except KeyboardInterrupt:
                print("\nInterrupted")
                sys.exit(1)
            except Exception as e:
                import traceback

                print(traceback.format_exc())
                print(f"  Category failed: {e}")
                all_ok = False
    finally:
        _close_minimax_h3_service(args)

    elapsed = time.time() - t0
    print(
        f"\n{'OK' if all_ok else 'COMPLETED WITH FAILURES'} — {elapsed:.0f}s ({elapsed/60:.1f}min)"
    )
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
