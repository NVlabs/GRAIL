"""Foreground, sharded Workbench previews with exact per-frame image caching.

This renders exported native samples, not Blender curve edits or subframes.
Rendering success is never a motion acceptance decision.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import time
import uuid
from ctypes.util import find_library


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_write(path, data):
    path = Path(path)
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def code_fingerprint():
    source = Path(__file__).resolve().parent
    return hashlib.sha256("".join(
        str(p.relative_to(source)) + sha256(p) for p in sorted(source.rglob("*.py"))
    ).encode()).hexdigest()


def sample_key(context, frame, qpos, view):
    """No tolerances: even a small pose change invalidates the image."""
    import numpy as np
    qpos = np.asarray(qpos, dtype="<f8")
    if qpos.ndim != 1 or not np.isfinite(qpos).all():
        raise ValueError("Expected one finite native pose")
    digest = hashlib.sha256(json.dumps(
        [context, frame, view], sort_keys=True, allow_nan=False).encode())
    digest.update(qpos.tobytes())
    return digest.hexdigest()


def cache_lookup(directory, key):
    directory = Path(directory)
    image, receipt = directory / (key + ".png"), directory / (key + ".json")
    try:
        data = json.loads(receipt.read_text())
        if data["key"] == key and data["sha256"] == sha256(image):
            return image, data
    except (OSError, ValueError, KeyError, TypeError):
        pass
    return None


def cache_store(directory, key, image, evidence):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / (key + ".png")
    temporary = directory / (key + "." + uuid.uuid4().hex + ".tmp")
    shutil.copyfile(image, temporary)
    temporary.replace(target)
    receipt = dict(evidence, key=key, sha256=sha256(target))
    json_write(directory / (key + ".json"), receipt)
    return receipt


def validate_config(config):
    if config.get("schema_version") != 1:
        raise ValueError("Expected schema_version 1")
    if type(config.get("require_hardware", False)) is not bool:
        raise ValueError("require_hardware must be boolean")
    if type(config.get("worker_timeout", 1800)) is not int or config.get("worker_timeout", 1800) <= 0:
        raise ValueError("worker_timeout must be a positive integer")
    for name in ("blend", "blender"):
        if not Path(config[name]).is_file():
            raise ValueError("Missing " + name)
    if config.get("native") and not Path(config["native"]).is_file():
        raise ValueError("Missing native NPZ")
    frames = config.get("frames")
    if (not isinstance(frames, list) or not frames or
            any(type(f) is not int or f < 0 for f in frames) or
            frames != sorted(set(frames))):
        raise ValueError("frames must be sorted, unique, nonnegative native integers")
    resolution = config.get("resolution", [480, 480])
    if (not isinstance(resolution, list) or len(resolution) != 2 or
            any(type(n) is not int or not 16 <= n <= 4096 for n in resolution)):
        raise ValueError("resolution must contain two integers in [16, 4096]")
    seen = set()
    if not config.get("views"):
        raise ValueError("Specify at least one view")
    for view in config["views"]:
        name = view.get("name", "")
        if not re.fullmatch(r"[A-Za-z0-9_-]+", name) or name in seen:
            raise ValueError("View names must be unique filename-safe identifiers")
        seen.add(name)
        if ("camera" in view) == ("follow_wrist" in view):
            raise ValueError("Each view needs a camera or follow_wrist body")
        if "follow_wrist" in view:
            import numpy as np
            direction = np.asarray(view.get("direction"), dtype=float)
            scale = view.get("ortho_scale", 0.55)
            if (direction.shape != (3,) or not np.isfinite(direction).all() or
                    np.linalg.norm(direction) == 0 or not 0 < scale < 100):
                raise ValueError("Invalid mounted camera direction/scale")
    return config


def graphics_environment(environment, directory, *, require_hardware):
    """Use the injected NVIDIA EGL library even if a container lacks its ICD.

    Pyxis may inject driver libraries but leave only Mesa's vendor JSON in the
    image. Write a worker-local GLVND manifest; never alter system drivers. An
    explicit caller selection wins. The actual render still must pass its guard.
    """
    env = environment.copy()
    if require_hardware and not env.get("__EGL_VENDOR_LIBRARY_FILENAMES"):
        library = find_library("EGL_nvidia")
        if library:
            manifest = Path(directory).resolve() / "nvidia_egl.json"
            json_write(manifest, {"file_format_version": "1.0.0",
                                 "ICD": {"library_path": library}})
            env["__EGL_VENDOR_LIBRARY_FILENAMES"] = str(manifest)
    return env


def run_render(config, output, *, cache, workers=4):
    validate_config(config)
    if type(workers) is not int or not 1 <= workers <= 32:
        raise ValueError("workers must be in [1, 32]")
    started = time.perf_counter()
    output, cache = Path(output).resolve(), Path(cache).resolve()
    output.mkdir(parents=True, exist_ok=False)
    cache.mkdir(parents=True, exist_ok=True)
    config = dict(config, blend=str(Path(config["blend"]).resolve()),
                  blender=str(Path(config["blender"]).resolve()))
    if config.get("native"):
        config["native"] = str(Path(config["native"]).resolve())
    bindings = {"blend": sha256(config["blend"]), "blender": sha256(config["blender"])}
    if config.get("native"):
        bindings["native"] = sha256(config["native"])
    source = Path(__file__).resolve().parent
    # Include the complete package implementation, including FK/native helpers.
    bindings["code"] = code_fingerprint()
    json_write(output / "config.json", config)
    count = min(workers, len(config["frames"]))

    def shard(index):
        directory = output / ("shard_%02d" % index)
        directory.mkdir()
        request = dict(config, frames=config["frames"][index::count],
                       output=str(directory), cache=str(cache), bindings=bindings)
        path = directory / "request.json"
        json_write(path, request)
        env = graphics_environment(os.environ, directory,
                                   require_hardware=config.get("require_hardware", False))
        # Profiles and child lifetimes are isolated; no detached jobs or shell.
        for name in ("CONFIG", "SCRIPTS", "EXTENSIONS"):
            profile = directory / ("profile_" + name.lower())
            profile.mkdir()
            env["BLENDER_USER_" + name] = str(profile)
        env.update(OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
        command = [config["blender"], "--background", "--factory-startup",
                   "--disable-autoexec", config["blend"], "--python-exit-code", "1",
                   "--python", str(source / "render_sequence_worker.py"), "--", str(path)]
        tick = time.perf_counter()
        with (directory / "worker.log").open("w") as stream:
            process = subprocess.run(command, env=env, stdout=stream, stderr=subprocess.STDOUT,
                                     timeout=config.get("worker_timeout", 1800), check=False)
        if process.returncode:
            raise RuntimeError("Render worker failed; inspect " + str(directory / "worker.log"))
        result = json.loads((directory / "result.json").read_text())
        result["process_seconds"] = time.perf_counter() - tick
        return result

    with ThreadPoolExecutor(max_workers=count) as pool:
        results = list(pool.map(shard, range(count)))
    images = output / "images"
    images.mkdir()
    rows = []
    for index, result in enumerate(results):
        for row in result["images"]:
            shutil.copyfile(output / ("shard_%02d" % index) / row["file"], images / row["file"])
            rows.append(row)
    rows.sort(key=lambda row: (row["frame"], row["view"]))
    expected = {(f, v["name"]) for f in config["frames"] for v in config["views"]}
    if len(rows) != len(expected) or {(r["frame"], r["view"]) for r in rows} != expected:
        raise RuntimeError("Incomplete frame/view coverage")
    # Prevent a mixed-input batch from being reported as a completed render.
    for key in ("blend", "blender", "native"):
        if key in bindings and sha256(config[key]) != bindings[key]:
            raise RuntimeError("Input changed while rendering: " + key)
    if code_fingerprint() != bindings["code"]:
        raise RuntimeError("Implementation changed while rendering")
    report = dict(status="RENDERED_UNREVIEWED", motion_accepted=False, workers=count,
                  wall_seconds=time.perf_counter() - started, bindings=bindings,
                  rendered=sum(not r["cached"] for r in rows),
                  reused=sum(r["cached"] for r in rows), images=rows, shards=results)
    json_write(output / "result.json", report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--cache", required=True)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args(argv)
    path = Path(args.config).resolve()
    config = json.loads(path.read_text())
    for key in ("blend", "blender", "native"):
        if config.get(key):
            config[key] = str((path.parent / config[key]).resolve())
    result = run_render(config, args.output, cache=args.cache, workers=args.workers)
    print(json.dumps({key: result[key] for key in
                      ("status", "workers", "wall_seconds", "rendered", "reused")}))


if __name__ == "__main__":
    main()
