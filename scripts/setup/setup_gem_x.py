#!/usr/bin/env python3
"""Install pinned GEM-X independently and download verified external assets.

Uses only the standard library before creating the new environment. Never imports
or modifies a prior GEP/GEM installation. Existing files must match their pinned
hashes; mismatches fail without overwriting them.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import urllib.request
import zipfile
from pathlib import Path

SOURCE_COMMIT = "20555befe2fbbe5b2015e1fff7ff214dc18ea9d6"
GEM_WEIGHTS_REVISION = "4fce057963671fd95cd4154297944397e7b5a9a0"
SOMA_ASSETS_REVISION = "d281db2d01553f7230c56e764cec00fe27ef23f7"
SOMA_VERSION = "0.3.3"
TORCH_VERSION = "2.10.0"
REPO_ROOT = Path(__file__).resolve().parents[2]

ASSETS = (
    {
        "relative_path": "weights/body_s240000.ckpt",
        "bytes": 3538753341,
        "sha256": "e244d27f2834eb6382c5f6b9cbf2a38681b685514a1b2e86f8f3d2592d614959",
        "url": "https://huggingface.co/nvidia/GEM-X/resolve/4fce057963671fd95cd4154297944397e7b5a9a0/body_s240000.ckpt",
        "zip_member_suffix": None,
    },
    {
        "relative_path": "weights/body_config.json",
        "bytes": 1086,
        "sha256": "61c29c72b93f88789ecb7beb21c526c78e92e24ffbce91d91e51984272b26368",
        "url": "https://huggingface.co/nvidia/GEM-X/resolve/4fce057963671fd95cd4154297944397e7b5a9a0/body_config.json",
        "zip_member_suffix": None,
    },
    {
        "relative_path": "weights/motion_native77_e774_s292175.ckpt",
        "bytes": 187134331,
        "sha256": "c7790f4587e518bcea40cb8281c9be2233d37f6caa668835462542760dcf82aa",
        "url": "https://huggingface.co/nvidia/GEM-X/resolve/4fce057963671fd95cd4154297944397e7b5a9a0/motion_native77_e774_s292175.ckpt",
        "zip_member_suffix": None,
    },
    {
        "relative_path": "weights/motion_stats_raw.npz",
        "bytes": 21320,
        "sha256": "b7719b634e0fc1f001005605d117d77448c8d4a110923149df58529c68112ed6",
        "url": "https://huggingface.co/nvidia/GEM-X/resolve/4fce057963671fd95cd4154297944397e7b5a9a0/motion_stats_raw.npz",
        "zip_member_suffix": None,
    },
    {
        "relative_path": "weights/motion_native77.yaml",
        "bytes": 823,
        "sha256": "5d5dff8ee849d3bcf53c28ce60340114980fe59062fa427c1f343655849cad8d",
        "url": "https://huggingface.co/nvidia/GEM-X/resolve/4fce057963671fd95cd4154297944397e7b5a9a0/motion_native77.yaml",
        "zip_member_suffix": None,
    },
    {
        "relative_path": "weights/body_video_native77.yaml",
        "bytes": 1286,
        "sha256": "796a306aca61e0c928888a23d74996740795bfa7cd1735bdc184d1e5274a02d7",
        "url": "https://huggingface.co/nvidia/GEM-X/resolve/4fce057963671fd95cd4154297944397e7b5a9a0/body_video_native77.yaml",
        "zip_member_suffix": None,
    },
    {
        "relative_path": "weights/vitpose.pth",
        "bytes": 3388483384,
        "sha256": "8e3eec71dbd6bb745eba8c6debe3e4b97bcfd1d01f300099f2eedf50b9a80c8e",
        "url": "https://huggingface.co/nvidia/GEM-X/resolve/4fce057963671fd95cd4154297944397e7b5a9a0/vitpose.pth",
        "zip_member_suffix": None,
    },
    {
        "relative_path": "soma/SOMA_neutral.npz",
        "bytes": 26608121,
        "sha256": "5829af4a5333abdba5ad449360b868faacee99d395971e62f35c0e7169432b63",
        "url": "https://huggingface.co/nvidia/SOMA-X/resolve/d281db2d01553f7230c56e764cec00fe27ef23f7/SOMA_neutral.npz",
        "zip_member_suffix": None,
    },
    {
        "relative_path": "soma/SOMA_template_rig.usda",
        "bytes": 28306038,
        "sha256": "a952cb7bdf4ccd801e355d53122d395b985b69d0d9181439edad5081bb171856",
        "url": "https://huggingface.co/nvidia/SOMA-X/resolve/d281db2d01553f7230c56e764cec00fe27ef23f7/SOMA_template_rig.usda",
        "zip_member_suffix": None,
    },
    {
        "relative_path": "soma/SOMA_procedural_transforms.json",
        "bytes": 20045,
        "sha256": "9853e19ab51625ffaa7d9e7ccf0e443272c2740837604e6bacd66604e8edf509",
        "url": "https://huggingface.co/nvidia/SOMA-X/resolve/d281db2d01553f7230c56e764cec00fe27ef23f7/SOMA_procedural_transforms.json",
        "zip_member_suffix": None,
    },
    {
        "relative_path": "soma/correctives_model.pt",
        "bytes": 70549157,
        "sha256": "411d061a651051a7a4ec941543495d35eac21e5a7a69c8ce042b89cd12772a43",
        "url": "https://huggingface.co/nvidia/SOMA-X/resolve/d281db2d01553f7230c56e764cec00fe27ef23f7/correctives_model.pt",
        "zip_member_suffix": None,
    },
    {
        "relative_path": "weights/yolox_humanart.onnx",
        "bytes": 396283264,
        "sha256": "8e9ea96a176bd48501eaaa77216e49ee30794d2f8ba80c7b9862beca4ea972da",
        "url": "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/yolox_x_8xb8-300e_humanart-a39d44ed.zip",
        "zip_member_suffix": "/end2end.onnx",
    },
)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_asset(path, asset):
    path = Path(path)
    if not path.is_file() or path.stat().st_size != asset["bytes"]:
        raise ValueError(f"Missing or wrong-sized asset: {path}")
    if sha256(path) != asset["sha256"]:
        raise ValueError(f"Asset SHA256 differs from the pinned release: {path}")


def download_file(url, target):
    # Only public, immutable model URLs are used; no credentials are printed.
    request = urllib.request.Request(url, headers={"User-Agent": "GRAIL-GEM-X-setup/1"})
    with urllib.request.urlopen(request, timeout=120) as response, Path(target).open("wb") as out:
        shutil.copyfileobj(response, out, length=8 * 1024 * 1024)


def ensure_asset(root, asset, *, verify_only=False):
    target = Path(root) / asset["relative_path"]
    if target.exists():
        verify_asset(target, asset)
        return {**asset, "path": str(target.resolve()), "operation": "verified_cache"}
    if verify_only:
        raise FileNotFoundError(f"Missing pinned asset: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".gem-x-download-", dir=target.parent) as temporary:
        download = Path(temporary) / "download"
        download_file(asset["url"], download)
        payload = download
        suffix = asset.get("zip_member_suffix")
        if suffix:
            with zipfile.ZipFile(download) as archive:
                members = [
                    x for x in archive.infolist() if not x.is_dir() and x.filename.endswith(suffix)
                ]
                if len(members) != 1:
                    raise ValueError("Expected exactly one YOLOX end2end.onnx archive member")
                payload = Path(temporary) / "verified.onnx"
                # Stream the one named member, never extract archive paths.
                with archive.open(members[0]) as source, payload.open("wb") as out:
                    shutil.copyfileobj(source, out, length=8 * 1024 * 1024)
        verify_asset(payload, asset)
        # A simultaneous installer must not replace an already selected file.
        try:
            os.link(payload, target)
        except FileExistsError:
            verify_asset(target, asset)
    return {**asset, "path": str(target.resolve()), "operation": "downloaded_verified"}


def verify_source(source):
    source = Path(source).resolve()
    commit = subprocess.check_output(
        ["git", "-C", str(source), "rev-parse", "HEAD"], text=True
    ).strip()
    if commit != SOURCE_COMMIT:
        raise ValueError(f"GEM-X checkout must be pinned to {SOURCE_COMMIT}, found {commit}")
    dirty = subprocess.check_output(
        ["git", "-C", str(source), "status", "--porcelain", "--untracked-files=no"], text=True
    ).strip()
    if dirty:
        raise ValueError(
            "GEM-X tracked source has local edits; preserve it and use the pinned submodule"
        )
    return source


def native77_config(assets, *, camera_mode="background"):
    if camera_mode not in ("static", "background"):
        raise ValueError("camera_mode must be static or background")
    weights = Path(assets).resolve() / "weights"
    return {
        "pose": {
            "checkpoint": str(weights / "body_s240000.ckpt"),
            "release": "body",
            "batch_size": 4,
        },
        "observations": {
            "checkpoint": str(weights / "vitpose.pth"),
            "release_id": "vitpose77-dinov3-huge-metrosim-256x192-v1",
            "batch_size": 4,
        },
        "detector": {"checkpoint": str(weights / "yolox_humanart.onnx"), "device": "cpu"},
        "motion": {
            "checkpoint": str(weights / "motion_native77_e774_s292175.ckpt"),
            "stats": str(weights / "motion_stats_raw.npz"),
            "network": {
                "output_dim": 655,
                "max_len": 120,
                "cliffcam_dim": 3,
                "cam_angvel_dim": 6,
                "imgseq_dim": 1024,
                "latent_dim": 512,
                "num_layers": 12,
                "num_heads": 8,
                "mlp_ratio": 4.0,
                "pred_cam_dim": 3,
                "static_conf_dim": 6,
                "dropout": 0.1,
                "avgbeta": True,
                "obs_num_joints": 77,
                "sequence_static_slices": [[456, 584], [584, 640]],
            },
            "condition_spec": {
                "pose_release_id": "gem-pose-body-s240000-v1:pose_tokens",
                "observation_source": "vitpose77-dinov3-huge-metrosim-256x192-v1",
                "observation_joint_set": "soma77_native_v1",
            },
            "time_options": {"reference_fps": 30.0, "default_fps": 30.0, "fps_by_data_name": {}},
        },
        "camera": {"mode": camera_mode},
    }


def write_identical_or_new(path, value):
    path = Path(path)
    if path.exists():
        if json.loads(path.read_text()) != value:
            raise ValueError(f"Preserving existing, different configuration/receipt: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as handle:
        json.dump(value, handle, indent=2)
        handle.write("\n")


def clean_subprocess_env():
    env = dict(os.environ)
    for key in ("PYTHONPATH", "PYTHONHOME", "VIRTUAL_ENV"):
        env.pop(key, None)
    env["PYTHONNOUSERSITE"] = "1"
    return env


def link_existing(target, link):
    """Make an optional ignored submodule link, preserving every existing entry."""
    target, link = Path(target).resolve(strict=True), Path(link)
    if link.is_symlink() or link.exists():
        if link.resolve() != target:
            raise ValueError(f"Preserving an existing different path: {link}")
        return
    link.parent.mkdir(parents=True, exist_ok=True)
    link.symlink_to(target, target_is_directory=target.is_dir())


def install_downstream_soma(python):
    """Explicit opt-in for the separate GRAIL fitting interpreter."""
    env = clean_subprocess_env()
    subprocess.run(
        [str(python), "-I", "-m", "pip", "install", f"py-soma-x=={SOMA_VERSION}"],
        check=True,
        env=env,
    )
    subprocess.run(
        [str(python), "-I", "-c", "import soma; assert soma.__version__ == '0.3.3'"],
        check=True,
        env=env,
    )


def ignore_local_links(source):
    raw = subprocess.check_output(
        ["git", "-C", str(source), "rev-parse", "--git-path", "info/exclude"], text=True
    ).strip()
    exclude = Path(raw)
    if not exclude.is_absolute():
        exclude = source / exclude
    previous = exclude.read_text() if exclude.exists() else ""
    # Directory-only upstream rules do not match symlinks. Keep this local to
    # the submodule Git metadata, with no change to upstream tracked files.
    missing = [
        entry
        for entry in ("/weights", "/assets/soma", "/.venv")
        if entry not in previous.splitlines()
    ]
    if missing:
        exclude.parent.mkdir(parents=True, exist_ok=True)
        with exclude.open("a") as handle:
            handle.write(
                ("\n" if previous and not previous.endswith("\n") else "")
                + "\n".join(missing)
                + "\n"
            )


def install_environment(destination, source, *, python, torch_index, render=False):
    destination = Path(destination).absolute()
    source = verify_source(source)
    marker = destination / ".grail-gem-x-setup.json"
    identity = {
        "source_commit": SOURCE_COMMIT,
        "torch": TORCH_VERSION,
        "torch_index": torch_index,
        "soma": SOMA_VERSION,
        "render": render,
    }
    env = clean_subprocess_env()
    if destination.exists():
        if not marker.is_file() or json.loads(marker.read_text()) != identity:
            raise ValueError(
                f"Refusing to modify an unowned/differently configured environment: {destination}"
            )
    else:
        version = subprocess.check_output(
            [python, "-I", "-c", "import sys; print(sys.version_info >= (3, 10))"],
            text=True,
            env=env,
        ).strip()
        if version != "True":
            raise ValueError("GEM-X needs Python 3.10 or newer")
        ensurepip = subprocess.check_output(
            [
                python,
                "-I",
                "-c",
                "import importlib.util; print(importlib.util.find_spec('ensurepip') is not None)",
            ],
            text=True,
            env=env,
        ).strip()
        if ensurepip == "True":
            command = [python, "-I", "-m", "venv", str(destination)]
        elif shutil.which("uv"):
            # Debian may omit ensurepip even with the interpreter installed.
            # uv still creates a separate, seeded venv without inherited sites.
            command = [shutil.which("uv"), "venv", "--seed", "--python", python, str(destination)]
        else:
            raise RuntimeError(
                "Install python3-venv/ensurepip or uv before creating the GEM-X environment"
            )
        subprocess.run(command, check=True, env=env)
        write_identical_or_new(marker, identity)
    settings = (destination / "pyvenv.cfg").read_text().lower()
    if "include-system-site-packages = false" not in settings:
        raise ValueError("The GEM-X environment must not inherit system-site packages")
    executable = destination / "bin/python"
    pip = [str(executable), "-I", "-m", "pip"]
    subprocess.run([*pip, "install", "--upgrade", "pip"], check=True, env=env)
    subprocess.run(
        [*pip, "install", f"torch=={TORCH_VERSION}", "--index-url", torch_index],
        check=True,
        env=env,
    )
    extras = "video,render" if render else "video"
    # Build a regular wheel from the pinned tree. Untracked checkout files must
    # not become installed code, and the runtime must not import an editable
    # checkout that could subsequently change underneath an existing cache.
    with tempfile.TemporaryDirectory(prefix="gem-x-pinned-source-") as temporary:
        archive = Path(temporary) / "source.zip"
        subprocess.run(
            [
                "git",
                "-C",
                str(source),
                "archive",
                "--format=zip",
                "--output",
                str(archive),
                SOURCE_COMMIT,
            ],
            check=True,
            env=env,
        )
        source_archive_sha256 = sha256(archive)
        build_source = Path(temporary) / "source"
        with zipfile.ZipFile(archive) as archive_file:
            archive_file.extractall(build_source)
        subprocess.run(
            [*pip, "install", f"{build_source}[{extras}]", f"py-soma-x=={SOMA_VERSION}"],
            check=True,
            env=env,
        )
    subprocess.run([*pip, "check"], check=True, env=env)
    probe = (
        "import importlib.metadata as m,json,sys,torch,soma,gem; from pathlib import Path; "
        "from transformers import DINOv3ViTConfig; from gem.pose.api import body_config; "
        "assert soma.__version__ == '0.3.3'; assert m.version('transformers') == '5.3.0'; "
        "assert Path(gem.__file__).resolve().is_relative_to(Path(sys.prefix).resolve()); "
        "assert body_config().task == 'body'; "
        "print(json.dumps({'python':sys.version.split()[0], 'torch':torch.__version__, "
        "'soma':soma.__version__, 'gem':gem.__file__, 'system_site_packages':False}))"
    )
    runtime = json.loads(
        subprocess.check_output([str(executable), "-I", "-c", probe], text=True, env=env)
    )
    subprocess.run(
        [str(destination / "bin/gem-infer"), "--help"],
        check=True,
        env=env,
        stdout=subprocess.DEVNULL,
    )
    runtime.update(
        source_commit=SOURCE_COMMIT,
        source_archive_sha256=source_archive_sha256,
        install_mode="wheel_from_pinned_git_archive",
    )
    return runtime


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=REPO_ROOT / "imports/GEM-X")
    parser.add_argument(
        "--assets",
        type=Path,
        help="External model root; about 7.7 GB, required except with --env-only",
    )
    parser.add_argument(
        "--env",
        type=Path,
        help="New isolated venv; required unless --download-only or --verify-only",
    )
    parser.add_argument(
        "--python", default=sys.executable, help="Python 3.10+ executable used to create the venv"
    )
    parser.add_argument(
        "--torch-index",
        choices=("https://download.pytorch.org/whl/cu128", "https://download.pytorch.org/whl/cpu"),
        default="https://download.pytorch.org/whl/cu128",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--env-only",
        action="store_true",
        help="Install runtime dependencies without model downloads",
    )
    mode.add_argument("--download-only", action="store_true")
    mode.add_argument(
        "--verify-only",
        action="store_true",
        help="Read-only verification; no downloads, installs or config writes",
    )
    parser.add_argument(
        "--render", action="store_true", help="Install optional Open3D 0.19 rendering dependencies"
    )
    parser.add_argument(
        "--link-submodule",
        action="store_true",
        help="Safely link ignored weights/, assets/soma and optional .venv to the external paths",
    )
    parser.add_argument(
        "--grail-python",
        type=Path,
        help="Explicitly install py-soma-x 0.3.3 into the separate downstream GRAIL interpreter",
    )
    parser.add_argument("--camera-mode", choices=("background", "static"), default="background")
    args = parser.parse_args(argv)
    if args.env is not None:
        args.env = args.env.expanduser().absolute()
    if args.verify_only and (args.link_submodule or args.grail_python):
        parser.error("--verify-only cannot install packages or create links")
    if args.env_only and (args.assets is not None or args.link_submodule):
        parser.error("--env-only cannot select model assets or create submodule links")
    if not args.env_only and args.assets is None:
        parser.error("--assets is required unless --env-only is selected")
    if not (args.download_only or args.verify_only) and args.env is None:
        parser.error("--env is required for a clean environment installation")
    source = verify_source(args.source)
    assets = args.assets.expanduser().resolve() if args.assets else None
    if assets is not None and (assets.is_relative_to(REPO_ROOT) or assets.is_relative_to(source)):
        parser.error("Place large model assets outside the repository/submodule")
    if not (args.download_only or args.verify_only):
        for command in ("ffmpeg", "ffprobe"):
            if shutil.which(command) is None:
                parser.error(f"Install the system prerequisite {command} first")
        runtime = install_environment(
            args.env, source, python=args.python, torch_index=args.torch_index, render=args.render
        )
    else:
        runtime = None
    if args.env_only:
        (args.env / "gem_x_runtime.json").write_text(json.dumps(runtime, indent=2) + "\n")
        if args.grail_python:
            install_downstream_soma(args.grail_python)
        print(json.dumps({"status": "PASS", "files_verified": 0, "runtime": runtime}, indent=2))
        return
    files = [ensure_asset(assets, asset, verify_only=args.verify_only) for asset in ASSETS]
    config_path = assets / f"body_video_native77_{args.camera_mode}.json"
    if not args.verify_only:
        write_identical_or_new(config_path, native77_config(assets, camera_mode=args.camera_mode))
        receipt = {
            "schema": "grail.gem_x.setup.v1",
            "source_commit": SOURCE_COMMIT,
            "gem_weights_revision": GEM_WEIGHTS_REVISION,
            "soma_assets_revision": SOMA_ASSETS_REVISION,
            "soma_version": SOMA_VERSION,
            "files": [{k: v for k, v in row.items() if k != "operation"} for row in files],
        }
        write_identical_or_new(assets / "gem_x_assets.json", receipt)
        if runtime is not None:
            (args.env / "gem_x_runtime.json").write_text(json.dumps(runtime, indent=2) + "\n")
        if args.grail_python:
            install_downstream_soma(args.grail_python)
        if args.link_submodule:
            link_existing(assets / "weights", source / "weights")
            link_existing(assets / "soma", source / "assets/soma")
            if args.env:
                link_existing(args.env, source / ".venv")
            ignore_local_links(source)
    print(
        json.dumps(
            {
                "status": "PASS",
                "files_verified": len(files),
                "assets": str(assets),
                "config": str(config_path),
                "runtime": runtime,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
