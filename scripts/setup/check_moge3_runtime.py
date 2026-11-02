"""Validate MoGe's installed sources without pretending image builds have a GPU."""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata as metadata
import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from urllib.parse import unquote, urlparse

MOGE_REVISION = "74fbce054ebed49800de42d0ad0e83495065719a"
DEPENDENCIES = {
    "utils3d_moge": "62f09d58509485564e24d5d9f6aac9ee9ebc0c37",
    "flex_gemm": "b2fadb29d41846c7981ade6801ffc689fae119cf",
}


def check_runtime(source, base_versions, *, no_gpu_import=False):
    """Check provenance/static code, then actual GPU-dependent imports if requested."""
    import torch
    import triton

    source = Path(source).resolve()
    if torch.__version__ != base_versions["torch"] or triton.__version__ != "3.4.0":
        raise ValueError("MoGe worker must retain base PyTorch and use isolated Triton 3.4.0")
    if (
        subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
        != MOGE_REVISION
    ):
        raise ValueError("MoGe source is not the pinned revision")
    if subprocess.check_output(
        ["git", "-C", str(source), "status", "--porcelain", "--untracked-files=no"], text=True
    ).strip():
        raise ValueError("MoGe source has tracked modifications")

    direct = json.loads(metadata.distribution("moge").read_text("direct_url.json") or "{}")
    location = urlparse(direct.get("url", ""))
    if (
        location.scheme != "file"
        or Path(unquote(location.path)).resolve() != source
        or direct.get("dir_info", {}).get("editable") is not True
    ):
        raise ValueError("Installed MoGe distribution does not bind the selected source")
    spec = importlib.util.find_spec("moge")
    if spec is None or spec.origin is None or Path(spec.origin).resolve().parent != source / "moge":
        raise ValueError("MoGe import path differs from the selected source")

    sources = list((source / "moge").rglob("*.py"))
    for package, revision in DEPENDENCIES.items():
        distribution = metadata.distribution(package)
        direct = json.loads(distribution.read_text("direct_url.json") or "{}")
        if direct.get("vcs_info", {}).get("commit_id") != revision:
            raise ValueError(f"Installed {package} is not the pinned revision")
        files = [
            Path(distribution.locate_file(entry))
            for entry in distribution.files or ()
            if str(entry).endswith(".py")
        ]
        if not files:
            raise ValueError(f"Installed {package} has no Python source inventory")
        sources.extend(files)
    for path in sources:
        compile(path.read_bytes(), str(path), "exec")

    if not no_gpu_import:
        if not torch.cuda.is_available():
            raise RuntimeError(
                "MoGe/FlexGEMM import requires a visible CUDA device; "
                "--no-gpu-import is only for image construction, not deployment preflight"
            )
        for module in ("utils3d_moge", "flex_gemm", "moge.model.v3"):
            importlib.import_module(module)

    return {
        "python": sys.executable,
        "base_python": str(Path(sys._base_executable).resolve()),
        "base_versions": base_versions,
        "moge_revision": MOGE_REVISION,
        "utils3d_moge_revision": DEPENDENCIES["utils3d_moge"],
        "flex_gemm_revision": DEPENDENCIES["flex_gemm"],
        "worker_packages": {
            name: metadata.version(name)
            for name in ("moge", "torch", "triton", "utils3d_moge", "flex_gemm")
        },
        "static_source_files_checked": len(sources),
        "gpu_dependent_imports_checked": not no_gpu_import,
        "gpu_inference_checked": False,
        "validation": (
            "Pinned distributions and Python compilation passed; "
            "GPU-dependent imports deferred to deployment preflight; inference not run"
            if no_gpu_import
            else "Pinned distributions, Python compilation and GPU-dependent imports passed; inference not run"
        ),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--base-versions", type=json.loads, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--no-gpu-import", action="store_true")
    args = parser.parse_args(argv)
    record = check_runtime(args.source, args.base_versions, no_gpu_import=args.no_gpu_import)
    args.output.write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
