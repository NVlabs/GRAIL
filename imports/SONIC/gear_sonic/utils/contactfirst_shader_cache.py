#!/usr/bin/env python3
"""Create immutable Kit shader-cache seeds and isolated writable copies per rank."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile


CACHE_NAMES = ("shadercache", "nv_shadercache")
MANIFEST = "SHADER_CACHE_SEED.json"


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def inventory(root: Path) -> dict:
    result = {}
    for name in CACHE_NAMES:
        directory = root / name
        if not directory.exists():
            continue
        for path in sorted(directory.rglob("*")):
            if path.is_symlink():
                raise ValueError("Shader cache seeds must not contain symlinks")
            if path.is_file():
                state = path.stat()
                result[path.relative_to(root).as_posix()] = [state.st_size, state.st_mtime_ns]
    return result


def snapshot(source: Path, destination: Path) -> dict:
    source = source.resolve(strict=True)
    destination = destination.absolute()
    if destination.exists() or source == destination or source in destination.parents:
        raise ValueError("Seed destination must be new and outside the writable source cache")
    before = inventory(source)
    if not before:
        raise ValueError("No shader cache files are available to snapshot")
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".shader-seed-", dir=destination.parent))
    try:
        files = {}
        for relative in before:
            target = staging / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source / relative, target)
            files[relative] = {"bytes": target.stat().st_size, "sha256": digest(target)}
        if before != inventory(source):
            raise RuntimeError("Shader cache changed during snapshot; retry after compilation is idle")
        receipt = {"version": 1, "source": str(source), "files": files,
                   "bytes": sum(item["bytes"] for item in files.values())}
        (staging / MANIFEST).write_text(json.dumps(receipt, indent=2) + "\n")
        # The producer never updates a published seed. Consumers open only for
        # reading and copy bytes, never hardlink into the seed.
        for path in staging.rglob("*"):
            if path.is_file():
                path.chmod(0o444)
        staging.rename(destination)
        return receipt
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def prepare(root: Path, rank: int, seed: Path | None = None) -> list[str]:
    if not root.is_absolute() or rank < 0:
        raise ValueError("Cache root must be absolute and rank nonnegative")
    root = root.resolve()
    if seed is not None:
        seed = seed.resolve(strict=True)
        if root == seed or seed in root.parents or root in seed.parents:
            raise ValueError("Writable cache and immutable seed must not contain each other")
    target = root / f"rank_{rank}"
    target.mkdir(parents=True, exist_ok=True)
    marker = target / "SEED_RECEIPT.json"
    seed_manifest = seed / MANIFEST if seed else None
    identity = {"seed": str(seed) if seed else None,
                "manifest_sha256": digest(seed_manifest) if seed else None}
    if marker.exists():
        if json.loads(marker.read_text()) != identity:
            raise ValueError("This rank cache was initialized from a different seed")
    else:
        if any(target.iterdir()):
            raise ValueError("Refusing to overwrite an unowned rank shader cache")
        if seed:
            receipt = json.loads(seed_manifest.read_text())
            if receipt.get("version") != 1 or not receipt.get("files"):
                raise ValueError("Invalid immutable shader cache seed manifest")
            for relative, expected in receipt["files"].items():
                entry = Path(relative)
                if entry.is_absolute() or ".." in entry.parts or entry.parts[0] not in CACHE_NAMES:
                    raise ValueError("Invalid path in shader cache seed manifest")
                source = seed / entry
                if source.is_symlink() or seed not in source.resolve().parents:
                    raise ValueError("Shader cache seed entry escapes its root")
                destination = target / entry
                destination.parent.mkdir(parents=True, exist_ok=True)
                # copyfile creates independent writable files even though seed
                # files are read-only. No shared inode links across job/ranks.
                shutil.copyfile(source, destination)
                if destination.stat().st_size != expected["bytes"] or digest(destination) != expected["sha256"]:
                    raise ValueError("Shader cache seed checksum mismatch")
        marker.write_text(json.dumps(identity, indent=2) + "\n")
    for name in CACHE_NAMES:
        (target / name).mkdir(exist_ok=True)
    print(f"[contactfirst_shader_cache] rank={rank} writable={target} seed={seed}", flush=True)
    return [f"--/rtx/shaderDb/shaderCachePath={target / 'shadercache'}",
            f"--/rtx/shaderDb/driverShaderCachePath={target / 'nv_shadercache'}"]


def prepare_from_environment() -> list[str]:
    root = os.environ.get("CONTACTFIRST_SHADER_CACHE_ROOT")
    seed = os.environ.get("CONTACTFIRST_SHADER_CACHE_SEED")
    if not root:
        if seed:
            raise ValueError("A shader cache seed requires an isolated writable cache root")
        return []
    if not Path(root).is_absolute() or any(character.isspace() for character in root):
        raise ValueError("Shader cache root must be absolute without whitespace for Kit arguments")
    return prepare(Path(root), int(os.environ.get("LOCAL_RANK", "0")), Path(seed) if seed else None)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["snapshot"])
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    options = parser.parse_args()
    result = snapshot(options.source, options.destination)
    print(json.dumps({"seed": str(options.destination), "files": len(result["files"]), "bytes": result["bytes"]}))
