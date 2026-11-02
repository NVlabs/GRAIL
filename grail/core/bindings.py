"""Content-addressed file and loaded-model bindings, without orchestration dependencies."""

import hashlib
import json
from pathlib import Path


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def bind(path, base=None):
    path = Path(path)
    if not path.is_absolute():
        path = Path(base or ".") / path
    path = path.resolve(strict=True)
    if not path.is_file():
        raise ValueError(f"Bind individual model/asset files, not a directory: {path}")
    return {"path": str(path), "sha256": sha256(path), "bytes": path.stat().st_size}


def loaded_model_hashes(human_model):
    """Bind the actual loaded/baked tensors, including implicit lite-model assets."""
    import numpy as np

    hashes = {}
    for name in ("model", "coco17_model"):
        model = getattr(human_model, name, None)
        if model is None or not hasattr(model, "state_dict"):
            continue
        digest = hashlib.sha256()
        for key, value in sorted(model.state_dict().items()):
            array = value.detach().cpu().contiguous().numpy()
            digest.update(key.encode() + b"\0")
            digest.update(str(array.dtype).encode() + b"\0")
            digest.update(np.asarray(array.shape, dtype="<i8").tobytes())
            digest.update(array.tobytes())
        if hasattr(model, "faces"):
            digest.update(np.asarray(model.faces, dtype="<i8").tobytes())
        hashes[name] = digest.hexdigest()
    return hashes


def write_json(path, value):
    """Exclusive creation avoids silently overwriting a prior checkpoint."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")
