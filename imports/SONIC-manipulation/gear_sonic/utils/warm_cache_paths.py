"""Stable filesystem links to initialize-once warm bundles (no fingerprints)."""
import os
import uuid
from pathlib import Path
from urllib.parse import quote


def warm_link(cache_root, kind, namespace, rank):
    explicit = os.environ.get("GEAR_WARM_CACHE_LINKS")
    root = Path(explicit).expanduser() if explicit else Path(cache_root) / "warm" / quote(str(namespace), safe="")
    revision = os.environ.get("GEAR_WARM_CACHE_REVISION", "initial")
    root = root / quote(revision, safe="")
    return root / ("%s-rank-%03d" % (kind, rank))


def publish_warm_link(link, entry):
    link.parent.mkdir(parents=True, exist_ok=True)
    temporary = link.with_name("." + link.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        temporary.symlink_to(entry.resolve(), target_is_directory=True)
        temporary.replace(link)
    finally:
        if temporary.is_symlink():
            temporary.unlink()
