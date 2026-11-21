"""Locations and constants used by the manipulation data production (handoff/pack).

Training and evaluation live in the unified ``imports/SONIC`` tree:
``gear_sonic/train_agent_trl.py`` with
``+exp=manager/universal_token/hoi/advanced_manip_ground_unified_contactfirst_finetune``
and ``gear_sonic/eval_contactfirst_trl.py`` (see
``scripts/tracking/manipulation_finetune.sh`` and the docs).
"""
from __future__ import annotations

import hashlib
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
# Unified vendored SONIC tree (the manipulation/ContactFirst extensions are folded in).
SONIC_MANIP_ROOT = REPO_ROOT / "imports" / "SONIC"
CONVERSION_DIR = Path(__file__).resolve().parent / "conversion"
# Shared 10-point BPS basis used by every released manipulation policy (observation input).
BPS_BASIS = Path(__file__).resolve().parent / "assets" / "bps_basis_10.npy"

# Upstream revision of the vendored manipulation extensions (recorded for provenance).
UPSTREAM_SONIC_COMMIT = "fb8563f9a5b5889d50bccba4dc07a607f2e4a71c"

# Checkpoints fetched by scripts/setup/download_checkpoints.sh (Hugging Face
# nvidia/PhysicalAI-Robotics-Locomanipulation-GRAIL, checkpoint/SONIC/models/...).
MODELS_DIR = REPO_ROOT / "imports" / "SONIC" / "models"
WARMSTART_ACTOR = MODELS_DIR / "manip_warmstart" / "last.pt"
SONIC_DECODER_CONFIG = MODELS_DIR / "sonic_manipulation_base" / "model_config.yaml"
SONIC_DECODER_CHECKPOINT = MODELS_DIR / "sonic_manipulation_base" / "last.pt"

# Object mass used for every released manipulation finetune (kg), written to physics.json.
OBJECT_MASS_KG = 0.1


def sha256(path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()
