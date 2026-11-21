"""Locations of the vendored manipulation tracking stack and its default checkpoints."""
from __future__ import annotations

import hashlib
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
# Vendored SONIC fork with the manipulation (ContactFirst) extensions; see its README.md.
SONIC_MANIP_ROOT = REPO_ROOT / "imports" / "SONIC-manipulation"
RUNTIME_DIR = Path(__file__).resolve().parent / "runtime"
CONVERSION_DIR = Path(__file__).resolve().parent / "conversion"
# Shared 10-point BPS basis used by every released manipulation policy (observation input).
BPS_BASIS = Path(__file__).resolve().parent / "assets" / "bps_basis_10.npy"

# Upstream revision of the vendored SONIC fork (recorded for provenance).
UPSTREAM_SONIC_COMMIT = "fb8563f9a5b5889d50bccba4dc07a607f2e4a71c"
EXPERIMENT = "manager/universal_token/hoi/advanced_manip_ground_unified_contactfirst_ablation_d"

# Checkpoints fetched by scripts/setup/download_checkpoints.sh (Hugging Face
# nvidia/PhysicalAI-Robotics-Locomanipulation-GRAIL, checkpoint/SONIC/models/...).
MODELS_DIR = REPO_ROOT / "imports" / "SONIC" / "models"
WARMSTART_ACTOR = MODELS_DIR / "manip_warmstart" / "last.pt"
SONIC_DECODER_CONFIG = MODELS_DIR / "sonic_manipulation_base" / "model_config.yaml"
SONIC_DECODER_CHECKPOINT = MODELS_DIR / "sonic_manipulation_base" / "last.pt"
# SHA-256 of the released files; the warm-start is the Teleop+GRAIL generalist actor (step 8325).
WARMSTART_ACTOR_SHA256 = "af687c9bf8c27226829ea2da58f307b50f4bd7da77c6dc26e9547e129b51ea01"
SONIC_DECODER_SHA256 = "62f3e336cc11cbfb7517fee1f053ed92c81d44accd3757faa0febf421d0a2cd4"

# Object mass used for every released manipulation finetune (kg), written to physics.json.
OBJECT_MASS_KG = 0.1

# PhysX GPU buffer capacities used by the 2048-environment production runs.
PHYSX_BUFFERS = {
    "gpu_collision_stack_size": 536870912,
    "gpu_found_lost_aggregate_pairs_capacity": 134217728,
    "gpu_found_lost_pairs_capacity": 134217728,
    "gpu_heap_capacity": 536870912,
    "gpu_max_rigid_contact_count": 67108864,
    "gpu_max_rigid_patch_count": 33554432,
    "gpu_temp_buffer_capacity": 134217728,
    "gpu_total_aggregate_pairs_capacity": 16777216,
}


def sha256(path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()
