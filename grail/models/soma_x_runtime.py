"""Audited SOMA-X 0.3.3 runtime for GRAIL's existing native identities.

The estimator's fixed-GS1 contract and a downstream manifest-authorized fixed
identity scale are separate. This wrapper supports the latter without changing
GEM-X's prediction adapter or the original identity manifests.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch
from torch import nn

DEFAULT_SOMA_ASSETS = Path(__file__).resolve().parents[2] / "imports/GEM-X/assets/soma"
COMPATIBILITY_PATH = Path(__file__).parents[1] / "constants/soma03_runtime_compatibility.json"


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def compatibility_record():
    record = json.loads(COMPATIBILITY_PATH.read_text())
    if record.get("schema") != "grail.soma03.runtime_compatibility.v1":
        raise ValueError("Unrecognized SOMA03 runtime compatibility record")
    return record


def validate_legacy_identity_binding(manifest, record):
    """Accept only the exact legacy model binding covered by the native audit."""
    for suffix, expected in record["legacy_identity_bindings"].items():
        matches = [
            value
            for path, value in manifest.get("sources", {}).items()
            if path.endswith("/" + suffix)
        ]
        if matches != [expected]:
            raise ValueError(f"SOMA03 identity model binding differs: {suffix}")


def identity_runtime_binding(record=None):
    """Portable, exact native binding for newly fitted SOMA-X identities."""
    record = compatibility_record() if record is None else record
    return {
        "schema": "grail.soma_x.identity_runtime_binding.v1",
        "runtime_version": record["runtime_version"],
        "source_sha256": record["official_source_sha256"],
        "asset_sha256": record["official_asset_sha256"],
        "convention": record["convention"],
    }


def validate_identity_binding(manifest, record):
    """Validate a new official binding or the exact audited historical binding.

    A present but invalid official binding cannot fall back to legacy metadata.
    The identity parameter schemas remain unchanged; geometry is independently
    bound to the producer runtime and never inferred from the filename.
    """
    if "runtime_binding" in manifest:
        if manifest["runtime_binding"] != identity_runtime_binding(record):
            raise ValueError("SOMA-X identity runtime binding differs")
    else:
        validate_legacy_identity_binding(manifest, record)


def validate_official_runtime(source_root, asset_root, version, record):
    """Refuse a different release, edited native source, or mixed geometry."""
    if version != record["runtime_version"]:
        raise ValueError(f"SOMA-X runtime version differs: {version!r}")
    for root, bindings, label in (
        (Path(source_root), record["official_source_sha256"], "source"),
        (Path(asset_root), record["official_asset_sha256"], "asset"),
    ):
        for relative, expected in bindings.items():
            path = root / relative
            if not path.is_file() or _sha256(path) != expected:
                raise ValueError(f"SOMA-X {label} binding differs: {relative}")


def official_runtime_inputs(asset_root):
    """Resolve and validate the actual official files used by native export/FK."""
    import soma

    record = compatibility_record()
    source = Path(soma.__file__).resolve().parent
    assets = Path(asset_root).expanduser().resolve()
    validate_official_runtime(source, assets, soma.__version__, record)
    paths = [Path(__file__).resolve(), COMPATIBILITY_PATH.resolve()]
    paths += [source / name for name in record["official_source_sha256"]]
    paths += [assets / name for name in record["official_asset_sha256"]]
    metadata = {
        "kind": "gem_x",
        "version": soma.__version__,
        "source_root": str(source),
        "asset_root": str(assets),
        "compatibility_id": record["id"],
        "convention": record["convention"],
    }
    return paths, metadata


class SomaXNativeLayer(nn.Module):
    """LOW-LOD native FK/skinning with the reviewed current-reference convention."""

    def __init__(self, asset_root, device="cpu", *, correctives=True):
        super().__init__()
        import soma
        from soma import SOMALayer, Unit

        self.compatibility = compatibility_record()
        self.asset_root = Path(asset_root).expanduser().resolve()
        self.source_root = Path(soma.__file__).resolve().parent
        validate_official_runtime(
            self.source_root, self.asset_root, soma.__version__, self.compatibility
        )
        kwargs = {} if correctives else {"correctives_model_path": None}
        self.soma = SOMALayer(
            data_root=str(self.asset_root),
            lod="low",
            mode="torch",
            device=device,
            identity_model_type="soma",
            output_unit=Unit.METERS,
            enable_procedural_transforms=True,
            reference_pose=None,
            **kwargs,
        )
        self.soma.requires_grad_(False)
        if self.soma.num_scale_params != 60 or tuple(self.soma.scale_param_names[-4:]) != tuple(
            self.compatibility["convention"]["appended_scale_names"]
        ):
            raise ValueError(
                "SOMA-X runtime scale controls differ from the reviewed local60 layout"
            )
        self.faces = self.soma.faces
        self.parents = [int(p) - 1 for p in self.soma.output_joint_parent_ids.cpu().tolist()][1:]
        self.validate_external_global_scale = True

    def static_forward(
        self,
        poses,
        identity_coeffs,
        scale_params,
        transl=None,
        pose2rot=True,
        return_joints_only=False,
        apply_correctives=True,
    ):
        if scale_params.shape[-1] != 57:
            raise ValueError("SOMA-X GRAIL adapter requires [global, local56] scale57")
        global_scale = scale_params[..., :1]
        if self.validate_external_global_scale and not torch.equal(
            global_scale, torch.ones_like(global_scale)
        ):
            raise ValueError("Raw SOMA-X prediction requires exact global scale 1")
        local = torch.cat((scale_params[..., 1:], torch.ones_like(scale_params[..., :4])), dim=-1)
        self.soma.prepare_identity(
            identity_coeffs.float(),
            local.float(),
            global_scale=global_scale.float(),
            repose_to_bind_pose=True,
        )
        return self.soma.pose(
            poses.float().contiguous(),
            transl=None if transl is None else transl.float(),
            pose2rot=pose2rot,
            apply_correctives=apply_correctives and not return_joints_only,
            fk_only=return_joints_only,
        )
