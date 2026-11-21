"""Build two offline Blender extension ZIPs from repository-owned edit kits.

No licensed models, case motions, historical artifact scripts or Python wheels
are included. Original kit files stay the source of truth; generated copies are
kept only in the explicit new output directory.
"""

import argparse
import hashlib
import json
import shutil
import zipfile
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10
    import tomli as tomllib

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
KITS = {
    "grail_hoi_refinement": (None, []),
    "grail_g1_retargeting": ("contact_edit_kit", ["native_io.py", "blender_curves.py"]),
}


def digest(data):
    return hashlib.sha256(data).hexdigest()


def bundled_source(name, filename, data):
    """Keep Blender embedded-script fallback, isolate normal package imports."""
    if name == "grail_g1_retargeting" and filename == "blender_curves.py":
        old = (
            "sys.path.insert(0, str(Path(__file__).resolve().parent))\n"
            "from native_io import load_npz, resolve_channels, merge_curve_samples, check_sample_limits"
        )
        new = (
            "if __package__:\n"
            "    from .native_io import load_npz, resolve_channels, merge_curve_samples, check_sample_limits\n"
            "else:\n"
            "    sys.path.insert(0, str(Path(__file__).resolve().parent))\n"
            "    from native_io import load_npz, resolve_channels, merge_curve_samples, check_sample_limits"
        )
        text = data.decode()
        if text.count(old) != 1:
            raise ValueError("Kit import adapter no longer matches; review before building")
        return (
            text.replace(old, new).encode(),
            "package-relative imports; historical embedded fallback retained",
        )
    return data, "byte-identical"


def build(out):
    out = Path(out).resolve()
    # Preflight everything before allocating an output directory.
    for name, (kit_name, files) in KITS.items():
        for path in [
            HERE / name / "__init__.py",
            HERE / name / "blender_manifest.toml",
            REPO / "LICENSE",
            *(REPO / "scripts/standalone" / kit_name / f for f in files),
        ]:
            if not path.is_file():
                raise FileNotFoundError(path)
        info = tomllib.loads((HERE / name / "blender_manifest.toml").read_text())
        if info["id"] != name:
            raise ValueError("Package ID mismatch")
    out.mkdir(parents=True, exist_ok=False)
    records = []
    for name, (kit_name, files) in KITS.items():
        target = out / "staging" / name
        target.mkdir(parents=True)
        bindings = []
        for path in sorted((HERE / name).rglob("*")):
            if (
                not path.is_file()
                or "__pycache__" in path.parts
                or path.suffix not in (".py", ".toml", ".md")
            ):
                continue
            if "tests" in path.relative_to(HERE / name).parts:
                continue
            rel = path.relative_to(HERE / name)
            dest = target / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            data = path.read_bytes()
            dest.write_bytes(data)
            bindings.append(
                dict(
                    source=str(path.relative_to(REPO)),
                    packaged=str(rel),
                    source_sha256=digest(data),
                    packaged_sha256=digest(data),
                    transform="byte-identical",
                )
            )
        if files:
            (target / "kit").mkdir()
            (target / "kit/__init__.py").write_text(
                '"""Bundled repository edit kit; see BUNDLE_PROVENANCE.json."""\n'
            )
        for filename in files:
            source = REPO / "scripts/standalone" / kit_name / filename
            data = source.read_bytes()
            packaged, transform = bundled_source(name, filename, data)
            (target / "kit" / filename).write_bytes(packaged)
            bindings.append(
                dict(
                    source=str(source.relative_to(REPO)),
                    packaged="kit/" + filename,
                    source_sha256=digest(data),
                    packaged_sha256=digest(packaged),
                    transform=transform,
                )
            )
        if name == "grail_hoi_refinement":
            source = REPO / "grail/agentic_kinematic/hoi_refinement/blender_worker.py"
            data = source.read_bytes()
            (target / "stage2_worker.py").write_bytes(data)
            bindings.append(
                dict(
                    source=str(source.relative_to(REPO)),
                    packaged="stage2_worker.py",
                    source_sha256=digest(data),
                    packaged_sha256=digest(data),
                    transform="byte-identical model-neutral Stage 2 worker",
                )
            )
            source = REPO / "grail/agentic_kinematic/hoi_refinement/preview.py"
            data = source.read_bytes()
            (target / "stage2_preview.py").write_bytes(data)
            bindings.append(
                dict(
                    source=str(source.relative_to(REPO)),
                    packaged="stage2_preview.py",
                    source_sha256=digest(data),
                    packaged_sha256=digest(data),
                    transform="byte-identical repair preview guards",
                )
            )
        shutil.copyfile(REPO / "LICENSE", target / "LICENSE")
        if (HERE / "README.md").is_file():
            shutil.copyfile(HERE / "README.md", target / "README.md")
        (target / "BUNDLE_PROVENANCE.json").write_text(
            json.dumps(
                dict(schema_version=1, package=name, model_assets_bundled=False, files=bindings),
                indent=2,
            )
            + "\n"
        )
        manifest = tomllib.loads((target / "blender_manifest.toml").read_text())
        archive = out / (name + "-" + manifest["version"] + ".zip")
        with zipfile.ZipFile(archive, "x", compression=zipfile.ZIP_DEFLATED) as z:
            for path in sorted(target.rglob("*")):
                if not path.is_file():
                    continue
                info = zipfile.ZipInfo(
                    path.relative_to(target).as_posix(), date_time=(2026, 1, 1, 0, 0, 0)
                )
                info.compress_type = zipfile.ZIP_DEFLATED
                info.external_attr = 0o100644 << 16
                z.writestr(info, path.read_bytes())
        records.append(
            dict(
                package=name,
                zip=str(archive),
                sha256=digest(archive.read_bytes()),
                bytes=archive.stat().st_size,
                staging=str(target),
                bundled_files=len(bindings),
            )
        )
    report = dict(
        schema_version=1,
        packages=records,
        note="Local research packages retain repository LICENSE; no model assets or external inference libraries bundled",
    )
    (out / "build_manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", required=True, help="New output directory, never overwritten")
    print(json.dumps(build(p.parse_args().out), indent=2))
