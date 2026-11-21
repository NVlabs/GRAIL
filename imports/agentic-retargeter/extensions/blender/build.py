"""Build the offline G1 Blender extension from the repository-owned edit kit.

No licensed models, case motions, historical artifact scripts or Python wheels
are included. Original kit files stay the source of truth; generated copies are
kept only in the explicit new output directory.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tomllib
import zipfile

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
INSPECTION_FILES = ('inspection.py', 'blender_inspection.py', 'collision.py',
                    'collision_backend.py', 'blender_collision.py',
                    'collision_sequence.py', 'collision_sequence_backend.py',
                    'blender_collision_sequence.py')
REPAIR_FILES = ('repair_bridge.py', 'repair_worker.py', 'local_repair.py',
                'motion_audit.py', 'joint_velocity.py', 'repair_geometry.py', 'repair_scope.py', 'repair_gpu.py', 'repair_numba.py',
                'repair_material.py', 'repair_tensor.py', 'repair_iteration.py', 'audit_cache.py',
                'contact_tolerance.py', 'static_scene.py', 'clearance_objective.py', 'motion_diagnostics.py')
REPAIR_CONTACT_FILES = ('native_io.py', 'pose.py')
KITS = {
    'grail_g1_retargeting': ('contact_edit_kit', ['native_io.py', 'blender_curves.py']),
}


def digest(data):
    return hashlib.sha256(data).hexdigest()


def bundled_source(name, filename, data):
    """Package-ready source retains the historical embedded-script fallback."""
    return data, 'byte-identical'


def build(out):
    out = Path(out).resolve()
    # Preflight everything before allocating an output directory.
    for name, (kit_name, files) in KITS.items():
        for path in [HERE/name/'__init__.py', HERE/name/'blender_manifest.toml',
                     REPO/'LICENSE', *(REPO/'src/contact_edit'/f for f in files),
                     *(REPO/'src'/f for f in INSPECTION_FILES + REPAIR_FILES),
                     *(REPO/'src/contact_edit'/f for f in REPAIR_CONTACT_FILES),
                     REPO/'tools/repair/README.md']:
            if not path.is_file():
                raise FileNotFoundError(path)
        info = tomllib.loads((HERE/name/'blender_manifest.toml').read_text())
        if info['id'] != name:
            raise ValueError('Package ID mismatch')
    out.mkdir(parents=True, exist_ok=False)
    records = []
    for name, (kit_name, files) in KITS.items():
        target = out/'staging'/name
        target.mkdir(parents=True)
        bindings = []
        for path in sorted((HERE/name).rglob('*')):
            if not path.is_file() or '__pycache__' in path.parts or path.suffix not in ('.py', '.toml', '.md'):
                continue
            if 'tests' in path.relative_to(HERE/name).parts:
                continue
            rel = path.relative_to(HERE/name)
            dest = target/rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            data = path.read_bytes()
            dest.write_bytes(data)
            bindings.append(dict(source=str(path.relative_to(REPO)), packaged=str(rel),
                                 source_sha256=digest(data), packaged_sha256=digest(data), transform='byte-identical'))
        (target/'kit').mkdir()
        (target/'kit/__init__.py').write_text('"""Bundled repository edit kit; see BUNDLE_PROVENANCE.json."""\n')
        for filename in files:
            source = REPO/'src/contact_edit'/filename
            data = source.read_bytes()
            packaged, transform = bundled_source(name, filename, data)
            (target/'kit'/filename).write_bytes(packaged)
            bindings.append(dict(source=str(source.relative_to(REPO)), packaged='kit/'+filename,
                                 source_sha256=digest(data), packaged_sha256=digest(packaged), transform=transform))
        (target/'inspection').mkdir()
        (target/'inspection/__init__.py').write_text('"""Bundled original-mesh inspection; see BUNDLE_PROVENANCE.json."""\n')
        for filename in INSPECTION_FILES:
            source = REPO/'src'/filename
            data = source.read_bytes()
            (target/'inspection'/filename).write_bytes(data)
            bindings.append(dict(source=str(source.relative_to(REPO)), packaged='inspection/'+filename,
                                 source_sha256=digest(data), packaged_sha256=digest(data), transform='byte-identical'))
        for subdir in ('repair_runtime', 'repair_runtime/contact_edit'):
            (target/subdir).mkdir()
            # Empty initializers keep importing the bridge inside Blender free
            # of external solver dependencies. Only the worker imports them.
            (target/subdir/'__init__.py').write_bytes(b'')
            bindings.append(dict(source=None, packaged=subdir+'/__init__.py',
                                 source_sha256=None, packaged_sha256=digest(b''),
                                 transform='generated-empty-package-initializer'))
        repair_sources = [(REPO/'src'/f, 'repair_runtime/'+f) for f in REPAIR_FILES]
        repair_sources += [(REPO/'src/contact_edit'/f, 'repair_runtime/contact_edit/'+f)
                           for f in REPAIR_CONTACT_FILES]
        for source, relative in repair_sources:
            data = source.read_bytes()
            (target/relative).write_bytes(data)
            bindings.append(dict(source=str(source.relative_to(REPO)), packaged=relative,
                                 source_sha256=digest(data), packaged_sha256=digest(data),
                                 transform='byte-identical'))
        shutil.copyfile(REPO/'LICENSE', target/'LICENSE')
        if (HERE/'README.md').is_file():
            readme = (HERE/'README.md').read_text().replace('../../docs/hand-inspection.md', 'HAND_INSPECTION.md').replace('../../docs/collision.md', 'COLLISION.md').replace('../../tools/repair/README.md', 'LOCAL_REPAIR.md')
            (target/'README.md').write_text(readme)
        inspection_guide = REPO/'docs/hand-inspection.md'
        if inspection_guide.is_file():
            shutil.copyfile(inspection_guide, target/'HAND_INSPECTION.md')
        collision_guide = REPO/'docs/collision.md'
        if collision_guide.is_file():
            shutil.copyfile(collision_guide, target/'COLLISION.md')
        shutil.copyfile(REPO/'tools/repair/README.md', target/'LOCAL_REPAIR.md')
        (target/'BUNDLE_PROVENANCE.json').write_text(json.dumps(dict(schema_version=1, package=name,
            model_assets_bundled=False, files=bindings), indent=2)+'\n')
        manifest = tomllib.loads((target/'blender_manifest.toml').read_text())
        archive = out/(name+'-'+manifest['version']+'.zip')
        with zipfile.ZipFile(archive, 'x', compression=zipfile.ZIP_DEFLATED) as z:
            for path in sorted(target.rglob('*')):
                if not path.is_file():
                    continue
                info = zipfile.ZipInfo(path.relative_to(target).as_posix(), date_time=(2026, 1, 1, 0, 0, 0))
                info.compress_type = zipfile.ZIP_DEFLATED
                info.external_attr = 0o100644 << 16
                z.writestr(info, path.read_bytes())
        records.append(dict(package=name, zip=str(archive), sha256=digest(archive.read_bytes()),
                            bytes=archive.stat().st_size, staging=str(target), bundled_files=len(bindings)))
    report = dict(schema_version=1, packages=records,
                  note='Local research packages retain repository LICENSE; no model assets, external solver runtimes or inference libraries bundled')
    (out/'build_manifest.json').write_text(json.dumps(report, indent=2)+'\n')
    return report


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', required=True, help='New output directory, never overwritten')
    print(json.dumps(build(p.parse_args().out), indent=2))
