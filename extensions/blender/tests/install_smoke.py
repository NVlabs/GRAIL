"""Test built ZIPs using Blender's real Install from Disk operator.

Use a fresh profile and a fresh output directory, e.g.:
  BLENDER_USER_CONFIG=/tmp/grail-test-profile/config \
  BLENDER_USER_SCRIPTS=/tmp/grail-test-profile/scripts \
  BLENDER_USER_EXTENSIONS=/tmp/grail-test-profile/extensions \
  /path/to/blender --background --factory-startup --python-exit-code 1 \
    --python extensions/blender/tests/install_smoke.py -- \
    --build /path/to/built-zips --out /path/to/new/test-output
"""
import argparse
import hashlib
import importlib
import json
import os
from pathlib import Path
import sys
import types

import addon_utils
import bpy
import numpy as np
from mathutils import Quaternion


def exercise_installed_native(fixtures, out, modules):
    """Run native kit imports/operators under the actual bl_ext namespace."""
    for fullname, unused in modules:
        addon_utils.enable(fullname, default_set=True)
    # This process owns disposable factory-startup data, never a user scene.
    for ob in list(bpy.context.scene.objects):
        bpy.data.objects.remove(ob, do_unlink=True)
    assert hasattr(bpy.context.scene, "grail_hoi")
    bpy.context.window.scene = bpy.data.scenes.new('Installed_robot_test')
    settings = bpy.context.scene.grail_g1
    settings.native_path = str(fixtures / 'robot' / 'native.npz')
    settings.model_path = str(fixtures / 'robot' / 'model.npz')
    settings.plan_path = str(fixtures / 'robot' / 'curves.json')
    settings.output_dir = str(out / 'installed_robot_candidate')
    assert bpy.ops.grail_g1.create_controls() == {'FINISHED'}
    assert bpy.ops.grail_g1.export_native() == {'FINISHED'}
    with np.load(out / 'installed_robot_candidate' / 'candidate_UNREVIEWED.npz') as candidate:
        with np.load(fixtures / 'robot' / 'native.npz') as source:
            assert not np.array_equal(candidate['qpos'][:, 7], source['qpos'][:, 7])
            assert np.array_equal(candidate['qpos'][:, :7], source['qpos'][:, :7])
            assert np.array_equal(candidate['qpos'][:, 8:], source['qpos'][:, 8:])
            for name in ('fps', 'obj_R', 'obj_t', 'object_scale'):
                assert np.array_equal(candidate[name], source[name]), name
            assert str(candidate['status']) == 'UNREVIEWED'
    for fullname, unused in reversed(modules):
        addon_utils.disable(fullname, default_set=True)
    assert not any(getattr(h, '__module__', '').startswith('bl_ext.grail_test.')
                   for h in bpy.app.handlers.frame_change_post)
    return dict(
                robot_actual_curve_export=True, robot_frozen_coordinates=True,
                cleanup=True, scope='Synthetic native transport, no contact-quality acceptance')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--build', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--fixtures', type=Path, help='Optional human/ and robot/ kit fixtures')
    args = parser.parse_args(sys.argv[sys.argv.index('--') + 1:])
    assert os.environ.get('BLENDER_USER_CONFIG'), 'An isolated Blender profile is required'
    assert (4, 2, 0) <= bpy.app.version < (4, 4, 0)
    args.out.mkdir(parents=True, exist_ok=False)
    build = json.loads((args.build / 'build_manifest.json').read_text())
    records = []
    sentinel_names = ('core', 'native_io', 'human_contact_core', 'human_numpy_model', 'human_key_runtime')
    sentinels = {name: types.ModuleType(name) for name in sentinel_names}
    for name, module in sentinels.items():
        assert name not in sys.modules
        sys.modules[name] = module
    # Install may legitimately add Blender repository paths. Neither extension
    # may put its vendored kit or the development checkout onto sys.path.
    repo_dir = args.out / 'extension_repository'
    repo_dir.mkdir()
    repo = bpy.context.preferences.extensions.repos.new(
        name='GRAIL isolated integration test', module='grail_test', custom_directory=str(repo_dir))
    cube = bpy.data.objects.get('Cube')
    text = bpy.data.texts.new('Keep_user_notes')
    expected_text = 'Existing user text must survive installation and lifecycle changes.'
    text.write(expected_text)
    other = bpy.data.scenes.new('Unrelated_user_scene')
    active = bpy.context.scene
    objects_before = sorted(bpy.data.objects.keys())
    modules = []
    for item in build['packages']:
        archive = Path(item['zip'])
        assert hashlib.sha256(archive.read_bytes()).hexdigest() == item['sha256']
        result = bpy.ops.extensions.package_install_files(
            'EXEC_DEFAULT', repo=repo.module, filepath=str(archive), enable_on_install=True)
        assert result == {'FINISHED'}, result
        fullname = 'bl_ext.' + repo.module + '.' + item['package']
        module = importlib.import_module(fullname)
        assert Path(module.__file__).is_relative_to(repo_dir)
        assert addon_utils.check(fullname)[1], fullname
        assert sorted(bpy.data.objects.keys()) == objects_before
        assert cube is bpy.data.objects.get('Cube')
        assert bpy.context.scene == active and other in bpy.data.scenes.values()
        assert bpy.data.texts['Keep_user_notes'].as_string() == expected_text
        assert all(sys.modules[name] is sentinel for name, sentinel in sentinels.items())
        assert not any('/kit' in path and str(repo_dir) in path for path in sys.path)
        modules.append((fullname, module))
        records.append(dict(package=item['package'], sha256=item['sha256'],
                            installed_module=fullname, zip_install=True,
                            user_scene_preserved=True, generic_import_isolation=True))
    assert hasattr(bpy.types.Scene, 'grail_hoi') and hasattr(bpy.types.Scene, 'grail_g1')
    for fullname, module in reversed(modules):
        before = (len(bpy.app.handlers.frame_change_post), len(bpy.app.handlers.load_pre))
        addon_utils.disable(fullname, default_set=True)
        addon_utils.enable(fullname, default_set=True)
        after = (len(bpy.app.handlers.frame_change_post), len(bpy.app.handlers.load_pre))
        assert after == before, (fullname, before, after)
        addon_utils.disable(fullname, default_set=True)
    assert not hasattr(bpy.types.Scene, 'grail_hoi')
    assert not hasattr(bpy.types.Scene, 'grail_g1')
    assert sorted(bpy.data.objects.keys()) == objects_before
    assert all(sys.modules[name] is sentinel for name, sentinel in sentinels.items())
    native_checks = exercise_installed_native(args.fixtures, args.out, modules) if args.fixtures else None
    assert all(sys.modules[name] is sentinel for name, sentinel in sentinels.items())
    result = dict(status='PASS', blender_version=bpy.app.version_string,
                  both_extensions_coexist=True, unregister_reregister_cleanup=True,
                  automatic_scene_script_execution=False, packages=records,
                  installed_native_operator_checks=native_checks,
                  scope='Packaging/lifecycle and optional synthetic native edits; real cases have separate tests')
    (args.out / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
    print('GRAIL_EXTENSION_INSTALL_SMOKE_PASS', json.dumps(result))


if __name__ == '__main__':
    main()
