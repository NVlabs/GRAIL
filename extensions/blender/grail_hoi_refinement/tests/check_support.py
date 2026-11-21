"""Native invariants for the addon-only Stage 1 path (no Blender required)."""

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

_FILE = Path(__file__).resolve().parents[1] / "_support.py"
_SPEC = importlib.util.spec_from_file_location("hoi_addon_support_test", _FILE)
support = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(support)


def fixture():
    n = 6
    motion = dict(poses=np.zeros((n, 66), dtype=np.float32),
                  trans=np.arange(n * 3, dtype=np.float32).reshape(n, 3) / 100,
                  left_hand_pose=np.zeros((n, 45), dtype=np.float32),
                  right_hand_pose=np.zeros((n, 45), dtype=np.float32),
                  betas=np.zeros(10), human_scale=np.array(1.),
                  obj_R=np.tile(np.eye(3), (n, 1, 1)), obj_t=np.zeros((n, 3)),
                  obj_scale=np.array([1., 2., 3.]), fps=np.array(24.))
    # Nontrivial camera orientation catches world-z assumptions.
    scene = dict(camera_R=np.array([[0., 0., -1.], [0., 1., 0.], [1., 0., 0.]]),
                 camera_K=np.array([[500., 0., 320.], [0., 500., 240.], [0., 0., 1.]]),
                 camera_t=np.zeros(3), image_width=np.array(640), image_height=np.array(480),
                 object_vertices=np.eye(3), object_faces=np.array([[0, 1, 2]], dtype=np.int32))
    args = dict(axis_world=[1., 0., 0.], offset_m=.02, bound_m=.03,
                contact_frame=2, prefix_end=3, translation_tolerance=1e-6,
                rotation_tolerance=1e-6, evidence="Synthetic camera and independently selected event")
    return motion, scene, args


def check_constant_depth_preserves_every_frozen_byte_and_input():
    motion, scene, args = fixture()
    before = {k: v.tobytes() for k, v in motion.items()}
    candidate, report = support.constant_depth_candidate(motion, scene, **args)
    assert report["status"] == "UNREVIEWED"
    assert report["contact_collision_acceptance"] is False
    assert candidate["trans"].dtype == motion["trans"].dtype
    assert np.allclose(candidate["trans"] - motion["trans"], [.02, 0, 0], atol=2e-8)
    assert np.allclose(np.diff(candidate["trans"], axis=0), np.diff(motion["trans"], axis=0), atol=2e-8)
    for name in motion:
        assert motion[name].tobytes() == before[name]
        if name != "trans":
            assert candidate[name].dtype == motion[name].dtype
            assert candidate[name].shape == motion[name].shape
            assert candidate[name].tobytes() == before[name]


def check_stationarity_requires_entire_prefix(test, mutation):
    motion, scene, args = fixture()
    if mutation == "translation":
        motion["obj_t"][1, 1] = .002
    else:
        angle = .01
        motion["obj_R"][1] = [[np.cos(angle), -np.sin(angle), 0],
                               [np.sin(angle), np.cos(angle), 0], [0, 0, 1]]
    with test.assertRaisesRegex(ValueError, "stationary-prefix"):
        support.constant_depth_candidate(motion, scene, **args)


INVALID_DECLARATIONS = [
    ({"axis_world": [0, 0, 1]}, "Declared depth axis"),
    ({"offset_m": .04}, "declared positive bound"),
    ({"offset_m": 0.}, "nonzero"),
    ({"contact_frame": 4}, "valid contact frame"),
    ({"evidence": ""}, "source evidence"),
]


def check_missing_or_inconsistent_declaration_rejected(test, change, match):
    motion, scene, args = fixture()
    args.update(change)
    with test.assertRaisesRegex(ValueError, match):
        support.constant_depth_candidate(motion, scene, **args)


def check_review_note_is_explicit_and_hash_bound_without_overwrite(test, tmp_path):
    motion, _, _ = fixture()
    candidate = tmp_path / "candidate.npz"
    np.savez_compressed(candidate, **motion)
    note_path = tmp_path / "manual_review.json"
    candidate_hash = support.digest(candidate)
    args = dict(reviewer="Test reviewer", status="PARTIAL", scope="Transport only",
                evidence="Synthetic roundtrip check", notes="Contact quality unreviewed")
    support.export_review_note(candidate, note_path, **args)
    note = json.loads(note_path.read_text())
    assert note["candidate"]["sha256"] == candidate_hash
    assert note["status"] == "PARTIAL" and not note["plugin_inferred_approval"]
    assert support.digest(candidate) == candidate_hash
    with test.assertRaises(FileExistsError):
        support.export_review_note(candidate, note_path, **args)


class NativeInvariantTests(unittest.TestCase):
    def test_frozen_fields(self):
        check_constant_depth_preserves_every_frozen_byte_and_input()

    def test_stationarity(self):
        for mutation in ("translation", "rotation"):
            with self.subTest(mutation=mutation):
                check_stationarity_requires_entire_prefix(self, mutation)

    def test_declarations(self):
        for change, match in INVALID_DECLARATIONS:
            with self.subTest(change=change):
                check_missing_or_inconsistent_declaration_rejected(self, change, match)

    def test_review(self):
        with tempfile.TemporaryDirectory() as directory:
            check_review_note_is_explicit_and_hash_bound_without_overwrite(self, Path(directory))


if __name__ == "__main__":
    unittest.main()
