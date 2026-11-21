import json

import numpy as np
import pytest

from grail.retargeting.manipulation import handoff


@pytest.mark.parametrize(
    ("meshes", "support"),
    [
        ([], "native"),
        ([{"role": "floor"}], "native"),
        ([{"role": "ground"}, {"role": "floor"}], "native"),
        ([{"role": "floor"}, {"role": "table"}], "table-block"),
        ([{"name": "legacy_support"}], "table-block"),
    ],
)
def test_native_support_selection(meshes, support):
    motion = dict(
        qpos=np.zeros((2, 50)), fps=24, timestamps=np.arange(2) / 24,
        dof_names=np.arange(43).astype(str), qposadr=np.arange(7, 50),
        obj_R=np.tile(np.eye(3), (2, 1, 1)), obj_t=np.zeros((2, 3)),
        object_scale=np.ones(3), static_scene_json=json.dumps({"meshes": meshes}),
    )
    summary = handoff.check_native(motion)
    assert summary["support"] == support
    assert summary["support_meshes"] == len(meshes)
