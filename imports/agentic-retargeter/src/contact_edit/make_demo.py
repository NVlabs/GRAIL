"""Create a small synthetic native fixture; no G1 asset or case is implied."""
from pathlib import Path
import argparse
import json
import numpy as np
import mujoco


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', required=True)
    parser.add_argument('--fps', type=float, default=24)
    args = parser.parse_args()
    if not np.isfinite(args.fps) or args.fps <= 0:
        raise ValueError('fps must be positive')
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=False)
    xml = '''<mujoco><compiler angle="radian"/><worldbody>
    <body name="base"><freejoint name="root"/><geom type="sphere" size=".1" mass="1"/>
      <body name="moving_pad"><joint name="pad_hinge" axis="0 0 1" range="-.6 .6"/>
        <geom type="capsule" fromto="0 0 0 .1 0 0" size=".01" mass=".1"/>
      </body>
      <body name="frozen_arm" pos="0 1 0"><joint name="frozen_hinge" axis="1 0 0" range="-1 1"/>
        <geom type="sphere" size=".02" mass=".1"/>
      </body></body></worldbody></mujoco>'''
    (out / 'robot.xml').write_text(xml)
    model = mujoco.MjModel.from_xml_string(xml)
    names = np.array([mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j) for j in range(model.njnt)])
    np.savez_compressed(out / 'model.npz', joint_names=names,
        jnt_type=model.jnt_type, jnt_limited=model.jnt_limited,
        jnt_qposadr=model.jnt_qposadr, jnt_range=model.jnt_range)
    qpos = np.tile(model.qpos0, (21, 1)); qpos[:, 8] = .17
    native = dict(qpos=qpos, fps=np.asarray(args.fps), timestamps=np.arange(21)/args.fps,
        obj_R=np.tile(np.eye(3), (21, 1, 1)), obj_t=np.zeros((21, 3)), object_scale=np.asarray(1.),
        dof_names=names[1:], qposadr=model.jnt_qposadr[1:],
        fixture_note=np.asarray('Synthetic geometry fixture; not a reviewed grasp'))
    np.savez_compressed(out / 'native.npz', **native)
    radial = np.array([np.cos(.3), np.sin(.3), 0.])
    target = radial * .1
    vertices = np.array([target, target+[0,0,.02], target+.02*radial])
    np.savez_compressed(out / 'object_mesh.npz', vertices=vertices, faces=np.array([[0,1,2]]))
    pose = dict(schema_version=1, model_xml='robot.xml', trajectory_npz='native.npz',
        object_mesh_npz='object_mesh.npz', frame=10,
        editable_joints={'pad_hinge': {'delta_bounds_rad': [-.4,.4]}},
        contacts=[dict(name='authored_material_contact', body_name='moving_pad', point_local=[.1,0,0],
            normal_local=[0,-1,0], object_face_id=0, object_barycentric=[1,0,0], normal_weight=0.)],
        solver=dict(max_evaluations=60, pose_weight=0))
    (out / 'pose.json').write_text(json.dumps(pose, indent=2)+'\n')
    plan = dict(schema_version=1, channels=[dict(joint='pad_hinge', window=[0,20],
        keys=[[0,0],[6,.3],[13,-.15],[20,0]], interpolation='BEZIER')])
    (out / 'curves.json').write_text(json.dumps(plan, indent=2)+'\n')
    plan['channels'][0]['keys'] = [[i,0] for i in range(21)]
    plan['channels'][0]['interpolation'] = 'LINEAR'
    (out / 'noop_curves.json').write_text(json.dumps(plan, indent=2)+'\n')
    print(out)


if __name__ == '__main__':
    main()
