"""Read-only object tracking diagnostics for stochastic training, in world space.

Samples the simulator at recorder post-step: after physics/rewards, before episode
reset and before command advancement. The target is the same current command pose
used by tracking rewards, not a nearest point/time. No state/action/RNG mutation.
"""
from __future__ import annotations


def pose_errors(actual_pos, actual_quat, target_pos, target_quat, valid):
    """Per-sample position norm [m] and shortest quaternion angle [rad], WXYZ.

    Missing scene slots are excluded, even if their accessor fallback equals the
    reference. A nonfinite valid pose or zero quaternion is an explicit error.
    """
    import torch
    if actual_pos.shape != target_pos.shape or actual_quat.shape != target_quat.shape:
        raise ValueError('Actual/reference pose shape mismatch')
    if actual_pos.shape[:-1] != actual_quat.shape[:-1] or actual_pos.shape[-1] != 3 or actual_quat.shape[-1] != 4 or valid.shape != actual_pos.shape[:-1]:
        raise ValueError('Pose/valid mask shape mismatch')
    pos = actual_pos.detach()[valid].to(torch.float64)
    ref_pos = target_pos.detach()[valid].to(torch.float64)
    quat = actual_quat.detach()[valid].to(torch.float64)
    ref_quat = target_quat.detach()[valid].to(torch.float64)
    # Finite/degenerate validation is batched at the PPO boundary, avoiding a
    # host/device synchronization on every simulator control step.
    pos_error = torch.linalg.vector_norm(pos - ref_pos, dim=-1)
    qnorm = torch.linalg.vector_norm(quat, dim=-1, keepdim=True)
    rnorm = torch.linalg.vector_norm(ref_quat, dim=-1, keepdim=True)
    quat = quat / qnorm
    ref_quat = ref_quat / rnorm
    dot = (quat * ref_quat).sum(dim=-1, keepdim=True)
    ref_quat = torch.where(dot < 0, -ref_quat, ref_quat)
    angle = 4.0 * torch.atan2(torch.linalg.vector_norm(quat - ref_quat, dim=-1),
                            torch.linalg.vector_norm(quat + ref_quat, dim=-1))
    return pos_error, angle


class ObjectTrackingAccumulator:
    """Per-PPO and current-worker-attempt statistics; one GPU→CPU per PPO."""
    def __init__(self):
        self.window = None
        self.total = None
        self.control_steps = 0
        self.total_control_steps = 0
        self.first_ppo_step = None

    def add(self, actual_pos, actual_quat, target_pos, target_quat, valid):
        import torch
        position, orientation = pose_errors(actual_pos, actual_quat, target_pos, target_quat, valid)
        values = torch.stack((position.sum(), position.square().sum(), orientation.sum(),
                              orientation.square().sum(), position.new_tensor(position.numel()),
                              position.new_tensor(valid.numel() - position.numel())))
        self.window = values if self.window is None else self.window + values
        self.control_steps += 1

    def sample_env(self, env):
        from gear_sonic.envs.manager_env.mdp.utils import get_all_object_current_state
        command = env.command_manager.get_term('motion')
        max_objects = int(getattr(command.motion_lib, 'max_num_objects', 1) or 1)
        if max_objects != 1:
            raise ValueError('This batch diagnostic requires exactly one dynamic tracked object')
        actual_pos, actual_quat, _, _, valid = get_all_object_current_state(env, command, max_objects=1)
        self.add(actual_pos, actual_quat, command.object_root_pos[:, :1],
                 command.object_root_quat[:, :1], valid)

    def flush(self, ppo_step):
        import math
        if self.window is None:
            raise RuntimeError('No object tracking samples before PPO callback')
        self.total = self.window if self.total is None else self.total + self.window
        values = self.window.detach().cpu().tolist()
        total = self.total.detach().cpu().tolist()
        if not all(math.isfinite(v) for v in values + total):
            raise RuntimeError('Nonfinite object pose or invalid quaternion in tracking samples')
        if values[4] < 1:
            raise RuntimeError('No valid dynamic object pose samples; cannot report zero error')
        self.total_control_steps += self.control_steps
        if self.first_ppo_step is None:
            self.first_ppo_step = int(ppo_step)
        def stats(v):
            n = v[4]
            return {'object_position_error_mean_m': v[0] / n,
                    'object_position_error_rms_m': math.sqrt(max(0.0, v[1] / n)),
                    'object_orientation_error_mean_rad': v[2] / n,
                    'object_orientation_error_rms_rad': math.sqrt(max(0.0, v[3] / n)),
                    'valid_object_samples': int(n), 'missing_object_samples': int(v[5])}
        receipt = {'status': 'sampled', 'ppo_step': int(ppo_step),
                   'kind': 'stochastic_training_rollout',
                   'sampling': 'every_control_step_all_envs_before_reset_and_command_advance',
                   'reference': 'motion.object_root_pos/object_root_quat at current command time',
                   'actual': 'get_all_object_current_state dynamic root pose',
                   'includes_terminal_frames': True, 'interaction_mask_filter': False,
                   'aggregation': 'sample_weighted; latest_window=one_PPO_rollout; cumulative=current_worker_attempt',
                   'cumulative_scope': 'current_worker_attempt_only',
                   'first_ppo_step': self.first_ppo_step,
                   'cumulative_from_step': self.first_ppo_step,
                   'cumulative_through_step': int(ppo_step),
                   'prior_attempt_samples_included': False,
                   'control_steps': self.control_steps, 'total_control_steps': self.total_control_steps,
                   'latest_window': stats(values), 'cumulative': stats(total)}
        self.window = None
        self.control_steps = 0
        return receipt
