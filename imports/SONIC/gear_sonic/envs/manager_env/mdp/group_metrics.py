"""Per-motion diagnostics sampled after physics and before reset/command advance.

Training success is completed-episode nonfailure under random reference starts;
it is distinct from periodic deterministic full-reference tracking success.
"""
from pathlib import Path
import json
from gear_sonic.envs.manager_env.mdp.tracking_metrics import pose_errors


class MotionAccumulator:
    def __init__(self, motion_ids):
        if not motion_ids or len(motion_ids) != len(set(motion_ids)):
            raise ValueError('Unique motion IDs required')
        self.motion_ids = list(motion_ids)
        self.window = self.total = None
        self.control_steps = self.total_control_steps = 0
        self.first_step = None
        self.first_env_counts = self.last_env_counts = None

    def add(self, group_ids, actual_pos, actual_quat, target_pos, target_quat, valid,
            terminated, truncated, reference_completed):
        import torch
        if valid.ndim != 2 or valid.shape[1] != 1 or group_ids.shape != valid.shape[:1]:
            raise ValueError('One physical object per environment required')
        n = len(self.motion_ids)
        position, orientation = pose_errors(actual_pos, actual_quat, target_pos, target_quat, valid)
        ids = group_ids[:, None].expand_as(valid)[valid]
        values = torch.zeros((n, 9), device=position.device, dtype=torch.float64)
        for col, samples in enumerate((position, position.square(), orientation, orientation.square(), torch.ones_like(position))):
            values[:, col].scatter_add_(0, ids, samples)
        values[:, 5].scatter_add_(0, group_ids, (~valid[:, 0]).to(torch.float64))
        done = terminated | truncated
        for col, samples in ((6, done), (7, terminated), (8, done & reference_completed & ~terminated)):
            values[:, col].scatter_add_(0, group_ids, samples.to(torch.float64))
        self.last_env_counts = torch.bincount(group_ids, minlength=n)
        if self.first_env_counts is None:
            self.first_env_counts = self.last_env_counts.clone()
        self.window = values if self.window is None else self.window + values
        self.control_steps += 1

    @staticmethod
    def statistics(v):
        import math
        samples, episodes, failures = v[4], v[6], v[7]
        return {'valid_object_samples': int(samples), 'missing_object_samples': int(v[5]),
                'object_position_error_mean_m': v[0] / samples if samples else None,
                'object_position_error_rms_m': math.sqrt(max(0.0, v[1] / samples)) if samples else None,
                'object_orientation_error_mean_rad': v[2] / samples if samples else None,
                'object_orientation_error_rms_rad': math.sqrt(max(0.0, v[3] / samples)) if samples else None,
                'completed_episodes': int(episodes), 'failure_episodes': int(failures),
                'nonfailure_episodes': int(episodes - failures),
                'completed_episode_nonfailure_rate': 1 - failures / episodes if episodes else None,
                'reference_end_nonfailure_episodes': int(v[8])}

    def flush(self, step):
        import math
        if self.window is None:
            raise RuntimeError('No per-motion samples')
        self.total = self.window if self.total is None else self.total + self.window
        window, total = self.window.cpu().tolist(), self.total.cpu().tolist()
        if not all(math.isfinite(x) for values in window + total for x in values):
            raise RuntimeError('Nonfinite per-motion sample')
        first, last = self.first_env_counts.cpu().tolist(), self.last_env_counts.cpu().tolist()
        self.first_step = int(step) if self.first_step is None else self.first_step
        self.total_control_steps += self.control_steps
        result = {'schema': 'contactfirst.objectgroup.training_metrics.v1', 'ppo_step': int(step),
            'kind': 'stochastic_training_rollout', 'cumulative_scope': 'current_worker_attempt_only',
            'sampling': 'all_envs_every_control_step_after_physics_before_reset_and_command_advance',
            'success_definition': '1 - native_failure_episodes / completed_episodes; random starts and timeouts are nonfailures, not full-reference success',
            'initial_reset_included': False, 'prior_attempt_samples_included': False,
            'first_ppo_step': self.first_step, 'control_steps': self.control_steps,
            'total_control_steps': self.total_control_steps,
            'motions': {key: {'motion_id': key, 'first_observed_env_count': first[i], 'last_observed_env_count': last[i],
                'latest_window': self.statistics(window[i]), 'cumulative': self.statistics(total[i])}
                for i, key in enumerate(self.motion_ids)}}
        self.window, self.control_steps = None, 0
        return result


def motion_ids_from_roster(path):
    """Motion aliases of a frozen cohort roster, in roster order."""
    roster = json.loads(Path(path).read_text())
    ids = [record['alias'] for record in roster['records']]
    if not ids or len(ids) != len(set(ids)):
        raise ValueError('Roster must list unique motion aliases')
    return ids


def make_recorders(output_root, roster_manifest):
    import torch
    from isaaclab.managers.recorder_manager import RecorderTerm, RecorderTermCfg, RecorderManagerBaseCfg, DatasetExportMode
    from gear_sonic.envs.manager_env.mdp.utils import get_all_object_current_state
    motion_ids = motion_ids_from_roster(roster_manifest)

    class PerMotionRecorder(RecorderTerm):
        def __init__(self, cfg, env):
            super().__init__(cfg, env)
            self.env = env
            self.metrics = MotionAccumulator(motion_ids)
            self.key_map = None

        def record_post_step(self):
            command = self.env.command_manager.get_term('motion')
            library = command.motion_lib
            # Native motion IDs address the currently loaded subset, not the
            # original unique-key array. Keep attribution correct across resets.
            if self.key_map is None:
                keys = [str(k) for k in library._motion_data_keys]
                if set(keys) != set(motion_ids) or len(keys) != len(motion_ids):
                    raise ValueError('Actual loaded motion keys differ from frozen group')
                self.key_map = torch.tensor([motion_ids.index(k) for k in keys], device=command.motion_ids.device, dtype=torch.long)
            ids = self.key_map[library._curr_motion_ids[command.motion_ids].long()]
            actual_pos, actual_quat, _, _, valid = get_all_object_current_state(self.env, command, max_objects=1)
            manager = self.env.termination_manager
            ref_complete = command.motion_start_time_steps + command.time_steps + 1 >= library.get_time_step_total(command.motion_ids)
            self.metrics.add(ids, actual_pos, actual_quat, command.object_root_pos[:, :1], command.object_root_quat[:, :1], valid,
                manager.terminated, manager.time_outs, ref_complete)
            return None, None

    # Configclass collects instance attributes as recorder terms. This does not
    # create render products or modify actions, resets, sampling, or RNG state.
    cfg = RecorderManagerBaseCfg()
    cfg.group_metrics = RecorderTermCfg(class_type=PerMotionRecorder)
    cfg.dataset_export_mode = DatasetExportMode.EXPORT_NONE
    cfg.export_in_record_pre_reset = False
    cfg.export_in_close = False
    return cfg
