from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
COMMANDS = REPO_ROOT / "imports" / "SONIC" / "gear_sonic" / "envs" / "manager_env" / "mdp" / "commands.py"


def _paired_reset_branch() -> str:
    source = COMMANDS.read_text()
    start = source.index("elif self.cfg.use_paired_motions:")
    end = source.index("elif self._multi_object_mode:", start)
    return source[start:end]


def test_paired_motion_resets_resample_start_times() -> None:
    branch = _paired_reset_branch()

    assert "_sample_start_time_steps_for_current_motions" in branch
    assert "self.motion_start_time_steps[env_ids]" in branch


def test_current_motion_start_time_sampling_applies_contact_window() -> None:
    source = COMMANDS.read_text()
    start = source.index("def _sample_start_time_steps_for_current_motions")
    end = source.index("\n    def ", start + 4)
    helper = source[start:end]

    assert "sample_time_steps" in helper
    assert "sample_from_n_initial_frames" in helper
    assert "start_from_first_frame" in helper
    assert "_enforce_valid_start_window" in helper

    # The contact bias is shared by all reset branches after their start times are chosen.
    start = source.index("def _resample_command(")
    reset = source[start:source.index("\n    def ", start + 4)]
    contact = reset.index("self.motion_start_time_steps[env_ids] = self._sample_before_contact(")
    assert reset.index("self._sample_from_reset_start_cache(") < contact
    assert "self.cfg.sample_before_contact" in reset[:contact]
    assert "not self.is_evaluating" in reset[:contact]
