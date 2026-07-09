"""Tests for Genesis-style backflip tasks."""

import pytest
import torch

import mjlab.tasks  # noqa: F401
from mjlab.envs import ManagerBasedRlEnv
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.tasks.registry import list_tasks, load_env_cfg, load_rl_cfg

BACKFLIP_TASK_IDS = (
  "Mjlab-Backflip-Flat-Unitree-Go2",
  "Mjlab-Backflip-Flat-BPX",
)

EXPECTED_ACTION_ORDER = {
  "Mjlab-Backflip-Flat-Unitree-Go2": [
    "FL_hip_joint",
    "FL_thigh_joint",
    "FL_calf_joint",
    "FR_hip_joint",
    "FR_thigh_joint",
    "FR_calf_joint",
    "RL_hip_joint",
    "RL_thigh_joint",
    "RL_calf_joint",
    "RR_hip_joint",
    "RR_thigh_joint",
    "RR_calf_joint",
  ],
  "Mjlab-Backflip-Flat-BPX": [
    "fl_hip_roll_joint",
    "fl_hip_pitch_joint",
    "fl_knee_joint",
    "fr_hip_roll_joint",
    "fr_hip_pitch_joint",
    "fr_knee_joint",
    "hl_hip_roll_joint",
    "hl_hip_pitch_joint",
    "hl_knee_joint",
    "hr_hip_roll_joint",
    "hr_hip_pitch_joint",
    "hr_knee_joint",
  ],
}


def test_backflip_tasks_are_registered() -> None:
  tasks = set(list_tasks())

  for task_id in BACKFLIP_TASK_IDS:
    assert task_id in tasks


@pytest.mark.parametrize("task_id", BACKFLIP_TASK_IDS)
def test_backflip_tasks_have_expected_training_cfg(task_id: str) -> None:
  cfg = load_env_cfg(task_id)

  assert cfg.scene.terrain is not None
  assert cfg.scene.terrain.terrain_type == "plane"
  assert "robot" in cfg.scene.entities
  assert {sensor.name for sensor in cfg.scene.sensors} == {"feet_ground_contact"}

  assert cfg.episode_length_s == 2.0
  assert cfg.decimation == 4
  assert set(cfg.terminations) == {"time_out"}

  action = cfg.actions["joint_pos"]
  assert isinstance(action, JointPositionActionCfg)
  assert action.scale == 0.5

  actor_terms = cfg.observations["actor"].terms
  for term_name in (
    "base_ang_vel",
    "projected_gravity",
    "joint_pos",
    "joint_vel",
    "actions",
    "previous_actions",
    "phase",
  ):
    assert term_name in actor_terms

  for reward_name in (
    "ang_vel_y",
    "ang_vel_z",
    "lin_vel_z",
    "orientation_control",
    "feet_height_before_backflip",
    "height_control",
    "actions_symmetry",
    "gravity_y",
    "feet_distance",
    "action_rate",
  ):
    assert reward_name in cfg.rewards

  rl_cfg = load_rl_cfg(task_id)
  assert rl_cfg.max_iterations == 1000
  assert rl_cfg.num_steps_per_env == 24
  assert rl_cfg.clip_actions == 100.0


@pytest.mark.parametrize("task_id", BACKFLIP_TASK_IDS)
def test_backflip_play_cfg_disables_corruption(task_id: str) -> None:
  cfg = load_env_cfg(task_id, play=True)

  assert cfg.scene.num_envs == 1
  assert cfg.observations["actor"].enable_corruption is False
  assert cfg.events["reset_base"].params["pose_range"]["yaw"] == (0.0, 0.0)
  assert cfg.events["reset_robot_joints"].params["position_range"] == (0.0, 0.0)


@pytest.mark.parametrize("task_id", BACKFLIP_TASK_IDS)
def test_backflip_env_resets_and_steps(task_id: str) -> None:
  cfg = load_env_cfg(task_id, play=True)
  env = ManagerBasedRlEnv(cfg=cfg, device="cpu", render_mode=None)
  try:
    obs, _ = env.reset()
    assert "actor" in obs
    joint_pos_action = env.action_manager.get_term("joint_pos")
    assert joint_pos_action.target_names == EXPECTED_ACTION_ORDER[task_id]

    action = torch.zeros(
      (env.num_envs, env.action_manager.total_action_dim),
      device=env.device,
    )
    obs, reward, terminated, truncated, _ = env.step(action)

    assert "actor" in obs
    assert reward.shape == (1,)
    assert terminated.shape == (1,)
    assert truncated.shape == (1,)
  finally:
    env.close()
