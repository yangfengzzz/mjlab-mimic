"""Tests specific to motion tracking tasks."""

from pathlib import Path

import numpy as np
import pytest
import torch

from mjlab.asset_zoo.robots import G1_ACTION_SCALE
from mjlab.envs import ManagerBasedRlEnv
from mjlab.envs import mdp as envs_mdp
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.tasks.registry import list_tasks, load_env_cfg
from mjlab.tasks.tracking.mdp import MotionCommandCfg


@pytest.fixture(scope="module")
def tracking_task_ids() -> list[str]:
  """Get all tracking task IDs."""
  return [t for t in list_tasks() if "Tracking" in t]


@pytest.fixture(scope="module")
def g1_tracking_task_ids(tracking_task_ids: list[str]) -> list[str]:
  """Get all G1 tracking task IDs."""
  return [t for t in tracking_task_ids if "G1" in t]


def test_tracking_tasks_have_motion_command(tracking_task_ids: list[str]) -> None:
  """All tracking tasks should have a 'motion' command of type MotionCommandCfg."""
  for task_id in tracking_task_ids:
    cfg = load_env_cfg(task_id)

    assert "motion" in cfg.commands, f"Task {task_id} missing 'motion' command"

    motion_cmd = cfg.commands["motion"]
    assert isinstance(motion_cmd, MotionCommandCfg), (
      f"Task {task_id} motion command is not MotionCommandCfg"
    )


def test_tracking_tasks_have_self_collision_sensor(
  tracking_task_ids: list[str],
) -> None:
  """Tracking tasks with self-collision rewards should have the matching sensor."""
  for task_id in tracking_task_ids:
    cfg = load_env_cfg(task_id)

    if "self_collisions" not in cfg.rewards:
      continue

    assert cfg.scene.sensors is not None, f"Task {task_id} has no sensors"

    sensor_names = {s.name for s in cfg.scene.sensors}
    assert "self_collision" in sensor_names, (
      f"Task {task_id} missing self_collision sensor"
    )


def test_tracking_no_state_estimation_observations() -> None:
  """No-state-estimation tasks remove observations that depend on state estimation."""
  task_id = "Mjlab-Tracking-Flat-Unitree-G1-No-State-Estimation"

  # Test both training and play modes
  for play_mode in [False, True]:
    cfg = load_env_cfg(task_id, play=play_mode)
    mode_str = "play mode" if play_mode else "training mode"

    assert "actor" in cfg.observations, (
      f"Task {task_id} ({mode_str}) missing policy observations"
    )
    actor_terms = cfg.observations["actor"].terms

    assert "motion_anchor_pos_b" not in actor_terms, (
      f"Task {task_id} ({mode_str}) has motion_anchor_pos_b in policy, "
      "expected it to be removed for no-state-estimation variant"
    )
    assert "base_lin_vel" not in actor_terms, (
      f"Task {task_id} ({mode_str}) has base_lin_vel in policy, "
      "expected it to be removed for no-state-estimation variant"
    )


def test_bpx_tracking_uses_entity_base_velocity_observations() -> None:
  """BPX tracking should not depend on missing imu_* builtin sensors."""
  cfg = load_env_cfg("Mjlab-Tracking-Flat-BPX")

  for group_name in ("actor", "critic"):
    terms = cfg.observations[group_name].terms
    assert terms["base_lin_vel"].func is envs_mdp.base_lin_vel
    assert terms["base_ang_vel"].func is envs_mdp.base_ang_vel


def test_bpx_tracking_sets_contact_buffer_for_sideflip() -> None:
  """BPX side-flip contacts exceed the base tracking nconmax heuristic."""
  cfg = load_env_cfg("Mjlab-Tracking-Flat-BPX")

  assert cfg.sim.nconmax == 128
  assert cfg.sim.njmax == 512


def test_bpx_tracking_keeps_self_collision_reward_and_sensor() -> None:
  """BPX tracking should expose Go2-style self-collision training signal."""
  cfg = load_env_cfg("Mjlab-Tracking-Flat-BPX")

  assert "self_collisions" in cfg.rewards
  assert cfg.scene.sensors is not None
  sensor_names = {s.name for s in cfg.scene.sensors}
  assert "self_collision" in sensor_names


def _write_minimal_bpx_motion(path: Path) -> None:
  fps = np.asarray([50.0], dtype=np.float32)
  joint_pos = np.tile(np.asarray([[0.0, 0.6, -0.9] * 4], dtype=np.float32), (2, 1))
  joint_vel = np.zeros_like(joint_pos)
  body_pos_w = np.zeros((2, 17, 3), dtype=np.float32)
  body_pos_w[:, :, 2] = 0.42
  body_quat_w = np.zeros((2, 17, 4), dtype=np.float32)
  body_quat_w[:, :, 0] = 1.0
  body_lin_vel_w = np.zeros((2, 17, 3), dtype=np.float32)
  body_ang_vel_w = np.zeros((2, 17, 3), dtype=np.float32)
  np.savez(
    path,
    fps=fps,
    joint_pos=joint_pos,
    joint_vel=joint_vel,
    body_pos_w=body_pos_w,
    body_quat_w=body_quat_w,
    body_lin_vel_w=body_lin_vel_w,
    body_ang_vel_w=body_ang_vel_w,
  )


def test_bpx_self_collision_sensor_detects_bad_pose_without_default_false_positive(
  tmp_path: Path,
) -> None:
  """BPX self-collision should be silent at nominal pose and active for bad poses."""
  motion_file = tmp_path / "motion.npz"
  _write_minimal_bpx_motion(motion_file)

  cfg = load_env_cfg("Mjlab-Tracking-Flat-BPX")
  cfg.scene.num_envs = 1
  cfg.commands["motion"].motion_file = str(motion_file)
  cfg.events = {}
  cfg.observations = {}
  cfg.rewards = {}
  cfg.terminations = {}
  env = ManagerBasedRlEnv(cfg, device="cpu")

  try:
    robot = env.scene["robot"]
    sensor = env.scene["self_collision"]

    def set_pose_and_read_contacts(joint_pos: list[float]) -> float:
      root_state = torch.zeros((1, 13), device=env.device)
      root_state[:, 2] = 0.42
      root_state[:, 3] = 1.0
      robot.write_root_state_to_sim(root_state)
      robot.write_joint_state_to_sim(
        torch.tensor([joint_pos], dtype=torch.float32, device=env.device),
        torch.zeros((1, 12), device=env.device),
      )
      env.sim.forward()
      env.scene.update(env.sim.mj_model.opt.timestep)
      assert sensor.data.found is not None
      return float(sensor.data.found.sum().item())

    nominal_pose = [0.0, 0.6, -0.9] * 4
    colliding_pose = [
      0.87,
      2.3,
      -0.56,
      -0.87,
      2.3,
      -0.56,
      0.0,
      0.6,
      -0.9,
      0.0,
      0.6,
      -0.9,
    ]

    assert set_pose_and_read_contacts(nominal_pose) == 0.0
    assert set_pose_and_read_contacts(colliding_pose) > 0.0
  finally:
    env.close()


def test_tracking_play_disables_rsi_randomization() -> None:
  """Tracking play tasks should disable RSI randomization."""
  tracking_tasks = [
    "Mjlab-Tracking-Flat-Unitree-G1",
    "Mjlab-Tracking-Flat-Unitree-G1-No-State-Estimation",
    "Mjlab-Tracking-Flat-BPX",
    "Mjlab-Tracking-Flat-BPX-No-State-Estimation",
  ]

  for task_id in tracking_tasks:
    cfg = load_env_cfg(task_id, play=True)

    motion_cmd = cfg.commands["motion"]
    assert isinstance(motion_cmd, MotionCommandCfg), (
      f"Task {task_id} (play mode) motion command is not MotionCommandCfg"
    )

    assert motion_cmd.pose_range == {}, (
      f"Task {task_id} (play mode) has non-empty pose_range={motion_cmd.pose_range}, "
      "expected empty dict for disabled RSI"
    )
    assert motion_cmd.velocity_range == {}, (
      f"Task {task_id} (play mode) has non-empty velocity_range={motion_cmd.velocity_range}, "
      "expected empty dict for disabled RSI"
    )


def test_tracking_play_uses_start_sampling_mode() -> None:
  """Tracking play tasks should use sampling_mode='start'."""
  tracking_tasks = [
    "Mjlab-Tracking-Flat-Unitree-G1",
    "Mjlab-Tracking-Flat-Unitree-G1-No-State-Estimation",
    "Mjlab-Tracking-Flat-BPX",
    "Mjlab-Tracking-Flat-BPX-No-State-Estimation",
  ]

  for task_id in tracking_tasks:
    cfg = load_env_cfg(task_id, play=True)

    motion_cmd = cfg.commands["motion"]
    assert isinstance(motion_cmd, MotionCommandCfg), (
      f"Task {task_id} (play mode) motion command is not MotionCommandCfg"
    )

    assert motion_cmd.sampling_mode == "start", (
      f"Task {task_id} (play mode) sampling_mode={motion_cmd.sampling_mode}, expected 'start'"
    )


def test_g1_tracking_has_correct_action_scale(g1_tracking_task_ids: list[str]) -> None:
  """G1 tracking tasks should use G1_ACTION_SCALE."""
  for task_id in g1_tracking_task_ids:
    cfg = load_env_cfg(task_id)

    assert "joint_pos" in cfg.actions, f"Task {task_id} missing 'joint_pos' action"

    joint_pos_action = cfg.actions["joint_pos"]
    assert isinstance(joint_pos_action, JointPositionActionCfg), (
      f"Task {task_id} joint_pos action is not JointPositionActionCfg"
    )

    assert joint_pos_action.scale == G1_ACTION_SCALE, (
      f"Task {task_id} action scale mismatch, expected G1_ACTION_SCALE"
    )
