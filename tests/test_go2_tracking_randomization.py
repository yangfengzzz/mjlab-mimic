"""Tests for Go2 tracking sim-to-real randomization config."""

from pathlib import Path

import torch

from mjlab.asset_zoo.robots import GO2_ACTION_SCALE
from mjlab.asset_zoo.robots.unitree_go2 import go2_constants
from mjlab.entity import Entity
from mjlab.tasks.tracking.config.go2.env_cfgs import (
  unitree_go2_flat_tracking_env_cfg,
)
from mjlab.tasks.tracking.mdp import MotionCommandCfg
from mjlab.tasks.tracking.mdp.commands import MotionLoader


def test_go2_tracking_sim2real_randomization_events() -> None:
  cfg = unitree_go2_flat_tracking_env_cfg()

  expected_events = {
    "push_robot",
    "base_com",
    "encoder_bias",
    "foot_friction",
    "robot_body_mass",
    "joint_armature",
    "joint_damping",
    "joint_frictionloss",
    "pd_gains",
    "effort_limits",
  }
  assert expected_events <= cfg.events.keys()

  assert cfg.events["robot_body_mass"].params["field"] == "body_mass"
  assert cfg.events["robot_body_mass"].params["operation"] == "scale"
  assert cfg.events["robot_body_mass"].params["ranges"] == (0.95, 1.10)
  assert cfg.events["robot_body_mass"].params["shared_random"] is True
  assert cfg.events["robot_body_mass"].params["asset_cfg"].body_names == (".*",)

  assert cfg.events["joint_armature"].params["field"] == "dof_armature"
  assert cfg.events["joint_armature"].params["ranges"] == (0.75, 1.25)
  assert cfg.events["joint_damping"].params["field"] == "dof_damping"
  assert cfg.events["joint_damping"].params["ranges"] == (0.0, 0.08)
  assert cfg.events["joint_frictionloss"].params["field"] == "dof_frictionloss"
  assert cfg.events["joint_frictionloss"].params["ranges"] == (0.0, 0.12)

  assert cfg.events["pd_gains"].params["kp_range"] == (0.85, 1.15)
  assert cfg.events["pd_gains"].params["kd_range"] == (0.75, 1.25)
  assert cfg.events["pd_gains"].params["operation"] == "scale"
  assert cfg.events["pd_gains"].params["asset_cfg"].actuator_ids == slice(None)

  assert cfg.events["effort_limits"].params["effort_limit_range"] == (0.85, 1.0)
  assert cfg.events["effort_limits"].params["operation"] == "scale"
  assert cfg.events["effort_limits"].params["asset_cfg"].actuator_ids == slice(None)

  assert cfg.events["encoder_bias"].params["bias_range"] == (-0.02, 0.02)
  assert cfg.events["foot_friction"].params["ranges"] == (0.3, 1.2)
  assert cfg.events["foot_friction"].params["shared_random"] is True
  assert cfg.events["base_com"].params["asset_cfg"].body_names == ("trunk",)


def test_go2_tracking_play_keeps_startup_randomization() -> None:
  cfg = unitree_go2_flat_tracking_env_cfg(play=True)

  assert "push_robot" not in cfg.events
  assert "robot_body_mass" in cfg.events
  assert "pd_gains" in cfg.events
  assert "effort_limits" in cfg.events
  assert cfg.events["encoder_bias"].params["bias_range"] == (-0.02, 0.02)


def test_go2_tracking_no_state_estimation_uses_same_randomization() -> None:
  cfg = unitree_go2_flat_tracking_env_cfg(has_state_estimation=False)

  assert "motion_anchor_pos_b" not in cfg.observations["actor"].terms
  assert "base_lin_vel" not in cfg.observations["actor"].terms
  assert "robot_body_mass" in cfg.events
  assert "pd_gains" in cfg.events


def test_go2_tracking_action_scale_unchanged() -> None:
  cfg = unitree_go2_flat_tracking_env_cfg()

  assert cfg.actions["joint_pos"].scale == GO2_ACTION_SCALE
  assert GO2_ACTION_SCALE == {
    ".*hip_.*": 0.29375,
    ".*thigh_.*": 0.29375,
    ".*calf_.*": 0.28125,
  }


def test_go2_tracking_enables_robot_self_collisions() -> None:
  cfg = unitree_go2_flat_tracking_env_cfg(has_state_estimation=False)
  robot = Entity(cfg.scene.entities["robot"])
  model = robot.spec.compile()

  for geom_id in range(model.ngeom):
    geom_name = model.geom(geom_id).name
    if "_collision" in geom_name:
      assert model.geom_contype[geom_id] == 1
      assert model.geom_conaffinity[geom_id] == 1


def test_go2_motion_loader_maps_compact_body_order() -> None:
  cfg = unitree_go2_flat_tracking_env_cfg()
  motion_cmd = cfg.commands["motion"]
  assert isinstance(motion_cmd, MotionCommandCfg)

  robot = Entity(go2_constants.get_go2_robot_cfg())
  body_ids, _ = robot.find_bodies(motion_cmd.body_names)

  loader = MotionLoader(
    str(Path("motion_file/go2_side_flip_motion.npz").resolve()),
    torch.tensor(body_ids, dtype=torch.long),
    motion_cmd.body_names,
    robot.body_names,
  )

  assert loader.body_pos_w.shape[1] == len(motion_cmd.body_names)
