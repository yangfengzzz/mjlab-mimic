"""Unitree Go2 flat tracking environment configurations."""

from mjlab.asset_zoo.robots import (
  GO2_ACTION_SCALE,
  get_go2_robot_cfg,
)
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.tasks.tracking import mdp
from mjlab.tasks.tracking.mdp import MotionCommandCfg
from mjlab.tasks.tracking.tracking_env_cfg import make_tracking_env_cfg


def _add_go2_sim2real_randomization(cfg: ManagerBasedRlEnvCfg) -> None:
  """Add medium-strength Go2 tracking randomization for sim-to-real training."""
  cfg.events["robot_body_mass"] = EventTermCfg(
    mode="startup",
    func=mdp.randomize_field,
    domain_randomization=True,
    params={
      "asset_cfg": SceneEntityCfg("robot", body_names=(".*",)),
      "operation": "scale",
      "field": "body_mass",
      "ranges": (0.9, 1.30),
      "shared_random": True,
    },
  )
  cfg.events["joint_armature"] = EventTermCfg(
    mode="startup",
    func=mdp.randomize_field,
    domain_randomization=True,
    params={
      "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)),
      "operation": "scale",
      "field": "dof_armature",
      "ranges": (0.75, 1.25),
    },
  )
  cfg.events["joint_damping"] = EventTermCfg(
    mode="startup",
    func=mdp.randomize_field,
    domain_randomization=True,
    params={
      "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)),
      "operation": "abs",
      "field": "dof_damping",
      "ranges": (0.0, 0.08),
    },
  )
  cfg.events["joint_frictionloss"] = EventTermCfg(
    mode="startup",
    func=mdp.randomize_field,
    domain_randomization=True,
    params={
      "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)),
      "operation": "abs",
      "field": "dof_frictionloss",
      "ranges": (0.0, 0.12),
    },
  )
  cfg.events["pd_gains"] = EventTermCfg(
    mode="startup",
    func=mdp.randomize_pd_gains,
    params={
      "asset_cfg": SceneEntityCfg("robot"),
      "operation": "scale",
      "kp_range": (0.85, 1.15),
      "kd_range": (0.75, 1.25),
    },
  )
  cfg.events["effort_limits"] = EventTermCfg(
    mode="startup",
    func=mdp.randomize_effort_limits,
    params={
      "asset_cfg": SceneEntityCfg("robot"),
      "operation": "scale",
      "effort_limit_range": (0.85, 1.0),
    },
  )
  cfg.events["encoder_bias"].params["bias_range"] = (-0.02, 0.02)


def unitree_go2_flat_tracking_env_cfg(
  has_state_estimation: bool = True,
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Create Unitree Go2 flat terrain tracking configuration."""
  cfg = make_tracking_env_cfg()

  cfg.scene.entities = {"robot": get_go2_robot_cfg(self_collisions=True)}

  cfg.scene.sensors = (
    ContactSensorCfg(
      name="self_collision",
      primary=ContactMatch(mode="subtree", pattern="trunk", entity="robot"),
      secondary=ContactMatch(mode="subtree", pattern="trunk", entity="robot"),
      fields=("found",),
      reduce="none",
      num_slots=1,
    ),
  )

  joint_pos_action = cfg.actions["joint_pos"]
  assert isinstance(joint_pos_action, JointPositionActionCfg)
  joint_pos_action.scale = GO2_ACTION_SCALE

  motion_cmd = cfg.commands["motion"]
  assert isinstance(motion_cmd, MotionCommandCfg)
  motion_cmd.anchor_body_name = "trunk"
  motion_cmd.body_names = (
    "trunk",
    "FL_hip",
    "FL_thigh",
    "FL_foot",
    "FR_hip",
    "FR_thigh",
    "FR_foot",
    "RL_hip",
    "RL_thigh",
    "RL_foot",
    "RR_hip",
    "RR_thigh",
    "RR_foot",
  )

  # Go2 has no joint torque support — remove tau reward term.
  cfg.rewards.pop("motion_joint_torque", None)

  # Go2 has no tau mode — remove tau obs terms.
  cfg.observations["actor"].terms.pop("joint_torque", None)
  cfg.observations["critic"].terms.pop("joint_torque", None)

  cfg.events["foot_friction"].params[
    "asset_cfg"
  ].geom_names = r"^[FR][LR]_foot_collision$"
  cfg.events["base_com"].params["asset_cfg"].body_names = ("trunk",)
  _add_go2_sim2real_randomization(cfg)

  cfg.terminations["ee_body_pos"].params["body_names"] = (
    "FL_foot",
    "FR_foot",
    "RL_foot",
    "RR_foot",
  )

  cfg.viewer.body_name = "trunk"

  if not has_state_estimation:
    new_actor_terms = {
      k: v
      for k, v in cfg.observations["actor"].terms.items()
      if k not in ["motion_anchor_pos_b", "base_lin_vel"]
    }
    cfg.observations["actor"] = ObservationGroupCfg(
      terms=new_actor_terms,
      concatenate_terms=True,
      enable_corruption=True,
    )

  if play:
    cfg.episode_length_s = int(1e9)
    cfg.observations["actor"].enable_corruption = False
    cfg.events.pop("push_robot", None)
    motion_cmd.pose_range = {}
    motion_cmd.velocity_range = {}
    motion_cmd.sampling_mode = "start"

  return cfg
