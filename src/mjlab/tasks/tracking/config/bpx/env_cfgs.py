"""Black Panther X flat tracking environment configurations."""

from mjlab.asset_zoo.robots import (
  BPX_ACTION_SCALE,
  get_bpx_robot_cfg,
)
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.tasks.tracking.mdp import MotionCommandCfg
from mjlab.tasks.tracking.tracking_env_cfg import make_tracking_env_cfg


def bpx_flat_tracking_env_cfg(
  has_state_estimation: bool = True,
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Create Black Panther X flat terrain tracking configuration."""
  cfg = make_tracking_env_cfg()

  cfg.scene.entities = {"robot": get_bpx_robot_cfg()}
  cfg.sim.nconmax = 128
  cfg.sim.njmax = 512

  cfg.scene.sensors = (
    ContactSensorCfg(
      name="self_collision",
      primary=ContactMatch(mode="subtree", pattern="torso", entity="robot"),
      secondary=ContactMatch(mode="subtree", pattern="torso", entity="robot"),
      fields=("found",),
      reduce="none",
      num_slots=1,
    ),
  )

  joint_pos_action = cfg.actions["joint_pos"]
  assert isinstance(joint_pos_action, JointPositionActionCfg)
  joint_pos_action.scale = BPX_ACTION_SCALE

  motion_cmd = cfg.commands["motion"]
  assert isinstance(motion_cmd, MotionCommandCfg)
  motion_cmd.anchor_body_name = "torso"
  motion_cmd.body_names = (
    "torso",
    "fl_hip_link",
    "fl_thigh_link",
    "fl_calf_link",
    "fl_toe_link",
    "fr_hip_link",
    "fr_thigh_link",
    "fr_calf_link",
    "fr_toe_link",
    "hl_hip_link",
    "hl_thigh_link",
    "hl_calf_link",
    "hl_toe_link",
    "hr_hip_link",
    "hr_thigh_link",
    "hr_calf_link",
    "hr_toe_link",
  )

  # BPX MJCF uses motor actuators without joint torque sensor support here.
  cfg.rewards.pop("motion_joint_torque", None)
  cfg.observations["actor"].terms.pop("joint_torque", None)
  cfg.observations["critic"].terms.pop("joint_torque", None)
  for group_name in ("actor", "critic"):
    cfg.observations[group_name].terms["base_lin_vel"].func = envs_mdp.base_lin_vel
    cfg.observations[group_name].terms["base_lin_vel"].params = {}
    cfg.observations[group_name].terms["base_ang_vel"].func = envs_mdp.base_ang_vel
    cfg.observations[group_name].terms["base_ang_vel"].params = {}

  cfg.events["foot_friction"].params[
    "asset_cfg"
  ].geom_names = r"^(fl|fr|hl|hr)_toe_link_collision_0$"
  cfg.events["base_com"].params["asset_cfg"].body_names = ("torso",)
  cfg.events["body_mass"] = EventTermCfg(
    mode="startup",
    func=envs_mdp.randomize_field,
    domain_randomization=True,
    params={
      "asset_cfg": SceneEntityCfg("robot", body_names=(".*",)),
      "operation": "scale",
      "field": "body_mass",
      "ranges": (0.9, 1.1),
    },
  )
  cfg.events["joint_armature"] = EventTermCfg(
    mode="startup",
    func=envs_mdp.randomize_field,
    domain_randomization=True,
    params={
      "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)),
      "operation": "scale",
      "field": "dof_armature",
      "ranges": (0.8, 1.2),
    },
  )
  cfg.events["joint_damping"] = EventTermCfg(
    mode="startup",
    func=envs_mdp.randomize_field,
    domain_randomization=True,
    params={
      "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)),
      "operation": "abs",
      "field": "dof_damping",
      "ranges": (0.0, 0.05),
    },
  )
  cfg.events["joint_frictionloss"] = EventTermCfg(
    mode="startup",
    func=envs_mdp.randomize_field,
    domain_randomization=True,
    params={
      "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)),
      "operation": "abs",
      "field": "dof_frictionloss",
      "ranges": (0.0, 0.2),
    },
  )
  cfg.events["pd_gains"] = EventTermCfg(
    mode="startup",
    func=envs_mdp.randomize_pd_gains,
    params={
      "asset_cfg": SceneEntityCfg("robot"),
      "kp_range": (0.9, 1.1),
      "kd_range": (0.8, 1.2),
      "operation": "scale",
    },
  )
  cfg.events["effort_limits"] = EventTermCfg(
    mode="startup",
    func=envs_mdp.randomize_effort_limits,
    params={
      "asset_cfg": SceneEntityCfg("robot"),
      "effort_limit_range": (0.9, 1.1),
      "operation": "scale",
    },
  )

  cfg.terminations["ee_body_pos"].params["body_names"] = (
    "fl_toe_link",
    "fr_toe_link",
    "hl_toe_link",
    "hr_toe_link",
  )

  cfg.viewer.body_name = "torso"

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
