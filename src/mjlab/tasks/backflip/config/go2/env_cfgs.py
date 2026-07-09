"""Unitree Go2 backflip environment configuration."""

from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.asset_zoo.robots import get_go2_robot_cfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.tasks.backflip.backflip_env_cfg import make_backflip_env_cfg

GO2_JOINT_NAMES = (
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
)

GO2_FOOT_BODY_NAMES = ("FL_foot", "FR_foot", "RL_foot", "RR_foot")
GO2_FOOT_CONTACT_PATTERN = GO2_FOOT_BODY_NAMES

_INIT_STATE = EntityCfg.InitialStateCfg(
  pos=(0.0, 0.0, 0.36),
  joint_pos={
    "FL_hip_joint": 0.0,
    "FR_hip_joint": 0.0,
    "RL_hip_joint": 0.0,
    "RR_hip_joint": 0.0,
    "FL_thigh_joint": 0.8,
    "FR_thigh_joint": 0.8,
    "RL_thigh_joint": 1.0,
    "RR_thigh_joint": 1.0,
    "FL_calf_joint": -1.5,
    "FR_calf_joint": -1.5,
    "RL_calf_joint": -1.5,
    "RR_calf_joint": -1.5,
  },
  joint_vel={".*": 0.0},
)

_HIP_THIGH_ACTUATOR = BuiltinPositionActuatorCfg(
  target_names_expr=(".*_hip_joint", ".*_thigh_joint"),
  stiffness=70.0,
  damping=3.0,
  effort_limit=23.7,
)
_CALF_ACTUATOR = BuiltinPositionActuatorCfg(
  target_names_expr=(".*_calf_joint",),
  stiffness=70.0,
  damping=3.0,
  effort_limit=45.43,
)


def _get_go2_backflip_robot_cfg() -> EntityCfg:
  robot_cfg = get_go2_robot_cfg()
  robot_cfg.init_state = _INIT_STATE
  robot_cfg.articulation = EntityArticulationInfoCfg(
    actuators=(_HIP_THIGH_ACTUATOR, _CALF_ACTUATOR),
    soft_joint_pos_limit_factor=0.9,
  )
  return robot_cfg


def unitree_go2_backflip_flat_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create Unitree Go2 flat-terrain backflip configuration."""
  return make_backflip_env_cfg(
    robot_cfg=_get_go2_backflip_robot_cfg(),
    joint_names=GO2_JOINT_NAMES,
    foot_body_names=GO2_FOOT_BODY_NAMES,
    foot_contact_pattern=GO2_FOOT_CONTACT_PATTERN,
    base_body_name="trunk",
    action_scale=0.5,
    init_joint_noise=0.02,
    init_yaw_noise=0.05,
    play=play,
  )
