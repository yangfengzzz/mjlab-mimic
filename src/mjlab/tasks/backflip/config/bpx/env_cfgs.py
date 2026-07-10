"""Black Panther X backflip environment configuration."""

from mjlab.asset_zoo.robots import BPX_JOINT_NAMES, get_bpx_robot_cfg
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.tasks.backflip.backflip_env_cfg import make_backflip_env_cfg

BPX_FOOT_BODY_NAMES = ("fl_toe_link", "fr_toe_link", "hl_toe_link", "hr_toe_link")
BPX_FOOT_CONTACT_PATTERN = BPX_FOOT_BODY_NAMES


def bpx_backflip_flat_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create BPX flat-terrain backflip configuration."""
  return make_backflip_env_cfg(
    robot_cfg=get_bpx_robot_cfg(),
    joint_names=BPX_JOINT_NAMES,
    foot_body_names=BPX_FOOT_BODY_NAMES,
    foot_contact_pattern=BPX_FOOT_CONTACT_PATTERN,
    base_body_name="torso",
    action_scale=0.5,
    height_target=0.36,
    stance_width=0.34244,
    nonfoot_contact_pattern=r".*_collision.*",
    nonfoot_contact_exclude=(r"^(fl|fr|hl|hr)_toe_link_collision_0$",),
    nonfoot_contact_weight=-2.0,
    init_joint_noise=0.02,
    init_yaw_noise=0.05,
    play=play,
  )
