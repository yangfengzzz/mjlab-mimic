"""Backflip task configuration shared by quadruped robots."""

from __future__ import annotations

from mjlab.entity import EntityCfg
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.scene import SceneCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.tasks.backflip import mdp
from mjlab.terrains import TerrainImporterCfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise
from mjlab.viewer import ViewerConfig

BACKFLIP_EPISODE_LENGTH_S = 2.0


def make_backflip_env_cfg(
  *,
  robot_cfg: EntityCfg,
  joint_names: tuple[str, ...],
  foot_body_names: tuple[str, str, str, str],
  foot_contact_pattern: tuple[str, ...],
  base_body_name: str,
  action_scale: float | dict[str, float],
  init_joint_noise: float = 0.02,
  init_yaw_noise: float = 0.05,
  num_envs: int = 4096,
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Create a flat-terrain backflip task config."""

  def robot_joints() -> SceneEntityCfg:
    return SceneEntityCfg(
      "robot",
      joint_names=list(joint_names),
      preserve_order=True,
    )

  def feet_bodies() -> SceneEntityCfg:
    return SceneEntityCfg(
      "robot",
      body_names=list(foot_body_names),
      preserve_order=True,
    )

  feet_ground_cfg = ContactSensorCfg(
    name="feet_ground_contact",
    primary=ContactMatch(mode="body", pattern=foot_contact_pattern, entity="robot"),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found", "force"),
    reduce="netforce",
    num_slots=1,
    track_air_time=True,
  )

  actor_terms = {
    "base_ang_vel": ObservationTermCfg(
      func=envs_mdp.base_ang_vel,
      noise=None if play else Unoise(n_min=-0.10, n_max=0.10),
      scale=0.25,
    ),
    "projected_gravity": ObservationTermCfg(
      func=envs_mdp.projected_gravity,
      noise=None if play else Unoise(n_min=-0.02, n_max=0.02),
    ),
    "joint_pos": ObservationTermCfg(
      func=envs_mdp.joint_pos_rel,
      params={"asset_cfg": robot_joints()},
      noise=None if play else Unoise(n_min=-0.01, n_max=0.01),
    ),
    "joint_vel": ObservationTermCfg(
      func=envs_mdp.joint_vel_rel,
      params={"asset_cfg": robot_joints()},
      noise=None if play else Unoise(n_min=-0.50, n_max=0.50),
      scale=0.05,
    ),
    "actions": ObservationTermCfg(func=envs_mdp.last_action),
    "previous_actions": ObservationTermCfg(func=mdp.previous_action),
    "phase": ObservationTermCfg(
      func=mdp.phase_encoding,
      params={"duration": BACKFLIP_EPISODE_LENGTH_S},
    ),
  }
  critic_terms = {
    **actor_terms,
    "base_lin_vel": ObservationTermCfg(func=envs_mdp.base_lin_vel, scale=2.0),
  }

  rewards = {
    "ang_vel_y": RewardTermCfg(func=mdp.ang_vel_y_reward, weight=5.0),
    "ang_vel_z": RewardTermCfg(func=mdp.ang_vel_z_penalty, weight=-1.0),
    "lin_vel_z": RewardTermCfg(func=mdp.lin_vel_z_reward, weight=20.0),
    "orientation_control": RewardTermCfg(
      func=mdp.orientation_control_penalty,
      weight=-1.0,
    ),
    "feet_height_before_backflip": RewardTermCfg(
      func=mdp.feet_height_before_backflip_penalty,
      weight=-30.0,
      params={"asset_cfg": feet_bodies()},
    ),
    "height_control": RewardTermCfg(
      func=mdp.height_control_penalty,
      weight=-10.0,
      params={"target_height": 0.3},
    ),
    "actions_symmetry": RewardTermCfg(
      func=mdp.action_symmetry_penalty,
      weight=-0.1,
    ),
    "gravity_y": RewardTermCfg(func=mdp.gravity_y_penalty, weight=-10.0),
    "feet_distance": RewardTermCfg(
      func=mdp.feet_distance_penalty,
      weight=-1.0,
      params={"asset_cfg": feet_bodies()},
    ),
    "action_rate": RewardTermCfg(func=envs_mdp.action_rate_l2, weight=-0.001),
  }

  events = {
    "reset_base": EventTermCfg(
      func=envs_mdp.reset_root_state_uniform,
      mode="reset",
      params={
        "pose_range": {
          "x": (0.0, 0.0),
          "y": (0.0, 0.0),
          "z": (0.0, 0.0),
          "yaw": (-init_yaw_noise, init_yaw_noise),
        },
        "velocity_range": {},
      },
    ),
    "reset_robot_joints": EventTermCfg(
      func=envs_mdp.reset_joints_by_offset,
      mode="reset",
      params={
        "position_range": (-init_joint_noise, init_joint_noise),
        "velocity_range": (0.0, 0.0),
        "asset_cfg": robot_joints(),
      },
    ),
  }

  if play:
    events["reset_base"].params["pose_range"]["yaw"] = (0.0, 0.0)
    events["reset_robot_joints"].params["position_range"] = (0.0, 0.0)

  return ManagerBasedRlEnvCfg(
    scene=SceneCfg(
      terrain=TerrainImporterCfg(terrain_type="plane"),
      entities={"robot": robot_cfg},
      sensors=(feet_ground_cfg,),
      num_envs=1 if play else num_envs,
      env_spacing=2.0,
      extent=2.0,
    ),
    observations={
      "actor": ObservationGroupCfg(
        terms=actor_terms,
        concatenate_terms=True,
        enable_corruption=not play,
        history_length=1,
      ),
      "critic": ObservationGroupCfg(
        terms=critic_terms,
        concatenate_terms=True,
        enable_corruption=False,
        history_length=1,
      ),
    },
    actions={
      "joint_pos": JointPositionActionCfg(
        entity_name="robot",
        actuator_names=list(joint_names),
        scale=action_scale,
        use_default_offset=True,
      )
    },
    events=events,
    rewards=rewards,
    terminations={
      "time_out": TerminationTermCfg(func=envs_mdp.time_out, time_out=True),
    },
    viewer=ViewerConfig(
      origin_type=ViewerConfig.OriginType.ASSET_BODY,
      entity_name="robot",
      body_name=base_body_name,
      distance=2.0,
      elevation=-10.0,
      azimuth=90.0,
    ),
    sim=SimulationCfg(
      nconmax=120,
      njmax=600,
      contact_sensor_maxmatch=128,
      mujoco=MujocoCfg(
        timestep=0.005,
        iterations=10,
        ls_iterations=20,
        ccd_iterations=80,
      ),
    ),
    decimation=4,
    episode_length_s=BACKFLIP_EPISODE_LENGTH_S,
  )
