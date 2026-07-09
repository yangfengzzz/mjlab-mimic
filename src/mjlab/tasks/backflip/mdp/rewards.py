"""Reward terms ported from the Genesis Go2 backflip example."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactSensor
from mjlab.utils.lab_api.math import quat_apply_inverse, quat_from_angle_axis

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def _robot(
  env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG
) -> Entity:
  return env.scene[asset_cfg.name]


def _elapsed_time(env: ManagerBasedRlEnv) -> torch.Tensor:
  return env.episode_length_buf.to(dtype=torch.float32) * env.step_dt


def _body_positions_b(asset: Entity, body_ids: list[int]) -> torch.Tensor:
  body_pos_w = asset.data.body_link_pos_w[:, body_ids, :]
  root_pos_w = asset.data.root_link_pos_w.unsqueeze(1)
  translated = body_pos_w - root_pos_w
  quat = asset.data.root_link_quat_w.unsqueeze(1).expand(-1, len(body_ids), -1)
  return quat_apply_inverse(quat, translated)


def orientation_control_penalty(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  start_time: float = 0.5,
  duration: float = 0.5,
) -> torch.Tensor:
  """Track a simple analytical one-rotation pitch reference."""
  asset = _robot(env, asset_cfg)
  elapsed = _elapsed_time(env)
  phase = torch.clamp(elapsed - start_time, min=0.0, max=duration)
  angle = 2.0 * torch.pi * phase / duration
  axis = torch.zeros((env.num_envs, 3), device=env.device)
  axis[:, 1] = 1.0
  quat_pitch = quat_from_angle_axis(angle, axis)
  gravity = asset.data.gravity_vec_w.reshape(-1, 3)
  if gravity.shape[0] == 1:
    gravity = gravity.expand(env.num_envs, -1)
  desired_gravity = quat_apply_inverse(quat_pitch, gravity)
  gravity_error = asset.data.projected_gravity_b - desired_gravity
  return torch.sum(torch.square(gravity_error), dim=1)


def ang_vel_y_reward(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  min_time: float = 0.5,
  max_time: float = 1.0,
  limit: float = 7.2,
) -> torch.Tensor:
  """Reward backward pitch angular velocity during takeoff/rotation."""
  asset = _robot(env, asset_cfg)
  elapsed = _elapsed_time(env)
  active = ((elapsed > min_time) & (elapsed < max_time)).float()
  ang_vel = torch.clamp(-asset.data.root_link_ang_vel_b[:, 1], min=-limit, max=limit)
  return ang_vel * active


def ang_vel_z_penalty(
  env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG
) -> torch.Tensor:
  asset = _robot(env, asset_cfg)
  return torch.abs(asset.data.root_link_ang_vel_b[:, 2])


def lin_vel_z_reward(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  min_time: float = 0.5,
  max_time: float = 0.75,
  limit: float = 3.0,
) -> torch.Tensor:
  """Reward upward velocity around takeoff."""
  asset = _robot(env, asset_cfg)
  elapsed = _elapsed_time(env)
  active = ((elapsed > min_time) & (elapsed < max_time)).float()
  return torch.clamp(asset.data.root_link_lin_vel_w[:, 2], max=limit) * active


def height_control_penalty(
  env: ManagerBasedRlEnv,
  target_height: float,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  pre_time: float = 0.4,
  post_time: float = 1.4,
) -> torch.Tensor:
  asset = _robot(env, asset_cfg)
  elapsed = _elapsed_time(env)
  active = ((elapsed < pre_time) | (elapsed > post_time)).float()
  return torch.square(target_height - asset.data.root_link_pos_w[:, 2]) * active


def action_symmetry_penalty(env: ManagerBasedRlEnv) -> torch.Tensor:
  actions = env.action_manager.action
  diff = torch.square(actions[:, 0] + actions[:, 3])
  diff += torch.sum(torch.square(actions[:, 1:3] - actions[:, 4:6]), dim=-1)
  diff += torch.square(actions[:, 6] + actions[:, 9])
  diff += torch.sum(torch.square(actions[:, 7:9] - actions[:, 10:12]), dim=-1)
  return diff


def gravity_y_penalty(
  env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG
) -> torch.Tensor:
  asset = _robot(env, asset_cfg)
  return torch.square(asset.data.projected_gravity_b[:, 1])


def feet_distance_penalty(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg,
  stance_width: float = 0.3,
) -> torch.Tensor:
  asset = _robot(env, asset_cfg)
  feet_pos_b = _body_positions_b(asset, asset_cfg.body_ids)
  desired_y = torch.tensor(
    (
      stance_width / 2.0,
      -stance_width / 2.0,
      stance_width / 2.0,
      -stance_width / 2.0,
    ),
    device=env.device,
  )
  return torch.sum(torch.square(feet_pos_b[:, :, 1] - desired_y), dim=1)


def feet_height_before_backflip_penalty(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg,
  min_time: float = 0.5,
  ground_offset: float = 0.02,
) -> torch.Tensor:
  """Penalize lifting feet before the launch window."""
  elapsed = _elapsed_time(env)
  active = (elapsed < min_time).float()
  asset = _robot(env, asset_cfg)
  foot_height = torch.clamp(
    asset.data.body_link_pos_w[:, asset_cfg.body_ids, 2],
    min=0.0,
  )
  foot_height = torch.clamp(foot_height - ground_offset, min=0.0)
  return torch.sum(foot_height, dim=1) * active


def nonfoot_contact_penalty(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
  """Penalize non-foot ground contacts."""
  sensor: ContactSensor = env.scene[sensor_name]
  assert sensor.data.found is not None
  return (sensor.data.found > 0).float().flatten(start_dim=1).sum(dim=1)
