"""Observation terms for backflip tasks."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


def phase_encoding(env: ManagerBasedRlEnv, duration: float = 2.0) -> torch.Tensor:
  """Genesis-style time encoding for a single backflip episode."""
  phase = torch.pi * env.episode_length_buf.to(dtype=torch.float32) * env.step_dt
  phase = (phase / duration).unsqueeze(-1)
  return torch.cat(
    (
      torch.sin(phase),
      torch.cos(phase),
      torch.sin(phase / 2.0),
      torch.cos(phase / 2.0),
      torch.sin(phase / 4.0),
      torch.cos(phase / 4.0),
    ),
    dim=-1,
  )


def previous_action(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Action from the previous policy step."""
  return env.action_manager.prev_action
