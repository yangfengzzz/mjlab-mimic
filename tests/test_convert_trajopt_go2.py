"""Tests for Go2 TrajOpt conversion helpers."""

from __future__ import annotations

import json
from pathlib import Path

import torch

from mjlab.scripts.convert_trajopt_go2 import TrajoptGo2MotionLoader


def _write_trajopt_json(path: Path) -> None:
  nodes = []
  for i in range(3):
    q = [
      float(i),
      0.0,
      0.3,
      0.0,
      0.0,
      0.0,
      1.0,
      *[float(i + j) for j in range(12)],
    ]
    v = [0.0] * 6 + [float(10 * i + j) for j in range(12)]
    nodes.append({"dt": 0.1, "q": q, "v": v})

  path.write_text(json.dumps({"solution": {"nodes": nodes}}))


def test_trajopt_go2_loader_resamples_and_uses_pinocchio_joint_velocity(
  tmp_path: Path,
) -> None:
  motion_file = tmp_path / "trajopt.json"
  _write_trajopt_json(motion_file)

  loader = TrajoptGo2MotionLoader(
    motion_file=str(motion_file),
    output_fps=10.0,
    device="cpu",
  )

  assert loader.output_frames == 2
  torch.testing.assert_close(
    loader.motion_base_poss,
    torch.tensor([[0.0, 0.0, 0.3], [1.0, 0.0, 0.3]]),
  )
  torch.testing.assert_close(
    loader.motion_base_rots,
    torch.tensor([[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]]),
  )
  torch.testing.assert_close(
    loader.motion_dof_poss,
    torch.tensor(
      [
        [float(j) for j in range(12)],
        [float(1 + j) for j in range(12)],
      ]
    ),
  )
  torch.testing.assert_close(
    loader.motion_dof_vels,
    torch.tensor(
      [
        [float(j) for j in range(12)],
        [float(10 + j) for j in range(12)],
      ]
    ),
  )
