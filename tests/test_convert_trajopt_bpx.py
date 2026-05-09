import json
from pathlib import Path

import numpy as np
import torch

from mjlab.scripts.convert_trajopt_bpx import (
  REQUIRED_NPZ_KEYS,
  TrajoptBpxMotionLoader,
  convert_file,
)


def _write_trajopt_solution(path: Path) -> None:
  nodes = []
  for i in range(3):
    q = np.zeros(19, dtype=float)
    q[:3] = (0.1 * i, 0.0, 0.42)
    q[3:7] = (0.0, 0.0, 0.0, 1.0)
    q[7:] = np.linspace(0.0, 0.11, 12) + 0.01 * i

    v = np.zeros(18, dtype=float)
    v[6:] = 0.1

    nodes.append(
      {
        "dt": 0.1,
        "q": q.tolist(),
        "v": v.tolist(),
        "a": np.zeros(18, dtype=float).tolist(),
        "forces": {},
      }
    )

  path.write_text(json.dumps({"info": {"model": "bpx"}, "solution": {"nodes": nodes}}))


def test_loader_converts_trajopt_xyzw_quat_to_mjlab_wxyz(tmp_path: Path) -> None:
  input_file = tmp_path / "bpx_solution.json"
  _write_trajopt_solution(input_file)

  loader = TrajoptBpxMotionLoader(
    motion_file=str(input_file),
    output_fps=10.0,
    device="cpu",
  )

  torch.testing.assert_close(
    loader.motion_base_rots[0],
    torch.tensor([1.0, 0.0, 0.0, 0.0]),
  )
  assert loader.motion_dof_poss.shape[1] == 12
  assert loader.motion_dof_vels.shape[1] == 12


def test_convert_file_writes_train_ready_npz(tmp_path: Path) -> None:
  input_file = tmp_path / "bpx_solution.json"
  output_file = tmp_path / "motion.npz"
  _write_trajopt_solution(input_file)

  convert_file(
    input_file=str(input_file),
    output_file=str(output_file),
    output_fps=10.0,
    device="cpu",
    render=False,
    upload_wandb=False,
  )

  data = np.load(output_file)
  assert set(REQUIRED_NPZ_KEYS).issubset(data.files)
  assert data["joint_pos"].shape[1] == 12
  assert data["joint_vel"].shape == data["joint_pos"].shape
  assert data["body_pos_w"].shape[0] == data["joint_pos"].shape[0]
  assert data["body_pos_w"].shape[2] == 3
  assert data["body_quat_w"].shape[2] == 4
