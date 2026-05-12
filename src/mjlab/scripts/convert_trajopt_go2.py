"""Convert Go2 TrajOpt JSON solutions to mjlab motion ``.npz`` files.

Input JSON format:
  ``solution.nodes[*].q`` is ``pos3 + quat_xyzw4 + 12 Go2 joints``.
  ``solution.nodes[*].v`` is optional velocity data in Pinocchio order.
"""

import json
from pathlib import Path
from typing import Annotated, Any

import numpy as np
import torch
import tyro
from tqdm import tqdm

import mjlab
from mjlab.entity import Entity
from mjlab.scene import Scene
from mjlab.sim.sim import Simulation, SimulationCfg
from mjlab.tasks.tracking.config.go2.env_cfgs import unitree_go2_flat_tracking_env_cfg
from mjlab.utils.lab_api.math import (
  axis_angle_from_quat,
  quat_conjugate,
  quat_mul,
  quat_slerp,
)
from mjlab.viewer.offscreen_renderer import OffscreenRenderer
from mjlab.viewer.viewer_config import ViewerConfig

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

REQUIRED_NPZ_KEYS = (
  "fps",
  "joint_pos",
  "joint_vel",
  "body_pos_w",
  "body_quat_w",
  "body_lin_vel_w",
  "body_ang_vel_w",
)

_Q_SIZE = 19
_V_SIZE = 18


class TrajoptGo2MotionLoader:
  """Load and resample a Go2 TrajOpt solution."""

  def __init__(
    self,
    motion_file: str,
    output_fps: float,
    device: torch.device | str,
  ):
    if output_fps <= 0.0:
      raise ValueError(f"output_fps must be positive, got {output_fps}.")
    self.motion_file = motion_file
    self.output_fps = output_fps
    self.output_dt = 1.0 / output_fps
    self.current_idx = 0
    self.device = device
    self._load_motion()
    self._interpolate_motion()
    self._compute_velocities()

  def _load_motion(self) -> None:
    with open(self.motion_file) as f:
      data = json.load(f)

    nodes = data.get("solution", {}).get("nodes")
    if not isinstance(nodes, list) or len(nodes) < 2:
      raise ValueError("TrajOpt JSON must contain at least two solution nodes.")

    q_values: list[np.ndarray] = []
    v_values: list[np.ndarray] = []
    dts: list[float] = []
    has_valid_v = True

    for i, node in enumerate(nodes):
      dt = float(node.get("dt", 0.0))
      if dt <= 0.0:
        raise ValueError(f"Node {i} has non-positive dt={dt}.")
      dts.append(dt)

      q = np.asarray(node.get("q"), dtype=np.float32)
      if q.shape != (_Q_SIZE,):
        raise ValueError(f"Node {i} q must have shape ({_Q_SIZE},), got {q.shape}.")
      q_values.append(q)

      v = np.asarray(node.get("v", []), dtype=np.float32)
      if v.shape == (_V_SIZE,):
        v_values.append(v)
      else:
        has_valid_v = False

    q_tensor = torch.tensor(np.stack(q_values), dtype=torch.float32, device=self.device)
    dts_np = np.asarray(dts[:-1], dtype=np.float32)
    node_times_np = np.concatenate(([0.0], np.cumsum(dts_np)))
    if node_times_np[-1] <= 0.0:
      raise ValueError("TrajOpt solution duration must be positive.")

    self.node_times = torch.tensor(
      node_times_np, dtype=torch.float32, device=self.device
    )
    self.duration = float(node_times_np[-1])
    self.motion_base_poss_input = q_tensor[:, :3]
    quat_xyzw = q_tensor[:, 3:7]
    self.motion_base_rots_input = quat_xyzw[:, [3, 0, 1, 2]]
    self.motion_base_rots_input = torch.nn.functional.normalize(
      self.motion_base_rots_input, dim=1
    )
    self.motion_dof_poss_input = q_tensor[:, 7:]

    self.motion_dof_vels_input: torch.Tensor | None = None
    if has_valid_v and len(v_values) == len(q_values):
      v_tensor = torch.tensor(
        np.stack(v_values), dtype=torch.float32, device=self.device
      )
      self.motion_dof_vels_input = v_tensor[:, 6:]

  def _interpolate_motion(self) -> None:
    output_frames = max(1, int(np.ceil(self.duration * self.output_fps - 1e-6)))
    times = (
      torch.arange(output_frames, device=self.device, dtype=torch.float32)
      * self.output_dt
    )
    self.output_times = times
    self.output_frames = int(times.shape[0])
    idx0, idx1, blend = self._compute_frame_blend(times)

    self.motion_base_poss = self._lerp(
      self.motion_base_poss_input[idx0],
      self.motion_base_poss_input[idx1],
      blend.unsqueeze(1),
    )
    self.motion_base_rots = self._slerp(
      self.motion_base_rots_input[idx0],
      self.motion_base_rots_input[idx1],
      blend,
    )
    self.motion_dof_poss = self._lerp(
      self.motion_dof_poss_input[idx0],
      self.motion_dof_poss_input[idx1],
      blend.unsqueeze(1),
    )
    if self.motion_dof_vels_input is not None:
      self.motion_dof_vels = self._lerp(
        self.motion_dof_vels_input[idx0],
        self.motion_dof_vels_input[idx1],
        blend.unsqueeze(1),
      )
    print(
      f"Loaded Go2 TrajOpt motion: {self.motion_base_poss_input.shape[0]} nodes "
      f"({self.duration:.2f}s) -> {self.output_frames} frames @ "
      f"{self.output_fps} fps"
    )

  def _lerp(
    self, a: torch.Tensor, b: torch.Tensor, blend: torch.Tensor
  ) -> torch.Tensor:
    return a * (1.0 - blend) + b * blend

  def _slerp(
    self, a: torch.Tensor, b: torch.Tensor, blend: torch.Tensor
  ) -> torch.Tensor:
    out = torch.zeros_like(a)
    for i in range(a.shape[0]):
      out[i] = quat_slerp(a[i], b[i], float(blend[i]))
    return out

  def _compute_frame_blend(
    self, times: torch.Tensor
  ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    idx1 = torch.searchsorted(self.node_times, times, right=True)
    idx1 = torch.clamp(idx1, 1, self.node_times.shape[0] - 1)
    idx0 = idx1 - 1
    duration = self.node_times[idx1] - self.node_times[idx0]
    blend = (times - self.node_times[idx0]) / duration
    return idx0, idx1, torch.clamp(blend, 0.0, 1.0)

  def _compute_velocities(self) -> None:
    self.motion_base_lin_vels = self._gradient(self.motion_base_poss)
    if not hasattr(self, "motion_dof_vels"):
      self.motion_dof_vels = self._gradient(self.motion_dof_poss)
    self.motion_base_ang_vels = self._so3_derivative(
      self.motion_base_rots, self.output_dt
    )

  def _gradient(self, values: torch.Tensor) -> torch.Tensor:
    if values.shape[0] < 2:
      return torch.zeros_like(values)
    return torch.gradient(values, spacing=self.output_dt, dim=0)[0]

  def _so3_derivative(self, rotations: torch.Tensor, dt: float) -> torch.Tensor:
    if rotations.shape[0] < 3:
      return torch.zeros((rotations.shape[0], 3), device=rotations.device)
    q_prev, q_next = rotations[:-2], rotations[2:]
    q_rel = quat_mul(q_next, quat_conjugate(q_prev))
    omega = axis_angle_from_quat(q_rel) / (2.0 * dt)
    return torch.cat([omega[:1], omega, omega[-1:]], dim=0)

  def get_next_state(
    self,
  ) -> tuple[
    tuple[
      torch.Tensor,
      torch.Tensor,
      torch.Tensor,
      torch.Tensor,
      torch.Tensor,
      torch.Tensor,
    ],
    bool,
  ]:
    state = (
      self.motion_base_poss[self.current_idx : self.current_idx + 1],
      self.motion_base_rots[self.current_idx : self.current_idx + 1],
      self.motion_base_lin_vels[self.current_idx : self.current_idx + 1],
      self.motion_base_ang_vels[self.current_idx : self.current_idx + 1],
      self.motion_dof_poss[self.current_idx : self.current_idx + 1],
      self.motion_dof_vels[self.current_idx : self.current_idx + 1],
    )
    self.current_idx += 1
    reset = self.current_idx >= self.output_frames
    if reset:
      self.current_idx = 0
    return state, reset


def _stack_motion_log(log: dict[str, Any]) -> dict[str, Any]:
  for key in REQUIRED_NPZ_KEYS:
    if key == "fps":
      continue
    log[key] = np.stack(log[key], axis=0)
  return log


def _upload_to_wandb(output_file: Path, wandb_name: str) -> None:
  import wandb

  run = wandb.init(project="trajopt_go2", name=wandb_name)
  registry = "motions"
  artifact = run.log_artifact(
    artifact_or_path=str(output_file),
    name=wandb_name,
    type=registry,
  )
  run.link_artifact(
    artifact=artifact,
    target_path=f"wandb-registry-{registry}/{wandb_name}",
  )
  wandb.finish()


def convert_file(
  input_file: str,
  output_file: str = "/tmp/go2_motion.npz",
  output_fps: float = 50.0,
  device: str = "cuda:0",
  render: bool = False,
  upload_wandb: bool = False,
  wandb_name: str | None = None,
) -> Path:
  """Convert a Go2 TrajOpt JSON solution to a train-ready ``motion.npz``."""
  if device.startswith("cuda") and not torch.cuda.is_available():
    print("[WARNING]: CUDA is not available. Falling back to CPU.")
    device = "cpu"

  scene = Scene(unitree_go2_flat_tracking_env_cfg().scene, device=device)
  model = scene.compile()

  sim_cfg = SimulationCfg()
  sim_cfg.mujoco.timestep = 1.0 / output_fps
  sim = Simulation(num_envs=1, cfg=sim_cfg, model=model, device=device)
  scene.initialize(sim.mj_model, sim.model, sim.data)

  renderer = None
  if render:
    viewer_cfg = ViewerConfig(
      height=480,
      width=640,
      origin_type=ViewerConfig.OriginType.ASSET_ROOT,
      distance=2.0,
      elevation=-5.0,
      azimuth=20,
    )
    renderer = OffscreenRenderer(model=sim.mj_model, cfg=viewer_cfg, scene=scene)
    renderer.initialize()

  output_path = Path(output_file)
  run_sim(
    sim=sim,
    scene=scene,
    input_file=input_file,
    output_file=output_path,
    output_fps=output_fps,
    render=render,
    renderer=renderer,
  )

  if upload_wandb:
    _upload_to_wandb(output_path, wandb_name or output_path.stem)

  return output_path


def run_sim(
  sim: Simulation,
  scene: Scene,
  input_file: str,
  output_file: Path,
  output_fps: float,
  render: bool,
  renderer: OffscreenRenderer | None,
) -> None:
  motion = TrajoptGo2MotionLoader(
    motion_file=input_file,
    output_fps=output_fps,
    device=sim.device,
  )

  robot: Entity = scene["robot"]
  robot_joint_indexes = robot.find_joints(GO2_JOINT_NAMES, preserve_order=True)[0]

  log: dict[str, Any] = {
    "fps": np.asarray([output_fps], dtype=np.float32),
    "joint_pos": [],
    "joint_vel": [],
    "body_pos_w": [],
    "body_quat_w": [],
    "body_lin_vel_w": [],
    "body_ang_vel_w": [],
  }

  frames = []
  scene.reset()
  file_saved = False

  pbar = tqdm(
    total=motion.output_frames,
    desc="Processing Go2 TrajOpt frames",
    unit="frame",
    ncols=100,
    bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
  )

  while not file_saved:
    (
      (
        base_pos,
        base_rot,
        base_lin_vel,
        base_ang_vel,
        dof_pos,
        dof_vel,
      ),
      reset_flag,
    ) = motion.get_next_state()

    root_states = robot.data.default_root_state.clone()
    root_states[:, 0:3] = base_pos
    root_states[:, :2] += scene.env_origins[:, :2]
    root_states[:, 3:7] = base_rot
    root_states[:, 7:10] = base_lin_vel
    root_states[:, 10:] = base_ang_vel
    robot.write_root_state_to_sim(root_states)

    joint_pos = robot.data.default_joint_pos.clone()
    joint_vel = robot.data.default_joint_vel.clone()
    joint_pos[:, robot_joint_indexes] = dof_pos
    joint_vel[:, robot_joint_indexes] = dof_vel
    robot.write_joint_state_to_sim(joint_pos, joint_vel)

    sim.forward()
    scene.update(sim.mj_model.opt.timestep)

    if render and renderer is not None:
      renderer.update(sim.data)
      frames.append(renderer.render())

    log["joint_pos"].append(robot.data.joint_pos[0].cpu().numpy().copy())
    log["joint_vel"].append(robot.data.joint_vel[0].cpu().numpy().copy())
    log["body_pos_w"].append(robot.data.body_link_pos_w[0].cpu().numpy().copy())
    log["body_quat_w"].append(robot.data.body_link_quat_w[0].cpu().numpy().copy())
    log["body_lin_vel_w"].append(robot.data.body_link_lin_vel_w[0].cpu().numpy().copy())
    log["body_ang_vel_w"].append(robot.data.body_link_ang_vel_w[0].cpu().numpy().copy())

    pbar.update(1)
    if reset_flag:
      file_saved = True
      pbar.close()

  output_file.parent.mkdir(parents=True, exist_ok=True)
  np.savez(output_file, **_stack_motion_log(log))
  print(f"[INFO]: Saved Go2 motion to {output_file}")

  if render and frames:
    import mediapy as media

    video_path = output_file.with_suffix(".mp4")
    media.write_video(str(video_path), frames, fps=output_fps)
    print(f"[INFO]: Saved Go2 motion video to {video_path}")


def main(
  input_file: Annotated[str, tyro.conf.Positional],
  output_file: str = "/tmp/go2_motion.npz",
  output_fps: float = 50.0,
  device: str = "cuda:0",
  render: bool = False,
  upload_wandb: bool = False,
  wandb_name: str | None = None,
) -> None:
  """Convert a Go2 TrajOpt JSON solution to a local ``motion.npz`` file."""
  convert_file(
    input_file=input_file,
    output_file=output_file,
    output_fps=output_fps,
    device=device,
    render=render,
    upload_wandb=upload_wandb,
    wandb_name=wandb_name,
  )


if __name__ == "__main__":
  tyro.cli(main, config=mjlab.TYRO_FLAGS)
