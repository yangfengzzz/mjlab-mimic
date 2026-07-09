"""Register Unitree Go2 backflip tasks."""

from mjlab.tasks.backflip.config.go2.env_cfgs import unitree_go2_backflip_flat_env_cfg
from mjlab.tasks.backflip.config.go2.rl_cfg import unitree_go2_backflip_ppo_runner_cfg
from mjlab.tasks.registry import register_mjlab_task

register_mjlab_task(
  task_id="Mjlab-Backflip-Flat-Unitree-Go2",
  env_cfg=unitree_go2_backflip_flat_env_cfg(),
  play_env_cfg=unitree_go2_backflip_flat_env_cfg(play=True),
  rl_cfg=unitree_go2_backflip_ppo_runner_cfg(),
)
