"""Register Black Panther X backflip tasks."""

from mjlab.tasks.backflip.config.bpx.env_cfgs import bpx_backflip_flat_env_cfg
from mjlab.tasks.backflip.config.bpx.rl_cfg import bpx_backflip_ppo_runner_cfg
from mjlab.tasks.registry import register_mjlab_task

register_mjlab_task(
  task_id="Mjlab-Backflip-Flat-BPX",
  env_cfg=bpx_backflip_flat_env_cfg(),
  play_env_cfg=bpx_backflip_flat_env_cfg(play=True),
  rl_cfg=bpx_backflip_ppo_runner_cfg(),
)
