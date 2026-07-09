"""RL configuration for Black Panther X backflip."""

from mjlab.rl import RslRlOnPolicyRunnerCfg
from mjlab.tasks.backflip.config.go2.rl_cfg import unitree_go2_backflip_ppo_runner_cfg


def bpx_backflip_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  """Create PPO runner configuration for BPX backflip."""
  cfg = unitree_go2_backflip_ppo_runner_cfg()
  cfg.experiment_name = "bpx_backflip"
  return cfg
