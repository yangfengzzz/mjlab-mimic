"""MDP terms for backflip tasks."""

from mjlab.tasks.backflip.mdp.observations import phase_encoding as phase_encoding
from mjlab.tasks.backflip.mdp.observations import previous_action as previous_action
from mjlab.tasks.backflip.mdp.rewards import (
  action_symmetry_penalty as action_symmetry_penalty,
)
from mjlab.tasks.backflip.mdp.rewards import ang_vel_y_reward as ang_vel_y_reward
from mjlab.tasks.backflip.mdp.rewards import ang_vel_z_penalty as ang_vel_z_penalty
from mjlab.tasks.backflip.mdp.rewards import (
  feet_distance_penalty as feet_distance_penalty,
)
from mjlab.tasks.backflip.mdp.rewards import (
  feet_height_before_backflip_penalty as feet_height_before_backflip_penalty,
)
from mjlab.tasks.backflip.mdp.rewards import gravity_y_penalty as gravity_y_penalty
from mjlab.tasks.backflip.mdp.rewards import (
  height_control_penalty as height_control_penalty,
)
from mjlab.tasks.backflip.mdp.rewards import lin_vel_z_reward as lin_vel_z_reward
from mjlab.tasks.backflip.mdp.rewards import (
  orientation_control_penalty as orientation_control_penalty,
)
