"""Black Panther X constants."""

from pathlib import Path

import mujoco

from mjlab import MJLAB_SRC_PATH
from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.utils.os import update_assets
from mjlab.utils.spec_config import CollisionCfg

##
# MJCF and assets.
##

BPX_XML: Path = MJLAB_SRC_PATH / "asset_zoo" / "robots" / "bpx" / "xmls" / "bpx.xml"
assert BPX_XML.exists()


def get_assets(meshdir: str) -> dict[str, bytes]:
  assets: dict[str, bytes] = {}
  update_assets(assets, BPX_XML.parent / "assets", meshdir)
  return assets


def get_spec() -> mujoco.MjSpec:
  spec = mujoco.MjSpec.from_file(str(BPX_XML))
  spec.assets = get_assets(spec.meshdir)
  # BPX's source MJCF includes a demo floor. Tasks add their own terrain, so
  # keeping this plane creates overlapping ground geoms in rendered scenes.
  spec.delete(spec.geom("floor"))
  # BPX ships torque motor actuators in MJCF. Delete them so mjlab owns the
  # actuator model and action space, matching the Go2 tracking setup.
  while spec.actuators:
    spec.delete(spec.actuators[0])
  return spec


##
# Actuator config.
##

BPX_JOINT_NAMES = (
  "fl_hip_roll_joint",
  "fl_hip_pitch_joint",
  "fl_knee_joint",
  "fr_hip_roll_joint",
  "fr_hip_pitch_joint",
  "fr_knee_joint",
  "hl_hip_roll_joint",
  "hl_hip_pitch_joint",
  "hl_knee_joint",
  "hr_hip_roll_joint",
  "hr_hip_pitch_joint",
  "hr_knee_joint",
)

BPX_ACTUATOR_CFG = BuiltinPositionActuatorCfg(
  target_names_expr=BPX_JOINT_NAMES,
  stiffness=30.0,
  damping=1.0,
  effort_limit=30.0,
  armature=0.01,
)

##
# Keyframes.
##

INIT_STATE = EntityCfg.InitialStateCfg(
  pos=(0.0, 0.0, 0.42),
  joint_pos={
    "fl_hip_roll_joint": 0.0,
    "fl_hip_pitch_joint": 0.6,
    "fl_knee_joint": -0.9,
    "fr_hip_roll_joint": 0.0,
    "fr_hip_pitch_joint": 0.6,
    "fr_knee_joint": -0.9,
    "hl_hip_roll_joint": 0.0,
    "hl_hip_pitch_joint": 0.6,
    "hl_knee_joint": -0.9,
    "hr_hip_roll_joint": 0.0,
    "hr_hip_pitch_joint": 0.6,
    "hr_knee_joint": -0.9,
  },
  joint_vel={".*": 0.0},
)

##
# Collision config.
##

_toe_regex = r"^(fl|fr|hl|hr)_toe_link_collision_0$"

FULL_COLLISION = CollisionCfg(
  geom_names_expr=(".*_collision.*",),
  condim={_toe_regex: 3, ".*_collision.*": 1},
  priority={_toe_regex: 1},
  friction={_toe_regex: (0.6,)},
  contype=1,
  conaffinity=1,
)

##
# Final config.
##

BPX_ARTICULATION = EntityArticulationInfoCfg(
  actuators=(
    BPX_ACTUATOR_CFG,
  ),
  soft_joint_pos_limit_factor=0.9,
)


def get_bpx_robot_cfg() -> EntityCfg:
  """Get a fresh BPX robot configuration instance."""
  return EntityCfg(
    init_state=INIT_STATE,
    collisions=(FULL_COLLISION,),
    spec_fn=get_spec,
    articulation=BPX_ARTICULATION,
  )


BPX_ACTION_SCALE: dict[str, float] = {}
for a in BPX_ARTICULATION.actuators:
  assert isinstance(a, BuiltinPositionActuatorCfg)
  e = a.effort_limit
  s = a.stiffness
  names = a.target_names_expr
  assert e is not None
  for n in names:
    BPX_ACTION_SCALE[n] = 0.25 * e / s


if __name__ == "__main__":
  import mujoco.viewer as viewer

  from mjlab.entity.entity import Entity

  robot = Entity(get_bpx_robot_cfg())

  viewer.launch(robot.spec.compile())
