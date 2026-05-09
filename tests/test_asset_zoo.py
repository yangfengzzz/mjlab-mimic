import mujoco
import numpy as np
import pytest

from mjlab.asset_zoo.robots import (
  get_bpx_robot_cfg,
  get_g1_robot_cfg,
  get_go1_robot_cfg,
)
from mjlab.entity import Entity


@pytest.mark.parametrize(
  "robot_name,robot_cfg_fn",
  [
    ("BPX", get_bpx_robot_cfg),
    ("G1", get_g1_robot_cfg),
    ("GO1", get_go1_robot_cfg),
  ],
)
def test_robot_compiles_parametrized(robot_name: str, robot_cfg_fn) -> None:
  """Tests that all robots in the asset zoo compile without errors."""
  robot_cfg = robot_cfg_fn()
  assert isinstance(Entity(robot_cfg).compile(), mujoco.MjModel)


def test_bpx_uses_project_position_actuators_only() -> None:
  """BPX should clear XML motors and use the project position actuators."""
  robot = Entity(get_bpx_robot_cfg())
  model = robot.compile()

  assert model.nu == 12
  assert not any(name.endswith("_motor") for name in robot.actuator_names)


def test_bpx_default_pose_matches_sideflip_nominal_pose() -> None:
  """BPX zero action should target the side-flip nominal starting pose."""
  robot = Entity(get_bpx_robot_cfg())
  model = robot.compile()

  expected = np.tile(np.array([0.0, 0.6, -0.9]), 4)
  np.testing.assert_allclose(model.key_qpos[0, 7:], expected, atol=1.0e-6)


def test_bpx_collision_geoms_are_enabled_for_contacts() -> None:
  """BPX collision geoms should be contact-enabled while visual geoms stay disabled."""
  robot = Entity(get_bpx_robot_cfg())
  model = robot.compile()

  for geom_id in range(model.ngeom):
    name = model.geom(geom_id).name
    contype = model.geom_contype[geom_id]
    conaffinity = model.geom_conaffinity[geom_id]
    if "_visual" in name:
      assert contype == 0
      assert conaffinity == 0
    elif "_collision" in name:
      assert contype != 0
      assert conaffinity != 0
