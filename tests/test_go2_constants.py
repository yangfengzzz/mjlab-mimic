"""Tests for go2_constants.py."""

import math
import re

import mujoco
import numpy as np
import pytest

from mjlab.asset_zoo.robots.unitree_go2 import go2_constants
from mjlab.entity import Entity
from mjlab.utils.string import resolve_expr


@pytest.fixture(scope="module")
def go2_entity() -> Entity:
  return Entity(go2_constants.get_go2_robot_cfg())


@pytest.fixture(scope="module")
def go2_model(go2_entity: Entity) -> mujoco.MjModel:
  return go2_entity.spec.compile()


@pytest.mark.parametrize(
  "actuator_config,stiffness,damping",
  [
    (go2_constants.GO2_ACTUATOR_HIP, 20.0, 1.0),
    (go2_constants.GO2_ACTUATOR_THIGH, 20.0, 1.0),
    (go2_constants.GO2_ACTUATOR_CALF, 40.0, 2.0),
  ],
)
def test_actuator_parameters(go2_model, actuator_config, stiffness, damping):
  """Go2 actuator constants should match the unitree_rl_mjlab profile."""
  for i in range(go2_model.nu):
    actuator = go2_model.actuator(i)
    matches = any(
      re.match(pattern, actuator.name)
      for pattern in actuator_config.target_names_expr
    )
    if matches:
      assert actuator.gainprm[0] == stiffness
      assert actuator.biasprm[1] == -stiffness
      assert actuator.biasprm[2] == -damping
      assert actuator.forcerange[0] == -actuator_config.effort_limit
      assert actuator.forcerange[1] == actuator_config.effort_limit


def test_joint_passive_properties_come_from_actuator_config(go2_model) -> None:
  """The MJCF should not add extra passive damping/friction on top of constants."""
  for i in range(go2_model.njnt):
    joint = go2_model.joint(i)
    if joint.name == "floating_base_joint":
      continue
    dof_id = joint.dofadr[0]
    expected_armature = 0.02 if "calf" in joint.name else 0.01
    assert go2_model.dof_armature[dof_id] == expected_armature
    assert go2_model.dof_damping[dof_id] == 0.0
    assert go2_model.dof_frictionloss[dof_id] == 0.0


def test_action_scale_matches_actuators() -> None:
  assert go2_constants.GO2_ACTION_SCALE == {
    ".*hip_.*": 0.29375,
    ".*thigh_.*": 0.29375,
    ".*calf_.*": 0.28125,
  }


def test_urdf_aligned_mass_distribution(go2_model) -> None:
  """Go2 MJCF should keep the generated-URDF physical mass distribution."""
  body_mass = {
    go2_model.body(i).name: float(go2_model.body_mass[i])
    for i in range(go2_model.nbody)
  }
  assert math.isclose(sum(body_mass.values()), 16.087, abs_tol=1e-6)

  expected = {"trunk": 6.921, "Head_upper": 0.001, "Head_lower": 0.001}
  for leg in ("FL", "FR", "RL", "RR"):
    expected[f"{leg}_hip"] = 0.678
    expected[f"{leg}_thigh"] = 1.152
    expected[f"{leg}_calf"] = 0.154
    expected[f"{leg}_foot"] = 0.04
    expected[f"{leg}_hip_rotor"] = 0.089
    expected[f"{leg}_thigh_rotor"] = 0.089
    expected[f"{leg}_calf_rotor"] = 0.089

  for body_name, expected_mass in expected.items():
    assert math.isclose(body_mass[body_name], expected_mass, abs_tol=1e-9)


def test_keyframe_base_position(go2_model) -> None:
  key = go2_model.key("init_state")
  np.testing.assert_allclose(key.qpos[:3], go2_constants.INIT_STATE.pos)


def test_keyframe_joint_positions(go2_entity, go2_model) -> None:
  expected_joint_pos = go2_constants.INIT_STATE.joint_pos
  assert expected_joint_pos is not None
  expected_values = resolve_expr(expected_joint_pos, go2_entity.joint_names, 0.0)
  key = go2_model.key("init_state")
  for joint_name, expected_value in zip(
    go2_entity.joint_names, expected_values, strict=True
  ):
    joint = go2_model.joint(joint_name)
    actual_value = key.qpos[joint.qposadr[0]]
    np.testing.assert_allclose(actual_value, expected_value, rtol=1e-5)


def test_go2_entity_creation(go2_entity) -> None:
  assert go2_entity.num_actuators == 12
  assert go2_entity.num_joints == 12
  assert go2_entity.is_actuated
  assert not go2_entity.is_fixed_base
