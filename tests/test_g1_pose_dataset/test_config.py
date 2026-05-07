"""Tests for g1_pose_dataset.config."""

from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np
import pytest
from g1_pose_dataset import config as cfg

import mink

XML_PATH = Path(__file__).resolve().parents[2] / "examples" / "unitree_g1" / "scene_g1_torso.xml"


@pytest.fixture(scope="module")
def model() -> mujoco.MjModel:
    return mujoco.MjModel.from_xml_path(XML_PATH.as_posix())


def test_extract_joint_names_returns_29(model: mujoco.MjModel) -> None:
    names = cfg.extract_joint_names(model)
    assert len(names) == 29
    # The free root joint must be excluded.
    for name in names:
        joint = model.joint(name)
        assert joint.type[0] != mujoco.mjtJoint.mjJNT_FREE


def test_extract_joint_names_includes_known_joints(model: mujoco.MjModel) -> None:
    names = cfg.extract_joint_names(model)
    for required in (
        "left_knee_joint",
        "right_knee_joint",
        "waist_yaw_joint",
        "waist_roll_joint",
        "waist_pitch_joint",
        "left_shoulder_pitch_joint",
        "right_wrist_yaw_joint",
    ):
        assert required in names


def test_extract_joint_names_in_qposadr_order(model: mujoco.MjModel) -> None:
    names = cfg.extract_joint_names(model)
    addrs = [int(model.joint(n).qposadr[0]) for n in names]
    assert addrs == sorted(addrs)


def test_build_posture_cost_overrides(model: mujoco.MjModel) -> None:
    cost = cfg.build_posture_cost(model)
    assert cost.shape == (model.nv,)
    # Default 1e-1 everywhere except the two overrides from the example.
    waist_roll_dof = int(model.joint("waist_roll_joint").dofadr[0])
    waist_pitch_dof = int(model.joint("waist_pitch_joint").dofadr[0])
    waist_yaw_dof = int(model.joint("waist_yaw_joint").dofadr[0])
    assert cost[waist_roll_dof] == pytest.approx(5.0)
    assert cost[waist_pitch_dof] == pytest.approx(1.0)
    # waist_yaw is NOT overridden (the example's override is commented out).
    assert cost[waist_yaw_dof] == pytest.approx(1e-1)


def test_build_ik_returns_expected_components(model: mujoco.MjModel) -> None:
    parts = cfg.build_ik(model)
    assert "tasks" in parts
    assert "limits" in parts
    assert "torso_task" in parts
    assert "joint_qposadrs" in parts
    # 4 tasks: torso, posture, two feet.
    assert len(parts["tasks"]) == 4
    # 2 limits: configuration limit, collision avoidance.
    assert len(parts["limits"]) == 2
    assert parts["torso_task"] is parts["tasks"][0]
    assert parts["joint_qposadrs"].shape == (29,)
    assert parts["joint_qposadrs"].dtype == np.int64

    # Knee lower-bound override actually applied to the configuration limit.
    config_limit = parts["limits"][0]
    assert isinstance(config_limit, mink.ConfigurationLimit)
    left_qpos = int(model.joint("left_knee_joint").qposadr[0])
    right_qpos = int(model.joint("right_knee_joint").qposadr[0])
    assert config_limit.lower[left_qpos] == pytest.approx(cfg.KNEE_LOWER_BOUND_RAD)
    assert config_limit.lower[right_qpos] == pytest.approx(cfg.KNEE_LOWER_BOUND_RAD)

    # Foot tasks point at the right sites in the right order.
    foot_tasks = parts["foot_tasks"]
    assert len(foot_tasks) == 2
    assert foot_tasks[0].frame_name == "right_foot"
    assert foot_tasks[1].frame_name == "left_foot"
    for ft in foot_tasks:
        assert ft.frame_type == "site"
