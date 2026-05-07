"""IK setup mirroring ``examples/humanoid_g1_torso.py`` for the dataset run.

Single source of truth for joint names, posture cost, tasks, and limits. If
the example changes, this module changes alongside it.
"""

from __future__ import annotations

from typing import TypedDict

import mujoco
import numpy as np

import mink

# Names match the example. Foot frames are MuJoCo sites; torso_link is a body.
TORSO_BODY = "torso_link"
FOOT_SITES = ("right_foot", "left_foot")

# Posture cost overrides. Values from the example.
POSTURE_COST_DEFAULT = 1e-1
POSTURE_COST_OVERRIDES_DOF = {
    "waist_roll_joint": 5.0,
    "waist_pitch_joint": 1.0,
}

# Tighter knee lower bound (matches the example).
KNEE_LOWER_BOUND_RAD = 0.17
KNEE_JOINTS = ("left_knee_joint", "right_knee_joint")

# Collision-avoidance pairs (matches the example).
COLLISION_PAIRS: list[tuple[list[str], list[str]]] = [
    (["left_hand_collision"], ["left_thigh_collision"]),
    (["right_hand_collision"], ["right_thigh_collision"]),
    (["torso_collision"], ["left_thigh_collision"]),
    (["torso_collision"], ["right_thigh_collision"]),
]
COLLISION_MIN_DISTANCE = 0.005
COLLISION_DETECTION_DISTANCE = 0.15


class IKParts(TypedDict):
    tasks: list
    limits: list
    torso_task: mink.FrameTask
    foot_tasks: list[mink.FrameTask]
    posture_task: mink.PostureTask
    joint_qposadrs: np.ndarray


def extract_joint_names(model: mujoco.MjModel) -> list[str]:
    """Return the 29 actuated-joint names in qposadr order (free root excluded)."""
    pairs: list[tuple[int, str]] = []
    for j in range(model.njnt):
        joint = model.joint(j)
        if joint.type[0] == mujoco.mjtJoint.mjJNT_FREE:
            continue
        pairs.append((int(joint.qposadr[0]), str(joint.name)))
    pairs.sort()
    return [name for _, name in pairs]


def build_posture_cost(model: mujoco.MjModel) -> np.ndarray:
    cost = np.full(model.nv, POSTURE_COST_DEFAULT)
    for jname, val in POSTURE_COST_OVERRIDES_DOF.items():
        cost[int(model.joint(jname).dofadr[0])] = val
    return cost


def build_ik(model: mujoco.MjModel, configuration: mink.Configuration) -> IKParts:
    """Build tasks + limits exactly as the example does.

    The torso task target is unset (caller sets it per cell). Foot and posture
    targets must be pinned by the caller after loading the keyframe.
    """
    torso_task = mink.FrameTask(
        frame_name=TORSO_BODY,
        frame_type="body",
        position_cost=[10.0, 10.0, 10.0],
        orientation_cost=10.0,
        lm_damping=1.0,
    )
    posture_task = mink.PostureTask(model, cost=build_posture_cost(model))

    foot_tasks: list[mink.FrameTask] = []
    for site_name in FOOT_SITES:
        foot_tasks.append(
            mink.FrameTask(
                frame_name=site_name,
                frame_type="site",
                position_cost=1000.0,
                orientation_cost=1000.0,
                lm_damping=1.0,
            )
        )

    tasks: list = [torso_task, posture_task, *foot_tasks]

    config_limit = mink.ConfigurationLimit(model)
    for jname in KNEE_JOINTS:
        config_limit.lower[int(model.joint(jname).qposadr[0])] = KNEE_LOWER_BOUND_RAD

    collision_limit = mink.CollisionAvoidanceLimit(
        model=model,
        geom_pairs=COLLISION_PAIRS,  # type: ignore[arg-type]
        minimum_distance_from_collisions=COLLISION_MIN_DISTANCE,
        collision_detection_distance=COLLISION_DETECTION_DISTANCE,
    )
    limits: list = [config_limit, collision_limit]

    joint_names = extract_joint_names(model)
    qposadrs = np.array(
        [int(model.joint(n).qposadr[0]) for n in joint_names], dtype=np.int64
    )

    return IKParts(
        tasks=tasks,
        limits=limits,
        torso_task=torso_task,
        foot_tasks=foot_tasks,
        posture_task=posture_task,
        joint_qposadrs=qposadrs,
    )
