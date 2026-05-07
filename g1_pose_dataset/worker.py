"""Per-process IK worker for the G1 dataset.

This module is import-clean (no top-level mujoco/mink work) so it can be
spawned via ``multiprocessing.get_context("spawn")``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import mujoco
import numpy as np

import mink
from g1_pose_dataset import config as cfg

KEYFRAME_NAME = "stand_drag"
TORSO_TARGET_X = 0.0
TORSO_TARGET_Y = 0.0
DEFAULT_DT = 1.0 / 200.0
DEFAULT_DAMPING = 1e-1
DEFAULT_SOLVER = "daqp"


@dataclass
class CellResult:
    converged: bool
    joints: np.ndarray  # shape (29,), float32
    final_norm: float
    iters: int


@dataclass
class WorkerState:
    model: mujoco.MjModel
    configuration: mink.Configuration
    tasks: list[mink.Task]
    limits: list[mink.Limit]
    torso_task: mink.FrameTask
    joint_qposadrs: np.ndarray  # shape (29,) int64
    joint_names: list[str]


def make_worker_state(model_path: str) -> WorkerState:
    """Load the model, build IK pieces, pin static targets at the keyframe."""
    model = mujoco.MjModel.from_xml_path(model_path)
    configuration = mink.Configuration(model)
    parts = cfg.build_ik(model)

    # Initialise to the standing keyframe and pin foot + posture targets.
    configuration.update_from_keyframe(KEYFRAME_NAME)
    parts["posture_task"].set_target_from_configuration(configuration)
    for foot_task, site_name in zip(parts["foot_tasks"], cfg.FOOT_SITES):
        foot_task.set_target(
            configuration.get_transform_frame_to_world(site_name, "site")
        )

    return WorkerState(
        model=model,
        configuration=configuration,
        tasks=parts["tasks"],
        limits=parts["limits"],
        torso_task=parts["torso_task"],
        joint_qposadrs=parts["joint_qposadrs"],
        joint_names=cfg.extract_joint_names(model),
    )


def solve_one_cell(
    state: WorkerState,
    command_rad_m: np.ndarray,  # shape (4,): roll, pitch, yaw, height
    threshold: float,
    max_iter: int,
    dt: float = DEFAULT_DT,
    damping: float = DEFAULT_DAMPING,
    solver: str = DEFAULT_SOLVER,
) -> CellResult:
    """Run the IK loop for one grid cell.

    Resets the configuration to the standing keyframe before iterating, so
    every cell starts from the same initial pose (reproducibility).
    """
    state.configuration.update_from_keyframe(KEYFRAME_NAME)

    roll, pitch, yaw, height = (
        float(command_rad_m[0]),
        float(command_rad_m[1]),
        float(command_rad_m[2]),
        float(command_rad_m[3]),
    )
    target = mink.SE3.from_rotation_and_translation(
        rotation=mink.SO3.from_rpy_radians(roll, pitch, yaw),
        translation=np.array([TORSO_TARGET_X, TORSO_TARGET_Y, height]),
    )
    state.torso_task.set_target(target)

    converged = False
    final_norm = math.inf
    iters = 0
    for it in range(max_iter):
        vel = mink.solve_ik(
            state.configuration,
            state.tasks,
            dt,
            solver,
            damping=damping,
            limits=state.limits,
        )
        final_norm = float(np.linalg.norm(vel))
        iters = it + 1
        if final_norm < threshold:
            converged = True
            break
        state.configuration.integrate_inplace(vel, dt)

    qpos = state.configuration.data.qpos
    joints = qpos[state.joint_qposadrs].astype(np.float32, copy=True)

    return CellResult(
        converged=converged, joints=joints, final_norm=final_norm, iters=iters
    )
