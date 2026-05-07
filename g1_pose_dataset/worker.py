"""Per-process IK worker for the G1 dataset.

This module is import-clean (no top-level mujoco/mink work) so it can be
spawned via ``multiprocessing.get_context("spawn")``.
"""

from __future__ import annotations

import json
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path

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


@dataclass
class SubshardPaths:
    commands: Path
    joints: Path
    diagnostics: Path
    done: Path


def subshard_paths(shards_dir: Path, chunk_id: int) -> SubshardPaths:
    base = Path(shards_dir) / f"subshard_{chunk_id:04d}"
    return SubshardPaths(
        commands=base.with_suffix(".commands.npy"),
        joints=base.with_suffix(".joints.npy"),
        diagnostics=base.with_suffix(".diagnostics.npy"),
        done=base.with_suffix(".done"),
    )


def is_chunk_done(shards_dir: Path, chunk_id: int) -> bool:
    return subshard_paths(shards_dir, chunk_id).done.exists()


def _atomic_write_done(done_path: Path, payload: dict) -> None:
    tmp = done_path.with_suffix(".done.tmp")
    tmp.write_text(json.dumps(payload))
    with open(tmp, "rb") as fh:
        os.fsync(fh.fileno())
    os.replace(tmp, done_path)


def process_chunk(
    state: WorkerState,
    chunk_id: int,
    commands: np.ndarray,  # (n_cells, 4) float32 — the chunk's cells
    shards_dir: Path,
    threshold: float,
    max_iter: int,
    save_diagnostics: bool,
    report_failed_commands: bool = False,
    dt: float = DEFAULT_DT,
    damping: float = DEFAULT_DAMPING,
    solver: str = DEFAULT_SOLVER,
) -> tuple[int, int]:
    """Process one chunk; return (n_attempted, n_converged).

    A chunk with an existing ``.done`` sentinel is skipped wholesale and
    returns ``(0, 0)``. Partial files (no sentinel) are silently overwritten.

    When ``report_failed_commands`` is true, the sentinel also includes a
    ``"failed_commands"`` list of ``[roll, pitch, yaw, height]`` for cells that
    did not converge — persisted per-shard so resumed runs preserve them.
    """
    shards_dir = Path(shards_dir)
    shards_dir.mkdir(parents=True, exist_ok=True)
    paths = subshard_paths(shards_dir, chunk_id)
    if paths.done.exists():
        return 0, 0

    n_cells = int(commands.shape[0])
    cmd_buf = np.empty((n_cells, 4), dtype=np.float32)
    jnt_buf = np.empty((n_cells, 29), dtype=np.float32)
    diag_buf = (
        np.empty((n_cells, 3), dtype=np.float32) if save_diagnostics else None
    )
    failed_commands: list[list[float]] = [] if report_failed_commands else []

    n_local = 0
    for i in range(n_cells):
        t0 = time.perf_counter()
        result = solve_one_cell(
            state,
            commands[i],
            threshold=threshold,
            max_iter=max_iter,
            dt=dt,
            damping=damping,
            solver=solver,
        )
        wall_ms = (time.perf_counter() - t0) * 1e3
        if diag_buf is not None:
            diag_buf[i] = (result.final_norm, float(result.iters), float(wall_ms))
        if result.converged:
            cmd_buf[n_local] = commands[i]
            jnt_buf[n_local] = result.joints
            n_local += 1
        elif report_failed_commands:
            failed_commands.append([float(x) for x in commands[i]])

    # Write data files first, then fsync, then sentinel.
    np.save(paths.commands, cmd_buf[:n_local])
    np.save(paths.joints, jnt_buf[:n_local])
    if diag_buf is not None:
        # diag_buf keeps all n_cells rows (converged + failed); commands.npy/joints.npy
        # are sliced to n_local. Not index-aligned with each other on purpose.
        np.save(paths.diagnostics, diag_buf)
    for f in (paths.commands, paths.joints) + ((paths.diagnostics,) if diag_buf is not None else ()):
        with open(f, "rb") as fh:
            os.fsync(fh.fileno())

    sentinel: dict = {"n_attempted": n_cells, "n_converged": int(n_local)}
    if report_failed_commands:
        sentinel["failed_commands"] = failed_commands
    _atomic_write_done(paths.done, sentinel)

    return n_cells, int(n_local)


def run_worker(
    rank: int,
    chunk_specs: list[tuple[int, np.ndarray]],  # [(chunk_id, commands), ...]
    shards_dir: Path,
    model_path: str,
    threshold: float,
    max_iter: int,
    save_diagnostics: bool,
    report_failed_commands: bool = False,
    progress_queue=None,  # multiprocessing.Queue or None
) -> None:
    """Top-level worker function suitable for ``mp.Process(target=...)``."""
    state = make_worker_state(model_path)
    for chunk_id, commands in chunk_specs:
        t0 = time.perf_counter()
        n_attempted, n_converged = process_chunk(
            state=state,
            chunk_id=chunk_id,
            commands=commands,
            shards_dir=shards_dir,
            threshold=threshold,
            max_iter=max_iter,
            save_diagnostics=save_diagnostics,
            report_failed_commands=report_failed_commands,
        )
        wall_s = time.perf_counter() - t0
        if progress_queue is not None:
            progress_queue.put(
                {
                    "rank": rank,
                    "chunk_id": chunk_id,
                    "n_attempted": n_attempted,
                    "n_converged": n_converged,
                    "wall_s": wall_s,
                }
            )
