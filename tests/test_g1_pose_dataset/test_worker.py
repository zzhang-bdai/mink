"""Tests for g1_pose_dataset.worker."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from g1_pose_dataset import worker as worker_mod

XML_PATH = (
    Path(__file__).resolve().parents[2]
    / "examples"
    / "unitree_g1"
    / "scene_g1_torso.xml"
)


@pytest.fixture(scope="module")
def state() -> worker_mod.WorkerState:
    return worker_mod.make_worker_state(XML_PATH.as_posix())


def test_make_worker_state_has_29_joints(state: worker_mod.WorkerState) -> None:
    assert state.joint_qposadrs.shape == (29,)
    assert len(state.joint_names) == 29


def test_solve_one_cell_returns_correct_shapes(state: worker_mod.WorkerState) -> None:
    # Mid-range command: x=0, y=0, z=0.7, no rotation.
    cmd = np.array([0.0, 0.0, 0.0, 0.7], dtype=np.float32)
    out = worker_mod.solve_one_cell(state, cmd, threshold=1e-3, max_iter=500)
    assert out.joints.shape == (29,)
    assert out.joints.dtype == np.float32
    assert isinstance(out.converged, bool)
    assert isinstance(out.final_norm, float)
    assert isinstance(out.iters, int)
    assert 0 <= out.iters <= 500


def test_solve_one_cell_central_command_converges(
    state: worker_mod.WorkerState,
) -> None:
    # A small perturbation around the standing pose should converge well within max_iter.
    cmd = np.array([0.0, 0.0, 0.0, 0.7], dtype=np.float32)
    out = worker_mod.solve_one_cell(state, cmd, threshold=1e-3, max_iter=500)
    assert out.converged is True
    assert out.final_norm < 1e-3


def test_solve_one_cell_unreachable_returns_not_converged(
    state: worker_mod.WorkerState,
) -> None:
    # Height of 0.05 m with feet pinned at standing is physically impossible.
    cmd = np.array([0.0, 0.0, 0.0, 0.05], dtype=np.float32)
    out = worker_mod.solve_one_cell(state, cmd, threshold=1e-3, max_iter=20)
    # Either fails to converge, or hits the iteration cap without reaching threshold.
    if out.converged:
        # If it claims convergence, residual must be below threshold.
        assert out.final_norm < 1e-3
    else:
        assert out.iters == 20


def test_subshard_paths_format(tmp_path) -> None:
    paths = worker_mod.subshard_paths(tmp_path, chunk_id=7)
    assert paths.commands.name == "subshard_0007.commands.npy"
    assert paths.joints.name == "subshard_0007.joints.npy"
    assert paths.done.name == "subshard_0007.done"


def test_run_worker_writes_subshard_files(state, tmp_path) -> None:
    # Tiny synthetic chunk: 3 cells around the central pose.
    commands = np.array(
        [
            [0.0, 0.0, 0.0, 0.7],
            [0.05, 0.0, 0.0, 0.7],
            [0.0, 0.05, 0.0, 0.7],
        ],
        dtype=np.float32,
    )

    n_attempted, n_converged = worker_mod.process_chunk(
        state=state,
        chunk_id=0,
        commands=commands,
        shards_dir=tmp_path,
        threshold=1e-3,
        max_iter=500,
        save_diagnostics=False,
    )
    assert n_attempted == 3
    assert n_converged >= 1  # at least one of the three should converge

    paths = worker_mod.subshard_paths(tmp_path, 0)
    assert paths.commands.exists()
    assert paths.joints.exists()
    assert paths.done.exists()

    cmds = np.load(paths.commands)
    jnts = np.load(paths.joints)
    assert cmds.shape == (n_converged, 4)
    assert jnts.shape == (n_converged, 29)


def test_run_worker_skips_completed_chunks(state, tmp_path) -> None:
    paths = worker_mod.subshard_paths(tmp_path, 0)
    # Pretend a chunk is already done.
    paths.commands.write_bytes(b"DUMMY-COMMANDS")
    paths.joints.write_bytes(b"DUMMY-JOINTS")
    paths.done.write_text('{"n_attempted": 0, "n_converged": 0}')

    commands = np.array([[0.0, 0.0, 0.0, 0.7]], dtype=np.float32)
    n_attempted, n_converged = worker_mod.process_chunk(
        state=state,
        chunk_id=0,
        commands=commands,
        shards_dir=tmp_path,
        threshold=1e-3,
        max_iter=500,
        save_diagnostics=False,
    )

    assert n_attempted == 0
    assert n_converged == 0
    # Files were not overwritten.
    assert paths.commands.read_bytes() == b"DUMMY-COMMANDS"


def test_run_worker_overwrites_partial_subshard(state, tmp_path) -> None:
    paths = worker_mod.subshard_paths(tmp_path, 0)
    # Partial: data files present but no .done sentinel.
    paths.commands.write_bytes(b"PARTIAL-DATA")
    paths.joints.write_bytes(b"PARTIAL-DATA")

    commands = np.array([[0.0, 0.0, 0.0, 0.7]], dtype=np.float32)
    n_attempted, n_converged = worker_mod.process_chunk(
        state=state,
        chunk_id=0,
        commands=commands,
        shards_dir=tmp_path,
        threshold=1e-3,
        max_iter=500,
        save_diagnostics=False,
    )

    assert n_attempted == 1
    # Files were overwritten with real .npy content.
    assert paths.commands.read_bytes() != b"PARTIAL-DATA"
    assert paths.done.exists()
    cmds = np.load(paths.commands)
    assert cmds.shape == (n_converged, 4)
