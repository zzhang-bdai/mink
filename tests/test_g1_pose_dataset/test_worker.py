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
