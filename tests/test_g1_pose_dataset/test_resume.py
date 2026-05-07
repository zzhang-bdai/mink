"""Resume-safety integration tests."""

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


def _make_tiny_grid() -> np.ndarray:
    # 6 reasonable commands; values chosen so most should converge.
    return np.array(
        [
            [0.0, 0.0, 0.0, 0.70],
            [0.0, 0.0, 0.0, 0.72],
            [0.0, 0.0, 0.0, 0.74],
            [0.0, 0.05, 0.0, 0.70],
            [0.05, 0.0, 0.0, 0.70],
            [0.0, 0.0, 0.05, 0.70],
        ],
        dtype=np.float32,
    )


def test_resume_after_dropping_a_chunk(state, tmp_path) -> None:
    grid = _make_tiny_grid()
    chunk_a = grid[:3]
    chunk_b = grid[3:]

    # Pass 1: process both chunks.
    worker_mod.process_chunk(
        state, 0, chunk_a, tmp_path, 1e-3, 500, save_diagnostics=False
    )
    worker_mod.process_chunk(
        state, 1, chunk_b, tmp_path, 1e-3, 500, save_diagnostics=False
    )

    # Snapshot results.
    paths1_a = worker_mod.subshard_paths(tmp_path, 0)
    paths1_b = worker_mod.subshard_paths(tmp_path, 1)
    cmds_a_first = np.load(paths1_a.commands).copy()
    jnts_a_first = np.load(paths1_a.joints).copy()
    cmds_b_first = np.load(paths1_b.commands).copy()

    # Simulate a crash: remove chunk B's .done sentinel and data files (partial).
    paths1_b.done.unlink()
    paths1_b.commands.unlink()
    paths1_b.joints.unlink()

    # Pass 2: rerun both chunks. Chunk A should be skipped entirely.
    worker_mod.process_chunk(
        state, 0, chunk_a, tmp_path, 1e-3, 500, save_diagnostics=False
    )
    worker_mod.process_chunk(
        state, 1, chunk_b, tmp_path, 1e-3, 500, save_diagnostics=False
    )

    # Chunk A's files were not touched (skipped via .done sentinel).
    np.testing.assert_array_equal(np.load(paths1_a.commands), cmds_a_first)
    np.testing.assert_array_equal(np.load(paths1_a.joints), jnts_a_first)

    # Chunk B was reprocessed — content should match the first pass exactly
    # (deterministic IK: same input cells → same output joints).
    np.testing.assert_allclose(
        np.load(paths1_b.commands), cmds_b_first, atol=1e-7
    )
    assert paths1_b.done.exists()
