"""Tests for g1_pose_dataset.concat."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from g1_pose_dataset import concat as concat_mod


def _write_subshard(
    shards_dir: Path,
    chunk_id: int,
    cmds: np.ndarray,
    jnts: np.ndarray,
    n_attempted: int,
    diag: np.ndarray | None = None,
) -> None:
    base = shards_dir / f"subshard_{chunk_id:04d}"
    np.save(base.with_suffix(".commands.npy"), cmds)
    np.save(base.with_suffix(".joints.npy"), jnts)
    if diag is not None:
        np.save(base.with_suffix(".diagnostics.npy"), diag)
    base.with_suffix(".done").write_text(
        json.dumps({"n_attempted": int(n_attempted), "n_converged": int(cmds.shape[0])})
    )


def test_concat_three_subshards(tmp_path) -> None:
    out_dir = tmp_path / "out"
    shards_dir = out_dir / "shards"
    shards_dir.mkdir(parents=True)

    rng = np.random.default_rng(0)
    chunks = [
        (rng.standard_normal((5, 4)).astype(np.float32),
         rng.standard_normal((5, 29)).astype(np.float32),
         5),
        (rng.standard_normal((3, 4)).astype(np.float32),
         rng.standard_normal((3, 29)).astype(np.float32),
         5),
        (rng.standard_normal((7, 4)).astype(np.float32),
         rng.standard_normal((7, 29)).astype(np.float32),
         5),
    ]
    for cid, (c, j, n_att) in enumerate(chunks):
        _write_subshard(shards_dir, cid, c, j, n_att)

    n_total = concat_mod.concat_shards(out_dir, save_diagnostics=False)
    assert n_total == 15

    cmds = np.load(out_dir / "commands.npy")
    jnts = np.load(out_dir / "joints.npy")
    np.testing.assert_array_equal(
        cmds, np.concatenate([c for c, _, _ in chunks], axis=0)
    )
    np.testing.assert_array_equal(
        jnts, np.concatenate([j for _, j, _ in chunks], axis=0)
    )


def test_concat_handles_empty_subshard(tmp_path) -> None:
    out_dir = tmp_path / "out"
    shards_dir = out_dir / "shards"
    shards_dir.mkdir(parents=True)

    _write_subshard(
        shards_dir, 0,
        np.zeros((4, 4), dtype=np.float32), np.zeros((4, 29), dtype=np.float32),
        n_attempted=10,
    )
    _write_subshard(
        shards_dir, 1,
        np.zeros((0, 4), dtype=np.float32), np.zeros((0, 29), dtype=np.float32),
        n_attempted=10,
    )
    _write_subshard(
        shards_dir, 2,
        np.zeros((2, 4), dtype=np.float32), np.zeros((2, 29), dtype=np.float32),
        n_attempted=5,
    )

    n_total = concat_mod.concat_shards(out_dir, save_diagnostics=False)
    assert n_total == 6


def test_concat_with_diagnostics(tmp_path) -> None:
    out_dir = tmp_path / "out"
    shards_dir = out_dir / "shards"
    shards_dir.mkdir(parents=True)

    cmds = np.zeros((2, 4), dtype=np.float32)
    jnts = np.zeros((2, 29), dtype=np.float32)
    diag0 = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
    diag1 = np.array([[7.0, 8.0, 9.0]], dtype=np.float32)

    _write_subshard(shards_dir, 0, cmds, jnts, 2, diag=diag0)
    _write_subshard(shards_dir, 1, np.zeros((1, 4), np.float32),
                    np.zeros((1, 29), np.float32), 1, diag=diag1)

    concat_mod.concat_shards(out_dir, save_diagnostics=True)

    diag = np.load(out_dir / "diagnostics.npy")
    np.testing.assert_array_equal(diag, np.concatenate([diag0, diag1]))


def test_concat_missing_done_raises(tmp_path) -> None:
    out_dir = tmp_path / "out"
    shards_dir = out_dir / "shards"
    shards_dir.mkdir(parents=True)
    # Only data files, no sentinel — concat must refuse.
    np.save(shards_dir / "subshard_0000.commands.npy",
            np.zeros((1, 4), np.float32))
    np.save(shards_dir / "subshard_0000.joints.npy",
            np.zeros((1, 29), np.float32))

    with pytest.raises(FileNotFoundError):
        concat_mod.concat_shards(out_dir, save_diagnostics=False)
