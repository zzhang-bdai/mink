"""Tests for g1_pose_dataset.grid."""

from __future__ import annotations

import numpy as np

from g1_pose_dataset import grid as grid_mod


def test_total_cells_is_8_505_000() -> None:
    assert grid_mod.total_cells() == 8_505_000


def test_axis_counts() -> None:
    assert grid_mod.axis_counts() == (20, 105, 90, 45)


def test_build_grid_shape_and_dtype() -> None:
    g = grid_mod.build_grid()
    assert g.shape == (8_505_000, 4)
    assert g.dtype == np.float32


def test_first_cell_is_min_corner_in_radians() -> None:
    g = grid_mod.build_grid()
    np.testing.assert_allclose(
        g[0],
        np.array([np.deg2rad(-10.0), np.deg2rad(-15.0), np.deg2rad(-45.0), 0.35], dtype=np.float32),
        atol=1e-7,
    )


def test_last_cell_is_max_minus_step() -> None:
    g = grid_mod.build_grid()
    # Half-open intervals: last roll is 9 deg, last pitch is 89, last yaw is 44, last height is 0.79.
    np.testing.assert_allclose(
        g[-1],
        np.array([np.deg2rad(9.0), np.deg2rad(89.0), np.deg2rad(44.0), 0.79], dtype=np.float32),
        atol=1e-6,
    )


def test_height_varies_fastest() -> None:
    g = grid_mod.build_grid()
    # First two rows differ only in height (last column).
    assert g[0, 3] != g[1, 3]
    np.testing.assert_array_equal(g[0, :3], g[1, :3])
    # Cell 45 advances yaw by one step (45 heights wrap), height resets.
    assert g[45, 2] != g[0, 2]
    assert g[45, 3] == g[0, 3]


def test_iter_cells_yields_correct_range() -> None:
    g = grid_mod.build_grid()
    cells = list(grid_mod.iter_cells(100, 105))
    assert len(cells) == 5
    for offset, (idx, cmd) in enumerate(cells):
        assert idx == 100 + offset
        np.testing.assert_array_equal(cmd, g[100 + offset])


def test_iter_cells_clips_at_total() -> None:
    cells = list(grid_mod.iter_cells(8_504_999, 8_505_010))
    assert len(cells) == 1


def test_iter_cells_empty_range() -> None:
    cells = list(grid_mod.iter_cells(500, 500))
    assert cells == []
