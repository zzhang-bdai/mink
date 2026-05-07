"""4D torso-pose command grid for the G1 dataset.

Ranges are specified in degrees / metres for human-readability, but commands
emitted by :func:`build_grid` and :func:`iter_cells` are already in
**radians + metres** (consumer-ready for ``mink.SO3.from_rpy_radians``).
Linearisation is C-order over ``(roll, pitch, yaw, height)`` so height varies
fastest; this is purely for reproducibility (the IK reset-each-cell logic is
order-independent).
"""

from __future__ import annotations

from typing import Iterator

import numpy as np

# (start, stop, step) in degrees for angles, metres for height. Half-open
# intervals (np.arange semantics).
ROLL_RANGE_DEG = (-10.0, 10.0, 1.0)
PITCH_RANGE_DEG = (-15.0, 90.0, 1.0)
YAW_RANGE_DEG = (-45.0, 45.0, 1.0)
HEIGHT_RANGE_M = (0.35, 0.80, 0.01)

DTYPE = np.float32


def axis_counts() -> tuple[int, int, int, int]:
    """Number of grid points along (roll, pitch, yaw, height)."""
    return (
        int(round((ROLL_RANGE_DEG[1] - ROLL_RANGE_DEG[0]) / ROLL_RANGE_DEG[2])),
        int(round((PITCH_RANGE_DEG[1] - PITCH_RANGE_DEG[0]) / PITCH_RANGE_DEG[2])),
        int(round((YAW_RANGE_DEG[1] - YAW_RANGE_DEG[0]) / YAW_RANGE_DEG[2])),
        int(round((HEIGHT_RANGE_M[1] - HEIGHT_RANGE_M[0]) / HEIGHT_RANGE_M[2])),
    )


def total_cells() -> int:
    nr, npi, ny, nh = axis_counts()
    return nr * npi * ny * nh


def _axis_values_radians_or_m() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    # np.arange can overshoot by one element due to floating-point accumulation
    # (e.g. HEIGHT_RANGE_M produces 46 elements instead of 45). Clip every axis
    # to axis_counts() — the authoritative source of truth.
    nr, npi, ny, nh = axis_counts()
    rolls = np.deg2rad(
        np.arange(ROLL_RANGE_DEG[0], ROLL_RANGE_DEG[1], ROLL_RANGE_DEG[2])[:nr]
    ).astype(DTYPE)
    pitches = np.deg2rad(
        np.arange(PITCH_RANGE_DEG[0], PITCH_RANGE_DEG[1], PITCH_RANGE_DEG[2])[:npi]
    ).astype(DTYPE)
    yaws = np.deg2rad(
        np.arange(YAW_RANGE_DEG[0], YAW_RANGE_DEG[1], YAW_RANGE_DEG[2])[:ny]
    ).astype(DTYPE)
    heights = np.arange(
        HEIGHT_RANGE_M[0], HEIGHT_RANGE_M[1], HEIGHT_RANGE_M[2]
    )[:nh].astype(DTYPE)
    return rolls, pitches, yaws, heights


def build_grid() -> np.ndarray:
    """Materialise the full (T, 4) command grid in radians + metres.

    Order: C-major over (roll, pitch, yaw, height) — height fastest.
    """
    rolls, pitches, yaws, heights = _axis_values_radians_or_m()
    # meshgrid with indexing="ij" then ravel preserves C-order with the named
    # axis order, so height (last) varies fastest.
    R, P, Y, H = np.meshgrid(rolls, pitches, yaws, heights, indexing="ij")
    result = np.stack(
        [R.ravel(order="C"), P.ravel(order="C"), Y.ravel(order="C"), H.ravel(order="C")],
        axis=1,
    )
    assert result.dtype == DTYPE
    return result


def iter_cells(start: int, stop: int) -> Iterator[tuple[int, np.ndarray]]:
    """Yield ``(linear_index, command_rad_m)`` for indices in ``[start, stop)``.

    Materialises the full grid then yields the requested slice. Callers that
    need many ranges should call :func:`build_grid` once and slice directly;
    that is what the worker does in production. Clips ``stop`` to
    :func:`total_cells`.
    """
    n = total_cells()
    stop = min(stop, n)
    if start >= stop:
        return
    g = build_grid()
    for i in range(start, stop):
        yield i, g[i]
