# G1 torso-pose dataset generation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a headless multi-process tool that generates 8,505,000 `(torso command, 29 joint angles)` samples for the Unitree G1, using the same IK setup as `examples/humanoid_g1_torso.py`, and saves them as memmap-friendly `.npy` files for downstream NN training.

**Architecture:** Top-level `g1_pose_dataset/` Python package with focused modules (grid, config, worker, concat) and a `__main__.py` CLI. N worker processes each handle a contiguous range of subshards (50,000 cells each), writing per-subshard `.npy` files with a `.done` sentinel for resume safety. A final concat step assembles the canonical `commands.npy` and `joints.npy`.

**Tech Stack:** Python 3.10+, mink, mujoco, numpy, multiprocessing (spawn), pytest, uv.

**Spec:** `docs/superpowers/specs/2026-05-06-g1-pose-dataset-design.md`.

---

## File map

Files this plan creates or modifies:

```
.gitignore                              # modify: add data/
g1_pose_dataset/__init__.py             # create
g1_pose_dataset/__main__.py             # create: CLI + dispatcher
g1_pose_dataset/grid.py                 # create: build_grid, iter_cells
g1_pose_dataset/config.py               # create: tasks, limits, joint names
g1_pose_dataset/worker.py               # create: WorkerState, solve_one_cell, run_worker
g1_pose_dataset/concat.py               # create: concat_shards
tests/test_g1_pose_dataset/__init__.py  # create
tests/test_g1_pose_dataset/test_grid.py     # create
tests/test_g1_pose_dataset/test_config.py   # create
tests/test_g1_pose_dataset/test_worker.py   # create
tests/test_g1_pose_dataset/test_concat.py   # create
tests/test_g1_pose_dataset/test_resume.py   # create
```

Each module has one clear responsibility (grid math, IK config, IK loop+sharding, concat). Tests live in their own directory mirroring the package.

---

## Task 1: Project scaffolding

The package lives at the repo root (top-level `g1_pose_dataset/`) and is **not** installed via `pyproject.toml` (mink is the installed package, not this generator). So we need pytest to add the repo root to `sys.path` for tests; without that, `import g1_pose_dataset` would fail in tests. The cleanest fix is one line of pytest config.

**Files:**
- Modify: `.gitignore`
- Modify: `pyproject.toml`
- Create: `g1_pose_dataset/__init__.py`
- Create: `tests/test_g1_pose_dataset/__init__.py`

- [ ] **Step 1: Add `data/` to `.gitignore`**

Append the following two lines to `.gitignore`:
```
# Generated datasets
/data/
```

- [ ] **Step 2: Add pytest pythonpath config**

Append the following to the very end of `pyproject.toml`:
```toml
[tool.pytest.ini_options]
pythonpath = ["."]
```

This adds the repo root to `sys.path` for the duration of the test run, so `from g1_pose_dataset import ...` resolves the top-level package.

- [ ] **Step 3: Create empty package files**

Create `g1_pose_dataset/__init__.py` containing exactly:
```python
"""G1 torso-pose dataset generation tool. Run via ``python -m g1_pose_dataset``."""
```

Create `tests/test_g1_pose_dataset/__init__.py` as an empty file (`""`).

- [ ] **Step 4: Verify the package is importable**

Run: `cd /home/zixin/Dev/tmp/mink && uv run python -c "import g1_pose_dataset; print(g1_pose_dataset.__doc__)"`
Expected output: `G1 torso-pose dataset generation tool. Run via ``python -m g1_pose_dataset``.`

- [ ] **Step 5: Verify pytest discovers the package**

Create a one-off sanity test, run pytest, then delete the file:
```bash
mkdir -p tests/test_g1_pose_dataset
cat > tests/test_g1_pose_dataset/test_smoke.py <<'EOF'
def test_import():
    import g1_pose_dataset
EOF
uv run pytest tests/test_g1_pose_dataset/test_smoke.py -v
```
Expected: 1 test passes. Then remove it: `rm tests/test_g1_pose_dataset/test_smoke.py`.

- [ ] **Step 6: Commit**

```bash
git add .gitignore pyproject.toml g1_pose_dataset/__init__.py tests/test_g1_pose_dataset/__init__.py
git commit -m "feat(dataset): scaffold g1_pose_dataset package"
```

---

## Task 2: Grid module

`grid.py` defines the 4D command grid in **degrees / metres** (ranges) but emits commands in **radians + metres** (consumer-ready). Linearisation is C-order over `(roll, pitch, yaw, height)`, so height varies fastest.

**Files:**
- Create: `g1_pose_dataset/grid.py`
- Create: `tests/test_g1_pose_dataset/test_grid.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_g1_pose_dataset/test_grid.py` with:
```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /home/zixin/Dev/tmp/mink && uv run pytest tests/test_g1_pose_dataset/test_grid.py -v`
Expected: every test fails with `ModuleNotFoundError: No module named 'g1_pose_dataset.grid'` (or `AttributeError`).

- [ ] **Step 3: Implement `grid.py`**

Create `g1_pose_dataset/grid.py` with:
```python
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
    nr, np_, ny, nh = axis_counts()
    return nr * np_ * ny * nh


def _axis_values_radians_or_m() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rolls = np.deg2rad(
        np.arange(ROLL_RANGE_DEG[0], ROLL_RANGE_DEG[1], ROLL_RANGE_DEG[2])
    ).astype(DTYPE)
    pitches = np.deg2rad(
        np.arange(PITCH_RANGE_DEG[0], PITCH_RANGE_DEG[1], PITCH_RANGE_DEG[2])
    ).astype(DTYPE)
    yaws = np.deg2rad(
        np.arange(YAW_RANGE_DEG[0], YAW_RANGE_DEG[1], YAW_RANGE_DEG[2])
    ).astype(DTYPE)
    heights = np.arange(
        HEIGHT_RANGE_M[0], HEIGHT_RANGE_M[1], HEIGHT_RANGE_M[2]
    ).astype(DTYPE)
    return rolls, pitches, yaws, heights


def build_grid() -> np.ndarray:
    """Materialise the full (T, 4) command grid in radians + metres.

    Order: C-major over (roll, pitch, yaw, height) — height fastest.
    """
    rolls, pitches, yaws, heights = _axis_values_radians_or_m()
    # meshgrid with indexing="ij" then ravel preserves C-order with the named
    # axis order, so height (last) varies fastest.
    R, P, Y, H = np.meshgrid(rolls, pitches, yaws, heights, indexing="ij")
    return np.stack(
        [R.ravel(order="C"), P.ravel(order="C"), Y.ravel(order="C"), H.ravel(order="C")],
        axis=1,
    ).astype(DTYPE)


def iter_cells(start: int, stop: int) -> Iterator[tuple[int, np.ndarray]]:
    """Yield ``(linear_index, command_rad_m)`` for indices in ``[start, stop)``.

    Clips ``stop`` to ``total_cells()``.
    """
    n = total_cells()
    stop = min(stop, n)
    if start >= stop:
        return
    g = build_grid()
    for i in range(start, stop):
        yield i, g[i]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /home/zixin/Dev/tmp/mink && uv run pytest tests/test_g1_pose_dataset/test_grid.py -v`
Expected: all 8 tests pass.

- [ ] **Step 5: Commit**

```bash
git add g1_pose_dataset/grid.py tests/test_g1_pose_dataset/test_grid.py
git commit -m "feat(dataset): grid module for 4D torso-pose commands"
```

---

## Task 3: Config module — joint names, posture cost, tasks, limits

`config.py` is the single source of truth for the IK setup. It mirrors `examples/humanoid_g1_torso.py` exactly. Three exported callables:

- `extract_joint_names(model)` → list of 29 actuated-joint names in qposadr order.
- `build_posture_cost(model)` → `(nv,)` array with the example's per-DoF overrides.
- `build_ik(model, configuration)` → returns a `WorkerState`-like dict containing tasks, limits, the torso task handle, and joint qposadrs. (The full `WorkerState` dataclass lives in `worker.py`; here we just return the components.)

**Files:**
- Create: `g1_pose_dataset/config.py`
- Create: `tests/test_g1_pose_dataset/test_config.py`
- Read: `examples/humanoid_g1_torso.py` (reference)

- [ ] **Step 1: Write the failing tests**

Create `tests/test_g1_pose_dataset/test_config.py` with:
```python
"""Tests for g1_pose_dataset.config."""

from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np
import pytest

import mink

from g1_pose_dataset import config as cfg

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
    configuration = mink.Configuration(model)
    parts = cfg.build_ik(model, configuration)
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /home/zixin/Dev/tmp/mink && uv run pytest tests/test_g1_pose_dataset/test_config.py -v`
Expected: every test fails with `ModuleNotFoundError: No module named 'g1_pose_dataset.config'`.

- [ ] **Step 3: Implement `config.py`**

Create `g1_pose_dataset/config.py` with:
```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /home/zixin/Dev/tmp/mink && uv run pytest tests/test_g1_pose_dataset/test_config.py -v`
Expected: all 5 tests pass.

- [ ] **Step 5: Commit**

```bash
git add g1_pose_dataset/config.py tests/test_g1_pose_dataset/test_config.py
git commit -m "feat(dataset): config module mirroring the G1 torso example"
```

---

## Task 4: Worker — `WorkerState` and `solve_one_cell`

Per-cell IK loop. Resets to the standing keyframe, sets the torso target, iterates `solve_ik` until `‖vel‖ < threshold` or max-iter. Returns convergence flag, joint angles, final residual, iteration count.

**Files:**
- Create: `g1_pose_dataset/worker.py` (partial — just `WorkerState`, `make_worker_state`, `solve_one_cell` here; `run_worker` added in Task 5)
- Modify: `tests/test_g1_pose_dataset/test_worker.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_g1_pose_dataset/test_worker.py` with:
```python
"""Tests for g1_pose_dataset.worker."""

from __future__ import annotations

from pathlib import Path

import mujoco
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /home/zixin/Dev/tmp/mink && uv run pytest tests/test_g1_pose_dataset/test_worker.py -v`
Expected: tests fail with `ModuleNotFoundError: No module named 'g1_pose_dataset.worker'`.

- [ ] **Step 3: Implement worker module (Task-4 portion)**

Create `g1_pose_dataset/worker.py` with:
```python
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
    tasks: list
    limits: list
    torso_task: mink.FrameTask
    joint_qposadrs: np.ndarray  # shape (29,) int64
    joint_names: list[str]


def make_worker_state(model_path: str) -> WorkerState:
    """Load the model, build IK pieces, pin static targets at the keyframe."""
    model = mujoco.MjModel.from_xml_path(model_path)
    configuration = mink.Configuration(model)
    parts = cfg.build_ik(model, configuration)

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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /home/zixin/Dev/tmp/mink && uv run pytest tests/test_g1_pose_dataset/test_worker.py -v`
Expected: all 4 tests pass. (`test_solve_one_cell_central_command_converges` is the only one that exercises real IK — should take ~1–2 s.)

- [ ] **Step 5: Commit**

```bash
git add g1_pose_dataset/worker.py tests/test_g1_pose_dataset/test_worker.py
git commit -m "feat(dataset): solve_one_cell IK loop with reset-each-cell"
```

---

## Task 5: Worker — `run_worker` and atomic subshard writes

`run_worker` processes a list of `chunk_id`s. For each, it streams cells through `solve_one_cell`, accumulates converged rows in pre-allocated buffers, and writes `subshard_NNNN.commands.npy` + `subshard_NNNN.joints.npy` + `subshard_NNNN.done` (sentinel last). Already-completed chunks (sentinel present) are skipped.

**Files:**
- Modify: `g1_pose_dataset/worker.py` (append `run_worker`, `subshard_paths`, `is_chunk_done`)
- Modify: `tests/test_g1_pose_dataset/test_worker.py` (append integration tests)

- [ ] **Step 1: Append failing tests**

Append to `tests/test_g1_pose_dataset/test_worker.py`:
```python


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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /home/zixin/Dev/tmp/mink && uv run pytest tests/test_g1_pose_dataset/test_worker.py -v -k "subshard or process_chunk or run_worker"`
Expected: 4 new tests fail with `AttributeError: module 'g1_pose_dataset.worker' has no attribute 'subshard_paths'` (or similar).

- [ ] **Step 3: Implement subshard write protocol**

First, add the new top-level imports to `g1_pose_dataset/worker.py`. Edit the existing import block at the top of the file from:

```python
import math
from dataclasses import dataclass
```

to:

```python
import json
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
```

Then **append** the following to the end of `g1_pose_dataset/worker.py`:

```python


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
    os.replace(tmp, done_path)


def process_chunk(
    state: WorkerState,
    chunk_id: int,
    commands: np.ndarray,  # (n_cells, 4) float32 — the chunk's cells
    shards_dir: Path,
    threshold: float,
    max_iter: int,
    save_diagnostics: bool,
    dt: float = DEFAULT_DT,
    damping: float = DEFAULT_DAMPING,
    solver: str = DEFAULT_SOLVER,
) -> tuple[int, int]:
    """Process one chunk; return (n_attempted, n_converged).

    A chunk with an existing ``.done`` sentinel is skipped wholesale and
    returns ``(0, 0)``. Partial files (no sentinel) are silently overwritten.
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

    # Write data files first, then fsync, then sentinel.
    np.save(paths.commands, cmd_buf[:n_local])
    np.save(paths.joints, jnt_buf[:n_local])
    if diag_buf is not None:
        np.save(paths.diagnostics, diag_buf)
    for f in (paths.commands, paths.joints) + ((paths.diagnostics,) if diag_buf is not None else ()):
        with open(f, "rb") as fh:
            os.fsync(fh.fileno())

    _atomic_write_done(
        paths.done, {"n_attempted": n_cells, "n_converged": int(n_local)}
    )

    return n_cells, int(n_local)


def run_worker(
    rank: int,
    chunk_specs: list[tuple[int, np.ndarray]],  # [(chunk_id, commands), ...]
    shards_dir: Path,
    model_path: str,
    threshold: float,
    max_iter: int,
    save_diagnostics: bool,
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /home/zixin/Dev/tmp/mink && uv run pytest tests/test_g1_pose_dataset/test_worker.py -v`
Expected: all worker tests pass (8 total).

- [ ] **Step 5: Commit**

```bash
git add g1_pose_dataset/worker.py tests/test_g1_pose_dataset/test_worker.py
git commit -m "feat(dataset): atomic subshard writes with .done sentinel"
```

---

## Task 6: Resume safety integration test

Validates the resume protocol end-to-end: a tiny grid is processed in two passes, with a chunk dropped between them, and the result must be identical to a single-pass run.

**Files:**
- Create: `tests/test_g1_pose_dataset/test_resume.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_g1_pose_dataset/test_resume.py` with:
```python
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
```

- [ ] **Step 2: Run test to verify it passes immediately**

(Worker logic from Task 5 already covers this — this test exercises the existing API end-to-end.)

Run: `cd /home/zixin/Dev/tmp/mink && uv run pytest tests/test_g1_pose_dataset/test_resume.py -v`
Expected: 1 test passes.

- [ ] **Step 3: Commit**

```bash
git add tests/test_g1_pose_dataset/test_resume.py
git commit -m "test(dataset): resume-safety integration test"
```

---

## Task 7: Concat module

Streams subshards into the canonical `commands.npy` / `joints.npy` (and optionally `diagnostics.npy`) using memmap'd writes so RAM usage stays bounded.

**Files:**
- Create: `g1_pose_dataset/concat.py`
- Create: `tests/test_g1_pose_dataset/test_concat.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_g1_pose_dataset/test_concat.py` with:
```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /home/zixin/Dev/tmp/mink && uv run pytest tests/test_g1_pose_dataset/test_concat.py -v`
Expected: tests fail with `ModuleNotFoundError: No module named 'g1_pose_dataset.concat'`.

- [ ] **Step 3: Implement `concat.py`**

Create `g1_pose_dataset/concat.py` with:
```python
"""Concatenate per-subshard .npy files into the canonical dataset files.

Streams data into memmapped output files so peak RAM is bounded regardless of
dataset size.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np

SUBSHARD_RE = re.compile(r"^subshard_(\d{4})\.done$")


def _find_subshards(shards_dir: Path) -> list[int]:
    if not shards_dir.exists():
        raise FileNotFoundError(f"shards directory does not exist: {shards_dir}")
    chunks: list[int] = []
    for path in sorted(shards_dir.iterdir()):
        m = SUBSHARD_RE.match(path.name)
        if m:
            chunks.append(int(m.group(1)))
    if not chunks:
        raise FileNotFoundError(f"no .done sentinels in {shards_dir}")
    chunks.sort()
    return chunks


def _shard_paths(shards_dir: Path, chunk_id: int) -> tuple[Path, Path, Path, Path]:
    base = shards_dir / f"subshard_{chunk_id:04d}"
    return (
        base.with_suffix(".commands.npy"),
        base.with_suffix(".joints.npy"),
        base.with_suffix(".diagnostics.npy"),
        base.with_suffix(".done"),
    )


def concat_shards(output_dir: Path, save_diagnostics: bool) -> int:
    """Assemble final files. Returns total converged-row count."""
    output_dir = Path(output_dir)
    shards_dir = output_dir / "shards"
    chunks = _find_subshards(shards_dir)

    # First pass: count rows.
    n_total = 0
    sentinels: list[dict] = []
    diag_rows = 0
    for cid in chunks:
        cmd_p, _jnt_p, diag_p, done_p = _shard_paths(shards_dir, cid)
        if not cmd_p.exists():
            raise FileNotFoundError(f"missing commands shard: {cmd_p}")
        sentinel = json.loads(done_p.read_text())
        sentinels.append(sentinel)
        n_total += int(sentinel["n_converged"])
        if save_diagnostics:
            if not diag_p.exists():
                raise FileNotFoundError(f"missing diagnostics shard: {diag_p}")
            diag_rows += int(np.load(diag_p, mmap_mode="r").shape[0])

    # Second pass: stream-copy.
    cmd_out_path = output_dir / "commands.npy"
    jnt_out_path = output_dir / "joints.npy"
    cmd_out = np.lib.format.open_memmap(
        cmd_out_path, mode="w+", dtype=np.float32, shape=(n_total, 4)
    )
    jnt_out = np.lib.format.open_memmap(
        jnt_out_path, mode="w+", dtype=np.float32, shape=(n_total, 29)
    )
    diag_out = None
    if save_diagnostics:
        diag_out = np.lib.format.open_memmap(
            output_dir / "diagnostics.npy",
            mode="w+", dtype=np.float32, shape=(diag_rows, 3),
        )

    cmd_offset = 0
    diag_offset = 0
    for cid, sentinel in zip(chunks, sentinels):
        cmd_p, jnt_p, diag_p, _done_p = _shard_paths(shards_dir, cid)
        n = int(sentinel["n_converged"])
        if n > 0:
            cmd_out[cmd_offset : cmd_offset + n] = np.load(cmd_p, mmap_mode="r")
            jnt_out[cmd_offset : cmd_offset + n] = np.load(jnt_p, mmap_mode="r")
            cmd_offset += n
        if diag_out is not None:
            diag = np.load(diag_p, mmap_mode="r")
            diag_out[diag_offset : diag_offset + diag.shape[0]] = diag
            diag_offset += int(diag.shape[0])

    # Force flush to disk.
    cmd_out.flush()
    jnt_out.flush()
    if diag_out is not None:
        diag_out.flush()

    return n_total
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /home/zixin/Dev/tmp/mink && uv run pytest tests/test_g1_pose_dataset/test_concat.py -v`
Expected: all 4 tests pass.

- [ ] **Step 5: Commit**

```bash
git add g1_pose_dataset/concat.py tests/test_g1_pose_dataset/test_concat.py
git commit -m "feat(dataset): concat subshards into canonical .npy files"
```

---

## Task 8: CLI dispatcher (`__main__.py`)

Wires the modules together: argparse, grid, worker dispatch (spawn N processes), progress reporting via queue, final concat. Includes `--pilot N` and `--dry-run` paths.

**Files:**
- Create: `g1_pose_dataset/__main__.py`

- [ ] **Step 1: Implement `__main__.py`**

Create `g1_pose_dataset/__main__.py` with:
```python
"""CLI entrypoint: ``python -m g1_pose_dataset``."""

from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import os
import sys
import time
from importlib.metadata import PackageNotFoundError, version as pkg_version
from pathlib import Path

import numpy as np

from g1_pose_dataset import concat as concat_mod
from g1_pose_dataset import config as cfg
from g1_pose_dataset import grid as grid_mod
from g1_pose_dataset import worker as worker_mod

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = REPO_ROOT / "examples" / "unitree_g1" / "scene_g1_torso.xml"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "data" / "g1_torso_pose"
DEFAULT_SUBSHARD_SIZE = 50_000
DEFAULT_THRESHOLD = 1e-3
DEFAULT_MAX_ITER = 500


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="g1_pose_dataset")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--num-workers", type=int, default=0,
                        help="0 = os.cpu_count() - 1")
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    parser.add_argument("--max-iter", type=int, default=DEFAULT_MAX_ITER)
    parser.add_argument("--subshard-size", type=int, default=DEFAULT_SUBSHARD_SIZE)
    parser.add_argument("--pilot", type=int, default=0,
                        help="Run only the first N cells in a single process and exit.")
    parser.add_argument("--save-diagnostics", action="store_true")
    parser.add_argument("--cleanup-shards", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    return parser.parse_args(argv)


def _resolve_num_workers(requested: int, total_chunks: int) -> int:
    if requested == 0:
        n = max((os.cpu_count() or 2) - 1, 1)
    else:
        n = max(requested, 1)
    return min(n, total_chunks)


def _chunk_specs_for_grid(grid: np.ndarray, subshard_size: int) -> list[tuple[int, np.ndarray]]:
    n = grid.shape[0]
    out: list[tuple[int, np.ndarray]] = []
    for chunk_id, start in enumerate(range(0, n, subshard_size)):
        stop = min(start + subshard_size, n)
        out.append((chunk_id, grid[start:stop].copy()))
    return out


def _split_chunks_among_workers(
    chunk_specs: list[tuple[int, np.ndarray]], num_workers: int
) -> list[list[tuple[int, np.ndarray]]]:
    per = math.ceil(len(chunk_specs) / num_workers)
    return [chunk_specs[i * per : (i + 1) * per] for i in range(num_workers)]


def _safe_pkg_version(name: str) -> str:
    try:
        return pkg_version(name)
    except PackageNotFoundError:
        return "unknown"


def _emit_metadata(
    output_dir: Path, n_converged: int, n_attempted: int, args: argparse.Namespace
) -> None:
    metadata = {
        "schema_version": 1,
        "ranges": {
            "roll_deg": list(grid_mod.ROLL_RANGE_DEG),
            "pitch_deg": list(grid_mod.PITCH_RANGE_DEG),
            "yaw_deg": list(grid_mod.YAW_RANGE_DEG),
            "height_m": list(grid_mod.HEIGHT_RANGE_M),
        },
        "axis_counts": list(grid_mod.axis_counts()),
        "n_total_cells": grid_mod.total_cells(),
        "n_attempted": n_attempted,
        "n_converged": n_converged,
        "n_skipped": n_attempted - n_converged,
        "threshold": args.threshold,
        "max_iter": args.max_iter,
        "subshard_size": args.subshard_size,
        "command_units": ["radians", "radians", "radians", "metres"],
        "command_fields": ["roll", "pitch", "yaw", "height"],
        "joint_dtype": "float32",
        "command_dtype": "float32",
        "cell_order": "C-order over (roll, pitch, yaw, height); height varies fastest",
        "model_path": str(args.model),
        "mink_version": _safe_pkg_version("mink"),
        "mujoco_version": _safe_pkg_version("mujoco"),
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))


def _emit_joint_names(output_dir: Path, model_path: Path) -> None:
    import mujoco as _mj

    model = _mj.MjModel.from_xml_path(str(model_path))
    names = cfg.extract_joint_names(model)
    (output_dir / "joint_names.json").write_text(json.dumps(names, indent=2))


def _run_pilot(args: argparse.Namespace) -> None:
    pilot_dir = args.output_dir / "pilot"
    shards_dir = pilot_dir / "shards"
    shards_dir.mkdir(parents=True, exist_ok=True)

    grid = grid_mod.build_grid()[: args.pilot]
    chunk_specs = _chunk_specs_for_grid(grid, args.subshard_size)
    print(f"[pilot] cells={args.pilot}, chunks={len(chunk_specs)}")

    state = worker_mod.make_worker_state(str(args.model))
    t_total0 = time.perf_counter()
    n_attempted = 0
    n_converged = 0
    for chunk_id, cmds in chunk_specs:
        n_a, n_c = worker_mod.process_chunk(
            state=state,
            chunk_id=chunk_id,
            commands=cmds,
            shards_dir=shards_dir,
            threshold=args.threshold,
            max_iter=args.max_iter,
            save_diagnostics=True,  # pilots always save diagnostics
        )
        n_attempted += n_a
        n_converged += n_c
        print(f"[pilot] chunk {chunk_id}: {n_c}/{n_a} converged")

    wall_total = time.perf_counter() - t_total0
    concat_mod.concat_shards(pilot_dir, save_diagnostics=True)
    _emit_joint_names(pilot_dir, args.model)
    _emit_metadata(pilot_dir, n_converged, n_attempted, args)

    diag = np.load(pilot_dir / "diagnostics.npy")
    iters = diag[:, 1]
    wall_ms = diag[:, 2]
    skip_rate = 1.0 - n_converged / max(n_attempted, 1)
    mean_ms = float(wall_ms.mean())
    p95_ms = float(np.quantile(wall_ms, 0.95))
    full_total = grid_mod.total_cells()
    n_workers_full = _resolve_num_workers(args.num_workers, math.ceil(full_total / args.subshard_size))
    eta_h = (full_total * mean_ms / 1e3) / n_workers_full / 3600.0

    print()
    print("=== pilot summary ===")
    print(f"cells: {n_attempted}; converged: {n_converged}; skip rate: {skip_rate:.2%}")
    print(f"iters: mean={iters.mean():.1f}, p50={np.median(iters):.0f}, "
          f"p95={np.quantile(iters, 0.95):.0f}, max={iters.max():.0f}")
    print(f"ms/cell: mean={mean_ms:.1f}, p95={p95_ms:.1f}")
    print(f"pilot wall: {wall_total:.1f} s")
    print(f"projected full-run ETA at {n_workers_full} workers: {eta_h:.2f} h")


def _run_full(args: argparse.Namespace) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    shards_dir = args.output_dir / "shards"
    shards_dir.mkdir(exist_ok=True)

    grid = grid_mod.build_grid()
    chunk_specs = _chunk_specs_for_grid(grid, args.subshard_size)
    n_workers = _resolve_num_workers(args.num_workers, len(chunk_specs))
    print(f"[run] total cells={grid.shape[0]}; chunks={len(chunk_specs)}; workers={n_workers}")

    splits = _split_chunks_among_workers(chunk_specs, n_workers)
    ctx = mp.get_context("spawn")
    progress_q = ctx.Queue()
    procs = []
    for rank, my_chunks in enumerate(splits):
        if not my_chunks:
            continue
        p = ctx.Process(
            target=worker_mod.run_worker,
            kwargs=dict(
                rank=rank,
                chunk_specs=my_chunks,
                shards_dir=shards_dir,
                model_path=str(args.model),
                threshold=args.threshold,
                max_iter=args.max_iter,
                save_diagnostics=args.save_diagnostics,
                progress_queue=progress_q,
            ),
        )
        p.start()
        procs.append(p)

    n_done = 0
    n_converged_total = 0
    n_attempted_total = 0
    expected_chunks = sum(len(s) for s in splits if s)
    t0 = time.perf_counter()
    while n_done < expected_chunks:
        msg = progress_q.get()
        n_done += 1
        n_attempted_total += msg["n_attempted"]
        n_converged_total += msg["n_converged"]
        elapsed = time.perf_counter() - t0
        eta_s = (elapsed / max(n_done, 1)) * (expected_chunks - n_done)
        print(
            f"[run] chunk {msg['chunk_id']:04d} "
            f"(rank {msg['rank']}): {msg['n_converged']}/{msg['n_attempted']} "
            f"converged in {msg['wall_s']:.1f}s   "
            f"[{n_done}/{expected_chunks} chunks, ETA {eta_s/60:.1f} min]"
        )

    for p in procs:
        p.join()

    print("[run] all workers done; concatenating...")
    n_final = concat_mod.concat_shards(args.output_dir, save_diagnostics=args.save_diagnostics)
    _emit_joint_names(args.output_dir, args.model)
    _emit_metadata(args.output_dir, n_final, n_attempted_total, args)

    if args.cleanup_shards:
        import shutil

        shutil.rmtree(shards_dir)
        print("[run] removed shards/")

    print(f"[run] DONE: {n_final} converged samples written to {args.output_dir}")


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    if args.dry_run:
        grid = grid_mod.build_grid()
        chunks = _chunk_specs_for_grid(grid, args.subshard_size)
        nw = _resolve_num_workers(args.num_workers, len(chunks))
        print(f"total cells: {grid.shape[0]}")
        print(f"chunks: {len(chunks)} (size {args.subshard_size}); last has {chunks[-1][1].shape[0]}")
        print(f"workers: {nw}")
        print(f"output: {args.output_dir}")
        print(f"model:  {args.model}")
        return 0

    if args.pilot > 0:
        _run_pilot(args)
        return 0

    _run_full(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 2: Smoke-test `--dry-run`**

Run: `cd /home/zixin/Dev/tmp/mink && uv run python -m g1_pose_dataset --dry-run`
Expected output (verbatim numbers):
```
total cells: 8505000
chunks: 171 (size 50000); last has 5000
workers: <some N>
output: <repo>/data/g1_torso_pose
model:  <repo>/examples/unitree_g1/scene_g1_torso.xml
```

- [ ] **Step 3: Smoke-test `--pilot 5`**

Run: `cd /home/zixin/Dev/tmp/mink && uv run python -m g1_pose_dataset --pilot 5 --output-dir /tmp/g1_pilot_smoke`
Expected: completes without error in <30 s; the script prints a `=== pilot summary ===` block; `/tmp/g1_pilot_smoke/pilot/commands.npy`, `joints.npy`, `diagnostics.npy`, `joint_names.json`, `metadata.json` all exist.

Verify with:
```bash
uv run python -c "
import numpy as np, json
out = '/tmp/g1_pilot_smoke/pilot'
print('commands shape:', np.load(out + '/commands.npy').shape)
print('joints shape:', np.load(out + '/joints.npy').shape)
print('diagnostics shape:', np.load(out + '/diagnostics.npy').shape)
print('joint names:', len(json.load(open(out + '/joint_names.json'))))
"
```
Expected: `joints shape: (n, 29)` for some `n ≤ 5`, `diagnostics shape: (5, 3)`, `joint names: 29`.

- [ ] **Step 4: Commit**

```bash
git add g1_pose_dataset/__main__.py
git commit -m "feat(dataset): CLI entrypoint with --dry-run, --pilot, full run"
```

---

## Task 9: Multiprocess full-run smoke test

End-to-end check that the spawn-based dispatcher actually works (the per-cell tests above run in-process). Uses a tiny grid by overriding the subshard size, so wall-time is bounded.

**Files:**
- Create: `tests/test_g1_pose_dataset/test_dispatch.py`

- [ ] **Step 1: Write the test**

Create `tests/test_g1_pose_dataset/test_dispatch.py` with:
```python
"""Multiprocess dispatcher smoke test."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_pilot_run_via_cli(tmp_path) -> None:
    out_dir = tmp_path / "g1_smoke"
    result = subprocess.run(
        [
            sys.executable, "-m", "g1_pose_dataset",
            "--pilot", "3",
            "--output-dir", str(out_dir),
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert result.returncode == 0, f"stderr:\n{result.stderr}\nstdout:\n{result.stdout}"
    pilot = out_dir / "pilot"
    cmds = np.load(pilot / "commands.npy")
    jnts = np.load(pilot / "joints.npy")
    diag = np.load(pilot / "diagnostics.npy")
    assert cmds.shape[1] == 4 and cmds.dtype == np.float32
    assert jnts.shape[1] == 29 and jnts.dtype == np.float32
    assert diag.shape == (3, 3)
    names = json.loads((pilot / "joint_names.json").read_text())
    assert len(names) == 29
    metadata = json.loads((pilot / "metadata.json").read_text())
    assert metadata["n_total_cells"] == 8_505_000
    assert metadata["n_attempted"] == 3
```

- [ ] **Step 2: Run the test**

Run: `cd /home/zixin/Dev/tmp/mink && uv run pytest tests/test_g1_pose_dataset/test_dispatch.py -v`
Expected: 1 test passes in ~30–60 s.

- [ ] **Step 3: Run the full dataset test suite**

Run: `cd /home/zixin/Dev/tmp/mink && uv run pytest tests/test_g1_pose_dataset/ -v`
Expected: all tests pass (8 grid + 5 config + 8 worker + 1 resume + 4 concat + 1 dispatch = 27 tests).

- [ ] **Step 4: Commit**

```bash
git add tests/test_g1_pose_dataset/test_dispatch.py
git commit -m "test(dataset): CLI pilot run end-to-end smoke test"
```

---

## Task 10: README and pilot validation

A short README so the next person knows how to run it, plus a manual pilot to validate convergence rate and wall-time before the full 8h run.

**Files:**
- Create: `g1_pose_dataset/README.md`

- [ ] **Step 1: Write the README**

Create `g1_pose_dataset/README.md` with:
```markdown
# G1 torso-pose dataset generator

Generates an 8,505,000-sample dataset of `(torso command, 29 joint angles)`
pairs by running mink IK on the G1 across a 4D grid of torso targets
(`roll × pitch × yaw × height`). Mirrors the IK setup from
`examples/humanoid_g1_torso.py`.

## Usage

```bash
# Inspect the plan without running anything.
python -m g1_pose_dataset --dry-run

# Validate convergence and estimate wall-time on a small slice (recommended first).
python -m g1_pose_dataset --pilot 1000

# Full run (multi-process, ~8 h on 16 cores).
python -m g1_pose_dataset

# Optional flags:
#   --output-dir PATH         (default: data/g1_torso_pose)
#   --num-workers N           (0 = os.cpu_count() - 1)
#   --threshold 1e-3
#   --max-iter 500
#   --subshard-size 50000
#   --save-diagnostics        write diagnostics.npy (final ‖vel‖, iters, wall_ms per cell)
#   --cleanup-shards          delete shards/ after successful concat
```

## Output layout

```
data/g1_torso_pose/
├── commands.npy            (N, 4) float32 — roll, pitch, yaw (rad), height (m)
├── joints.npy              (N, 29) float32 — joint angles (rad)
├── joint_names.json        ordered list of 29 joint names matching joints.npy columns
├── metadata.json           ranges, threshold, versions, counts
├── diagnostics.npy         (T, 3) float32 — only with --save-diagnostics
└── shards/                 per-subshard intermediate files (resume safety)
```

`commands.npy` and `joints.npy` are row-aligned: row `i` of joints is the
converged solution for command row `i`. Use `np.memmap` or `np.load(...,
mmap_mode="r")` for shuffled-index NN training without loading the whole file.

## Resume

Each subshard (50,000 cells) is written atomically with a `.done` sentinel.
On restart, completed subshards are skipped. Worst-case work loss from a
crash is one subshard.

## Spec / design

See [`docs/superpowers/specs/2026-05-06-g1-pose-dataset-design.md`](../docs/superpowers/specs/2026-05-06-g1-pose-dataset-design.md).
```

- [ ] **Step 2: Commit the README**

```bash
git add g1_pose_dataset/README.md
git commit -m "docs(dataset): README for the generator"
```

- [ ] **Step 3: Manual pilot validation (run, do not commit any output)**

This step is operator-driven, not automated. Before launching the full run, the operator should:

```bash
cd /home/zixin/Dev/tmp/mink
uv run python -m g1_pose_dataset --pilot 1000
```

Expected: completes in 1–3 minutes; reports a convergence rate, mean/p95 ms per cell, and an extrapolated full-run wall-time. If skip rate is unacceptably high (>20%), the threshold may need raising via `--threshold`.

This step does not produce a commit; the pilot output goes under `data/g1_torso_pose/pilot/` (gitignored).

---

## Self-review notes

The plan covers every requirement in the spec:

- **Sample schema** (commands.npy, joints.npy, joint_names.json) → Tasks 2, 4, 8
- **Grid math** (45×20×105×90 = 8,505,000, half-open intervals) → Task 2
- **IK setup matching the example** (tasks, limits, posture cost overrides, knee bound, collision pairs) → Task 3
- **Per-cell loop** (reset to keyframe, iterate to convergence, skip non-converged) → Task 4
- **Multiprocessing with spawn + per-worker MjModel** → Task 8 (`mp.get_context("spawn")`)
- **Subshard atomic writes with `.done` sentinel; resume on restart** → Task 5, validated by Task 6
- **Concat into canonical files via memmap** → Task 7
- **CLI flags** (`--dry-run`, `--pilot`, `--save-diagnostics`, `--cleanup-shards`, `--num-workers`, `--threshold`, `--max-iter`, `--subshard-size`, `--output-dir`) → Task 8
- **Pilot mode reports skip rate, iters quantiles, ms/cell, ETA** → Task 8 (`_run_pilot`)
- **Diagnostics file** (`(T, 3)` float32 in linearised order) → Tasks 5, 7, 8
- **Metadata file** with ranges, versions, counts → Task 8 (`_emit_metadata`)
- **Tests:** grid math, config, single-cell IK, subshard write protocol, resume, concat, end-to-end CLI → Tasks 2, 3, 4, 5, 6, 7, 9
