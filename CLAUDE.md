# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

`AGENTS.md` covers project structure, the `make` targets (`sync`, `format`, `type`, `test`, `coverage`, `doc`, `build`), coding style, and commit/PR conventions — read it for the basics. This file documents the architectural flow and quirks that take multiple files to piece together.

## Big-picture IK pipeline

A typical control step assembles three things and integrates the result:

1. **`Configuration`** (`src/mink/configuration.py`) wraps `mujoco.MjModel` + `MjData`. `update()` runs `mj_kinematics` + `mj_comPos` (and `mj_makeConstraint` if `model.neq > 0`). It exposes per-frame transforms, body-frame Jacobians, and the joint-space inertia matrix.
2. **`Task`s** (`src/mink/tasks/`) each implement `compute_error(config)` and `compute_jacobian(config)`. The `Task` base class turns those into a weighted QP objective `(H, c)` via `_assemble_qp` (cost vector + gain α + optional Levenberg–Marquardt damping that activates only on large errors). `BaseTask` is the lighter contract — `DampingTask` and `KineticEnergyRegularizationTask` skip error/Jacobian and contribute `(H, c)` directly.
3. **`Limit`s** (`src/mink/limits/`) each implement `compute_qp_inequalities(config, dt)` returning a `Constraint(G, h, inactive)`.

`solve_ik(config, tasks, dt, solver, damping, limits, constraints)` (`src/mink/solve_ik.py`) stacks the tasks' objectives, the limits' inequalities, and any tasks passed via `constraints=` (treated as **equality** constraints `A Δq = b` rather than least-squares), hands the result to `qpsolvers.solve_problem`, and returns a **velocity in tangent space** (`Δq / dt`). The caller is responsible for `configuration.integrate_inplace(vel, dt)` — `solve_ik` does not step the configuration. The default backend in examples is `daqp`.

## Body-frame Jacobian convention

`Configuration.get_frame_jacobian` (configuration.py:166–173) takes MuJoCo's world-aligned-but-body-centered Jacobian and left-multiplies it by the SE3 adjoint of the inverse frame transform to return a true **body Jacobian**. This is intentional and matches the `Task` math; do not "simplify" it back to the raw MuJoCo call.

## Lie groups and the native C extension

`src/mink/lie/` defines `SO3` and `SE3` as immutable dataclasses. SE3's internal parameterization is `wxyz_xyz` — `(qw, qx, qy, qz, x, y, z)` — and its tangent is `(vx, vy, vz, ωx, ωy, ωz)`. Most other robotics libraries use `xyzw`; preserve this when interoperating.

`src/mink/lie/_lie_ops_c.c` is a NumPy C extension providing fused SE3/SO3 operations on raw `double` arrays for the IK hot path. `Configuration` and `FrameTask`/`RelativeFrameTask` import it as `_native` and pass raw `ndarray[7]` end-to-end when it's available. Two ways to fall back to pure Python:

- Set `MINK_DISABLE_NATIVE=1` (also referenced in `tool.coverage.report.exclude_also`) — useful when debugging numeric mismatches between the two paths.
- Source-install without a C compiler: scikit-build-core's `if.failed = true` override in `pyproject.toml` falls back to a pure-Python wheel.

Equation-number comments in `lie/base.py` (e.g. "Eqn. 25") refer to Solà/Deray/Atchuthan, *A micro Lie theory for state estimation in robotics* (arXiv:1812.01537).

## Tests and types

- Single test file: `uv run pytest tests/test_frame_task.py`. Single test method: `uv run pytest tests/test_frame_task.py::TestFrameTask::test_<name>`. Most test classes use `absltest.TestCase` with `setUpClass` loading a robot via `robot_descriptions.loaders.mujoco.load_robot_description` — that's the canonical way to bring up a real `MjModel` in tests.
- CI runs **both** `pyright` and `ty` (the preview Astral type checker) across Python 3.10–3.13. `make type` runs both. Don't drop one when fixing the other.
- Coverage is currently 100%; new code generally needs tests. Patterns intentionally excluded from coverage live under `tool.coverage.report.exclude_also` in `pyproject.toml` (Taylor-series branches, native-disabled fallbacks, abstract methods, etc.).

## Examples and macOS

Run examples with `uv run examples/<name>.py`. On macOS, `mjpython` is required for the viewer; run `./fix_mjpython_macos.sh` once after `uv sync` to symlink `libpython` into `.venv/lib/`, then `uv run mjpython examples/<name>.py`.

## In-progress design docs

`docs/superpowers/specs/` and `docs/superpowers/plans/` hold date-stamped design specs and implementation plans for active work (e.g. the G1 torso-pose dataset generator). The implementation packages described in those docs (e.g. top-level `g1_pose_dataset/`) are not in the working tree yet — only stale `__pycache__/` from earlier iterations remains. Treat the spec + plan in `docs/superpowers/` as the source of truth, and re-read both before resuming work on that effort.
