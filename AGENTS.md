# Repository Guidelines

## Project Structure & Module Organization

`src/mink/` contains the Python package. Core IK logic lives in `configuration.py`, `solve_ik.py`, `tasks/`, `limits/`, and `lie/`; optional integrations live under `src/mink/contrib/`. Tests are in `tests/` and generally mirror the module under test, for example `tests/test_frame_task.py`. Robot demos and MuJoCo assets are in `examples/`, with per-robot XML, meshes, images, licenses, and README files grouped together. Sphinx documentation is in `docs/`, type stubs are in `typings/`, and performance scripts are in `benchmarks/`. Keep generated outputs, caches, `dist/`, `_build/`, and `data/` out of commits.

## Build, Test, and Development Commands

- `make sync`: install all project, dev, and package extras with `uv`.
- `make format`: run `ruff format` and `ruff check --fix`.
- `make type`: run both `ty check` and `pyright`.
- `make test`: run the pytest suite with `uv run pytest`.
- `make test-all`: run formatting fixes, type checks, and tests.
- `make coverage`: run pytest under coverage and print the report.
- `make doc`: build Sphinx docs with warnings treated as errors.
- `make build`: build wheel/sdist and test the built artifacts.

Run examples with `uv run examples/arm_ur5e.py`; on macOS, run `./fix_mjpython_macos.sh` first when `mjpython` is needed.

## Coding Style & Naming Conventions

Use Python 3.10+ syntax and type annotations for public APIs. Formatting and import order are controlled by Ruff; do not hand-format around it. Follow existing naming: modules and functions use `snake_case`, classes use `PascalCase`, constants use `UPPER_SNAKE_CASE`, and tests use `test_<behavior>`. Keep numerical code explicit and local to the relevant task, limit, or Lie group module.

## Testing Guidelines

Tests use pytest with `absltest` in many files. Add or update focused tests when changing behavior, especially for tasks, limits, Lie operations, and solver behavior. Name files `tests/test_<module>.py` and test methods `test_<expected_behavior>`. Run `make test` for quick validation and `make test-all` before opening a PR.

## Commit & Pull Request Guidelines

Recent history uses short, imperative commits with optional conventional prefixes, such as `feat(example): ...`, `docs: ...`, and `docs(spec): ...`. Keep commits scoped to one logical change. Pull requests should describe the change, link related issues when applicable, update `CHANGELOG.md`, include docs or examples for user-facing behavior, and note local verification commands. Include screenshots or short clips for visual robot/example changes when useful.
