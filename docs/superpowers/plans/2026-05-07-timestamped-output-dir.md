# Timestamped Default Output Dir — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `python -m g1_pose_dataset` write to `data/g1_torso_pose/{YYYYMMDD_HHMMSS}[_{dataset_name}]` by default, with `--resume` to reuse the latest matching folder and `--output-dir` as a verbatim override.

**Architecture:** Add a pure helper `_resolve_output_dir` in `g1_pose_dataset/__main__.py` that takes the parsed flags plus an injectable `now` clock and a `base` directory and returns the resolved `Path`. `main()` calls it once after `_parse_args` and writes the result back to `args.output_dir` so the rest of the pipeline is untouched. Three new CLI flags (`--output-dir` default → `None`, new `--dataset-name`, new `--resume`).

**Tech Stack:** Python 3 (existing project), argparse, datetime, regex, pytest.

**Spec:** [`docs/superpowers/specs/2026-05-07-timestamped-output-dir-design.md`](../specs/2026-05-07-timestamped-output-dir-design.md)

---

### Task 1: Add `_resolve_output_dir` (TDD)

**Files:**
- Modify: `g1_pose_dataset/__main__.py` (add helper + imports)
- Create: `tests/test_g1_pose_dataset/test_output_dir.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_g1_pose_dataset/test_output_dir.py` with this exact content:

```python
"""Unit tests for _resolve_output_dir in g1_pose_dataset.__main__."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pytest

from g1_pose_dataset.__main__ import _resolve_output_dir

FROZEN = datetime(2026, 5, 7, 0, 4, 10)


def _now() -> datetime:
    return FROZEN


def test_default_no_flags_returns_timestamp_only(tmp_path: Path) -> None:
    out = _resolve_output_dir(
        output_dir=None,
        dataset_name="",
        resume=False,
        base=tmp_path,
        now=_now,
    )
    assert out == tmp_path / "20260507_000410"


def test_default_with_dataset_name_appends_suffix(tmp_path: Path) -> None:
    out = _resolve_output_dir(
        output_dir=None,
        dataset_name="final",
        resume=False,
        base=tmp_path,
        now=_now,
    )
    assert out == tmp_path / "20260507_000410_final"


def test_resume_picks_lexicographically_largest_match(tmp_path: Path) -> None:
    (tmp_path / "20260101_000000_final").mkdir()
    (tmp_path / "20260507_000410_final").mkdir()
    # A name-suffix mismatch must not be picked even though it is "later".
    (tmp_path / "20270101_000000_other").mkdir()
    out = _resolve_output_dir(
        output_dir=None,
        dataset_name="final",
        resume=True,
        base=tmp_path,
        now=_now,
    )
    assert out == tmp_path / "20260507_000410_final"


def test_resume_empty_name_does_not_match_suffixed_folders(tmp_path: Path) -> None:
    (tmp_path / "20260101_000000_final").mkdir()
    with pytest.raises(SystemExit):
        _resolve_output_dir(
            output_dir=None,
            dataset_name="",
            resume=True,
            base=tmp_path,
            now=_now,
        )


def test_resume_with_no_matches_raises(tmp_path: Path) -> None:
    with pytest.raises(SystemExit):
        _resolve_output_dir(
            output_dir=None,
            dataset_name="",
            resume=True,
            base=tmp_path,
            now=_now,
        )


def test_explicit_output_dir_is_returned_verbatim(tmp_path: Path) -> None:
    explicit = tmp_path / "anywhere" / "i" / "want"
    out = _resolve_output_dir(
        output_dir=explicit,
        dataset_name="final",  # ignored
        resume=True,           # ignored
        base=tmp_path,
        now=_now,
    )
    assert out == explicit


def test_resume_ignores_non_matching_filenames(tmp_path: Path) -> None:
    # A regular file (not a dir) and a non-conforming subdir must be skipped.
    (tmp_path / "20260507_000410_final").mkdir()
    (tmp_path / "not_a_timestamp").mkdir()
    (tmp_path / "20260507_000410_final.txt").write_text("ignore me")
    out = _resolve_output_dir(
        output_dir=None,
        dataset_name="final",
        resume=True,
        base=tmp_path,
        now=_now,
    )
    assert out == tmp_path / "20260507_000410_final"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /home/zixin/Dev/tmp/mink && uv run pytest tests/test_g1_pose_dataset/test_output_dir.py -v`

Expected: All seven tests fail at import time with
`ImportError: cannot import name '_resolve_output_dir' from 'g1_pose_dataset.__main__'`.

- [ ] **Step 3: Add the imports and helper to `__main__.py`**

Edit `g1_pose_dataset/__main__.py`. After the existing `from pathlib import Path` line, add these two lines:

```python
import re
from datetime import datetime
```

Then, immediately above `def _parse_args(...)` (currently around line 34), insert:

```python
def _resolve_output_dir(
    output_dir: Path | None,
    dataset_name: str,
    resume: bool,
    *,
    base: Path,
    now=datetime.now,
) -> Path:
    """Resolve the run's output directory.

    See docs/superpowers/specs/2026-05-07-timestamped-output-dir-design.md.
    """
    if output_dir is not None:
        return output_dir

    suffix_re = rf"_{re.escape(dataset_name)}" if dataset_name else ""
    pattern = re.compile(rf"^\d{{8}}_\d{{6}}{suffix_re}$")

    if resume:
        if not base.exists():
            raise SystemExit(
                f"--resume: base directory {base} does not exist"
            )
        matches = sorted(
            p.name for p in base.iterdir()
            if p.is_dir() and pattern.match(p.name)
        )
        if not matches:
            raise SystemExit(
                f"--resume: no folder matching {pattern.pattern} in {base}"
            )
        return base / matches[-1]

    ts = now().strftime("%Y%m%d_%H%M%S")
    name = ts if dataset_name == "" else f"{ts}_{dataset_name}"
    return base / name
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /home/zixin/Dev/tmp/mink && uv run pytest tests/test_g1_pose_dataset/test_output_dir.py -v`

Expected: 7 passed.

- [ ] **Step 5: Run the full test suite to make sure nothing else broke**

Run: `cd /home/zixin/Dev/tmp/mink && uv run pytest tests/test_g1_pose_dataset -v`

Expected: all green (the existing dispatch tests pass `--output-dir` explicitly so they should be unaffected).

- [ ] **Step 6: Commit**

```bash
git -C /home/zixin/Dev/tmp/mink add g1_pose_dataset/__main__.py tests/test_g1_pose_dataset/test_output_dir.py
git -C /home/zixin/Dev/tmp/mink commit -m "feat(dataset): add _resolve_output_dir helper for timestamped runs

Pure helper with injected clock and base dir. Returns:
- explicit --output-dir verbatim
- base/{ts}[_{name}] for fresh runs
- base/<latest matching subdir> for --resume
SystemExit on resume miss.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 2: Wire up new CLI flags and call the resolver from `main()`

**Files:**
- Modify: `g1_pose_dataset/__main__.py` (rename constant; update `_parse_args`; call resolver in `main`; add log lines in `_run_pilot` and `_run_full`)
- Modify: `tests/test_g1_pose_dataset/test_output_dir.py` (add three parser-default tests)

- [ ] **Step 1: Add parser-default unit tests**

Append these to the bottom of `tests/test_g1_pose_dataset/test_output_dir.py`:

```python
from g1_pose_dataset.__main__ import _parse_args


def test_parse_args_default_output_dir_is_none() -> None:
    args = _parse_args([])
    assert args.output_dir is None
    assert args.dataset_name == ""
    assert args.resume is False


def test_parse_args_dataset_name_and_resume_flags() -> None:
    args = _parse_args(["--dataset-name", "final", "--resume"])
    assert args.dataset_name == "final"
    assert args.resume is True


def test_parse_args_explicit_output_dir() -> None:
    args = _parse_args(["--output-dir", "/tmp/foo"])
    assert args.output_dir == Path("/tmp/foo")
```

- [ ] **Step 2: Run the new tests to confirm they fail**

Run: `cd /home/zixin/Dev/tmp/mink && uv run pytest tests/test_g1_pose_dataset/test_output_dir.py -v -k "parse_args"`

Expected: 3 failures. The first will fail with `assert <PosixPath('.../data/g1_torso_pose')> is None` (current default), the second with `AttributeError: 'Namespace' object has no attribute 'dataset_name'`.

- [ ] **Step 3: Rename the constant and update `_parse_args`**

In `g1_pose_dataset/__main__.py`:

Find this line (currently line 28):

```python
DEFAULT_OUTPUT_DIR = REPO_ROOT / "data" / "g1_torso_pose"
```

Replace with:

```python
DEFAULT_OUTPUT_BASE = REPO_ROOT / "data" / "g1_torso_pose"
```

Then in `_parse_args`, replace this line:

```python
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
```

with these three argument definitions (replacing the single line):

```python
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Output directory. If unset, defaults to "
            "data/g1_torso_pose/{YYYYMMDD_HHMMSS}[_{dataset_name}]. "
            "When set, used verbatim (--dataset-name and --resume are ignored)."
        ),
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        default="",
        help=(
            "Suffix appended to the default timestamped folder name. "
            "Ignored when --output-dir is set."
        ),
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Reuse the latest existing folder under data/g1_torso_pose/ "
            "matching the timestamp[_{dataset_name}] pattern instead of "
            "creating a new one. Ignored when --output-dir is set."
        ),
    )
```

- [ ] **Step 4: Call the resolver from `main()`**

In `g1_pose_dataset/__main__.py`, find `def main` (currently around line 340) and replace its body with:

```python
def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    args.output_dir = _resolve_output_dir(
        args.output_dir,
        args.dataset_name,
        args.resume,
        base=DEFAULT_OUTPUT_BASE,
    )

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
```

(The only change vs. the original is the new `args.output_dir = _resolve_output_dir(...)` block right after `_parse_args`.)

- [ ] **Step 5: Add an early "output_dir = …" log line to both run functions**

In `_run_pilot` (currently around line 157), insert this line as the first line of the function body, before `pilot_dir = args.output_dir / "pilot"`:

```python
    print(f"[pilot] output_dir = {args.output_dir}")
```

In `_run_full` (currently around line 226), insert this line as the first line of the function body, before `args.output_dir.mkdir(parents=True, exist_ok=True)`:

```python
    print(f"[run] output_dir = {args.output_dir}")
```

- [ ] **Step 6: Run the parser tests + full suite**

Run: `cd /home/zixin/Dev/tmp/mink && uv run pytest tests/test_g1_pose_dataset -v`

Expected: all green, including the three new parser tests and the existing dispatch smoke tests.

- [ ] **Step 7: Manual smoke check of `--dry-run`**

Run: `cd /home/zixin/Dev/tmp/mink && uv run python -m g1_pose_dataset --dry-run`

Expected: prints `output:` line ending in a path that looks like `data/g1_torso_pose/<14-digit timestamp>` (no trailing underscore, no name suffix).

Run: `cd /home/zixin/Dev/tmp/mink && uv run python -m g1_pose_dataset --dry-run --dataset-name final`

Expected: `output:` line ends with `data/g1_torso_pose/<timestamp>_final`.

Run: `cd /home/zixin/Dev/tmp/mink && uv run python -m g1_pose_dataset --dry-run --resume`

Expected: SystemExit with `--resume: ... no folder matching ...` (no matching folder exists yet).

- [ ] **Step 8: Commit**

```bash
git -C /home/zixin/Dev/tmp/mink add g1_pose_dataset/__main__.py tests/test_g1_pose_dataset/test_output_dir.py
git -C /home/zixin/Dev/tmp/mink commit -m "feat(dataset): wire CLI to timestamped default output dir

- --output-dir default becomes None; explicit value used verbatim
- new --dataset-name STR (default \"\")
- new --resume flag picks up the latest matching folder
- main() resolves output_dir once via _resolve_output_dir
- both run functions log the resolved output_dir up front
- rename DEFAULT_OUTPUT_DIR -> DEFAULT_OUTPUT_BASE for clarity

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 3: Update README

**Files:**
- Modify: `g1_pose_dataset/README.md`

- [ ] **Step 1: Update the Optional flags block**

In `g1_pose_dataset/README.md`, find the `# Optional flags:` block (currently lines 20-29). Replace the existing `--output-dir` line with these three lines and keep the rest intact:

```
#   --output-dir PATH         override; default: data/g1_torso_pose/{YYYYMMDD_HHMMSS}[_{name}]
#   --dataset-name NAME       suffix appended to the default timestamped folder
#   --resume                  reuse the latest existing matching folder instead of starting fresh
```

So the resulting block reads:

```bash
# Optional flags:
#   --output-dir PATH         override; default: data/g1_torso_pose/{YYYYMMDD_HHMMSS}[_{name}]
#   --dataset-name NAME       suffix appended to the default timestamped folder
#   --resume                  reuse the latest existing matching folder instead of starting fresh
#   --num-workers N           (0 = os.cpu_count() - 1)
#   --threshold 1e-3
#   --max-iter 500
#   --subshard-size 50000
#   --save-diagnostics        write diagnostics.npy (final ‖vel‖, iters, wall_ms per cell)
#   --cleanup-shards          delete shards/ after successful concat
#   --report-failed-commands  record non-converging commands in each .done sentinel
#                             and aggregate them into metadata.json (off by default)
```

- [ ] **Step 2: Update the Output layout example**

In `g1_pose_dataset/README.md`, find the `## Output layout` section (currently around line 32). Replace the path on the first line of the code block from:

```
data/g1_torso_pose/
```

to:

```
data/g1_torso_pose/20260507_000410_final/
```

Leave the rest of the tree (commands.npy, joints.npy, etc.) unchanged.

- [ ] **Step 3: Update the Resume section**

In `g1_pose_dataset/README.md`, replace the entire `## Resume` section with:

```markdown
## Resume

Each subshard (50,000 cells) is written atomically with a `.done` sentinel.
Re-running with `--resume` (and matching `--dataset-name`, if any) picks up
the latest existing folder under `data/g1_torso_pose/` and skips completed
subshards. Worst-case work loss from a crash is one subshard.

Without `--resume` every run lands in a fresh timestamped folder, so an
unintended re-run never overwrites a finished dataset.
```

- [ ] **Step 4: Sanity-check the README renders**

Run: `cd /home/zixin/Dev/tmp/mink && head -60 g1_pose_dataset/README.md`

Expected: the three new flags appear in the Optional flags block, the layout shows a timestamped folder, and the Resume section mentions `--resume`.

- [ ] **Step 5: Commit**

```bash
git -C /home/zixin/Dev/tmp/mink add g1_pose_dataset/README.md
git -C /home/zixin/Dev/tmp/mink commit -m "docs(dataset): document --dataset-name, --resume, and timestamped output dir

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Final verification

- [ ] **Step 1: Run the full test suite end-to-end**

Run: `cd /home/zixin/Dev/tmp/mink && uv run pytest tests/test_g1_pose_dataset -v`

Expected: all green.

- [ ] **Step 2: Confirm git status is clean**

Run: `git -C /home/zixin/Dev/tmp/mink status`

Expected: working tree clean (or only the pre-existing modifications listed at the start of the session, which this plan does not touch).
