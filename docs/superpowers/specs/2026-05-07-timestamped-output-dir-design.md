# Timestamped default output dir for `g1_pose_dataset`

Date: 2026-05-07

## Background

`g1_pose_dataset` (see [2026-05-06-g1-pose-dataset-design.md](2026-05-06-g1-pose-dataset-design.md))
currently writes every collection run into the same fixed default folder,
`data/g1_torso_pose/`. That is convenient for resuming an in-progress run but
makes it easy to clobber a finished dataset with the next run, and gives no
hint about *when* a given dataset was produced. As we start collecting more
than one dataset variant (different ranges, thresholds, names like "final"
versus "ablation"), each run should land in its own clearly identified
folder by default.

## Goal

Change the default output directory to
`data/g1_torso_pose/{YYYYMMDD_HHMMSS}[_{dataset_name}]`, so that a fresh run
with no flags lands in a new timestamped folder, and the user can label runs
with a short suffix (`--dataset-name final` →
`data/g1_torso_pose/20260507_000410_final`). Resuming an existing run becomes
an opt-in `--resume` flag rather than the default. Explicit `--output-dir`
remains a complete override.

## Non-goals

- Changing how shards are written, concatenated, or resumed *within* an
  output directory. The .done sentinel + skip logic stays as-is.
- Cross-run merging or any kind of dataset registry/index file.
- Auto-detecting "the previous run" without `--resume`. Default behaviour is
  always: create a fresh timestamped folder.
- Any change to `g1_pose_dataset/play.py`'s default path semantics. (It still
  defaults to `data/g1_torso_pose/pilot/joints.npy`; users that want to
  visualise a timestamped run pass the path explicitly. Updating play.py is
  out of scope for this spec.)

## CLI surface

In `g1_pose_dataset/__main__.py::_parse_args`:

| flag | type | new default | meaning |
|---|---|---|---|
| `--output-dir` | `Path` | `None` (was `data/g1_torso_pose`) | When provided, used verbatim; bypasses timestamp/resume logic entirely. |
| `--dataset-name` | `str` | `""` | Suffix appended to the timestamp when `--output-dir` is `None`. Empty → no suffix and no underscore. |
| `--resume` | flag | `False` | When set and `--output-dir` is `None`, reuse the latest matching subfolder under `data/g1_torso_pose/` instead of creating a new one. Errors if no matching folder exists. |

Examples (assuming `now == 2026-05-07 00:04:10`):

| invocation | resolved output dir |
|---|---|
| `python -m g1_pose_dataset` | `data/g1_torso_pose/20260507_000410` |
| `python -m g1_pose_dataset --dataset-name final` | `data/g1_torso_pose/20260507_000410_final` |
| `python -m g1_pose_dataset --resume` | latest `\d{8}_\d{6}` folder under `data/g1_torso_pose/` |
| `python -m g1_pose_dataset --resume --dataset-name final` | latest `\d{8}_\d{6}_final` folder |
| `python -m g1_pose_dataset --output-dir /tmp/foo` | `/tmp/foo` (verbatim) |
| `python -m g1_pose_dataset --output-dir /tmp/foo --dataset-name final` | `/tmp/foo` (suffix ignored — `--output-dir` is the override) |

## Path resolution

A new private helper in `__main__.py`:

```python
def _resolve_output_dir(
    output_dir: Path | None,
    dataset_name: str,
    resume: bool,
    *,
    base: Path,
    now: Callable[[], datetime] = datetime.now,
) -> Path:
    ...
```

Behaviour:

1. If `output_dir is not None`, return `output_dir` unchanged. `dataset_name`
   and `resume` are silently ignored (documented in `--help`/README).
2. Otherwise, build the regex
   `^\d{8}_\d{6}` + (`f"_{re.escape(dataset_name)}$"` if `dataset_name` else
   `$`).
3. If `resume`:
   - Scan `base` for direct subdirectories whose name matches the regex.
   - If none, raise `SystemExit` with a message:
     `"--resume: no folder matching <pattern> in <base>"`.
   - Otherwise return the lexicographically-largest match. (Lex sort is
     correct for `YYYYMMDD_HHMMSS` because every component is fixed-width.)
4. If not `resume`: format `now()` as `"%Y%m%d_%H%M%S"`; return
   `base / (ts if dataset_name == "" else f"{ts}_{dataset_name}")`.

`main()` calls this once after `_parse_args` and writes the result back to
`args.output_dir` so the rest of `_run_pilot` / `_run_full` is unchanged.

The dataset-name pattern intentionally requires an *exact* `_<name>` match.
That means folders for *different* names never collide on `--resume`, and
`--resume` with the empty default does not pick up name-suffixed folders
either.

## Logging

Currently `_run_full` only prints the resolved output dir at the very end.
Add a one-line print near the top of both `_run_pilot` and `_run_full` so
the user sees where the run is writing as soon as it starts:

```
[run] output_dir = data/g1_torso_pose/20260507_000410_final
```

This matters more once the path is non-deterministic: the user can `ls` it
mid-run, tail logs to it, etc.

## Pilot interaction

`_run_pilot` continues to write under `<output_dir>/pilot/`. With the new
default, that means e.g. `data/g1_torso_pose/20260507_000410/pilot/`.
`--resume --pilot N` works identically to a full `--resume`: the latest
matching timestamped folder is selected, and the existing sentinel-skip
logic in `worker.process_chunk` handles "this subshard is already done".

A pilot run and a full run sharing the same timestamped folder is fine —
they live in `pilot/` and `shards/` subdirs respectively and never overlap.

## Tests

New unit tests in a new file `tests/test_g1_pose_dataset/test_output_dir.py`
that exercise `_resolve_output_dir` directly, with `now` injected for
determinism and `base=tmp_path`:

1. **Default path, no flags:** returns `tmp_path / "20260507_000410"`.
2. **`--dataset-name final`:** returns `tmp_path / "20260507_000410_final"`.
3. **Empty name does not match name-suffixed folders on resume:**
   pre-create `20260101_000000_final`; `_resolve_output_dir(resume=True,
   dataset_name="")` raises `SystemExit`.
4. **Resume picks the latest matching folder:**
   pre-create `20260101_000000_final` and `20260507_000410_final`;
   `_resolve_output_dir(resume=True, dataset_name="final")` returns the
   later one.
5. **Resume errors with no matches:** empty `base`, `resume=True` →
   `SystemExit`.
6. **Explicit `output_dir` is returned verbatim** even with `dataset_name`
   set and `resume=True` (i.e. timestamp/resume logic is fully bypassed).

The existing `test_dispatch.py` smoke tests already pass `--output-dir`
explicitly (line 22, line 53), so they bypass the new logic and need no
changes.

## README updates

Update `g1_pose_dataset/README.md`:
- The `Optional flags:` block lists `--dataset-name STR` and `--resume`.
- The `--output-dir` line notes the new default behaviour: timestamped
  folder per run; `--resume` to pick up a previous run.
- The `Output layout` example uses a representative timestamped path.

## Out of scope / non-goals (re-stated)

- No env var or config file for `dataset_name` — flag only.
- No changes to `worker.py`, `concat.py`, `grid.py`, or `play.py`.
- No retroactive migration of existing `data/g1_torso_pose/` contents into
  a timestamped folder.
- No deletion or warning about pre-existing un-timestamped runs.
