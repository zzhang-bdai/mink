# G1 torso-pose dataset generation

Date: 2026-05-06

## Background

`examples/humanoid_g1_torso.py` solves whole-body IK on the Unitree G1 to track a
6-DoF torso pose target. The user supplies a target via mouse drag, sliders, or
the `R` key (random sample). For learning a controller / policy that maps a
torso pose command to a whole-body joint configuration, we want a large offline
dataset of `(command, joints)` pairs produced by the same IK pipeline.

## Goal

Generate a dataset of 8,505,000 samples spanning a 4D grid over the torso
command `(roll, pitch, yaw, height)`, where each sample's joint configuration is
the converged solution of the IK problem set up exactly as in the example.
Output two `.npy` files plus a JSON sidecar so the dataset is directly memmap-
friendly for PyTorch / JAX training pipelines.

## Non-Goals

- Real-time generation in a viewer. There is no viewer; the script is headless.
- Floating-base / pelvis pose recording. We store only the 29 actuated joint
  angles (the user explicitly chose joints-only).
- Generalising to other robots or other task setups. The script is specific to
  the G1 + torso-pose + standing-feet configuration of the example.
- A reusable `mink.contrib` dataset library. The first iteration lives under
  `scripts/` + `src/g1_pose_dataset/` and is project-local.

## Sample schema

Each saved row contains:
- **command**: `(roll, pitch, yaw, height)` — radians, radians, radians, metres,
  in that order. Stored in radians (not degrees) so consumers feed `commands`
  directly into `mink.SO3.from_rpy_radians` or any other downstream model
  without unit conversion.
- **joints**: 29 actuated-joint angles in radians, ordered to match
  `joint_names.json`.

The 4D grid (half-open `np.arange(start, stop, step)` semantics, in degrees and
metres):

| dim    | start | stop | step | count |
|--------|-------|------|------|-------|
| roll   | -10   | 10   | 1    | 20    |
| pitch  | -15   | 90   | 1    | 105   |
| yaw    | -45   | 45   | 1    | 90    |
| height | 0.35  | 0.80 | 0.01 | 45    |

`20 × 105 × 90 × 45 = 8,505,000` cells. Endpoint convention is documented in
`metadata.json`. Linearisation uses C-order over `(roll, pitch, yaw, height)`,
i.e. height varies fastest. Order is irrelevant to correctness (we reset to the
keyframe between cells) but is fixed for reproducibility.

## IK setup

Mirrors `examples/humanoid_g1_torso.py` exactly:

- Model: `examples/unitree_g1/scene_g1_torso.xml`.
- Tasks:
  - `torso_task = FrameTask("torso_link", "body", position_cost=[10,10,10],
    orientation_cost=10, lm_damping=1.0)`.
  - `posture_task = PostureTask(model, cost=posture_cost)` where
    `posture_cost = np.full(model.nv, 1e-1)`, then
    `posture_cost[model.joint("waist_roll_joint").dofadr[0]] = 5.0`,
    `posture_cost[model.joint("waist_pitch_joint").dofadr[0]] = 1.0`.
    `waist_yaw_joint` stays at the default `1e-1` (the example's override is
    commented out — preserved as commented out).
  - Two foot tasks: `FrameTask("right_foot"|"left_foot", "site",
    position_cost=1000, orientation_cost=1000, lm_damping=1.0)`.
- Limits:
  - `ConfigurationLimit(model)` with the example's tightened knee lower bound
    of 0.17 rad on `left_knee_joint` / `right_knee_joint`.
  - `CollisionAvoidanceLimit` with the example's four geom-pair groups
    (`left_hand_collision`↔`left_thigh_collision`,
    `right_hand_collision`↔`right_thigh_collision`,
    `torso_collision`↔`left_thigh_collision`,
    `torso_collision`↔`right_thigh_collision`),
    `minimum_distance_from_collisions=0.005`,
    `collision_detection_distance=0.15`.
- Solver: `daqp`, `damping=1e-1`, `dt = 1/200 = 0.005`.

**Per-worker initialisation (once):**

1. Load the model, build `mink.Configuration`.
2. `configuration.update_from_keyframe("stand_drag")`.
3. `posture_task.set_target_from_configuration(configuration)`.
4. For each foot, capture its world transform now and call
   `foot_task.set_target(transform)`. Foot targets are static for the rest of
   the worker's lifetime.

**Per-cell loop:**

```python
configuration.update_from_keyframe("stand_drag")
T = mink.SE3.from_rotation_and_translation(
    mink.SO3.from_rpy_radians(roll, pitch, yaw),
    np.array([0.0, 0.0, height]),
)
torso_task.set_target(T)
converged = False
final_norm = math.inf
for it in range(max_iter):
    vel = mink.solve_ik(configuration, tasks, dt, "daqp",
                        damping=1e-1, limits=limits)
    final_norm = float(np.linalg.norm(vel))
    if final_norm < threshold:
        converged = True
        break
    configuration.integrate_inplace(vel, dt)
if converged:
    record (command, joint_angles)
# else: skip
```

**Convergence threshold:** default `1e-3`. Note: the `lm_damping=1.0` and
`damping=1e-1` regularisers create a damping-induced floor on `‖vel‖`, so on
some cells the residual may plateau above the threshold even at a sensible
local minimum. The pilot run (see below) measures this empirically; if skip
rate is unacceptable, raise the threshold via `--threshold` rather than
weakening the regularisation.

**Max iterations:** default `500`. Bounds the wall-time of any one cell to
roughly 0.5 s, so unreachable cells cannot stall the run.

**Joint-value extraction:** for each name in `joint_names.json`, read
`configuration.q[model.joint(name).qposadr[0]]`. The 29 actuated joint names
are the joints listed under the `<body>` hierarchy of `g1.xml` excluding the
free root — collected once at startup by walking
`model.joint(i).name for i in range(model.njnt)` and skipping the free joint.

## Output layout

```
data/g1_torso_pose/
├── commands.npy            # (N, 4) float32 — radians, radians, radians, metres
├── joints.npy              # (N, 29) float32 — radians, joint_names order
├── joint_names.json        # ordered list of 29 actuated joint names
├── metadata.json           # ranges, threshold, max_iter, mink/mujoco versions,
│                           # n_total_cells (8,505,000), n_converged, n_skipped,
│                           # cell_order, dtype, schema_version
├── diagnostics.npy         # (T, 3) float32, only if --save-diagnostics:
│                           #   final_vel_norm, iter_count, wall_ms — one row
│                           #   per grid cell in linearised order, including
│                           #   non-converged
└── shards/                 # kept by default (resume insurance); --cleanup-shards removes
    ├── subshard_0000.commands.npy
    ├── subshard_0000.joints.npy
    ├── subshard_0000.done                # JSON: n_converged, n_attempted
    ├── subshard_0001.commands.npy
    └── ...
```

Float precision: `float32` throughout. Joint angles need ~5–7 significant
digits; doubling to `float64` doubles file size for no NN-training benefit.

## Architecture

```
g1_pose_dataset/                  # top-level package, importable from repo root
├── __init__.py
├── __main__.py   # CLI + dispatcher + concat (run: python -m g1_pose_dataset)
├── config.py     # build_tasks_and_limits(model), extract_joint_names(model) —
│                 # single source of truth, mirrors the example
├── grid.py       # build_grid() -> (T, 4); iter_cells(start, end)
├── worker.py     # run_worker(rank, chunk_ids, args) — one per process
└── concat.py     # concat_shards(output_dir) -> writes commands.npy + joints.npy
tests/
└── test_g1_pose_dataset/
    ├── __init__.py
    ├── test_grid.py    # math: total count, axis counts, linearisation order
    ├── test_worker.py  # 10-cell smoke run with real model
    ├── test_concat.py  # write fake subshards, concat, verify final
    └── test_resume.py  # tiny grid, kill mid-chunk, restart, no dups/gaps
```

The package lives at repo top level (not under `src/`) because it is not part
of the published mink wheel — it is a project-local generation tool.
`scripts/` is gitignored in this repo, so the entrypoint becomes the package's
`__main__.py` (run via `python -m g1_pose_dataset ...`). Splitting into focused
modules keeps grid / worker / concat testable independently.

## Multiprocessing & sharding

**Process model:** `multiprocessing.get_context("spawn").Process`. Spawn (not
fork) avoids inheriting partially-initialised mujoco / mink C state, which can
be flaky on some platforms. `N` defaults to `os.cpu_count() - 1`, overridable
via `--num-workers`.

**Subshard layout:** subshard size `S = 50_000` cells.
`ceil(8_505_000 / 50_000) = 171` subshards total — subshards 0..169 hold 50,000
cells each, subshard 170 holds the remaining 5,000. Subshard `chunk_id` covers
linearised cells `[chunk_id * S, min((chunk_id+1) * S, T))`. Workers own
contiguous ranges of `chunk_id`s assigned at startup; if `--num-workers`
exceeds the subshard count (171), it is clamped down so no worker has an empty
range.

**Per-subshard write protocol** (atomic, resume-safe):

1. Worker preallocates `cmd_buf = np.empty((S, 4), float32)` and
   `jnt_buf = np.empty((S, 29), float32)` (~6.5 MB total per worker).
2. For each cell in the chunk, run the IK loop. If converged, append
   `(command, joints)` to the buffers; increment `n_local`.
3. After all cells in the chunk: `np.save(subshard_{chunk:04d}.commands.npy,
   cmd_buf[:n_local])` (and joints), `os.fsync` both, then write
   `subshard_{chunk:04d}.done` containing
   `{"n_attempted": ..., "n_converged": n_local}`.

The `.done` sentinel is written **last** and only after the data files are
fsynced. On startup, a chunk with `.done` present is skipped wholesale; without
it, any partial files are silently overwritten.

**Final concat** (after all workers exit):

1. Walk `shards/` in canonical order (chunk_id 0..170), summing
   `n_converged` from each `.done`. This yields total `N`.
2. Open `commands.npy` and `joints.npy` via `np.lib.format.open_memmap` with
   shape `(N, 4)` / `(N, 29)`.
3. Stream-copy each subshard into the next slice of the destination memmap.
4. If `--save-diagnostics`, similarly assemble `diagnostics.npy` from
   per-subshard diagnostic files (always written when `--save-diagnostics` is
   on, separate from the converged-only commands/joints subshards).

Concat takes seconds and uses bounded RAM regardless of dataset size.

**Progress reporting:** main process owns a `multiprocessing.Queue`. Workers
post `(chunk_id, n_attempted, n_converged, wall_s)` after each subshard. Main
prints a progress line with cumulative cells / converged / ETA. No shared
counters or locks needed.

## CLI

```
python -m g1_pose_dataset \
    [--output-dir data/g1_torso_pose] \
    [--num-workers AUTO]            # default: os.cpu_count() - 1
    [--threshold 1e-3]              # ‖vel‖ convergence threshold
    [--max-iter 500]                # IK iteration cap
    [--subshard-size 50000]
    [--pilot N]                     # run only first N cells, single worker
    [--save-diagnostics]            # write diagnostics.npy
    [--cleanup-shards]              # delete shards/ after successful concat
    [--dry-run]                     # build grid, print plan, exit
```

**`--pilot N`:** runs a single worker on the first `N` linearised cells, writes
to `pilot/` with the same layout (diagnostics always on). Reports skip rate,
mean/p50/p95/max iter counts, mean/p95 ms per cell, and an extrapolated
wall-time for the full run at `N_workers × observed throughput`. Required
sanity check before committing to a multi-hour run.

**Recommended validation sequence:**

1. `--dry-run` — confirm grid math and output paths.
2. `--pilot 1000` (~30–90 s) — confirm convergence and threshold are sensible.
3. `--pilot 50000` (~one full subshard) — tight wall-time estimate.
4. Full run, no flags.

## Testing

Real tests using the real model — no mocks of IK behaviour, since IK behaviour
is the thing being relied upon.

- `tests/test_dataset_grid.py`: pure-numpy unit tests on `build_grid()` and
  `iter_cells()`. Asserts total count `= 8_505_000`, per-axis counts, dtype,
  endpoint convention, linearisation order. <1 s.
- `tests/test_dataset_worker.py`: instantiates a real `MjModel` and runs
  `run_worker` over a 10-cell range covering the standing-target case (zero
  pose at standing height) and one off-axis case. Asserts converged outputs are
  within configuration limits and joint names match
  `model.joint(name).qposadr[0]` lookups. ~5 s.
- `tests/test_dataset_concat.py`: writes three synthetic subshards of known
  shape and content, calls `concat_shards`, asserts final files match the
  concatenated input. No IK. <1 s.
- `tests/test_dataset_resume.py`: tiny grid (e.g. 200 cells, S=50 → 4 chunks),
  runs worker, kills mid-chunk, restarts, asserts no duplicates and no gaps in
  the final concat. ~3 s.

No multiprocess orchestration test — orchestration is thin (process spawn +
queue) and gets covered by the pilot run + the full run itself.

## Risks and mitigations

- **Damping floor on ‖vel‖.** `lm_damping=1.0` may push the achievable residual
  above `1e-3` on some cells. *Mitigation:* the pilot reveals this empirically;
  `--threshold` is a CLI knob.
- **Unreachable grid cells.** Pitch up to 90° at heights as low as 0.35 m may
  be physically infeasible with feet pinned at standing positions, leading to
  high skip rate in some grid regions. *Mitigation:* skip rate is reported per
  pilot and per full run; non-converged cells are dropped (per user choice),
  and the linearised diagnostics file (if enabled) makes the failure region
  visible for post-hoc analysis.
- **Worker crash mid-chunk.** *Mitigation:* atomic per-subshard writes with
  `.done` sentinel and the resume protocol. Crashed chunks lose at most 50,000
  cells of work.
- **Long wall-time.** Single-thread is ~120 h at 50 ms/cell; even with 16
  workers a full run is ~8 h. *Mitigation:* required pilot validates the
  per-cell cost before committing; resume capability covers interruptions.
- **Mujoco state in subprocesses.** *Mitigation:* `spawn` start method; each
  worker builds its own `MjModel` and `mink.Configuration` in its own process.

## Open questions

None remaining at design time. Pilot results may surface tuning needs
(threshold, max_iter, regulariser values) handled via CLI flags.
