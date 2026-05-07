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
