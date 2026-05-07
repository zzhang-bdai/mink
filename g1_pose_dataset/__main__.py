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
