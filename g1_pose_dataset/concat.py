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
