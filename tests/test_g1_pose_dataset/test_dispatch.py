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
        check=False,
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
