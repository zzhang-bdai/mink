# Tkinter slider torso teleop — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a Tkinter slider window to `examples/humanoid_g1_torso.py` that gives 6-DOF pose control over the torso IK target, replacing imprecise mouse-drag.

**Architecture:** Single-file change to an example script. A Tkinter `Tk` window is created after `move_mocap_to_frame`, holding 6 `DoubleVar` sliders (3 position, 3 RPY in degrees) plus a Reset button. Inside the existing `while viewer.is_running():` loop, pump Tk events with `root.update()`, read sliders, convert RPY→quat via `mink.SO3.from_rpy_radians(...).wxyz`, and write to `data.mocap_pos[torso_mid]` / `data.mocap_quat[torso_mid]`. No threading. No new dependencies.

**Tech Stack:** Python 3.10+, `tkinter` (stdlib), `mujoco`, `mink`, `numpy`, `loop_rate_limiters`.

**Spec:** [`docs/superpowers/specs/2026-05-06-humanoid-torso-sliders-design.md`](../specs/2026-05-06-humanoid-torso-sliders-design.md)

---

## Files

- **Modify:** `examples/humanoid_g1_torso.py` (add Tk import; add slider-window setup after `move_mocap_to_frame`; add per-tick slider→mocap update inside the loop).

This is example/script code. The `mink` test suite already covers `SO3.from_rpy_radians` / `as_rpy_radians` round-trip, so we don't add new files in `tests/`. Verification for the example is by running the script and exercising the UI; an explicit verification task is included below.

## Prerequisites

The working tree currently has uncommitted local changes adding `CollisionAvoidanceLimit` to `examples/humanoid_g1_torso.py`. These are unrelated to the slider work. **Before starting Task 1**, decide:

- **Option A (recommended):** Commit those changes as their own commit so the slider commits are clean. Suggested message: `feat(example): add collision avoidance to G1 torso example`. If you take this path, do it once, then proceed.
- **Option B:** Leave them uncommitted; they'll be bundled into the slider-feature commit at Task 7. Acceptable but mixes two unrelated changes in one commit.

The plan below assumes Option A. If you choose B, the only difference is that Task 7's commit will also contain the collision-avoidance hunks.

---

## Task 1: Verify the SO3 RPY round-trip

A one-time sanity check before relying on the conversion in the example. Validates both directions of the path the example will use: `from_rpy_radians(...).wxyz` (write) and `SO3(wxyz=...).as_rpy_radians()` (read).

**Files:**
- Read-only check, no file edits.

- [ ] **Step 1: Run the round-trip check**

Run from the repo root:

```bash
.venv/bin/python -c "
import numpy as np
import mink
cases = [(0,0,0), (15,-30,45), (-90,10,170), (5,5,5)]
for rpy_deg in cases:
    rpy_rad = tuple(np.deg2rad(rpy_deg))
    q = mink.SO3.from_rpy_radians(*rpy_rad).wxyz
    rt = mink.SO3(wxyz=q).as_rpy_radians()
    rt_deg = (np.rad2deg(rt.roll), np.rad2deg(rt.pitch), np.rad2deg(rt.yaw))
    err = max(abs(a-b) for a, b in zip(rpy_deg, rt_deg))
    print(f'rpy={rpy_deg} round-trip={tuple(round(v,4) for v in rt_deg)} max_err={err:.6f}')
"
```

Expected: each line prints `max_err` below ~1e-4. If any line shows a large error, **stop and report**: the conversion path is not what we think and the design needs revisiting before further code changes.

---

## Task 2: Add `tkinter` import

**Files:**
- Modify: `examples/humanoid_g1_torso.py` (top imports, around line 6).

- [ ] **Step 1: Add the import**

In the import block at the top of the file, add `import tkinter as tk` immediately after `from loop_rate_limiters import RateLimiter` so the imports group as: stdlib, third-party, local.

The imports section should now look like:

```python
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np
from loop_rate_limiters import RateLimiter

import tkinter as tk

import mink
```

- [ ] **Step 2: Verify the file still parses**

```bash
.venv/bin/python -c "import ast; ast.parse(open('examples/humanoid_g1_torso.py').read())"
```

Expected: no output (success).

---

## Task 3: Capture initial pose after `move_mocap_to_frame`

Capture the initial torso target pose so the slider window initializes to it and the Reset button has a target to return to.

**Files:**
- Modify: `examples/humanoid_g1_torso.py` (inside `with mujoco.viewer.launch_passive(...) as viewer:`, immediately after the existing `mink.move_mocap_to_frame(...)` line).

- [ ] **Step 1: Add the pose-capture block**

Find the line:

```python
        # Place the torso target mocap at the current torso pose.
        mink.move_mocap_to_frame(model, data, "torso_target", "torso_link", "body")
```

Immediately after it, add:

```python

        # Capture initial torso target pose (used for slider init + reset).
        init_pos = data.mocap_pos[torso_mid].copy()
        _init_rpy = mink.SO3(wxyz=data.mocap_quat[torso_mid].copy()).as_rpy_radians()
        init_rpy_deg = (
            float(np.rad2deg(_init_rpy.roll)),
            float(np.rad2deg(_init_rpy.pitch)),
            float(np.rad2deg(_init_rpy.yaw)),
        )
```

- [ ] **Step 2: Verify the file still parses**

```bash
.venv/bin/python -c "import ast; ast.parse(open('examples/humanoid_g1_torso.py').read())"
```

Expected: no output.

---

## Task 4: Build the Tk slider window

Create the Tk window with 6 sliders + Reset button right after the pose-capture block. The window is built once, lives for the duration of the viewer, and stays simple — no callbacks, no threading.

**Files:**
- Modify: `examples/humanoid_g1_torso.py` (after the Task 3 block, before the existing `rate = RateLimiter(...)` line).

- [ ] **Step 1: Insert the Tk window setup**

Find the line:

```python
        rate = RateLimiter(frequency=200.0, warn=False)
```

Immediately before it, add:

```python

        # ----- Tkinter slider window for 6-DOF torso pose ------------------
        tk_root = tk.Tk()
        tk_root.title("Torso pose")

        pos_vars = [tk.DoubleVar(value=float(v)) for v in init_pos]
        rpy_vars = [tk.DoubleVar(value=v) for v in init_rpy_deg]

        pos_ranges = [
            (float(init_pos[i] - 0.5), float(init_pos[i] + 0.5)) for i in range(3)
        ]
        for label, var, (lo, hi) in zip(("x", "y", "z"), pos_vars, pos_ranges):
            tk.Label(tk_root, text=f"{label} (m)").pack(anchor="w", padx=8)
            tk.Scale(
                tk_root,
                from_=lo,
                to=hi,
                resolution=0.001,
                orient="horizontal",
                variable=var,
                length=320,
            ).pack(fill="x", padx=8)

        for label, var in zip(("roll", "pitch", "yaw"), rpy_vars):
            tk.Label(tk_root, text=f"{label} (deg)").pack(anchor="w", padx=8)
            tk.Scale(
                tk_root,
                from_=-180.0,
                to=180.0,
                resolution=0.5,
                orient="horizontal",
                variable=var,
                length=320,
            ).pack(fill="x", padx=8)

        def _reset_sliders() -> None:
            for var, val in zip(pos_vars, init_pos):
                var.set(float(val))
            for var, val in zip(rpy_vars, init_rpy_deg):
                var.set(val)

        tk.Button(tk_root, text="Reset", command=_reset_sliders).pack(pady=8)

        tk_alive = True  # cleared if the window is closed
        # -------------------------------------------------------------------

```

- [ ] **Step 2: Verify the file still parses**

```bash
.venv/bin/python -c "import ast; ast.parse(open('examples/humanoid_g1_torso.py').read())"
```

Expected: no output.

---

## Task 5: Drive mocap from sliders inside the loop

Pump Tk events each tick, read slider values, and write them to the mocap arrays before the existing `torso_task.set_target(...)` line picks them up.

**Files:**
- Modify: `examples/humanoid_g1_torso.py` (top of the body of `while viewer.is_running():`).

- [ ] **Step 1: Replace the loop body's first line**

Find the loop body:

```python
        while viewer.is_running():
            # Update the torso target from the (user-draggable) mocap body.
            torso_task.set_target(mink.SE3.from_mocap_id(data, torso_mid))
```

Replace those three lines (loop header + comment + set_target) with:

```python
        while viewer.is_running():
            # Pump Tk events; read sliders into the mocap target.
            # Mouse-drag of the mocap is overwritten on the next tick — sliders win.
            if tk_alive:
                try:
                    tk_root.update()
                    data.mocap_pos[torso_mid] = [v.get() for v in pos_vars]
                    rpy_rad = np.deg2rad([v.get() for v in rpy_vars])
                    data.mocap_quat[torso_mid] = mink.SO3.from_rpy_radians(
                        *rpy_rad
                    ).wxyz
                except tk.TclError:
                    tk_alive = False
            torso_task.set_target(mink.SE3.from_mocap_id(data, torso_mid))
```

- [ ] **Step 2: Verify the file still parses**

```bash
.venv/bin/python -c "import ast; ast.parse(open('examples/humanoid_g1_torso.py').read())"
```

Expected: no output.

---

## Task 6: Manual verification

Automated tests don't cover this kind of GUI loop. Run the example and exercise the UI to confirm acceptance criteria.

**Files:**
- None. Read-only run.

- [ ] **Step 1: Launch the example**

```bash
.venv/bin/python examples/humanoid_g1_torso.py
```

Expected: a MuJoCo viewer window AND a "Torso pose" Tk window with 6 sliders and a Reset button. Console may show MuJoCo / GL info; no Python tracebacks.

- [ ] **Step 2: Exercise position sliders**

Move each of `x`, `y`, `z` sliders in turn. Confirm:
- The translucent red box (the mocap target) translates along the corresponding world axis.
- The G1 torso tracks the box, with the feet remaining pinned and posture limits respected.
- No tracebacks in the console.

- [ ] **Step 3: Exercise rotation sliders**

Move each of `roll`, `pitch`, `yaw` sliders. Confirm:
- The mocap box rotates as expected.
- The torso link rotates toward the box; IK doesn't oscillate.

- [ ] **Step 4: Reset**

After moving sliders, click **Reset**. Confirm:
- All six sliders snap back to the captured initial values.
- The mocap box returns to the standing position; the torso follows.

- [ ] **Step 5: Close the slider window**

Click the OS close button on the Tk window. Confirm:
- The Tk window disappears.
- The MuJoCo viewer keeps running, with the torso target frozen at the last commanded pose.
- No unhandled exceptions in the console.

- [ ] **Step 6: Quit the viewer**

Close the MuJoCo viewer. The script should exit cleanly (return code 0).

If any of these checks fail, **stop and report**: do not commit until the failure mode is understood.

---

## Task 7: Commit

- [ ] **Step 1: Stage and commit the slider feature**

```bash
git add examples/humanoid_g1_torso.py
git status --short
```

Expected `git status --short` shows only `M examples/humanoid_g1_torso.py` (and no other unintended files).

```bash
git commit -m "$(cat <<'EOF'
feat(example): tkinter sliders for torso pose teleop in G1 example

Replaces mouse-drag mocap manipulation with a Tk slider window for
six-DOF control over the torso IK target. Three position sliders
(meters, init ± 0.5 m) and three RPY rotation sliders (deg, ±180)
plus a Reset button. Tk events are pumped from the existing viewer
loop; closing the slider window leaves the viewer running with the
last commanded pose.

Spec: docs/superpowers/specs/2026-05-06-humanoid-torso-sliders-design.md
EOF
)"
```

- [ ] **Step 2: Confirm clean tree**

```bash
git status --short
```

Expected: empty output (working tree clean).

---

## Self-review notes (for the planner — already applied)

- Spec coverage: every "Acceptance criteria" bullet in the spec maps to a verification step in Task 6.
- Quaternion convention is explicit (wxyz on both ends); Task 1 verifies it before code is written.
- No "TODO"/"TBD" placeholders. Every code step shows the exact code to insert.
- Type / name consistency: `pos_vars`, `rpy_vars`, `init_pos`, `init_rpy_deg`, `tk_alive` are introduced once and referenced consistently.
- Pre-existing `CollisionAvoidanceLimit` modification is acknowledged in the Prerequisites section so the executor isn't surprised.
