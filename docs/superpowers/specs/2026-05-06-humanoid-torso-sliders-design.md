# Tkinter sliders for torso pose target — `examples/humanoid_g1_torso.py`

Date: 2026-05-06

## Background

`examples/humanoid_g1_torso.py` demonstrates whole-body IK on the Unitree G1 by
tracking a torso pose target. Today, the target is a mocap body the user drags
with the mouse in the MuJoCo viewer. Mouse drag is awkward for precise pose
specification — small adjustments are jittery, and rotation handles are
difficult to use without overshooting.

## Goal

Add a Tkinter slider window alongside the MuJoCo viewer that gives the user
direct numeric control over the 6-DOF torso pose target. Sliders provide
smooth, repeatable input that mouse drag does not.

## Non-Goals

- Removing the mocap body from the scene XML.
- Disabling mouse-drag manipulation in the MuJoCo viewer.
- Replicating this pattern across other examples.
- Building a reusable `mink.contrib` helper. (Possible follow-up; the example
  stays self-contained for now.)

## Design

### Slider window

A single Tkinter `Tk` window with:

- Three position sliders: `x`, `y`, `z` in meters. Range = `init_pos ± 0.5 m`
  per axis, resolution `0.001`.
- Three rotation sliders: `roll`, `pitch`, `yaw` in degrees. Range `±180°`,
  resolution `0.5°`.
- One **Reset** button that snaps all six sliders back to the captured initial
  pose.

The window is created **after** the existing `mink.move_mocap_to_frame(...)`
call, so the captured initial torso pose is available to populate slider
initial values and the reset reference.

### Quaternion convention

`data.mocap_quat` uses `wxyz` ordering. mink's `SO3` also stores `wxyz`. So
the conversion is direct:

```python
quat_wxyz = mink.SO3.from_rpy_radians(roll, pitch, yaw).wxyz
```

No reordering required.

### Initial Euler from initial quat

The initial mocap quaternion (set by `move_mocap_to_frame`) is converted to
RPY for slider initial values via:

```python
init_quat_wxyz = data.mocap_quat[torso_mid].copy()
init_rpy = mink.SO3(wxyz=init_quat_wxyz).as_rpy_radians()  # RollPitchYaw NamedTuple
init_roll_deg  = np.rad2deg(init_rpy.roll)
init_pitch_deg = np.rad2deg(init_rpy.pitch)
init_yaw_deg   = np.rad2deg(init_rpy.yaw)
```

### Per-tick update path

Inside the existing `while viewer.is_running():` loop, before the existing
`torso_task.set_target(...)` line:

1. `root.update()` — pump Tk events. Wrapped in `try/except tk.TclError` so
   closing the slider window doesn't crash the simulation viewer.
2. Read 6 slider values from `tk.DoubleVar` instances.
3. Write `data.mocap_pos[torso_mid]` from the three position sliders.
4. Write `data.mocap_quat[torso_mid]` from
   `mink.SO3.from_rpy_radians(...).wxyz`.

After step 4, the existing
`torso_task.set_target(mink.SE3.from_mocap_id(data, torso_mid))` line picks
up the updated mocap and IK runs as before.

### Reset behavior

The Reset button calls a closure that sets each `DoubleVar` back to its
captured initial value. The per-tick path reads sliders unconditionally, so
the next tick restores the mocap target.

### Window-close handling

If the user closes the slider window via the OS window manager, `root.update()`
raises `tk.TclError`. We catch it once, set a flag, and skip further Tk
interaction. The viewer remains running; mocap stays at its last commanded
pose. No noisy log spam.

## Things that stay unchanged

- `examples/unitree_g1/scene_g1_torso.xml`: the mocap body remains as-is.
- IK tasks: `torso_task`, `posture_task`, the two foot frame tasks.
- Limits: `ConfigurationLimit`, `CollisionAvoidanceLimit`.
- Keyframe init (`stand_drag`), `posture_task.set_target_from_configuration()`,
  foot pinning at initial standing positions.
- Solver, damping, rate limiter.

## Dependencies

`tkinter` is in the Python standard library for the supported versions
(3.10–3.13). No change to `pyproject.toml`.

## Risks

- Tk on a system without a display server may fail. Acceptable: this is an
  interactive GUI example; if the user can run the MuJoCo viewer they can run
  Tk too.
- Concurrent mouse drag of the mocap will be instantly overwritten by the
  next slider read. This is acceptable — sliders are the authority. Mentioned
  in an inline comment.

## Acceptance criteria

- Running `python examples/humanoid_g1_torso.py` opens the MuJoCo viewer and
  a Tk slider window.
- Each position slider translates the torso target smoothly along its axis;
  the IK solution follows.
- Each rotation slider rotates the torso target smoothly about its axis; the
  IK solution follows.
- Reset returns sliders and torso target to the captured standing pose.
- Closing the slider window does not crash the viewer; the simulation
  continues with the last commanded pose.

## Out of scope / follow-ups

- A reusable slider-teleop helper in `mink.contrib`, analogous to
  `TeleopMocap`.
- Disabling mouse-drag mocap manipulation in the MuJoCo viewer.
- Porting the slider UI to other mocap-based examples.
