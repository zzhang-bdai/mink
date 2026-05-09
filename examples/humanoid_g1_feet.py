import tkinter as tk
from collections.abc import Callable, Sequence
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np
from loop_rate_limiters import RateLimiter

import mink

_HERE = Path(__file__).parent
XML = _HERE / "unitree_g1" / "scene.xml"

FEET = ("right_foot", "left_foot")
_FLOATING_BASE = "floating_base_joint"
_TORSO = "torso_link"
TARGETS = (_TORSO, *FEET)
TORSO_TARGET = mink.SE3.from_rotation_and_translation(
    rotation=mink.SO3.identity(),
    translation=np.array([0.0, 0.0, 0.8], dtype=np.float64),
)


def pose_from_xyz_rpy_degrees(
    xyz: Sequence[float],
    rpy_degrees: Sequence[float],
) -> mink.SE3:
    rpy = np.deg2rad(np.asarray(rpy_degrees, dtype=np.float64))
    return mink.SE3.from_rotation_and_translation(
        rotation=mink.SO3.from_rpy_radians(*rpy),
        translation=np.asarray(xyz, dtype=np.float64),
    )


def _format_qpos_value(value: float) -> str:
    if abs(value) < 1e-6:
        value = 0.0
    return f"{value:.6f}"


def format_slider_entry_value(value: float) -> str:
    return _format_qpos_value(value)


def _qpos_labels(model: mujoco.MjModel) -> list[tuple[int, str]]:
    labels = []
    for joint_id in range(model.njnt):
        joint_name = mujoco.mj_id2name(
            model,
            mujoco.mjtObj.mjOBJ_JOINT,
            joint_id,
        )
        if joint_name is None:
            joint_name = f"joint_{joint_id}"

        qadr = model.jnt_qposadr[joint_id]
        joint_type = model.jnt_type[joint_id]
        if joint_type == mujoco.mjtJoint.mjJNT_FREE:
            components = ("x", "y", "z", "qw", "qx", "qy", "qz")
            labels.extend(
                (qadr + idx, f"{joint_name}.{component}")
                for idx, component in enumerate(components)
            )
        elif joint_type == mujoco.mjtJoint.mjJNT_BALL:
            components = ("qw", "qx", "qy", "qz")
            labels.extend(
                (qadr + idx, f"{joint_name}.{component}")
                for idx, component in enumerate(components)
            )
        else:
            labels.append((qadr, joint_name))
    return sorted(labels)


def format_qpos_for_terminal(model: mujoco.MjModel, qpos: np.ndarray) -> str:
    values = np.asarray(qpos, dtype=np.float64).copy()
    lines = ["qpos:"]
    for qpos_id, label in _qpos_labels(model):
        lines.append(f"  {label} = {_format_qpos_value(values[qpos_id])}")
    return "\n".join(lines)


def zero_configuration_qpos(model: mujoco.MjModel) -> np.ndarray:
    q_zero = np.zeros(model.nq)
    qadr = model.joint(_FLOATING_BASE).qposadr[0]
    q_zero[qadr + 3] = 1.0

    data = mujoco.MjData(model)
    data.qpos[:] = q_zero
    mujoco.mj_forward(model, data)

    torso_id = model.body(_TORSO).id
    torso_at_zero = mink.SE3.from_rotation_and_translation(
        rotation=mink.SO3.from_matrix(data.xmat[torso_id].reshape(3, 3)),
        translation=data.xpos[torso_id].copy(),
    )
    floating_base = TORSO_TARGET @ torso_at_zero.inverse()

    q = q_zero.copy()
    q[qadr : qadr + 3] = floating_base.translation()
    q[qadr + 3 : qadr + 7] = floating_base.rotation().wxyz
    return q


def initialize_zero_configuration(configuration: mink.Configuration) -> None:
    configuration.update(q=zero_configuration_qpos(configuration.model))


def get_initial_foot_targets(
    configuration: mink.Configuration,
) -> dict[str, mink.SE3]:
    return {
        foot: configuration.get_transform_frame_to_world(foot, "site") for foot in FEET
    }


def get_initial_pose_targets(
    configuration: mink.Configuration,
) -> dict[str, mink.SE3]:
    return {
        _TORSO: TORSO_TARGET,
        **get_initial_foot_targets(configuration),
    }


def _rpy_degrees(pose: mink.SE3) -> tuple[float, float, float]:
    rpy = pose.rotation().as_rpy_radians()
    return (
        float(np.rad2deg(rpy.roll)),
        float(np.rad2deg(rpy.pitch)),
        float(np.rad2deg(rpy.yaw)),
    )


def _set_mocap_pose(data: mujoco.MjData, mocap_id: int, pose: mink.SE3) -> None:
    data.mocap_pos[mocap_id] = pose.translation()
    data.mocap_quat[mocap_id] = pose.rotation().wxyz


def _add_slider_row(
    parent: tk.Widget,
    row: int,
    label: str,
    var: tk.DoubleVar,
    lower: float,
    upper: float,
    resolution: float,
) -> None:
    tk.Label(parent, text=label).grid(row=row, column=0, sticky="w", padx=(8, 6))
    tk.Scale(
        parent,
        from_=lower,
        to=upper,
        resolution=resolution,
        orient="horizontal",
        variable=var,
        length=320,
        showvalue=False,
    ).grid(row=row, column=1, sticky="ew")

    entry_var = tk.StringVar(value=format_slider_entry_value(var.get()))
    entry = tk.Entry(parent, textvariable=entry_var, width=10)
    entry.grid(row=row, column=2, sticky="e", padx=(6, 8))

    def sync_entry(*_: str) -> None:
        try:
            entry_var.set(format_slider_entry_value(var.get()))
        except tk.TclError:
            pass

    def commit_entry(_: tk.Event | None = None) -> None:
        try:
            value = float(entry_var.get())
        except ValueError:
            sync_entry()
            return
        value = min(max(value, lower), upper)
        var.set(value)
        entry_var.set(format_slider_entry_value(value))

    var.trace_add("write", sync_entry)
    entry.bind("<Return>", commit_entry)
    entry.bind("<FocusOut>", commit_entry)


def _make_slider_window(
    initial_targets: dict[str, mink.SE3],
    save_command: Callable[[], None] | None = None,
):
    root = tk.Tk()
    root.title("G1 pose targets")

    slider_vars = {}
    for target_name, target in initial_targets.items():
        target_frame = tk.LabelFrame(root, text=target_name.replace("_", " "))
        target_frame.pack(fill="x", padx=8, pady=(8, 4))
        target_frame.columnconfigure(1, weight=1)

        pos = target.translation()
        rpy_deg = _rpy_degrees(target)
        pos_vars = [tk.DoubleVar(value=float(v)) for v in pos]
        rpy_vars = [tk.DoubleVar(value=v) for v in rpy_deg]
        slider_vars[target_name] = (pos_vars, rpy_vars)

        row = 0
        for label, var, center in zip(("x", "y", "z"), pos_vars, pos):
            _add_slider_row(
                target_frame,
                row,
                f"{label} (m)",
                var,
                lower=float(center - 0.5),
                upper=float(center + 0.5),
                resolution=0.001,
            )
            row += 1

        for label, var in zip(("roll", "pitch", "yaw"), rpy_vars):
            _add_slider_row(
                target_frame,
                row,
                f"{label} (deg)",
                var,
                lower=-180.0,
                upper=180.0,
                resolution=0.5,
            )
            row += 1

    def reset_sliders() -> None:
        for target_name, target in initial_targets.items():
            pos_vars, rpy_vars = slider_vars[target_name]
            for var, val in zip(pos_vars, target.translation()):
                var.set(float(val))
            for var, val in zip(rpy_vars, _rpy_degrees(target)):
                var.set(val)

    buttons = tk.Frame(root)
    buttons.pack(pady=8)
    if save_command is not None:
        tk.Button(buttons, text="Save", command=save_command).pack(side="left", padx=4)
    tk.Button(buttons, text="Reset", command=reset_sliders).pack(side="left", padx=4)
    return root, slider_vars


def _read_slider_targets(slider_vars) -> dict[str, mink.SE3]:
    targets = {}
    for foot, (pos_vars, rpy_vars) in slider_vars.items():
        targets[foot] = pose_from_xyz_rpy_degrees(
            xyz=[var.get() for var in pos_vars],
            rpy_degrees=[var.get() for var in rpy_vars],
        )
    return targets


def main() -> None:
    model = mujoco.MjModel.from_xml_path(XML.as_posix())
    configuration = mink.Configuration(model)

    posture_cost = np.full(model.nv, 1e-2)
    for joint_name in ("waist_roll_joint", "waist_pitch_joint"):
        posture_cost[model.joint(joint_name).dofadr[0]] = 1.0

    tasks = [
        torso_task := mink.FrameTask(
            frame_name=_TORSO,
            frame_type="body",
            position_cost=100.0,
            orientation_cost=100.0,
            lm_damping=1.0,
        ),
        posture_task := mink.PostureTask(model, cost=posture_cost),
    ]

    feet_tasks = []
    for foot in FEET:
        task = mink.FrameTask(
            frame_name=foot,
            frame_type="site",
            position_cost=100.0,
            orientation_cost=100.0,
            lm_damping=1.0,
        )
        feet_tasks.append(task)
    tasks.extend(feet_tasks)

    limits = [mink.ConfigurationLimit(model)]
    feet_mid = [model.body(f"{foot}_target").mocapid[0] for foot in FEET]

    initialize_zero_configuration(configuration)
    posture_task.set_target_from_configuration(configuration)
    torso_task.set_target(TORSO_TARGET)
    initial_targets = get_initial_pose_targets(configuration)

    model = configuration.model
    data = configuration.data
    for foot, mocap_id in zip(FEET, feet_mid):
        _set_mocap_pose(data, mocap_id, initial_targets[foot])

    solver = "daqp"

    with mujoco.viewer.launch_passive(
        model=model,
        data=data,
        show_left_ui=False,
        show_right_ui=False,
    ) as viewer:
        mujoco.mjv_defaultFreeCamera(model, viewer.cam)

        def save_qpos() -> None:
            print(
                format_qpos_for_terminal(model, configuration.data.qpos.copy()),
                flush=True,
            )

        tk_root, slider_vars = _make_slider_window(
            initial_targets,
            save_command=save_qpos,
        )
        tk_alive = True

        rate = RateLimiter(frequency=200.0, warn=False)
        while viewer.is_running():
            if tk_alive:
                try:
                    tk_root.update()
                    slider_targets = _read_slider_targets(slider_vars)
                    torso_task.set_target(slider_targets[_TORSO])
                    for foot, mocap_id in zip(FEET, feet_mid):
                        _set_mocap_pose(data, mocap_id, slider_targets[foot])
                except tk.TclError:
                    tk_alive = False

            for foot_task, mocap_id in zip(feet_tasks, feet_mid):
                foot_task.set_target(mink.SE3.from_mocap_id(data, mocap_id))

            vel = mink.solve_ik(
                configuration,
                tasks,
                rate.dt,
                solver,
                damping=1e-1,
                limits=limits,
            )
            configuration.integrate_inplace(vel, rate.dt)
            mujoco.mj_camlight(model, data)

            viewer.sync()
            rate.sleep()


if __name__ == "__main__":
    main()
