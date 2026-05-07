from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np
from loop_rate_limiters import RateLimiter

import tkinter as tk

import mink

_HERE = Path(__file__).parent
_XML = _HERE / "unitree_g1" / "scene_g1_torso.xml"


if __name__ == "__main__":
    model = mujoco.MjModel.from_xml_path(_XML.as_posix())

    configuration = mink.Configuration(model)
    feet = ["right_foot", "left_foot"]

    posture_cost = np.full(model.nv, 1e-1)
    # posture_cost[model.joint("waist_yaw_joint").dofadr[0]] = 1.0
    posture_cost[model.joint("waist_roll_joint").dofadr[0]] = 5.0
    posture_cost[model.joint("waist_pitch_joint").dofadr[0]] = 1.0

    tasks = [
        torso_task := mink.FrameTask(
            frame_name="torso_link",
            frame_type="body",
            position_cost=[10.0, 10.0, 10.0],
            orientation_cost=10.0,
            lm_damping=1.0,
        ),
        posture_task := mink.PostureTask(model, cost=posture_cost),
    ]

    feet_tasks = []
    for foot in feet:
        task = mink.FrameTask(
            frame_name=foot,
            frame_type="site",
            position_cost=1000.0,
            orientation_cost=1000.0,
            lm_damping=1.0,
        )
        feet_tasks.append(task)
    tasks.extend(feet_tasks)

    collision_pairs = [
        (["left_hand_collision"], ["left_thigh_collision"]),
        (["right_hand_collision"], ["right_thigh_collision"]),
        (["torso_collision"], ["left_thigh_collision"]),
        (["torso_collision"], ["right_thigh_collision"]),
    ]
    collision_avoidance_limit = mink.CollisionAvoidanceLimit(
        model=model,
        geom_pairs=collision_pairs,  # type: ignore
        minimum_distance_from_collisions=0.005,
        collision_detection_distance=0.15,
    )

    config_limit = mink.ConfigurationLimit(model)
    # Keep knees above 0.17 rad (tighter than the model's native lower bound).
    for jname in ("left_knee_joint", "right_knee_joint"):
        config_limit.lower[model.joint(jname).qposadr[0]] = 0.17

    limits = [
        config_limit,
        collision_avoidance_limit,
    ]

    torso_mid = model.body("torso_target").mocapid[0]

    model = configuration.model
    data = configuration.data
    solver = "daqp"

    # Pressing 'R' in the viewer requests a fresh random pose target. We only
    # flip a flag here; Tk isn't thread-safe so the main loop does the work.
    resample_requested = [False]

    def _key_callback(keycode: int) -> None:
        if keycode == ord("R"):
            resample_requested[0] = True

    with mujoco.viewer.launch_passive(
        model=model,
        data=data,
        show_left_ui=False,
        show_right_ui=False,
        key_callback=_key_callback,
    ) as viewer:
        mujoco.mjv_defaultFreeCamera(model, viewer.cam)

        # Initialize to the standing keyframe.
        configuration.update_from_keyframe("stand_drag")
        posture_task.set_target_from_configuration(configuration)

        # Pin the feet at their initial standing positions.
        for foot, foot_task in zip(feet, feet_tasks):
            foot_task.set_target(
                configuration.get_transform_frame_to_world(foot, "site")
            )

        # Place the torso target mocap at the current torso pose.
        mink.move_mocap_to_frame(model, data, "torso_target", "torso_link", "body")

        # Capture initial torso target pose (used for slider init + reset).
        init_pos = data.mocap_pos[torso_mid].copy()
        _init_rpy = mink.SO3(wxyz=data.mocap_quat[torso_mid].copy()).as_rpy_radians()
        init_rpy_deg = (
            float(np.rad2deg(_init_rpy.roll)),
            float(np.rad2deg(_init_rpy.pitch)),
            float(np.rad2deg(_init_rpy.yaw)),
        )

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

        tk_alive = True  # cleared if the slider window is closed
        # -------------------------------------------------------------------

        rate = RateLimiter(frequency=200.0, warn=False)
        while viewer.is_running():
            if resample_requested[0]:
                resample_requested[0] = False
                # Reset to the standing keyframe so each random target is
                # approached from the same initial joint configuration.
                configuration.update_from_keyframe("stand_drag")
                sampled_pos = (
                    0.0,
                    0.0,
                    float(np.random.uniform(0.35, 0.8)),
                )
                sampled_rpy_deg = (
                    float(np.random.uniform(-10.0, 10.0)),
                    float(np.random.uniform(-15.0, 90.0)),
                    float(np.random.uniform(-45.0, 45.0)),
                )
                if tk_alive:
                    for var, val in zip(pos_vars, sampled_pos):
                        var.set(val)
                    for var, val in zip(rpy_vars, sampled_rpy_deg):
                        var.set(val)
                else:
                    data.mocap_pos[torso_mid] = sampled_pos
                    data.mocap_quat[torso_mid] = mink.SO3.from_rpy_radians(
                        *np.deg2rad(sampled_rpy_deg)
                    ).wxyz
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

            vel = mink.solve_ik(
                configuration, tasks, rate.dt, solver, damping=1e-1, limits=limits
            )
            configuration.integrate_inplace(vel, rate.dt)
            mujoco.mj_camlight(model, data)

            viewer.sync()
            rate.sleep()
