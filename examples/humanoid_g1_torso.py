from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np
from loop_rate_limiters import RateLimiter

import mink

_HERE = Path(__file__).parent
_XML = _HERE / "unitree_g1" / "scene_g1_torso.xml"


if __name__ == "__main__":
    model = mujoco.MjModel.from_xml_path(_XML.as_posix())

    configuration = mink.Configuration(model)
    feet = ["right_foot", "left_foot"]

    posture_cost = np.full(model.nv, 1e-1)
    posture_cost[model.joint("waist_yaw_joint").dofadr[0]] = 5.0
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

    limits = [mink.ConfigurationLimit(model)]

    torso_mid = model.body("torso_target").mocapid[0]

    model = configuration.model
    data = configuration.data
    solver = "daqp"

    with mujoco.viewer.launch_passive(
        model=model, data=data, show_left_ui=False, show_right_ui=False
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

        rate = RateLimiter(frequency=200.0, warn=False)
        while viewer.is_running():
            # Update the torso target from the (user-draggable) mocap body.
            torso_task.set_target(mink.SE3.from_mocap_id(data, torso_mid))

            vel = mink.solve_ik(
                configuration, tasks, rate.dt, solver, damping=1e-1, limits=limits
            )
            configuration.integrate_inplace(vel, rate.dt)
            mujoco.mj_camlight(model, data)

            viewer.sync()
            rate.sleep()
