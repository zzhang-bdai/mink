"""Visualize joint configurations from a G1 pose dataset .npy file.

Loads ``joints.npy`` (shape ``(N, 29)``), the sibling ``commands.npy``
(shape ``(N, 4)`` — roll, pitch, yaw, height) and ``joint_names.json``,
and lets you step through saved configurations one at a time. For each
pose the floating-base (pelvis) pose is computed so the ``torso_link``
body lands at the commanded SE(3) target ``(0, 0, height)`` with the
commanded RPY orientation.

    Right arrow: next pose
    Left arrow:  previous pose
    R:           jump back to pose 0

Usage::

    uv run python -m g1_pose_dataset.play
    uv run python -m g1_pose_dataset.play data/g1_torso_pose/pilot/joints.npy

On macOS the MuJoCo viewer must be launched with ``mjpython`` instead of
``python``.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np

import mink

_HERE = Path(__file__).parent
_REPO = _HERE.parent
_DEFAULT_XML = _REPO / "examples" / "unitree_g1" / "scene_g1_torso.xml"
_DEFAULT_JOINTS = _REPO / "data" / "g1_torso_pose" / "pilot" / "joints.npy"

# GLFW key codes (MuJoCo's viewer key_callback receives raw GLFW keycodes).
_KEY_RIGHT = 262
_KEY_LEFT = 263
_KEY_R = ord("R")


def _resolve_sibling(
    joints_path: Path, override: Path | None, name: str, flag: str
) -> Path:
    if override is not None:
        return override
    sibling = joints_path.parent / name
    if not sibling.exists():
        raise SystemExit(
            f"Could not find {name} next to {joints_path}. "
            f"Pass {flag} explicitly."
        )
    return sibling


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "joints",
        nargs="?",
        type=Path,
        default=_DEFAULT_JOINTS,
        help=f"Path to joints.npy (default: {_DEFAULT_JOINTS})",
    )
    parser.add_argument(
        "--xml",
        type=Path,
        default=_DEFAULT_XML,
        help=f"G1 MuJoCo scene XML (default: {_DEFAULT_XML})",
    )
    parser.add_argument(
        "--joint-names",
        type=Path,
        default=None,
        help="Path to joint_names.json (default: sibling of joints.npy)",
    )
    parser.add_argument(
        "--commands",
        type=Path,
        default=None,
        help="Path to commands.npy (default: sibling of joints.npy)",
    )
    parser.add_argument(
        "--torso-body",
        default="torso_link",
        help="Body whose pose follows the commanded SE(3) target.",
    )
    parser.add_argument(
        "--root-joint",
        default="floating_base_joint",
        help="Free joint that controls the floating-base pose.",
    )
    parser.add_argument(
        "--keyframe",
        default="stand_drag",
        help="Keyframe used to seed an initial valid root quaternion.",
    )
    args = parser.parse_args()

    joints_path: Path = args.joints
    if not joints_path.exists():
        raise SystemExit(f"Joints file not found: {joints_path}")
    if not args.xml.exists():
        raise SystemExit(f"Model XML not found: {args.xml}")

    joint_names_path = _resolve_sibling(
        joints_path, args.joint_names, "joint_names.json", "--joint-names"
    )
    commands_path = _resolve_sibling(
        joints_path, args.commands, "commands.npy", "--commands"
    )
    with joint_names_path.open() as f:
        joint_names: list[str] = json.load(f)

    joints = np.load(joints_path)
    commands = np.load(commands_path)
    if joints.ndim != 2 or joints.shape[1] != len(joint_names):
        raise SystemExit(
            f"joints array shape {joints.shape} does not match "
            f"{len(joint_names)} joint names from {joint_names_path.name}"
        )
    if commands.shape != (joints.shape[0], 4):
        raise SystemExit(
            f"commands shape {commands.shape} does not match joints "
            f"({joints.shape[0]} rows, expected 4 columns: roll, pitch, yaw, height)"
        )
    n_poses = joints.shape[0]
    print(
        f"Loaded {n_poses} poses x {joints.shape[1]} joints from {joints_path} "
        f"(commands from {commands_path.name})"
    )

    model = mujoco.MjModel.from_xml_path(args.xml.as_posix())
    data = mujoco.MjData(model)

    qpos_addrs = np.array(
        [model.joint(name).qposadr[0] for name in joint_names], dtype=np.int32
    )
    torso_id = model.body(args.torso_body).id
    root_id = model.joint(args.root_joint).bodyid[0]

    keyframe_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_KEY, args.keyframe
    )
    if keyframe_id < 0:
        raise SystemExit(
            f"Keyframe '{args.keyframe}' not found in {args.xml}. "
            "Use --keyframe to choose another."
        )
    mujoco.mj_resetDataKeyframe(model, data, keyframe_id)

    state = {"index": -1, "dirty": True}

    def show(idx: int) -> None:
        # Apply joint angles with whatever root pose is currently set; the
        # absolute torso pose computed via FK is then snapped to the command
        # by overriding the floating-base qpos.
        data.qpos[qpos_addrs] = joints[idx]
        mujoco.mj_forward(model, data)

        R_torso_now = data.xmat[torso_id].reshape(3, 3)
        p_torso_now = data.xpos[torso_id]
        R_root_now = data.xmat[root_id].reshape(3, 3)
        p_root_now = data.xpos[root_id]

        roll, pitch, yaw, height = (float(v) for v in commands[idx])
        R_torso_des = mink.SO3.from_rpy_radians(roll, pitch, yaw).as_matrix()
        p_torso_des = np.array([0.0, 0.0, height])

        # T_root_new = T_torso_des @ T_torso_now^-1 @ T_root_now
        R_mid = R_torso_now.T @ R_root_now
        p_mid = R_torso_now.T @ (p_root_now - p_torso_now)
        R_root_new = R_torso_des @ R_mid
        p_root_new = R_torso_des @ p_mid + p_torso_des

        data.qpos[0:3] = p_root_new
        data.qpos[3:7] = mink.SO3.from_matrix(R_root_new).wxyz
        mujoco.mj_forward(model, data)

        state["index"] = idx
        state["dirty"] = True
        print(
            f"Pose {idx + 1:>5d} / {n_poses}  "
            f"r={np.rad2deg(roll):+6.1f}° p={np.rad2deg(pitch):+6.1f}° "
            f"y={np.rad2deg(yaw):+6.1f}° h={height:.3f}m"
        )

    def key_callback(keycode: int) -> None:
        cur = state["index"]
        if keycode == _KEY_RIGHT:
            new = min(cur + 1, n_poses - 1)
        elif keycode == _KEY_LEFT:
            new = max(cur - 1, 0)
        elif keycode == _KEY_R:
            new = 0
        else:
            return
        if new != cur:
            show(new)

    print(
        "Controls: Right = next pose, Left = previous pose, R = reset to pose 0. "
        "Close the window to exit."
    )

    with mujoco.viewer.launch_passive(
        model=model,
        data=data,
        show_left_ui=False,
        show_right_ui=False,
        key_callback=key_callback,
    ) as viewer:
        mujoco.mjv_defaultFreeCamera(model, viewer.cam)
        show(0)
        while viewer.is_running():
            viewer.sync()
            time.sleep(1.0 / 60.0)


if __name__ == "__main__":
    main()
