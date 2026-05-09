from importlib import util
from pathlib import Path

import mujoco
import numpy as np

import mink

_ROOT = Path(__file__).resolve().parents[1]
_EXAMPLE = _ROOT / "examples" / "humanoid_g1_feet.py"


def _load_example():
    spec = util.spec_from_file_location("humanoid_g1_feet", _EXAMPLE)
    assert spec is not None
    assert spec.loader is not None
    module = util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_zero_joint_configuration_places_torso_at_requested_world_pose():
    example = _load_example()
    model = mujoco.MjModel.from_xml_path(example.XML.as_posix())
    configuration = mink.Configuration(model)

    example.initialize_zero_configuration(configuration)

    torso = configuration.get_transform_frame_to_world("torso_link", "body")
    np.testing.assert_allclose(
        torso.wxyz_xyz,
        example.TORSO_TARGET.wxyz_xyz,
        atol=1e-12,
    )
    np.testing.assert_allclose(configuration.data.qpos[7:], 0.0, atol=1e-12)


def test_pose_from_xyz_rpy_degrees_uses_mujoco_wxyz_quaternion_order():
    example = _load_example()

    pose = example.pose_from_xyz_rpy_degrees(
        xyz=(0.1, -0.2, 0.3),
        rpy_degrees=(90.0, 0.0, 0.0),
    )

    np.testing.assert_allclose(pose.translation(), [0.1, -0.2, 0.3], atol=1e-12)
    np.testing.assert_allclose(
        pose.rotation().wxyz,
        [np.sqrt(0.5), np.sqrt(0.5), 0.0, 0.0],
        atol=1e-12,
    )


def test_foot_targets_start_at_initialized_foot_sites():
    example = _load_example()
    model = mujoco.MjModel.from_xml_path(example.XML.as_posix())
    configuration = mink.Configuration(model)
    example.initialize_zero_configuration(configuration)

    targets = example.get_initial_foot_targets(configuration)

    assert tuple(targets) == example.FEET
    for foot, target in targets.items():
        site_pose = configuration.get_transform_frame_to_world(foot, "site")
        np.testing.assert_allclose(target.wxyz_xyz, site_pose.wxyz_xyz, atol=1e-12)


def test_initial_pose_targets_include_torso_and_feet():
    example = _load_example()
    model = mujoco.MjModel.from_xml_path(example.XML.as_posix())
    configuration = mink.Configuration(model)
    example.initialize_zero_configuration(configuration)

    targets = example.get_initial_pose_targets(configuration)

    assert tuple(targets) == ("torso_link", *example.FEET)
    np.testing.assert_allclose(
        targets["torso_link"].wxyz_xyz,
        example.TORSO_TARGET.wxyz_xyz,
        atol=1e-12,
    )
    for foot in example.FEET:
        site_pose = configuration.get_transform_frame_to_world(foot, "site")
        np.testing.assert_allclose(targets[foot].wxyz_xyz, site_pose.wxyz_xyz)


def test_format_qpos_for_terminal_prints_named_qpos_entries():
    example = _load_example()
    model = mujoco.MjModel.from_xml_string(
        """
        <mujoco>
          <worldbody>
            <body>
              <freejoint name="root"/>
              <geom type="sphere" size="0.01" mass="1"/>
              <body>
                <joint name="knee" type="hinge"/>
                <geom type="sphere" size="0.01" mass="1"/>
              </body>
            </body>
          </worldbody>
        </mujoco>
        """
    )
    qpos = np.array([0.1, 0.2, -2e-12, 1.0, 0.0, 0.0, 0.0, 1.23456789])

    text = example.format_qpos_for_terminal(model, qpos)

    assert text == "\n".join(
        [
            "qpos:",
            "  root.x = 0.100000",
            "  root.y = 0.200000",
            "  root.z = 0.000000",
            "  root.qw = 1.000000",
            "  root.qx = 0.000000",
            "  root.qy = 0.000000",
            "  root.qz = 0.000000",
            "  knee = 1.234568",
        ]
    )


def test_format_slider_entry_value_uses_fixed_precision_without_negative_zero():
    example = _load_example()

    assert example.format_slider_entry_value(1.23456789) == "1.234568"
    assert example.format_slider_entry_value(-2e-12) == "0.000000"
