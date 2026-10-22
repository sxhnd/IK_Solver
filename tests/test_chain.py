import numpy as np
import pytest

from ik.chain import Chain2D, wrap_angle


def test_straight_arm_lies_along_x_axis():
    chain = Chain2D([1.0, 2.0])
    np.testing.assert_allclose(
        chain.joint_positions([0, 0]),
        [[0, 0], [1, 0], [3, 0]],
    )


def test_first_joint_at_90_degrees_points_arm_straight_up():
    chain = Chain2D([1.0, 2.0])
    np.testing.assert_allclose(
        chain.joint_positions([np.pi / 2, 0]),
        [[0, 0], [0, 1], [0, 3]],
        atol=1e-12,
    )


def test_angles_are_relative_to_previous_link():
    chain = Chain2D([1.0, 2.0])
    np.testing.assert_allclose(
        chain.joint_positions([1, 0]),
        [
            [0, 0],
            [np.cos(1), np.sin(1)],
            [np.cos(1) + 2 * np.cos(1), np.sin(1) + 2 * np.sin(1)],
        ],
        atol=1e-12,
    )


def test_base_offset_shifts_every_point():
    chain = Chain2D([1.0, 1.0], base=(5.0, -1.0))
    np.testing.assert_allclose(
        chain.joint_positions([0, 0]),
        [[5, -1], [6, -1], [7, -1]],
    )


def test_link_lengths_are_preserved_for_any_angles():
    chain = Chain2D([1.0, 0.7, 0.3])
    points = chain.joint_positions([0.4, -2.1, 1.3])
    gaps = np.linalg.norm(np.diff(points, axis=0), axis=1)
    np.testing.assert_allclose(gaps, chain.lengths)


def test_reach_is_sum_of_lengths():
    assert Chain2D([1.0, 0.7, 0.3]).reach == pytest.approx(2.0)


@pytest.mark.parametrize(
    "bad",
    [[], [1.0, 0.0], [1.0, -1.0], [float("nan")], [float("inf")], [[1.0, 2.0]]],
)
def test_rejects_bad_lengths(bad):
    with pytest.raises(ValueError):
        Chain2D(bad)


@pytest.mark.parametrize("bad", [(1.0,), (1.0, 2.0, 3.0), [[1.0, 2.0]]])
def test_rejects_base_without_two_coordinates(bad):
    with pytest.raises(ValueError):
        Chain2D([1.0], base=bad)


@pytest.mark.parametrize("bad", [[0.5], [0.5, 0.0, 0.0]])
def test_joint_positions_wrong_number_of_angles_raises_value_error(bad):
    with pytest.raises(ValueError):
        Chain2D([1.0, 2.0]).joint_positions(bad)


def test_end_effector_is_the_last_joint_position():
    chain = Chain2D([1.0, 0.7, 0.3])
    q = [0.4, -2.1, 1.3]
    np.testing.assert_allclose(chain.end_effector(q), chain.joint_positions(q)[-1])


def test_n_joints_counts_links():
    assert Chain2D([1.0, 0.7, 0.3]).n_joints == 3


def test_wrap_angle():
    np.testing.assert_allclose(
        wrap_angle([0.0, 3 * np.pi / 2, -3 * np.pi / 2]),
        [0.0, -np.pi / 2, np.pi / 2],
    )


def test_wrap_angle_maps_pi_to_minus_pi():
    np.testing.assert_allclose(wrap_angle([np.pi, -np.pi]), [-np.pi, -np.pi])
