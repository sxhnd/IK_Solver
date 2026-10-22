from typing import Final

import numpy as np
import numpy.typing as npt
import pytest
from ik.analytic import solve_two_link
from ik.chain import Chain2D

# Forward kinematics of an exact solution is accurate to machine
# precision; 1e-9 leaves room for arccos losing digits near a straight
# elbow.
POSITION_TOLERANCE_M: Final = 1e-9


def test_solve_two_link_reachable_target_both_solutions_reach_it() -> None:
    chain = Chain2D([1.0, 0.6])
    target = np.array([0.9, 0.7])

    solutions = solve_two_link(chain, target)

    assert len(solutions) == 2
    for q in solutions:
        np.testing.assert_allclose(
            chain.end_effector(q), target, rtol=0, atol=POSITION_TOLERANCE_M
        )


def test_solve_two_link_reachable_target_elbows_are_mirror_images() -> None:
    q_down, q_up = solve_two_link(Chain2D([1.0, 0.6]), [0.9, 0.7])

    assert q_down[1] == pytest.approx(-q_up[1], abs=POSITION_TOLERANCE_M)


def test_solve_two_link_random_reachable_targets_are_all_reached() -> None:
    chain = Chain2D([1.0, 0.6], base=(0.5, -0.2))
    rng = np.random.default_rng(0)

    for q_true in rng.uniform(-np.pi, np.pi, size=(50, 2)):
        target = chain.end_effector(q_true)
        for q in solve_two_link(chain, target):
            np.testing.assert_allclose(
                chain.end_effector(q), target, rtol=0, atol=POSITION_TOLERANCE_M
            )


def test_solve_two_link_fully_stretched_target_gives_straight_elbow() -> None:
    for q in solve_two_link(Chain2D([1.0, 0.6]), [1.6, 0.0]):
        assert q[1] == pytest.approx(0.0, abs=1e-6)


@pytest.mark.parametrize("target", [[2.0, 0.0], [0.1, 0.1]])  # Too far, too close.
def test_solve_two_link_unreachable_target_returns_no_solutions(
    target: npt.ArrayLike,
) -> None:
    assert solve_two_link(Chain2D([1.0, 0.6]), target) == []


def test_solve_two_link_three_links_raises_value_error() -> None:
    with pytest.raises(ValueError, match="2 links"):
        solve_two_link(Chain2D([1.0, 1.0, 1.0]), [1.0, 1.0])
