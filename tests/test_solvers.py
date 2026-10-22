"""Behavior every iterative solver must have.

Each solver in SOLVERS runs through all of these tests.
"""

from collections.abc import Callable
from typing import Final

import numpy as np
import numpy.typing as npt
import pytest
from ik.ccd import solve_ccd
from ik.chain import Chain2D
from ik.ik_result import IKResult

SOLVERS = [solve_ccd]
# Solvers that must handle unreachable targets. A solver known to fail
# is left out of this list.
UNREACHABLE_SOLVERS = SOLVERS

# Position tolerance, in the same units as the link lengths.
TOL: Final = 1e-4


@pytest.fixture
def chain() -> Chain2D:
    return Chain2D([1.0, 0.8, 0.5, 0.3])


def random_poses(chain: Chain2D, n: int, seed: int) -> npt.NDArray[np.float64]:
    rng = np.random.default_rng(seed)
    return rng.uniform(-np.pi, np.pi, size=(n, chain.n_joints))


def random_reachable_targets(
    chain: Chain2D, n: int, seed: int
) -> list[npt.NDArray[np.float64]]:
    # The tip of any pose is reachable. Many poses reach the same point,
    # tests compare tip positions, not joint angles.
    return [chain.end_effector(q) for q in random_poses(chain, n, seed)]


@pytest.mark.parametrize("solve", SOLVERS)
def test_reaches_random_targets_from_random_starts(
    solve: Callable[..., IKResult], chain: Chain2D
) -> None:
    starts = random_poses(chain, 25, seed=5)
    targets = random_reachable_targets(chain, 25, seed=1)

    for q0, target in zip(starts, targets, strict=True):
        result = solve(chain, target, q0=q0, tol=TOL)

        assert result.is_converged, f"missed {target} from {q0}: error {result.error:.2e}"
        assert np.linalg.norm(chain.end_effector(result.q) - target) < TOL


@pytest.mark.parametrize("solve", UNREACHABLE_SOLVERS)
def test_unreachable_target_gives_straight_arm_and_no_convergence(
    solve: Callable[..., IKResult], chain: Chain2D
) -> None:
    # A far target gives a large error vector, exposing solvers whose
    # steps are too big.
    target = 3 * chain.reach * np.array([0.6, 0.8])

    # A single start can hide a bad solver, so try several.
    for q0 in [None, *random_poses(chain, 10, seed=6)]:
        result = solve(chain, target, q0=q0, tol=TOL)

        assert not result.is_converged
        # The best pose points straight at the target, so the error is
        # distance - reach, which is 2 * reach here.
        assert result.error == pytest.approx(2 * chain.reach, abs=1e-3)


@pytest.mark.parametrize("solve", SOLVERS)
def test_already_at_target_takes_zero_iterations(
    solve: Callable[..., IKResult], chain: Chain2D
) -> None:
    q0 = np.array([0.3, -0.4, 0.5, 0.2])

    result = solve(chain, chain.end_effector(q0), q0=q0, tol=TOL)

    assert result.is_converged
    assert result.iterations == 0


@pytest.mark.parametrize("solve", SOLVERS)
def test_does_not_modify_callers_q0(solve: Callable[..., IKResult], chain: Chain2D) -> None:
    q0 = np.array([0.3, -0.4, 0.5, 0.2])
    before = q0.copy()

    solve(chain, [0.5, 1.2], q0=q0, tol=TOL)

    np.testing.assert_array_equal(q0, before)


@pytest.mark.parametrize("solve", SOLVERS)
def test_reported_error_matches_forward_kinematics(
    solve: Callable[..., IKResult], chain: Chain2D
) -> None:
    target = np.array([0.5, 1.2])

    result = solve(chain, target, tol=TOL)

    expected_error = np.linalg.norm(chain.end_effector(result.q) - target)
    assert result.error == pytest.approx(expected_error, abs=1e-12)


@pytest.mark.parametrize("solve", SOLVERS)
def test_works_with_offset_base(solve: Callable[..., IKResult]) -> None:
    chain = Chain2D([1.0, 0.8, 0.5], base=(3.0, -2.0))

    for target in random_reachable_targets(chain, 10, seed=2):
        assert solve(chain, target, tol=TOL).is_converged


@pytest.mark.parametrize("solve", UNREACHABLE_SOLVERS)
def test_units_do_not_matter(solve: Callable[..., IKResult]) -> None:
    # The same arm in centimetres. A constant that assumes links are
    # 1 long fails here.
    chain = Chain2D([100.0, 80.0, 50.0, 30.0])

    for target in random_reachable_targets(chain, 10, seed=4):
        assert solve(chain, target, tol=TOL).is_converged

    far = solve(chain, [0.0, 3 * chain.reach], tol=TOL)
    assert far.error == pytest.approx(2 * chain.reach, rel=1e-3)


@pytest.mark.parametrize("target", [[0.0, 1.0, 2.0], [np.nan, 0.0]])
def test_rejects_invalid_target(chain: Chain2D, target: list[float]) -> None:
    with pytest.raises(ValueError, match="target"):
        solve_ccd(chain, target)


@pytest.mark.parametrize("q0", [[0.1], [0.1, 0.2, np.nan, 0.4]])
def test_rejects_invalid_starting_angles(chain: Chain2D, q0: list[float]) -> None:
    with pytest.raises(ValueError, match="starting angles"):
        solve_ccd(chain, [0.0, 1.0], q0=q0)


@pytest.mark.parametrize("tol", [0.0, -1.0, np.nan])
def test_rejects_invalid_tolerance(chain: Chain2D, tol: float) -> None:
    with pytest.raises(ValueError, match="tolerance"):
        solve_ccd(chain, [0.0, 1.0], tol=tol)


def test_rejects_negative_iteration_limit(chain: Chain2D) -> None:
    with pytest.raises(ValueError, match="max_iters"):
        solve_ccd(chain, [0.0, 1.0], max_iters=-1)


def test_zero_iteration_limit_returns_the_starting_pose(chain: Chain2D) -> None:
    q0 = np.zeros(chain.n_joints)

    result = solve_ccd(chain, [0.0, 1.0], q0=q0, max_iters=0)

    np.testing.assert_array_equal(result.q, q0)
    assert result.iterations == 0
    assert result.history == []
