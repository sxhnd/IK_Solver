"""Tests for the Jacobian transpose solver and the J^T step it uses.

The solver moves the joints by alpha * J^T e, where J is the Jacobian
and e is the tip-to-target error. The tests check the Jacobian against
finite differences, check the J^T products, check the update rule and
step size, and check convergence and edge cases.
"""

from typing import Final

import numpy as np
import numpy.typing as npt
import pytest
from ik.chain import Chain2D
from ik.jacobian import jacobian
from ik.jacobian_transpose import solve_jacobian_transpose

# Central-difference step. Its truncation error is about h^2 ~ 1e-12 and
# its rounding error about eps / h ~ 1e-10, so 1e-6 is a safe bound.
FD_STEP: Final = 1e-6
FD_ATOL: Final = 1e-6

# Exact linear algebra on values of order 1.
ALGEBRA_ATOL: Final = 1e-12

# Same position tolerance as tests/test_solvers.py.
TOL: Final = 1e-4

# Jacobian transpose converges slowly. In a reference run on the 4-link
# arm below, a cap of 200 iterations converged 5 of 25 cases, while a
# cap of 2000 converged all 25 (about 140 to 740 iterations each).
MAX_ITERS: Final = 2000


def finite_difference_jacobian(
    chain: Chain2D, q: npt.ArrayLike, h: float = FD_STEP
) -> npt.NDArray[np.float64]:
    """Differentiate end_effector numerically, without ik.jacobian."""
    q = np.asarray(q, dtype=np.float64)
    columns = []
    for i in range(q.size):
        dq = np.zeros_like(q)
        dq[i] = h
        columns.append((chain.end_effector(q + dq) - chain.end_effector(q - dq)) / (2 * h))
    return np.column_stack(columns)


def explicit_transpose_times(jac: npt.NDArray[np.float64], v: npt.ArrayLike) -> npt.NDArray:
    """Compute J^T v with explicit loops, not .T or @."""
    v = np.asarray(v, dtype=np.float64)
    rows, cols = jac.shape
    out = np.zeros(cols)
    for i in range(cols):
        for j in range(rows):
            out[i] += jac[j, i] * v[j]
    return out


def expected_first_step(
    chain: Chain2D, target: npt.ArrayLike, q0: npt.ArrayLike, *, alpha: float
) -> npt.NDArray[np.float64]:
    """Return q0 + alpha * J^T e, the update the solver should make."""
    q0 = np.asarray(q0, dtype=np.float64)
    error = np.asarray(target, dtype=np.float64) - chain.end_effector(q0)
    return q0 + alpha * jacobian(chain, q0).T @ error


def random_poses(chain: Chain2D, n: int, seed: int) -> npt.NDArray[np.float64]:
    rng = np.random.default_rng(seed)
    return rng.uniform(-np.pi, np.pi, size=(n, chain.n_joints))


@pytest.fixture
def chain() -> Chain2D:
    return Chain2D([1.0, 0.8, 0.5, 0.3])


# --- The Jacobian itself -----------------------------------------


def test_jacobian_matches_finite_differences_at_random_configs(chain: Chain2D) -> None:
    for q in random_poses(chain, 10, seed=11):
        np.testing.assert_allclose(
            jacobian(chain, q), finite_difference_jacobian(chain, q), atol=FD_ATOL
        )


def test_jacobian_matches_finite_differences_with_offset_base() -> None:
    chain = Chain2D([1.0, 0.7], base=[3.0, -2.0])
    for q in random_poses(chain, 10, seed=12):
        np.testing.assert_allclose(
            jacobian(chain, q), finite_difference_jacobian(chain, q), atol=FD_ATOL
        )


# --- Applying the transpose --------------------------------------


def test_jt_times_task_vector_has_one_entry_per_joint(chain: Chain2D) -> None:
    jac = jacobian(chain, [0.1, 0.2, 0.3, 0.4])
    assert (jac.T @ np.array([0.5, -0.2])).shape == (chain.n_joints,)


def test_jt_matches_explicit_transpose(chain: Chain2D) -> None:
    rng = np.random.default_rng(13)
    for q in random_poses(chain, 10, seed=14):
        jac = jacobian(chain, q)
        v = rng.normal(size=2)
        np.testing.assert_allclose(jac.T @ v, explicit_transpose_times(jac, v), atol=ALGEBRA_ATOL)


def test_jt_is_power_consistent(chain: Chain2D) -> None:
    # Work done by a task-space force f on joint velocity qdot must be
    # the same whether you map f to torques first or push qdot forward.
    rng = np.random.default_rng(15)
    for q in random_poses(chain, 10, seed=16):
        jac = jacobian(chain, q)
        f = rng.normal(size=2)
        qdot = rng.normal(size=chain.n_joints)
        np.testing.assert_allclose(
            np.dot(jac.T @ f, qdot), np.dot(f, jac @ qdot), atol=ALGEBRA_ATOL
        )


# --- The solver's update rule and step size ----------------------


@pytest.mark.parametrize("lengths", [[1.0, 0.6], [1.0, 0.8, 0.5]])
def test_first_step_is_alpha_times_jt_error(lengths: list[float]) -> None:
    # The 2-link case runs without a shape error, so it isolates a wrong
    # formula (for example J @ e instead of J.T @ e) from a shape bug.
    chain = Chain2D(lengths)
    q0 = np.array([0.2, -0.3, 0.4])[: chain.n_joints]
    target = [0.5, 1.2]

    result = solve_jacobian_transpose(chain, target, q0=q0, alpha=0.1, max_iters=1)

    np.testing.assert_allclose(
        result.history[0],
        expected_first_step(chain, target, q0, alpha=0.1),
        atol=ALGEBRA_ATOL,
    )


def test_first_step_scales_linearly_with_alpha(chain: Chain2D) -> None:
    q0 = np.array([0.2, -0.3, 0.4, 0.1])
    target = [0.5, 1.2]

    small = solve_jacobian_transpose(chain, target, q0=q0, alpha=0.05, max_iters=1)
    large = solve_jacobian_transpose(chain, target, q0=q0, alpha=0.10, max_iters=1)

    # Doubling the gain doubles the step.
    np.testing.assert_allclose(
        large.history[0] - q0, 2 * (small.history[0] - q0), atol=ALGEBRA_ATOL
    )


def test_small_gain_reduces_error_every_step(chain: Chain2D) -> None:
    # Gradient descent on 0.5 |e|^2 with a small enough step cannot
    # raise the error. alpha = 0.01 is small enough for these arms.
    alpha = 0.01
    for q0, target_q in zip(
        random_poses(chain, 3, seed=17), random_poses(chain, 3, seed=18), strict=True
    ):
        target = chain.end_effector(target_q)
        result = solve_jacobian_transpose(
            chain, target, q0=q0, alpha=alpha, tol=1e-12, max_iters=50
        )
        errors = [np.linalg.norm(chain.end_effector(q) - target) for q in [q0, *result.history]]
        assert np.all(np.diff(errors) <= 1e-12), f"error rose: {errors}"


# --- Convergence and edge cases ----------------------------------


def test_converges_to_random_reachable_targets(chain: Chain2D) -> None:
    starts = random_poses(chain, 25, seed=5)
    targets = [chain.end_effector(q) for q in random_poses(chain, 25, seed=1)]

    for q0, target in zip(starts, targets, strict=True):
        result = solve_jacobian_transpose(chain, target, q0=q0, tol=TOL, max_iters=MAX_ITERS)

        assert result.is_converged, f"missed {target} from {q0}: error {result.error:.2e}"
        assert result.iterations <= MAX_ITERS
        assert np.linalg.norm(chain.end_effector(result.q) - target) < TOL


def test_target_already_reached_takes_zero_iterations(chain: Chain2D) -> None:
    q0 = np.array([0.3, -0.4, 0.5, 0.2])

    result = solve_jacobian_transpose(chain, chain.end_effector(q0), q0=q0, tol=TOL)

    assert result.is_converged
    assert result.iterations == 0
    assert result.history == []
    np.testing.assert_array_equal(result.q, q0)


def test_unreachable_target_stays_finite_and_within_physical_bounds(chain: Chain2D) -> None:
    # The target is 3 * reach from the base, so no pose can reach it.
    # This solver can settle in a local minimum, so the test checks
    # physical bounds rather than the best possible error.
    target = 3 * chain.reach * np.array([0.6, 0.8])
    distance_to_base = float(np.linalg.norm(target - chain.base))
    lowest_possible = distance_to_base - chain.reach
    highest_possible = distance_to_base + chain.reach

    for q0 in [np.full(chain.n_joints, 0.1), *random_poses(chain, 5, seed=6)]:
        result = solve_jacobian_transpose(chain, target, q0=q0, tol=TOL, max_iters=MAX_ITERS)

        assert not result.is_converged
        assert np.isfinite(result.q).all()
        assert np.isfinite(result.error)
        assert lowest_possible - 1e-9 <= result.error <= highest_possible + 1e-9


def test_singular_straight_arm_with_target_on_its_axis_stalls_without_nan(
    chain: Chain2D,
) -> None:
    # Straight arm, target on the arm's axis. The error lies along the
    # arm, so J^T e is exactly zero and the solver cannot move. This is
    # a known limit of the method: it must report failure honestly.
    q0 = np.zeros(chain.n_joints)
    target = [1.0, 0.0]

    result = solve_jacobian_transpose(chain, target, q0=q0, tol=TOL, max_iters=MAX_ITERS)

    assert not result.is_converged
    np.testing.assert_array_equal(result.q, q0)
    assert np.isfinite(result.error)
    assert result.error == pytest.approx(abs(chain.reach - target[0]), abs=1e-9)
