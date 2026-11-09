"""Tests for the exact and numeric Jacobians."""

import numpy as np
import pytest
from ik.chain import Chain2D
from ik.jacobian import jacobian, jacobian_numeric


def test_exact_jacobian_matches_finite_differences() -> None:
    chain = Chain2D([1.0, 0.8, 0.5, 0.3])
    rng = np.random.default_rng(3)
    for q in rng.uniform(-np.pi, np.pi, size=(20, 4)):
        np.testing.assert_allclose(jacobian(chain, q), jacobian_numeric(chain, q), atol=1e-6)


def test_matches_finite_differences_with_offset_base() -> None:
    chain = Chain2D([1.0, 0.7], base=[3.0, -2.0])
    q = [0.4, -1.1]
    np.testing.assert_allclose(jacobian(chain, q), jacobian_numeric(chain, q), atol=1e-6)


def test_shape_is_2_by_n() -> None:
    chain = Chain2D([1.0, 1.0, 1.0])
    assert jacobian(chain, [0.1, 0.2, 0.3]).shape == (2, 3)
    assert jacobian_numeric(chain, [0.1, 0.2, 0.3]).shape == (2, 3)


def test_straight_arm_is_singular() -> None:
    # Every joint can only push the tip sideways, never along the arm.
    jac = jacobian(Chain2D([1.0, 0.8, 0.5]), [0.0, 0.0, 0.0])
    np.testing.assert_allclose(jac[0], 0.0, atol=1e-12)
    assert np.linalg.matrix_rank(jac) == 1


def test_bent_arm_is_not_singular() -> None:
    assert np.linalg.matrix_rank(jacobian(Chain2D([1.0, 0.8, 0.5]), [0.1, 0.1, 0.1])) == 2


def test_rejects_wrong_number_of_angles() -> None:
    chain = Chain2D([1.0, 1.0])
    with pytest.raises(ValueError, match="joint angles"):
        jacobian(chain, [0.0])
    with pytest.raises(ValueError, match="joint angles"):
        jacobian_numeric(chain, [0.0, 0.0, 0.0])
