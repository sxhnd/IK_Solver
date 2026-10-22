from dataclasses import FrozenInstanceError

import numpy as np
import pytest
from ik.chain import Chain2D
from ik.ik_result import finish, initial_q

ERROR_TOLERANCE = 1e-12


def test_default_start_is_bent_not_straight() -> None:
    q = initial_q(Chain2D([1.0, 1.0, 1.0]), None)

    assert q.shape == (3,)
    # A straight arm is singular.
    assert np.all(q != 0)


def test_initial_q_copies_the_callers_array() -> None:
    q0 = np.array([0.1, 0.2])

    q = initial_q(Chain2D([1.0, 1.0]), q0)
    q[0] = 99.0

    np.testing.assert_array_equal(q0, [0.1, 0.2])


def test_finish_measures_error_with_forward_kinematics() -> None:
    chain = Chain2D([1.0, 1.0])
    straight = np.array([0.0, 0.0])  # Tip at (2, 0).

    miss = finish(chain, straight, np.array([2.0, 0.5]), iterations=3, tol=1e-4, history=[])
    hit = finish(chain, straight, np.array([2.0, 0.0]), iterations=3, tol=1e-4, history=[])

    assert miss.error == pytest.approx(0.5, abs=ERROR_TOLERANCE)
    assert not miss.is_converged
    assert miss.iterations == 3
    assert hit.is_converged


def test_ik_result_fields_cannot_be_reassigned() -> None:
    result = finish(Chain2D([1.0]), np.array([0.0]), np.array([1.0, 0.0]), 0, 1e-4, [])

    with pytest.raises(FrozenInstanceError):
        # mypy rejects assignment to a frozen field; the test needs
        # check that the runtime rejects it too.
        result.error = 5.0  # type: ignore[misc]
