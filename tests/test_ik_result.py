import numpy as np
import pytest

from ik.chain import Chain2D
from ik.ik_result import finish, initial_q


def test_default_start_is_bent_not_straight():
    q = initial_q(Chain2D([1.0, 1.0, 1.0]), None)
    assert q.shape == (3,)
    assert np.all(q != 0)  # the straight pose is a singularity (see Task 7)


def test_initial_q_copies_the_callers_array():
    q0 = np.array([0.1, 0.2])
    q = initial_q(Chain2D([1.0, 1.0]), q0)
    q[0] = 99.0
    assert q0[0] == 0.1


def test_finish_measures_error_with_forward_kinematics():
    chain = Chain2D([1.0, 1.0])
    straight = np.array([0.0, 0.0])  # tip at (2, 0)
    miss = finish(chain, straight, np.array([2.0, 0.5]), iterations=3, tol=1e-4, history=[])
    assert miss.error == pytest.approx(0.5)
    assert not miss.is_converged
    assert miss.iterations == 3
    hit = finish(chain, straight, np.array([2.0, 0.0]), iterations=3, tol=1e-4, history=[])
    assert hit.is_converged
