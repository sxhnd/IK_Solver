"""The Jacobian of a planar arm: how each joint moves the tip."""

import numpy as np
import numpy.typing as npt

from ik.array_types import FloatArray
from ik.chain import Chain2D

# Step size for the central difference. Smaller loses precision to
# rounding; larger loses it to curvature.
_DEFAULT_STEP = 1e-6


def jacobian_numeric(chain: Chain2D, q: npt.ArrayLike, h: float = _DEFAULT_STEP) -> FloatArray:
    """Return the 2 x n Jacobian by nudging each joint by +-h.

    Column i is (tip(q + h e_i) - tip(q - h e_i)) / 2h: the Calc 1
    derivative, with a small h in place of a limit.

    Raises
    ------
    ValueError
        If q does not have one angle per joint.
    """
    angles = np.asarray(q, dtype=np.float64)
    if angles.shape != chain.lengths.shape:
        raise ValueError(f"expected {chain.n_joints} joint angles, got shape {angles.shape}")
    columns = np.zeros((2, chain.n_joints))
    for i in range(chain.n_joints):
        nudge = np.zeros_like(angles)
        nudge[i] = h
        columns[:, i] = (
            chain.end_effector(angles + nudge) - chain.end_effector(angles - nudge)
        ) / (2 * h)
    return columns


def jacobian(chain: Chain2D, q: npt.ArrayLike) -> FloatArray:
    """Return the exact 2 x n Jacobian.

    Turning joint i spins the tip around joint i. The tip moves
    perpendicular to r = tip - joint_i, at speed |r| per radian, so
    column i is r rotated by 90 degrees: (-r_y, r_x).

    Raises
    ------
    ValueError
        If q does not have one angle per joint.
    """
    # joint_positions checks the shape of q for us.
    points = chain.joint_positions(q)
    r = points[-1] - points[:-1]
    result: FloatArray = np.vstack((-r[:, 1], r[:, 0]))
    return result
