"""What every iterative solver returns, and the helpers they share."""

from dataclasses import dataclass, field
from typing import Final

import numpy as np
import numpy.typing as npt

from ik.array_types import FloatArray
from ik.chain import Chain2D

# A fully extended arm is singular: the Jacobian loses rank and
# Jacobian solvers can stall there. 0.1 rad is about 6 degrees of bend
# per joint.
DEFAULT_JOINT_ANGLE_RAD: Final = 0.1

# Distance from tip to target that counts as reached, in the units of
# the chain's link lengths: 0.1 mm for an arm measured in metres.
DEFAULT_TOLERANCE: Final = 1e-4

# Starting limit on iterations. The benchmark should replace it with a
# bound measured on the test arms.
DEFAULT_MAX_ITERATIONS: Final = 200


@dataclass(frozen=True)
class IKResult:
    """Outcome of one solver run."""

    q: FloatArray  # Final joint angles, radians.
    is_converged: bool
    iterations: int
    # Tip-to-target distance, recomputed by forward kinematics.
    error: float
    # q after each iteration.
    history: list[FloatArray] = field(default_factory=list)


def initial_q(chain: Chain2D, q0: npt.ArrayLike | None) -> FloatArray:
    """Return a copy of the starting angles, bent if q0 is None."""
    if q0 is None:
        return np.full(chain.n_joints, DEFAULT_JOINT_ANGLE_RAD)
    return np.array(q0, dtype=np.float64)


# Every solver ends through this function, so it takes the whole outcome
# of a run as arguments instead of being split to fit the limits.
def finish(  # noqa: PLR0913, PLR0917
    chain: Chain2D,
    q: FloatArray,
    target: FloatArray,
    iterations: int,
    tol: float,
    history: list[FloatArray],
) -> IKResult:
    """Build the solver's result, measuring the error from q."""
    # The error is measured here, not taken from the solver, so a buggy
    # solver cannot report success it did not reach.
    error = float(np.linalg.norm(chain.end_effector(q) - target))
    return IKResult(
        q=q,
        is_converged=error < tol,
        iterations=iterations,
        error=error,
        history=history,
    )
