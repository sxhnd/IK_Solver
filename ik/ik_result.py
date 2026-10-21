"""What every iterative solver returns, and the helpers they share."""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Final

import numpy as np

from ik.chain import Chain2D

# A fully extended arm is singular: the Jacobian loses rank and Jacobian
# solvers can stall there. About 6 degrees of bend at every joint avoids it.
DEFAULT_JOINT_ANGLE_RAD: Final = 0.1


@dataclass(frozen=True)
class IKResult:
    """Outcome of one solver run."""

    q: np.ndarray  # Final joint angles, radians.
    is_converged: bool
    iterations: int
    error: float  # Tip-to-target distance, recomputed by forward kinematics.
    history: list[np.ndarray] = field(default_factory=list)  # q after each iteration.


def initial_q(chain: Chain2D, q0: Sequence[float] | None) -> np.ndarray:
    """Return a private copy of the starting angles, bent slightly if q0 is None."""
    if q0 is None:
        return np.full(chain.n_joints, DEFAULT_JOINT_ANGLE_RAD)
    return np.array(q0, dtype=np.float64)


def finish(
    chain: Chain2D,
    q: np.ndarray,
    target: np.ndarray,
    iterations: int,
    tol: float,
    history: list[np.ndarray],
) -> IKResult:
    """Build the solver's result, measuring the error from the final angles."""
    # The error comes from forward kinematics, never from a solver's own
    # bookkeeping, so a buggy solver cannot report a false success.
    error = float(np.linalg.norm(chain.end_effector(q) - target))
    return IKResult(
        q=q,
        is_converged=error < tol,
        iterations=iterations,
        error=error,
        history=history,
    )
