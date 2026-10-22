"""Cyclic Coordinate Descent for planar arms."""

import numpy as np
import numpy.typing as npt

from ik.array_types import FloatArray
from ik.chain import Chain2D
from ik.ik_result import (
    DEFAULT_MAX_ITERATIONS,
    DEFAULT_TOLERANCE,
    IKResult,
    finish,
    initial_q,
)


def _signed_angle(a: FloatArray, b: FloatArray) -> float:
    """Return the angle in (-pi, pi] that rotates a onto b."""
    # np.cross on 2-element vectors is deprecated in numpy 2.
    cross = a[0] * b[1] - a[1] * b[0]
    return float(np.arctan2(cross, np.dot(a, b)))


# Keep this signature consistent with the other solvers.
def solve_ccd(  # noqa: PLR0917
    chain: Chain2D,
    target: npt.ArrayLike,
    q0: npt.ArrayLike | None = None,
    tol: float = DEFAULT_TOLERANCE,
    max_iters: int = DEFAULT_MAX_ITERATIONS,
) -> IKResult:
    """Find joint angles that move the end effector toward ``target``.

    Parameters
    ----------
    chain
        Arm geometry to solve.
    target
        Target point with shape ``(2,)``.
    q0
        Optional initial joint angles with shape ``(chain.n_joints,)``.
    tol
        Tip-to-target distance that counts as convergence.
    max_iters
        Maximum number of complete joint sweeps.

    Raises
    ------
    ValueError
        If the target, starting angles, tolerance, or iteration limit is
        invalid.
    """
    target_array = np.asarray(target, dtype=np.float64)
    if target_array.shape != (2,) or not np.isfinite(target_array).all():
        raise ValueError(f"expected a finite target with shape (2,), got {target_array!r}")
    if not np.isfinite(tol) or tol <= 0:
        raise ValueError(f"expected a finite positive tolerance, got {tol!r}")
    if max_iters < 0:
        raise ValueError(f"expected a non-negative integer max_iters, got {max_iters!r}")

    q = initial_q(chain, q0)
    if q.shape != chain.lengths.shape or not np.isfinite(q).all():
        raise ValueError(
            f"expected {chain.n_joints} finite starting angles, got {q!r}"
        )

    history: list[FloatArray] = []
    for iteration in range(max_iters + 1):
        error = float(np.linalg.norm(chain.end_effector(q) - target_array))
        if error < tol:
            return finish(chain, q, target_array, iteration, tol, history)
        if iteration == max_iters:
            break

        for joint_index in reversed(range(chain.n_joints)):
            points = chain.joint_positions(q)
            to_end = points[-1] - points[joint_index]
            to_target = target_array - points[joint_index]
            q[joint_index] += _signed_angle(to_end, to_target)
        history.append(q.copy())

    return finish(chain, q, target_array, max_iters, tol, history)
