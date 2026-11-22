"""FABRIK for planar arms: move joints as points, find angles last."""

import numpy as np
import numpy.typing as npt

from ik.array_types import FloatArray
from ik.chain import Chain2D, wrap_angle
from ik.ik_result import (
    DEFAULT_MAX_ITERATIONS,
    DEFAULT_TOLERANCE,
    IKResult,
    finish,
    initial_q,
)

# Below this, two joints count as the same point.
_MIN_NORM = 1e-12


def _unit(v: FloatArray) -> FloatArray:
    """Return v scaled to length 1, or the x axis if v is near zero."""
    norm = float(np.linalg.norm(v))
    if norm < _MIN_NORM:
        return np.array([1.0, 0.0])
    unit: FloatArray = v / norm
    return unit


def _positions_to_angles(points: FloatArray) -> FloatArray:
    """Return the relative joint angles of the chain through points."""
    links = np.diff(points, axis=0)
    absolute = np.arctan2(links[:, 1], links[:, 0])
    # Each relative angle is the turn from the previous link.
    return wrap_angle(np.diff(absolute, prepend=0.0))


def _straighten(chain: Chain2D, points: FloatArray, target: FloatArray) -> None:
    """Lay the whole arm in a line from the base toward target."""
    for i in range(chain.n_joints):
        points[i + 1] = points[i] + chain.lengths[i] * _unit(target - points[i])


def _sweep(chain: Chain2D, points: FloatArray, target: FloatArray) -> None:
    """Run one forward pass and one backward pass over points."""
    lengths = chain.lengths
    # Forward: pin the tip to the target, walk back to the base.
    points[-1] = target
    for i in reversed(range(chain.n_joints)):
        points[i] = points[i + 1] + lengths[i] * _unit(points[i] - points[i + 1])
    # Backward: pin the base in place, walk out to the tip.
    points[0] = chain.base
    for i in range(chain.n_joints):
        points[i + 1] = points[i] + lengths[i] * _unit(points[i + 1] - points[i])


# Keep this signature consistent with the other solvers.
def solve_fabrik(  # noqa: PLR0917
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
        Maximum number of forward-and-backward passes.

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
        raise ValueError(f"expected {chain.n_joints} finite starting angles, got {q!r}")

    points = chain.joint_positions(q)
    is_unreachable = float(np.linalg.norm(target_array - chain.base)) >= chain.reach

    history: list[FloatArray] = []
    for iteration in range(max_iters + 1):
        error = float(np.linalg.norm(chain.end_effector(q) - target_array))
        if error < tol:
            return finish(chain, q, target_array, iteration, tol, history)
        if iteration == max_iters:
            break

        if is_unreachable:
            # The best pose is a straight line toward the target.
            _straighten(chain, points, target_array)
            q = _positions_to_angles(points)
            history.append(q.copy())
            return finish(chain, q, target_array, iteration + 1, tol, history)

        _sweep(chain, points, target_array)
        q = _positions_to_angles(points)
        history.append(q.copy())

    return finish(chain, q, target_array, max_iters, tol, history)
