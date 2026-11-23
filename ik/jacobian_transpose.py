
"""Jacobian transpose for planar arms."""
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
from ik.jacobian import jacobian

# fixed step size
DEFAULT_ALPHA = 0.1


def solve_jacobian_transpose(
    chain: Chain2D,
    target: npt.ArrayLike,
    q0: npt.ArrayLike | None = None, 
    tol: float = DEFAULT_TOLERANCE,
    max_iters: int = DEFAULT_MAX_ITERATIONS,
    alpha: float = DEFAULT_ALPHA,
) -> IKResult:
    target_array = np.asarray(target, dtype=np.float64)
    
    if target_array.shape != (2,) or not np.isfinite(target_array).all():
        raise ValueError(f"expected a finite target with shape (2,), got {target_array!r}")
    if not np.isfinite(tol) or tol <= 0:
        raise ValueError(f"expected a finite positive tolerance, got {tol!r}")
    if max_iters < 0:
        raise ValueError(f"expected a non-negative integer max_iters, got {max_iters!r}")
    if not np.isfinite(alpha) or alpha <= 0:
        raise ValueError(f"expected a finite positive alpha, got {alpha!r}")

    q = initial_q(chain, q0)
    
    if q.shape != chain.lengths.shape or not np.isfinite(q).all():
        raise ValueError(f"expected {chain.n_joints} finite starting angles, got {q!r}")
    
    history: list[FloatArray] = []
    
    for iteration in range(max_iters + 1):
        # vector from tip to target
        e = target_array - chain.end_effector(q)
        if float(np.linalg.norm(e)) < tol:
            return finish(chain, q, target_array, iteration, tol, history)
        if iteration == max_iters:
            break
        
        J = jacobian(chain,q)
        q = q + alpha * J.T @ e
        history.append(q.copy())
        
    return finish(chain, q, target_array, max_iters, tol, history)
        
        
    
