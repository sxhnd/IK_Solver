"""A planar serial arm: link lengths and forward kinematics."""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np


class Chain2D:
    """N revolute joints in a plane.

    Joint angles are relative: q[i] is measured from the direction of
    link i - 1, or from the +x axis for q[0]. Joint i sits at the start
    of link i.
    """

    def __init__(
        self, lengths: Sequence[float], base: Sequence[float] = (0.0, 0.0)
    ) -> None:
        self.lengths = np.array(lengths, dtype=np.float64)
        if self.lengths.ndim != 1 or self.lengths.size == 0 or (self.lengths <= 0).any():
            raise ValueError(
                f"expected a non-empty 1D sequence of positive lengths, got {lengths!r}"
            )
        self.base = np.array(base, dtype=np.float64)

    @property
    def n_joints(self) -> int:
        return int(self.lengths.size)

    @property
    def reach(self) -> float:
        """Farthest distance the end effector can get from the base."""
        return float(self.lengths.sum())

    def joint_positions(self, q: Sequence[float]) -> np.ndarray:
        """Return the base, every joint, and the end effector, shape (n + 1, 2)."""
        # Angles are relative, so the absolute direction of link i is the
        # running sum of q[0] .. q[i].
        headings = np.cumsum(np.asarray(q, dtype=np.float64))
        steps = np.column_stack(
            (self.lengths * np.cos(headings), self.lengths * np.sin(headings))
        )
        return np.vstack((self.base, self.base + np.cumsum(steps, axis=0)))

    def end_effector(self, q: Sequence[float]) -> np.ndarray:
        """Return the tip position, shape (2,)."""
        return self.joint_positions(q)[-1]


def wrap_angle(angles: Sequence[float] | np.ndarray) -> np.ndarray:
    """Map angles in radians onto the half-open interval [-pi, pi)."""
    return np.mod(np.asarray(angles) + np.pi, 2 * np.pi) - np.pi
