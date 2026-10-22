"""A planar serial arm: link lengths and forward kinematics."""

import numpy as np
import numpy.typing as npt

from ik.array_types import FloatArray


class Chain2D:
    """N revolute joints in a plane.

    Joint angles are relative: q[i] is measured from the direction of
    link i - 1, or from the +x axis for q[0]. Joint i sits at the start
    of link i.

    Raises
    ------
    ValueError
        If lengths is not a non-empty 1D sequence of finite positive
        numbers, or base does not have two coordinates.
    """

    def __init__(self, lengths: npt.ArrayLike, base: npt.ArrayLike = (0.0, 0.0)) -> None:
        self.lengths: FloatArray = np.array(lengths, dtype=np.float64)
        is_valid_lengths = (
            self.lengths.ndim == 1
            and self.lengths.size > 0
            and np.isfinite(self.lengths).all()
            and (self.lengths > 0).all()
        )
        if not is_valid_lengths:
            raise ValueError(
                f"expected a non-empty 1D sequence of finite positive lengths, got {lengths!r}"
            )
        self.base: FloatArray = np.array(base, dtype=np.float64)
        if self.base.shape != (2,):
            raise ValueError(f"expected base of shape (2,), got {self.base.shape}")
        # Read-only, so no caller can bypass the checks above by
        # assigning into the arrays later.
        self.lengths.flags.writeable = False
        self.base.flags.writeable = False

    @property
    def n_joints(self) -> int:
        """Number of joints, equal to the number of links."""
        return int(self.lengths.size)

    @property
    def reach(self) -> float:
        """Farthest distance the end effector can get from the base."""
        return float(self.lengths.sum())

    def joint_positions(self, q: npt.ArrayLike) -> FloatArray:
        """Return the base, every joint, and the end effector.

        Parameters
        ----------
        q : shape (n,)
            Relative joint angles in radians.

        Returns
        -------
        positions : shape (n + 1, 2)
            Row 0 is the base and the last row is the end effector.

        Raises
        ------
        ValueError
            If q does not have one angle per joint.
        """
        angles = np.asarray(q, dtype=np.float64)
        if angles.shape != self.lengths.shape:
            raise ValueError(f"expected {self.n_joints} joint angles, got shape {angles.shape}")
        # Angles are relative, so the absolute direction of link i is
        # the running sum of q[0] .. q[i].
        headings = np.cumsum(angles)
        steps = np.column_stack((self.lengths * np.cos(headings), self.lengths * np.sin(headings)))
        positions: FloatArray = np.vstack((self.base, self.base + np.cumsum(steps, axis=0)))
        return positions

    def end_effector(self, q: npt.ArrayLike) -> FloatArray:
        """Return the tip position, shape (2,)."""
        tip: FloatArray = self.joint_positions(q)[-1]
        return tip


def wrap_angle(angles: npt.ArrayLike) -> FloatArray:
    """Map angles in radians onto the half-open interval [-pi, pi)."""
    wrapped: FloatArray = np.mod(np.asarray(angles) + np.pi, 2 * np.pi) - np.pi
    return wrapped
