"""Closed-form IK for a two-link arm. Exact, so it is the ground truth for the rest."""
from __future__ import annotations

from collections.abc import Sequence
from typing import Final

import numpy as np

from ik.chain import Chain2D

# Rounding can push cos(q2) just past 1 for a target exactly at full reach.
# A margin of 1e-12 is about 4500 times machine epsilon for doubles.
REACH_MARGIN: Final = 1e-12


def solve_two_link(chain: Chain2D, target: Sequence[float]) -> list[np.ndarray]:
    """Return every joint solution that puts the end effector on target.

    The list is empty when the target is out of reach, otherwise
    [elbow_down, elbow_up]. The two are equal when the arm is straight.

    Raises
    ------
    ValueError
        If the chain does not have exactly two links.
    """
    if chain.n_joints != 2:
        raise ValueError(f"expected a chain with 2 links, got {chain.n_joints}")
    l1, l2 = chain.lengths
    x, y = np.asarray(target, dtype=np.float64) - chain.base
    # Law of cosines on the triangle base, elbow, target.
    cos_q2 = (x * x + y * y - l1 * l1 - l2 * l2) / (2 * l1 * l2)
    if abs(cos_q2) > 1 + REACH_MARGIN:
        return []
    cos_q2 = np.clip(cos_q2, -1.0, 1.0)
    elbow_angle = np.arccos(cos_q2)
    solutions = []
    for q2 in (elbow_angle, -elbow_angle):
        # The bent elbow turns the arm away from the target direction by
        # atan2(l2 sin q2, l1 + l2 cos q2), so q1 subtracts that angle.
        q1 = np.arctan2(y, x) - np.arctan2(l2 * np.sin(q2), l1 + l2 * np.cos(q2))
        solutions.append(np.array([q1, q2]))
    return solutions
