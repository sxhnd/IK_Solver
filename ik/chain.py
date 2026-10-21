import numpy as np


class Chain2D:
    def __init__(self, lengths, base=(0.0, 0.0)):
        self.lengths = np.array(lengths, dtype=float)
        if self.lengths.size == 0 or (self.lengths <= 0).any():
            raise ValueError("lengths must be a non-empty list of positive numbers")
        self.base = np.array(base, dtype=float)

    @property
    def reach(self):
        return self.lengths.sum()

    def joint_positions(self, q):
        headings = np.cumsum(q)
        steps = np.column_stack((
            self.lengths * np.cos(headings),
            self.lengths * np.sin(headings),
        ))
        return np.vstack((self.base, self.base + np.cumsum(steps, axis=0)))


def wrap_angle(a):
    return np.mod(np.asarray(a) + np.pi, 2 * np.pi) - np.pi
