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


def wrap_angle(a):
    pass
