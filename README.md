# Inverse kinematics solver

I’m learning robotics by building a solver that finds the joint angles needed to put an arm’s tip on a target.

## What works

- Forward kinematics for a planar arm with any number of links: joint angles in, joint positions out.
- An analytic inverse kinematics solver for a two-link arm, returning both elbow positions when the target is reachable and an empty list otherwise.
- Tests for the geometry, reach limits, base offsets, and result helpers.

This is still a 2D geometry project. It doesn’t handle obstacles, joint limits, motor dynamics, or reinforcement learning yet.

## Try it

Use Python 3.12 or later with [uv](https://docs.astral.sh/uv/). From a terminal:

```sh
git clone https://github.com/sxhnd/IK_Solver.git
cd IK_Solver
uv sync
```

From the repository directory, start `uv run python` and run:

```python
from ik.analytic import solve_two_link
from ik.chain import Chain2D

arm = Chain2D([1.0, 0.6])
target = [0.9, 0.7]

for angles in solve_two_link(arm, target):
    print("angles (radians):", angles.round(4))
    print("tip:", arm.end_effector(angles).round(4))
```

You get two angle pairs, each placing the tip at `[0.9, 0.7]`. Angles are in radians and relative to the previous link; the first angle is measured from the positive x-axis. Use the same units for link lengths and target coordinates.

The interactive arm on [my site](https://sxhnd.github.io/) uses a JavaScript port of this two-link solver.

## Run the tests

From the repository directory, run:

```sh
uv run pytest -q
```

## Code

- [`ik/chain.py`](ik/chain.py): arm geometry and forward kinematics.
- [`ik/analytic.py`](ik/analytic.py): the two-link inverse kinematics solver.
- [`ik/ccd.py`](ik/ccd.py): the cyclic coordinate descent solver for arms with any number of links.
- [`ik/ik_result.py`](ik/ik_result.py): result storage and helpers for future iterative solvers.
