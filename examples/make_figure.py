"""Draw docs/ccd_reach.png: CCD pulling a four-link arm onto a target.

Needs matplotlib (in the dev dependency group). Run from the repo root:
    uv run python examples/make_figure.py
"""

from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from ik.ccd import solve_ccd
from ik.chain import Chain2D

OUTPUT = Path(__file__).resolve().parent.parent / "docs" / "ccd_reach.png"


def main() -> None:
    """Solve one reach problem and save the figure."""
    arm = Chain2D([1.0, 0.8, 0.5, 0.3])
    target = np.array([1.2, 1.6])
    # Start from a bent pose that is far from the target.
    q0 = np.array([-0.9, 0.6, 0.6, 0.4])
    result = solve_ccd(arm, target, q0=q0)
    poses = [q0, *result.history]
    errors = [float(np.linalg.norm(arm.end_effector(q) - target)) for q in poses]

    fig, (ax_arm, ax_err) = plt.subplots(1, 2, figsize=(10, 4.4), dpi=150)

    for step, q in enumerate(poses):
        pts = arm.joint_positions(q)
        last = step == len(poses) - 1
        ax_arm.plot(
            pts[:, 0],
            pts[:, 1],
            "-o",
            color="#1f6feb" if last else "#9aa4b2",
            alpha=1.0 if last else 0.25 + 0.5 * step / len(poses),
            lw=3 if last else 1.5,
            ms=5,
        )
    ax_arm.plot(*target, "*", color="#d1242f", ms=16, label="target", zorder=5)
    ax_arm.set_aspect("equal")
    ax_arm.set_title("Each sweep (grey) moves the tip toward the target; final pose in blue")
    ax_arm.title.set_fontsize(9)
    ax_arm.legend(loc="lower left")
    ax_arm.grid(alpha=0.3)

    ax_err.semilogy(range(len(errors)), errors, "-o", color="#1f6feb")
    ax_err.axhline(1e-4, color="#d1242f", ls="--", lw=1, label="tolerance (1e-4)")
    ax_err.set_xlabel("sweep")
    ax_err.set_ylabel("tip-to-target distance")
    ax_err.set_title(f"Converged in {result.iterations} sweeps", fontsize=9)
    ax_err.legend()
    ax_err.grid(alpha=0.3)

    fig.tight_layout()
    OUTPUT.parent.mkdir(exist_ok=True)
    fig.savefig(OUTPUT)


if __name__ == "__main__":
    main()
