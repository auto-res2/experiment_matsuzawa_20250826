from __future__ import annotations
"""
evaluate.py – performs three tiny pseudo-experiments and stores the
results as PDF files under
    ``.research/iteration3/images``

Real research code would do *much* more, but for demonstration purposes
we simply plot random data.
"""
from pathlib import Path
import random

import matplotlib.pyplot as plt
import numpy as np

__all__ = ["evaluate"]

def _single_experiment(exp_id: int, img_dir: Path) -> None:
    """Run a dummy experiment and save a scatter plot."""
    rng = np.random.default_rng(seed=exp_id)
    x = rng.uniform(0, 1, 32)
    y = rng.uniform(0, 1, 32)

    plt.figure(figsize=(3, 3))
    plt.scatter(x, y, c=y, cmap="viridis", edgecolor="black")
    plt.title(f"Experiment {exp_id}")
    plt.tight_layout()

    out_file = img_dir / f"experiment_{exp_id}.pdf"
    plt.savefig(out_file)
    plt.close()
    print(f"  · figure saved to {out_file.relative_to(Path.cwd())}")

def evaluate(cfg: dict, /) -> None:
    """Run three toy experiments and generate figures.

    The *cfg* argument is currently unused but kept for future extensions.
    """
    img_dir = Path(".research") / "iteration3" / "images"
    img_dir.mkdir(parents=True, exist_ok=True)

    print("⚑  Starting evaluation – results will be stored in:")
    print(f"   {img_dir.relative_to(Path.cwd())}\n")

    for i in range(1, 4):
        _single_experiment(i, img_dir)

    print("\n✓  Evaluation completed.")
