"""
evaluate.py – model evaluation + visualisations (PDF)  according to the rules.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Any

import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

from .train import ActivationTracer

# ---------------------------------------------------------------------------


def accuracy(logits: torch.Tensor, y: torch.Tensor) -> float:
    return (logits.argmax(1) == y).float().mean().item()


# ---------------------------------------------------------------------------
# Public API – evaluate()  (called by src/main.py)
# ---------------------------------------------------------------------------


def evaluate(model: torch.nn.Module, loader: DataLoader, cfg: Dict[str, Any]):
    device = next(model.parameters()).device
    tracer = ActivationTracer(model)
    model.eval()
    accs = []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits, _ = tracer(x)
            accs.append(accuracy(logits, y))
    tracer.remove()

    m = float(np.mean(accs))
    print(f"Evaluation accuracy: {m*100:.2f} %")

    # ------------- plot & save --------------------------------------------------
    img_dir = Path(cfg["image_dir"])
    img_dir.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(3.5, 3))
    plt.hist(accs, bins=20, color="#4C72B0")
    plt.xlabel("Batch accuracy")
    plt.ylabel("Count")
    plt.title("Accuracy distribution")
    plt.tight_layout()
    outfile = img_dir / "accuracy_hist.pdf"
    plt.savefig(outfile)
    print(f"Saved → {outfile.relative_to(Path.cwd())}")
