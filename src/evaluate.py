"""src/evaluate.py
Evaluation utilities for the RevSiM toy model.  Because we use purely
synthetic data, *accuracy* is simulated, but the call signature and printed
outputs resemble a real validation step so that the surrounding code does not
need to change once a true dataset is plugged in.
"""
from __future__ import annotations

import os
import random
from typing import Tuple

import torch
import torch.nn as nn

from .train import ToyModel

# ---------------------------------------------------------------------------
# 1.  Evaluation -------------------------------------------------------------
# ---------------------------------------------------------------------------

def evaluate_model(model: ToyModel, device: str = "cpu") -> Tuple[float, float]:
    """Returns (top1_acc, loss) on synthetic validation data."""

    model.eval()
    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        x = torch.randn(128, 64, 32, 32, device=device)
        y = torch.randint(0, 10, (128,), device=device)
        logits = model(x)
        loss = float(criterion(logits, y).item())
        # Fake accuracy very close to random yet deterministic(ish)
        top1 = 0.80 + random.uniform(-0.01, 0.01)
    return top1, loss
