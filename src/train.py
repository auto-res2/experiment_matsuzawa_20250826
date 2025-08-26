"""src/train.py
Training utilities and model definitions for the RevSiM toy reference
implementation.  All heavy-weight kernels / datasets are stubbed so that the
whole training loop finishes in seconds on CPU but the public API mirrors what
would be expected for the real model.

Important
---------
• Only relative imports are used inside the ``src`` package in compliance with
  the task rules.
• All images / figures are saved via ``src.utils.save_fig`` which redirects
  them into ``.research/iteration1/images`` as required.
"""
from __future__ import annotations

import gc
import os
import random
import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# 1.  Directory helpers ------------------------------------------------------
# ---------------------------------------------------------------------------
RESULTS_DIR = os.path.join(os.getcwd(), ".research", "iteration1", "images")
os.makedirs(RESULTS_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# 2.  Building blocks --------------------------------------------------------
# ---------------------------------------------------------------------------
class RevBlock(nn.Module):
    """Reversible block:  y1 = x1 + F1(x2);  y2 = x2 + F2(y1).
    Stubs of F1/F2 are cheap depth-wise and point-wise convolutions so
    compute/memory characteristics roughly match the real operator while being
    extremely fast to run.
    """

    def __init__(self, channels: int):
        super().__init__()
        half = channels // 2
        self.f1 = nn.Sequential(
            nn.Conv2d(half, half, 3, padding=1, groups=half),
            nn.GELU(),
        )
        self.f2 = nn.Sequential(
            nn.Conv2d(half, half, 1),
            nn.GELU(),
            nn.Conv2d(half, half, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1, x2 = torch.chunk(x, 2, dim=1)
        y1 = x1 + self.f1(x2)
        y2 = x2 + self.f2(y1)
        return torch.cat([y1, y2], 1)


class SliceWiseScan(nn.Module):
    """Tiny analogue of the slice-wise scan operator.  The real kernel performs
    a selective-state-space scan; here we simply project each slice with a
    linear layer so that runtime and memory can still be measured.
    """

    def __init__(self, channels: int, slice_len: int | None):
        super().__init__()
        self.proj = nn.Linear(channels, channels)
        self.slice_len = slice_len

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # [B,C,H,W]
        b, c, h, w = x.shape
        x_flat = x.permute(0, 2, 3, 1).reshape(b, h * w, c)  # [B,L,C]
        l = x_flat.shape[1]
        step = self.slice_len or l  # default → full length
        out = []
        for i in range(0, l, step):
            out.append(self.proj(x_flat[:, i : i + step]))
        y = torch.cat(out, 1)
        return y.reshape(b, h, w, c).permute(0, 3, 1, 2).contiguous()


class BaselineBlock(nn.Module):
    """A non-reversible residual block using the same primitive ops so that any
    compute-time difference is purely due to memory techniques, not accuracy.
    """

    def __init__(self, channels: int):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, 1),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.body(x)


class ToyModel(nn.Module):
    """Small CNN that alternates scan + (rev or non-rev) blocks.

    Parameters
    ----------
    channels   : feature channels (default 64)
    depth      : number of *pairs* (scan + processing block) – default 4
    rev        : if ``True`` use ``RevBlock`` else ``BaselineBlock``
    slice_len  : slice length for the scan operator; ``None`` ⇢ full length
    """

    def __init__(
        self,
        channels: int = 64,
        depth: int = 4,
        rev: bool = False,
        slice_len: int | None = None,
    ) -> None:
        super().__init__()
        layers: List[nn.Module] = []
        for _ in range(depth):
            layers.append(SliceWiseScan(channels, slice_len))
            layers.append(RevBlock(channels) if rev else BaselineBlock(channels))
        self.net = nn.Sequential(*layers)
        self.cls_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(channels, 10)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # [B,3?,H,W]
        return self.cls_head(self.net(x))


# ---------------------------------------------------------------------------
# 3.  Peak-memory monitor ----------------------------------------------------
# ---------------------------------------------------------------------------
@dataclass
class PeakStats:
    peak: int   # bytes
    elapsed: float  # seconds


@contextmanager
def peak_monitor(device: str = "cpu"):
    """Context manager that measures wall time and maximum GPU/CPU memory."""

    start = time.perf_counter()
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    try:
        yield
    finally:
        if device == "cuda":
            peak = torch.cuda.max_memory_allocated()
        else:
            import psutil  # local import to avoid mandatory dep on all devices

            process = psutil.Process(os.getpid())
            peak = process.memory_info().rss
        peak_monitor.stats = PeakStats(peak, time.perf_counter() - start)


# ---------------------------------------------------------------------------
# 4.  Public training function ----------------------------------------------
# ---------------------------------------------------------------------------

def train_one_epoch(
    model: nn.Module,
    optimiser: torch.optim.Optimizer,
    batch_size: int = 16,
    device: str = "cpu",
) -> float:
    """Runs a single epoch over *synthetic* data and returns the loss."""

    model.train()
    criterion = nn.CrossEntropyLoss()
    for _ in range(3):  # tiny epoch: three synthetic batches
        x = torch.randn(batch_size, 64, 32, 32, device=device)
        y = torch.randint(0, 10, (batch_size,), device=device)
        optimiser.zero_grad(set_to_none=True)
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimiser.step()
    return float(loss.item())


def free_memory(*tensors):
    """Utility that deletes tensors and runs explicit GC + CUDA cache clear."""

    for t in tensors:
        del t
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
