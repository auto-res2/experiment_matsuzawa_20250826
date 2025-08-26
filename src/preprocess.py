"""src/preprocess.py
For the toy reference implementation, *pre-processing* simply generates random
tensors.  The module exists so that researchers can later drop-in real data
loading pipelines (e.g. ImageNet, CAMELYON WSI) without changing the public
interface.
"""
from __future__ import annotations

from typing import Generator, Tuple

import torch

# ---------------------------------------------------------------------------
# 1.  Synthetic dataloader ---------------------------------------------------
# ---------------------------------------------------------------------------

def get_dummy_dataloader(batch_size: int = 16, device: str = "cpu") -> Generator[Tuple[torch.Tensor, torch.Tensor], None, None]:
    """Infinite generator that yields random (images, labels)."""

    while True:
        x = torch.randn(batch_size, 64, 32, 32, device=device)
        y = torch.randint(0, 10, (batch_size,), device=device)
        yield x, y
