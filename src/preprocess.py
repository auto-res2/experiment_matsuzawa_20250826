"""
preprocess.py – data preparation util.  The quick experiment downloads
CIFAR-10, applies basic transforms and returns train & test loaders.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Any, Tuple

import torchvision.transforms as T
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader, Subset
import torch

# ---------------------------------------------------------------------------


def get_dataloaders(cfg: Dict[str, Any]) -> Tuple[DataLoader, DataLoader]:
    root = Path("data")
    root.mkdir(exist_ok=True)

    transform = T.Compose([
        T.ToTensor(),
        T.Normalize((0.491, 0.482, 0.447), (0.247, 0.243, 0.262))])

    train_ds = CIFAR10(root, train=True, download=True, transform=transform)
    test_ds  = CIFAR10(root, train=False, download=True, transform=transform)

    # for the smoke-test keep only 3 classes and 600 samples
    if cfg.get("quick", True):
        idx_train = torch.where(torch.tensor(train_ds.targets) < 3)[0][:600]
        idx_test  = torch.where(torch.tensor(test_ds.targets) < 3)[0][:300]
        train_ds  = Subset(train_ds, idx_train)
        test_ds   = Subset(test_ds, idx_test)
        cfg["num_classes"] = 3
    else:
        cfg["num_classes"] = 10

    train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True, num_workers=2)
    test_loader  = DataLoader(test_ds,  batch_size=cfg["batch_size"], shuffle=False, num_workers=2)
    return train_loader, test_loader
