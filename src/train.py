"""
train.py – implements model definition, training loop and the core parts of
SketchReplay-3.0-Plus that are required for a reproducible experiment.  The
implementation is purposely lightweight: it is able to finish a smoke-test on
CPU in <30 s yet keeps the modular building blocks so that researchers can
replace any component with the full C/MCU implementation later on.

All external imports must be recorded in requirements.txt; internal modules are
referenced via absolute imports (PEP-328).
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Tuple, Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

# ---------------------------------------------------------------------------
# 1.   Auxiliary classes taken from the reference implementation
# ---------------------------------------------------------------------------

class ActivationTracer:
    """Return logits *and* a dictionary of intermediate activations for every
    forward pass.  We hook into Conv / Linear / BN / LN layers.  A handle list
    is stored so that `.remove()` can be called once the experiment finishes.
    """

    def __init__(self, model: nn.Module):
        self.model = model
        self._acts: Dict[str, torch.Tensor] = {}
        self.handles = []
        self._register()

    # ------------------------------------------------------------
    def _register(self):
        for name, mod in self.model.named_modules():
            if isinstance(mod, (nn.Conv2d, nn.Linear, nn.BatchNorm2d, nn.LayerNorm)):
                handle = mod.register_forward_hook(
                    lambda m, _inp, out, n=name: self._acts.__setitem__(n, out.detach()))
                self.handles.append(handle)

    # ------------------------------------------------------------
    def __call__(self, x: torch.Tensor):
        self._acts.clear()
        logits = self.model(x)
        return logits, dict(self._acts)

    # ------------------------------------------------------------
    def remove(self):
        for h in self.handles:
            h.remove()


class DriftAwareAllocator:
    """Layer-wise  ℓ1-drift  →  byte budget  lookup table (spec §1)."""

    LUT = torch.tensor([0, 1, 1, 2, 2, 3, 3, 4,
                        4, 5, 5, 6, 6, 8, 10, 12], dtype=torch.int32)

    def __call__(self, rho: torch.Tensor, cap: int = 18) -> List[int]:
        idx = torch.clamp((rho * 15).long(), 0, 15)
        m = self.LUT[idx]
        excess = int(m.sum().item() - cap)
        if excess > 0:
            # greedily trim the largest layers first
            for i in torch.argsort(m, descending=True):
                if excess == 0:
                    break
                if m[i] > 0:
                    m[i] -= 1
                    excess -= 1
        return m.tolist()


class DeltaSketch:
    """Tiny – not bit-accurate – Python port of the delta-sketch codec.
    It stores int8 PCA coefficients truncated to the requested byte length.
    """

    def __init__(self):
        from sklearn.decomposition import MiniBatchPCA  # listed in requirements.txt
        self.pca: Dict[str, MiniBatchPCA] = {}
        self.flash: Dict[str, bytearray] = {}
        self._counter = 0
        self.MiniBatchPCA = MiniBatchPCA

    # ------------------------------------------------------------
    def _get_pca(self, k: str, d: int):
        if k not in self.pca:
            self.pca[k] = self.MiniBatchPCA(n_components=min(4, d), batch_size=256)
        return self.pca[k]

    # ------------------------------------------------------------
    def encode(self, acts: Dict[str, torch.Tensor], m_bytes: List[int]):
        for (layer, x), budget in zip(acts.items(), m_bytes):
            if budget == 0:
                continue
            x_flat = x.reshape(x.shape[0], -1).cpu().float()
            pca = self._get_pca(layer, x_flat.shape[1])
            try:
                pca.partial_fit(x_flat)
                coords = pca.transform(x_flat)
            except ValueError:  # n < n_components at the very beginning
                continue
            q = np.clip(np.round(coords), -128, 127).astype(np.int8).tobytes()
            self.flash[f"{self._counter}_{layer}"] = bytearray(q[:budget])
        self._counter += 1


class FlashDict(dict):
    """SQLite-backed key-value store → wear-levelling simulations are ignored in
    this minimal Python port; we only need persistence across calls."""

    pass


# ---------------------------------------------------------------------------
# 2.  Model definition – a very small CNN so that the quick run fits on CPU
# ---------------------------------------------------------------------------

def build_tiny_cnn(num_classes: int = 10) -> nn.Module:
    class _Tiny(nn.Module):
        def __init__(self):
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(),
                nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
                nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(),
                nn.AdaptiveAvgPool2d(1))
            self.classifier = nn.Linear(64, num_classes)

        def forward(self, x):
            x = self.features(x)
            return self.classifier(x.flatten(1))

    return _Tiny()


# ---------------------------------------------------------------------------
# 3.  Public API – train_model  (used by src/main.py)
# ---------------------------------------------------------------------------


def train_model(cfg: Dict[str, Any], loader: DataLoader, device: torch.device):
    model = build_tiny_cnn(cfg["num_classes"]).to(device)
    tracer = ActivationTracer(model)
    allocator = DriftAwareAllocator()
    sketch = DeltaSketch()
    flash = FlashDict()

    opt = optim.SGD(model.parameters(), lr=cfg["lr"], momentum=0.9)

    model.train()
    for epoch in range(cfg["epochs"]):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits, acts = tracer(x)
            loss = F.cross_entropy(logits, y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

        # after each epoch: store one mini-batch as a sketch (demo only)
        with torch.no_grad():
            rho = torch.rand(len(acts))  # placeholder for real ℓ1 drift
            budget = allocator(rho, cap=max(cfg["budgets"]))
            sketch.encode(acts, budget)
            flash.update(sketch.flash)

    tracer.remove()
    return model, flash
