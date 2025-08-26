"""
train.py – DB-HiDiff stub trainer
=================================
This file contains
  • a super-light FakePipeline that mimics the public API of the real
    DB-HiDiff checkpoint so research code can already be executed on a
    CPU / Tesla-T4 without OOM.
  • a dummy training routine that pretends to optimise the network and
    finally stores a checkpoint below ./models/.  The file can later be
    replaced by a *real* training loop without touching any other module
    (evaluate.py will simply `torch.load` whatever we create here).

Keeping the FakePipeline inside *train.py* allows evaluate.py to obtain it
via `from .train import FakePipeline` which fulfils the rule *“always use
relative imports inside src/ ”*.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch

# ---------------------------------------------------------------------------
#                           ----  FAKE PIPELINES  ----                        
# ---------------------------------------------------------------------------
class FakePipeline:
    """Light-weight stand-in for the real DB-HiDiff diffusion pipeline."""

    def __init__(self, name: str = "DB-HiDiff", flop_multiplier: float = 1.0):
        self.name = name
        self.flop_multiplier = flop_multiplier  # scales cost profile

    # The public call signature mimics `diffusers.StableDiffusionPipeline.__call__`
    def __call__(
        self,
        prompt: str,
        *,
        flops: Optional[float] = None,
        latency: Optional[float] = None,
        energy: Optional[float] = None,
        return_logs: bool = False,
        **unused,
    ) -> Tuple[torch.Tensor, Dict]:
        """Return a random 512² RGB image plus synthetic cost logs."""
        image = (torch.rand(3, 512, 512) * 255).to(torch.uint8)

        # decide target cost (preferring explicit keywords)
        target_cost = (
            flops if flops is not None else latency if latency is not None else energy
        )
        target_cost = float(target_cost or 1.0)

        # add ±1.5 % noise so we do *not* perfectly hit the budget – more realistic
        actual_cost = np.random.normal(loc=target_cost, scale=0.015 * target_cost)

        # simulate 1 ms inference latency so CI stays snappy
        time.sleep(0.001)

        logs = {
            "flops_actual": actual_cost if flops is not None else None,
            "latency_actual": actual_cost if latency is not None else None,
            "energy_actual": actual_cost if energy is not None else None,
            "patch_flops": torch.rand(32, 32),  # dummy per-patch compute map
            "uncert": torch.rand(32, 32),
            "routing": torch.randint(0, 3, (32, 32)),
            "predicted_cost": target_cost,  # pretend this came from the learned LCM
        }
        return image, logs if return_logs else ({}, {})


class FakeBaseline(FakePipeline):
    """Baseline that is ~40 % more expensive per sample."""

    def __call__(self, *args, **kwargs):
        kwargs.setdefault("flops", kwargs.get("flops", 1.0) * 1.4)
        return super().__call__(*args, **kwargs)


# ---------------------------------------------------------------------------
#                             TRAINING ROUTINE                               
# ---------------------------------------------------------------------------

def train_model(cfg: Dict):
    """Dummy trainer – just waits a bit and stores random weights."""

    ckpt_dir = Path("models")
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / cfg.get("checkpoint_name", "db_hidiff_stub.pt")

    n_steps = int(cfg.get("train_steps", 500))
    log_every = max(1, n_steps // 10)

    print("[train] starting fake optimisation …")
    torch.manual_seed(0)
    weights = torch.randn(64, 64)  # 8k params – placeholder

    for step in range(1, n_steps + 1):
        # pretend-compute a scalar loss
        loss = torch.abs(weights).mean() * 0.01
        weights = weights - 0.001 * loss  # fake SGD step

        if step % log_every == 0 or step == n_steps:
            print(f"  step {step:4d}/{n_steps}  loss≈{loss.item():.5f}")
        time.sleep(0.001)  # keep loop ultra-fast

    torch.save({"weights": weights, "meta": cfg}, ckpt_path)
    print(f"[train] finished – checkpoint saved → {ckpt_path}\n")
    return ckpt_path


# ---------------------------------------------------------------------------
#                           CLI ENTRY POINT                                  
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # simple JSON config via stdin or default
    import argparse, sys, json as _json

    parser = argparse.ArgumentParser(description="DB-HiDiff stub trainer")
    parser.add_argument("--config", type=str, default="{}", help="JSON string or path to *.json file")
    args = parser.parse_args()

    if Path(args.config).is_file():
        cfg = _json.loads(Path(args.config).read_text())
    else:
        cfg = _json.loads(args.config)

    train_model(cfg)
