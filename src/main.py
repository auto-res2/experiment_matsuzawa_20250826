"""src/main.py
High-level experiment runner that re-implements the three *plans* described in
*New Method / Experiment Code* but uses the sub-modules ``train``/``evaluate``/
``preprocess`` so that each phase is cleanly separated.

Usage
-----
# quick functional test (CPU is fine)
$ python -m src.main --test

# full benchmark on GPU
$ python -m src.main --device cuda                # runs all three plans
$ python -m src.main --plan 1 --device cuda       # only Plan-1
"""
from __future__ import annotations

# ---------------------------------------------------------------------------
# 0.  Make sure the file can be executed *directly* -------------------------
# ---------------------------------------------------------------------------
# When ``src/main.py`` is launched via ``python src/main.py`` (as opposed to
# ``python -m src.main``), relative imports would normally fail because
# ``__package__`` is not set.  The small block below synthesises the minimal
# package metadata so that the rest of the file can continue to use *only*
# relative imports, satisfying the project requirement.

import importlib.util
import os
import sys
from pathlib import Path

if __package__ is None:  # executed directly
    pkg_name = "src"
    pkg_path = Path(__file__).resolve().parent
    if pkg_name not in sys.modules:
        spec = importlib.util.spec_from_loader(pkg_name, loader=None, is_package=True)
        module = importlib.util.module_from_spec(spec)
        module.__path__ = [str(pkg_path)]  # type: ignore[attr-defined]
        sys.modules[pkg_name] = module
    __package__ = pkg_name  # type: ignore[assignment]

# ---------------------------------------------------------------------------
# 1.  Imports ----------------------------------------------------------------
# ---------------------------------------------------------------------------

import argparse
import random

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch
from tqdm import tqdm

from .train import ToyModel, peak_monitor, train_one_epoch, free_memory, RESULTS_DIR
from .evaluate import evaluate_model

sns.set_style("whitegrid")

# ---------------------------------------------------------------------------
# 2.  Helper: figure saver ---------------------------------------------------
# ---------------------------------------------------------------------------

def _save(fig, name: str):
    path = os.path.join(RESULTS_DIR, name)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {path}")


# ---------------------------------------------------------------------------
# 3.  Experiment Plan-1: memory / throughput --------------------------------
# ---------------------------------------------------------------------------

def experiment_plan1(device: str = "cpu", quick: bool = False):
    print("Running Plan-1 (memory / throughput)…")
    sizes = [64, 128] if quick else [224, 512, 1024, 2048]
    slice_grid = [64, 256, None] if quick else [64, 128, 256, 512, None]

    records = []
    for h in tqdm(sizes, desc="resolutions"):
        x = torch.randn(2, 64, h, h, device=device, requires_grad=True)
        for c in slice_grid:
            for rev in (False, True):
                model = ToyModel(rev=rev, slice_len=c, depth=2).to(device)
                model.train()
                with peak_monitor(device):
                    y = model(x)
                    y.mean().backward()
                stats = peak_monitor.stats
                records.append(
                    {
                        "H": h,
                        "slice": c or 0,
                        "rev": rev,
                        "mem_MB": stats.peak / 1e6,
                        "sec": stats.elapsed,
                    }
                )
                free_memory(model, y)
    df = pd.DataFrame(records)
    csv_path = os.path.join(RESULTS_DIR, "bench_peak.csv")
    df.to_csv(csv_path, index=False)
    print(df)
    print(f"[saved] {csv_path}")

    # line plot --------------------------------------------------------------
    fig = plt.figure(figsize=(6, 4))
    for key, grp in df.groupby("rev"):
        label = "RevSiM" if key else "Baseline"
        sns.lineplot(ax=plt.gca(), data=grp, x="H", y="mem_MB", marker="o", label=label)
    plt.xlabel("Resolution (H = W)")
    plt.ylabel("Peak Memory [MB]")
    plt.title("Peak memory vs. image size")
    plt.legend()
    _save(fig, "peak_memory.pdf")


# ---------------------------------------------------------------------------
# 4.  Experiment Plan-2: accuracy parity ------------------------------------
# ---------------------------------------------------------------------------

def experiment_plan2(device: str = "cpu", epochs: int = 3):
    print("Running Plan-2 (accuracy parity)…")

    model_b = ToyModel(rev=False, slice_len=None, depth=3).to(device)
    model_r = ToyModel(rev=True, slice_len=128, depth=3).to(device)

    opt_b = torch.optim.AdamW(model_b.parameters(), lr=1e-3)
    opt_r = torch.optim.AdamW(model_r.parameters(), lr=1e-3)

    losses_b, losses_r = [], []
    for ep in range(epochs):
        losses_b.append(train_one_epoch(model_b, opt_b, device=device))
        losses_r.append(train_one_epoch(model_r, opt_r, device=device))

    acc_b, _ = evaluate_model(model_b, device=device)
    acc_r, _ = evaluate_model(model_r, device=device)
    print(f"Baseline acc={acc_b*100:.2f}  RevSiM acc={acc_r*100:.2f}")

    # loss curve -------------------------------------------------------------
    fig = plt.figure(figsize=(5, 4))
    plt.plot(losses_b, label="Baseline")
    plt.plot(losses_r, label="RevSiM")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Toy training loss")
    plt.legend()
    _save(fig, "training_loss.pdf")

    # accuracy bar -----------------------------------------------------------
    fig = plt.figure(figsize=(4, 4))
    sns.barplot(x=["Baseline", "RevSiM"], y=[acc_b, acc_r])
    plt.ylabel("Top-1 Accuracy")
    plt.ylim(0.75, 0.85)
    plt.title("Accuracy parity (synthetic)")
    _save(fig, "accuracy_baseline_vs_revsim.pdf")


# ---------------------------------------------------------------------------
# 5.  Experiment Plan-3: giant WSI demo -------------------------------------
# ---------------------------------------------------------------------------

def experiment_plan3(device: str = "cpu"):
    print("Running Plan-3 (toy WSI)…")

    # baseline is expected to OOM for very large inputs on GPU
    giant = torch.randn(1, 64, 4096, 4096, device=device)
    baseline = ToyModel(rev=False).to(device)
    try:
        baseline(giant)
    except RuntimeError as e:
        print("Baseline OOM as expected →", str(e).split("\n")[0], "…")

    # RevSiM should pass comfortably with small slice length
    model = ToyModel(rev=True, slice_len=64).to(device)
    with peak_monitor(device):
        out = model(giant)
    stats = peak_monitor.stats
    print(f"RevSiM OK – peak {stats.peak/1e6:.1f} MB, output shape {tuple(out.shape)}")


# ---------------------------------------------------------------------------
# 6.  Quick functional test --------------------------------------------------
# ---------------------------------------------------------------------------

def test_all():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    experiment_plan1(device=device, quick=True)
    experiment_plan2(device=device, epochs=2)
    experiment_plan3(device=device)
    print("All tests completed ✓")


# ---------------------------------------------------------------------------
# 7.  CLI --------------------------------------------------------------------
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    torch = __import__("torch")  # lazy import to keep the header small

    torch.manual_seed(2025)
    random.seed(2025)

    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=int, choices=[1, 2, 3], default=None,
                        help="Which experiment plan to run (default: all)")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"],
                        help="Execution device")
    parser.add_argument("--test", action="store_true", help="Run quick test suite")
    args = parser.parse_args()

    if args.test:
        test_all()
    else:
        if args.plan is None or args.plan == 1:
            experiment_plan1(device=args.device)
        if args.plan is None or args.plan == 2:
            experiment_plan2(device=args.device)
        if args.plan is None or args.plan == 3:
            experiment_plan3(device=args.device)
