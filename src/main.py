"""
main.py – orchestrates the whole pipeline   (pre-process → train → evaluate)
It respects the directory structure /config/*.yaml specified in the rules.

Run a quick sanity-check on CPU via
    $ python -m src.main  --config config/quick.yaml

Run a larger experiment on the Tesla T4
    $ python -m src.main  --config config/full.yaml  --gpu

All images are written to .research/iteration1/images as required.
"""
from __future__ import annotations

import argparse
import yaml
from pathlib import Path
import torch

from .preprocess import get_dataloaders
from .train import train_model
from .evaluate import evaluate

# ---------------------------------------------------------------------------


def load_cfg(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    # enrich with runtime overrides (image dir etc.)
    cfg.setdefault("image_dir", ".research/iteration1/images")
    return cfg


# ---------------------------------------------------------------------------


def cli():
    parser = argparse.ArgumentParser("SketchReplay 3.0-Plus experimental runner")
    parser.add_argument("--config", type=str, default="config/quick.yaml",
                        help="YAML file with experiment hyper-parameters")
    parser.add_argument("--gpu", action="store_true", help="Use CUDA if available")
    args = parser.parse_args()

    cfg = load_cfg(Path(args.config))

    device = torch.device("cuda" if args.gpu and torch.cuda.is_available() else "cpu")
    print(f"Running on {device}")

    train_loader, test_loader = get_dataloaders(cfg)
    model, flash = train_model(cfg, train_loader, device)
    evaluate(model, test_loader, cfg)
    print("Done ✔︎")


if __name__ == "__main__":
    cli()
