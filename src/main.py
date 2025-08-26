"""
main.py – orchestrates the whole pipeline   (pre-process → train → evaluate)
It respects the directory structure /config/*.yaml specified in the rules.

Run a quick sanity-check on CPU via
    $ python src/main.py

Run a larger experiment on the Tesla T4
    $ python src/main.py  --config config/full.yaml  --gpu

All images are written to .research/iteration2/images as required.
"""
from __future__ import annotations

import argparse
import yaml
from pathlib import Path
import sys
import torch

# Ensure the project root (parent of src) is on the path so that
# `import src.*` works even when this file is executed as a script.
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.preprocess import get_dataloaders  # type: ignore  # noqa: E402
from src.train import train_model  # type: ignore  # noqa: E402
from src.evaluate import evaluate  # type: ignore  # noqa: E402

# ---------------------------------------------------------------------------

def load_cfg(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Config file '{path}' not found.")
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    # enrich with runtime overrides (image dir etc.)
    cfg.setdefault("image_dir", ".research/iteration2/images")
    return cfg


# ---------------------------------------------------------------------------

def cli():
    parser = argparse.ArgumentParser("SketchReplay 3.0-Plus experimental runner")
    parser.add_argument("--config", type=str, default="config/config.yaml",
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
