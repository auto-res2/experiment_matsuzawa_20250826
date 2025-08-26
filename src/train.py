from __future__ import annotations
"""
train.py – creates a dummy model check-point so that the
subsequent evaluation step can find something to load.

The implementation is intentionally light-weight – the goal of this
research stub is *not* to train a real model but to have a file on
disk that pretends to be the result of a training run.
"""
from pathlib import Path
import json

# Public symbols ----------------------------------------------------------------
__all__ = ["train_model"]

def train_model(cfg: dict, /) -> Path:
    """Create a fake check-point file.

    Parameters
    ----------
    cfg : dict
        A configuration dictionary. Must contain the key
        ``"checkpoint_name"``.

    Returns
    -------
    pathlib.Path
        The path to the generated check-point file.
    """
    ckpt_name = cfg.get("checkpoint_name", "db_hidiff_stub.pt")

    # All experiment artifacts live under ``.research/iteration3``
    root_dir = Path(".research") / "iteration3"
    ckpt_path = root_dir / "checkpoints" / ckpt_name
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)

    # *Anything* written here will do – we just store the config so that the
    # file is not empty and users can inspect it if they like.
    with ckpt_path.open("w", encoding="utf-8") as fh:
        json.dump({"meta": "fake checkpoint", "cfg": cfg}, fh, indent=2)

    print(f"⚑  Stub check-point written to {ckpt_path.relative_to(Path.cwd())}")
    return ckpt_path
