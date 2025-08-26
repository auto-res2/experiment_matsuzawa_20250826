"""
main.py – orchestrates the entire experimental pipeline
=======================================================
The flow is:
  1. Pre-processing (creates a toy prompt file)
  2. Training      (stub – creates a fake checkpoint)
  3. Evaluation    (runs three experiments & saves plots as PDF)

All directories and filenames follow the structure given in the user rules.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from . import preprocess as _prep
from . import train as _train
from . import evaluate as _eval


# ---------------------------------------------------------------------------
#                         DEFAULT CONFIG  (can be overridden)                
# ---------------------------------------------------------------------------
_DEFAULT_CFG = {
    "train_steps": 300,
    "checkpoint_name": "db_hidiff_stub.pt",
}


# ---------------------------------------------------------------------------
#                               MAIN                                         
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="DB-HiDiff experimental runner")
    parser.add_argument("--config", type=str, default="{}", help="Path to JSON/YAML or JSON string with parameters")
    args = parser.parse_args()

    # minimal config parsing – only JSON for brevity
    cfg = _DEFAULT_CFG.copy()
    if args.config.strip():
        if Path(args.config).is_file():
            cfg.update(json.loads(Path(args.config).read_text()))
        else:
            cfg.update(json.loads(args.config))

    print("====  DB-HiDiff Research Suite  ====")
    print("configuration:")
    for k, v in cfg.items():
        print(f"  {k}: {v}")
    print("====================================\n")

    # 1. preprocessing ------------------------------------------------------
    prompt_file = _prep.preprocess("data")

    # 2. training -----------------------------------------------------------
    ckpt = _train.train_model(cfg)

    # 3. evaluation ---------------------------------------------------------
    _eval.evaluate(cfg)

    print("⚑  Done.  All figures are located under .research/iteration1/images")


if __name__ == "__main__":
    main()
