from __future__ import annotations

"""
main.py – orchestrates the entire experimental pipeline
=======================================================
The flow is:
  1. Pre-processing (creates a toy prompt file)
  2. Training      (stub – creates a fake checkpoint)
  3. Evaluation    (runs three experiments & saves plots as PDF)

All directories and filenames follow the structure given in the user rules.
"""

# ---------------------------------------------------------------------------
#  Bootstrap so relative imports also work when the file is executed directly
# ---------------------------------------------------------------------------
import types
import sys
from pathlib import Path

if __package__ in {None, ""}:  # pragma: no cover – only executed as script
    pkg_path = Path(__file__).resolve().parent          # …/src
    pkg_name = pkg_path.name                            # "src"

    root = str(pkg_path.parent)
    if root not in sys.path:
        sys.path.insert(0, root)

    if pkg_name not in sys.modules:
        pkg = types.ModuleType(pkg_name)
        pkg.__path__ = [str(pkg_path)]
        sys.modules[pkg_name] = pkg

    __package__ = pkg_name

# ---------------------------------------------------------------------------
#  Now the regular imports will succeed in *module* as well as *script* mode
# ---------------------------------------------------------------------------
import argparse
import json

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
    parser.add_argument(
        "--config",
        type=str,
        default="{}",
        help="Path to JSON/YAML or JSON string with parameters",
    )
    args = parser.parse_args()

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
    _prep.preprocess("data")

    # 2. training -----------------------------------------------------------
    _train.train_model(cfg)

    # 3. evaluation ---------------------------------------------------------
    _eval.evaluate(cfg)

    print("\n⚑  Done.  All figures are located under .research/iteration3/images")


if __name__ == "__main__":
    main()
