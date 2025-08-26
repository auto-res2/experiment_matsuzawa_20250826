"""
preprocess.py – minimal dummy pre-processing for DB-HiDiff demo
================================================================
For real research you would load LAION-Aesthetic, resize / centre-crop, build
CLIP embeddings, etc.  Here we merely create a *tiny* JSONL with a handful of
text prompts so that the rest of the pipeline has something to work with.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import List

PROMPTS = [
    "A photo of a cute corgi in a field of sunflowers",
    "An astronaut lounging in a tropical resort in space, vaporwave",
    "Snow-covered mountain under clear blue sky",
]

def preprocess(data_dir: str | Path = "data") -> Path:
    path = Path(data_dir)
    path.mkdir(parents=True, exist_ok=True)
    out_file = path / "train_prompts.jsonl"

    with out_file.open("w") as f:
        for p in PROMPTS:
            f.write(json.dumps({"prompt": p}) + "\n")

    print(f"[preprocess] wrote {len(PROMPTS)} prompts → {out_file}")
    return out_file


if __name__ == "__main__":
    preprocess()
