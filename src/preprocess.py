from __future__ import annotations
"""
preprocess.py – produces the input data that the (dummy) training step
expects.  For this minimal example we only generate a *prompt* file with
a few lines of text.
"""
from pathlib import Path

__all__ = ["preprocess"]

def preprocess(data_root: str | Path, /) -> Path:
    """Generate a toy prompt file.

    The file is written to ``<data_root>/prompts.txt``.

    Returns
    -------
    pathlib.Path
        The path to the generated prompt file.
    """
    data_root = Path(data_root)
    data_root.mkdir(parents=True, exist_ok=True)

    prompt_path = data_root / "prompts.txt"
    sample_prompts = [
        "Translate the following English text to French:",
        "Summarise the key points from this article:",
        "Write a short poem about the sea:",
    ]

    prompt_path.write_text("\n".join(sample_prompts), encoding="utf-8")
    print(f"⚑  Toy prompt file created at {prompt_path.relative_to(Path.cwd())}")
    return prompt_path
