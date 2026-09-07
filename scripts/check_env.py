"""Verify the experiment environment is importable.

Checks PyTorch, HuggingFace transformers, TransformerLens, and seed control.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.seed import set_seed


def main() -> None:
    seed = set_seed(0)
    print(f"python: {sys.version.split()[0]}")

    import torch

    print(f"torch: {torch.__version__}  cuda={torch.cuda.is_available()}")

    import transformers

    print(f"transformers: {transformers.__version__}")

    import transformer_lens

    tl_ver = getattr(transformer_lens, "__version__", "unknown")
    print(f"transformer_lens: {tl_ver}")

    from scripts.signals import SignalLogger

    print(f"signals: {SignalLogger.__name__}")
    print(f"seed: {seed}")
    print("env ok")


if __name__ == "__main__":
    main()
