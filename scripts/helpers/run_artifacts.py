"""Shim → ``spg.run_artifacts``."""
import sys
from pathlib import Path

_SDK = Path(__file__).resolve().parents[2] / "sdk"
if str(_SDK) not in sys.path:
    sys.path.insert(0, str(_SDK))

from spg import run_artifacts as _m

sys.modules[__name__] = _m
