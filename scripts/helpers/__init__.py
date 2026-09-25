"""Backward-compatible shims — real code lives in ``spg`` (``sdk/spg``). """

from __future__ import annotations

import sys
from pathlib import Path

_SDK = Path(__file__).resolve().parents[2] / "sdk"
if str(_SDK) not in sys.path:
    sys.path.insert(0, str(_SDK))

from spg.geometry import (  # noqa: E402, F401
    DEFAULT_PROBE_PROMPTS,
    collect_mlp_out_bank,
    collect_mlp_out_banks,
    compute_geometry_metrics,
)
from spg.signals import SignalLogger, format_live_line  # noqa: E402, F401
from spg.diff_geometry import (  # noqa: E402, F401
    diff_snapshots,
    snapshot_neighborhood,
    plot_neighborhood_diff,
)
from spg.seed import set_seed  # noqa: E402, F401

__all__ = [
    "DEFAULT_PROBE_PROMPTS",
    "SignalLogger",
    "collect_mlp_out_bank",
    "collect_mlp_out_banks",
    "compute_geometry_metrics",
    "diff_snapshots",
    "format_live_line",
    "plot_neighborhood_diff",
    "set_seed",
    "snapshot_neighborhood",
]
