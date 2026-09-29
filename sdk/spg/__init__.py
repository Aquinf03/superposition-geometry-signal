"""spg — Superposition Geometry as a live training signal beside loss.

Docs: docs/documentation/
"""

from __future__ import annotations

from spg.tracker import Tracker, SignalLogger, StepRecord, format_live_line
from spg.diff import Diff
from spg.plot import plot
from spg.view import write_view_html, build_view_payload
from spg.control import (
    GeoControl,
    ControlStatus,
    FeatureRelativeControl,
    FeatureRelativeStatus,
)
from spg.live import format_control_line
from spg import tui
from spg.grad_geometry import collect_mlp_out_bank_grad, mean_interference_tensor
from spg.geometry import (
    DEFAULT_PROBE_PROMPTS,
    collect_mlp_out_bank,
    collect_mlp_out_banks,
    compute_geometry_metrics,
)
from spg.seed import set_seed

__version__ = "0.1.1"

__all__ = [
    "Tracker",
    "Diff",
    "plot",
    "GeoControl",
    "ControlStatus",
    "FeatureRelativeControl",
    "FeatureRelativeStatus",
    "format_control_line",
    "tui",
    "collect_mlp_out_bank_grad",
    "mean_interference_tensor",
    "write_view_html",
    "build_view_payload",
    "SignalLogger",
    "StepRecord",
    "format_live_line",
    "DEFAULT_PROBE_PROMPTS",
    "collect_mlp_out_bank",
    "collect_mlp_out_banks",
    "compute_geometry_metrics",
    "set_seed",
    "__version__",
]
