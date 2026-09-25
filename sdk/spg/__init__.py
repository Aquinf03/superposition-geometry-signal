"""spg — Superposition Geometry as a live training signal beside loss.

Quickstart::

    from spg import Tracker, Diff, plot

    tracker = Tracker(results_dir="runs", run_name="demo", live_print=True)
    tracker.log(step=0, loss=1.2, geometry={
        "interference_mean": 0.4,
        "spectral_participation": 0.2,
        "coactivation_overlap": 0.1,
    })
    plot(tracker.csv_path)
"""

from __future__ import annotations

from spg.tracker import Tracker, SignalLogger, StepRecord, format_live_line
from spg.diff import Diff
from spg.plot import plot
from spg.view import write_view_html, build_view_payload
from spg.geometry import (
    DEFAULT_PROBE_PROMPTS,
    collect_mlp_out_bank,
    collect_mlp_out_banks,
    compute_geometry_metrics,
)
from spg.seed import set_seed

__version__ = "0.1.0"

__all__ = [
    "Tracker",
    "Diff",
    "plot",
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
