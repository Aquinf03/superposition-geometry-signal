"""Diff — compare superposition geometry for selected features / neighborhoods."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import torch

from spg.diff_geometry import (
    diff_snapshots,
    diff_step_queries,
    extract_mlp_out_feature,
    plot_neighborhood_diff,
    save_diff,
    snapshot_neighborhood,
)


class Diff:
    """Static helpers for neighborhood snapshots and deltas."""

    snapshot = staticmethod(snapshot_neighborhood)
    snapshots = staticmethod(diff_snapshots)
    step_queries = staticmethod(diff_step_queries)
    save = staticmethod(save_diff)
    plot = staticmethod(plot_neighborhood_diff)
    extract_mlp_out = staticmethod(extract_mlp_out_feature)

    @classmethod
    def compare(
        cls,
        before: Dict[str, Any],
        after: Dict[str, Any],
        *,
        mode: str = "pre_post",
        out: Optional[Path] = None,
        plot_path: Optional[Path] = None,
    ) -> Dict[str, Any]:
        """Diff two snapshots; optionally write JSON / neighborhood plot."""
        result = cls.snapshots(before, after, mode=mode)
        if out is not None:
            cls.save(result, Path(out))
        if plot_path is not None:
            cls.plot(result, Path(plot_path))
        return result


__all__ = [
    "Diff",
    "diff_snapshots",
    "diff_step_queries",
    "extract_mlp_out_feature",
    "plot_neighborhood_diff",
    "save_diff",
    "snapshot_neighborhood",
]
