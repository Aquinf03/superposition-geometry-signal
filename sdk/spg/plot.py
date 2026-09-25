"""Plot loss beside geometry curves from a signals.csv."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

from spg.plot_signals import plot_signals, read_signals


def plot(
    csv: Union[str, Path],
    out: Optional[Union[str, Path]] = None,
) -> Path:
    """Plot ``loss`` + ``geometry_*`` panels; write PNG next to CSV by default."""
    csv_path = Path(csv)
    out_path = Path(out) if out is not None else csv_path.parent / "loss_geometry.png"
    return plot_signals(csv_path, out_path)


__all__ = ["plot", "read_signals", "plot_signals"]
