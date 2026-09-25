"""Plot loss beside geometry_* columns from a signals.csv (track dashboard).

Single-layer CSVs: one panel per metric.
Multi-layer CSVs (geometry_L{n}_*): one panel per metric, one line per layer.
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.helpers.paths import RESULTS

_LAYER_COL_RE = re.compile(r"^geometry_L(?P<layer>\d+)_(?P<metric>.+)$")


def read_signals(path: Path) -> Dict[str, List[float]]:
    with path.open() as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"empty signals file: {path}")
    cols = rows[0].keys()
    out: Dict[str, List[float]] = {}
    for col in cols:
        out[col] = [float(r[col]) if r.get(col) not in (None, "") else float("nan") for r in rows]
    return out


def _group_geo_cols(geo_cols: List[str]) -> Tuple[Dict[str, List[Tuple[int, str]]], List[str]]:
    """Return ({metric: [(layer, col), ...]}, plain_cols)."""
    by_metric: Dict[str, List[Tuple[int, str]]] = defaultdict(list)
    plain: List[str] = []
    for col in geo_cols:
        m = _LAYER_COL_RE.match(col)
        if m:
            by_metric[m.group("metric")].append((int(m.group("layer")), col))
        else:
            plain.append(col)
    for metric in by_metric:
        by_metric[metric].sort(key=lambda t: t[0])
    return dict(by_metric), plain


def plot_signals(csv_path: Path, out_path: Path) -> Path:
    import matplotlib.pyplot as plt

    data = read_signals(csv_path)
    steps = data["step"]
    geo_cols = sorted(c for c in data if c.startswith("geometry_"))
    by_metric, plain = _group_geo_cols(geo_cols)

    if by_metric:
        metric_order = sorted(by_metric.keys())
        n = 1 + len(metric_order)
        fig, axes = plt.subplots(n, 1, figsize=(8, 2.4 * n), sharex=True)
        if n == 1:
            axes = [axes]

        axes[0].plot(steps, data["loss"], color="black", lw=1.5)
        axes[0].set_ylabel("loss")
        axes[0].set_title("track: loss + geometry (multi-layer)")

        for ax, metric in zip(axes[1:], metric_order):
            for layer, col in by_metric[metric]:
                ax.plot(steps, data[col], lw=1.5, label=f"L{layer}")
            ax.set_ylabel(metric)
            ax.legend(loc="best", fontsize=8, frameon=False)
    else:
        n = 1 + len(plain)
        fig, axes = plt.subplots(n, 1, figsize=(8, 2.2 * n), sharex=True)
        if n == 1:
            axes = [axes]

        axes[0].plot(steps, data["loss"], color="black", lw=1.5)
        axes[0].set_ylabel("loss")
        axes[0].set_title("track: loss + geometry")

        for ax, col in zip(axes[1:], plain):
            ax.plot(steps, data[col], lw=1.5)
            ax.set_ylabel(col.replace("geometry_", ""))

    axes[-1].set_xlabel("step")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot loss + geometry from signals.csv")
    parser.add_argument(
        "--csv",
        type=Path,
        default=RESULTS / "rome_gpt2_small_baseline" / "signals.csv",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output PNG (default: next to csv as loss_geometry.png)",
    )
    args = parser.parse_args()
    out = args.out or (args.csv.parent / "loss_geometry.png")
    path = plot_signals(args.csv, out)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
