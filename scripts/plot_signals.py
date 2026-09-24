"""Plot loss beside geometry_* columns from a signals.csv (track dashboard)."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


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


def plot_signals(csv_path: Path, out_path: Path) -> Path:
    import matplotlib.pyplot as plt

    data = read_signals(csv_path)
    steps = data["step"]
    geo_cols = sorted(c for c in data if c.startswith("geometry_"))

    n = 1 + len(geo_cols)
    fig, axes = plt.subplots(n, 1, figsize=(8, 2.2 * n), sharex=True)
    if n == 1:
        axes = [axes]

    axes[0].plot(steps, data["loss"], color="black", lw=1.5)
    axes[0].set_ylabel("loss")
    axes[0].set_title("track: loss + geometry")

    for ax, col in zip(axes[1:], geo_cols):
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
        default=ROOT / "results" / "rome_gpt2_small_baseline" / "signals.csv",
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
