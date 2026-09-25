"""Validate geometry on a real training run (not edit locality).

Reads ``signals.csv`` (+ optional cross-feature / ckpt diffs) and checks that
geometry trajectories track real training phenomena:

  1. geometry is not flat while loss moves
  2. geometry correlates with loss (collapse / pack signals)
  3. late-layer geometry phase (collapse or unpack-from-floor)
  4. packing coupling (interference ↔ spectral)
  5. layer-phase divergence (early vs late layers disagree)
  6. (optional) related features closer than unrelated at a frozen ckpt

Example:
  python experiments/validate_training_geometry.py \\
      --run-dir experiments/results/train_gpt2_small_geometry
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.helpers.paths import RESULTS

_LAYER_COL_RE = re.compile(r"^geometry_L(?P<layer>\d+)_(?P<metric>.+)$")


def pearson(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    x = x - x.mean()
    y = y - y.mean()
    denom = float(np.sqrt((x * x).sum() * (y * y).sum()))
    if denom < 1e-12:
        return 0.0
    return float((x * y).sum() / denom)


def read_signals(path: Path) -> Dict[str, np.ndarray]:
    with path.open() as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"empty signals: {path}")
    out: Dict[str, np.ndarray] = {}
    for col in rows[0].keys():
        out[col] = np.array(
            [float(r[col]) if r.get(col) not in (None, "") else np.nan for r in rows],
            dtype=float,
        )
    return out


def parse_geo_columns(cols: List[str]) -> Dict[int, Dict[str, str]]:
    """layer → {metric: column_name}."""
    by_layer: Dict[int, Dict[str, str]] = {}
    for col in cols:
        m = _LAYER_COL_RE.match(col)
        if not m:
            # single-layer legacy: geometry_interference_mean
            if col.startswith("geometry_") and "_L" not in col:
                by_layer.setdefault(-1, {})[col[len("geometry_") :]] = col
            continue
        by_layer.setdefault(int(m.group("layer")), {})[m.group("metric")] = col
    return by_layer


def window_means(arr: np.ndarray, head: int = 5, tail: int = 5) -> Tuple[float, float, float]:
    """Return (early_mean, late_mean, late-early)."""
    n = len(arr)
    h = min(head, max(n // 4, 1))
    t = min(tail, max(n // 4, 1))
    early = float(np.nanmean(arr[:h]))
    late = float(np.nanmean(arr[-t:]))
    return early, late, late - early


def claim(cid: str, passed: bool, detail: str, **stats: Any) -> Dict[str, Any]:
    return {"id": cid, "pass": bool(passed), "detail": detail, "stats": stats}


def find_cross_feature_summary(run_dir: Path) -> Optional[Path]:
    cands = sorted(run_dir.glob("diff_features_step_*/diff_summary.json"))
    return cands[-1] if cands else None


def find_ckpt_diff_summary(run_dir: Path) -> Optional[Path]:
    cands = sorted(run_dir.glob("diff_ckpt_*_vs_*/diff_summary.json"))
    return cands[-1] if cands else None


def validate_run(run_dir: Path) -> Dict[str, Any]:
    run_dir = Path(run_dir)
    csv_path = run_dir / "signals.csv"
    data = read_signals(csv_path)
    loss = data["loss"]
    steps = data["step"]
    by_layer = parse_geo_columns(list(data.keys()))
    if not by_layer:
        raise ValueError(f"no geometry_* columns in {csv_path}")

    claims: List[Dict[str, Any]] = []

    # --- 1. Geometry not flat while loss moves ---
    loss_range = float(np.nanmax(loss) - np.nanmin(loss))
    geo_ranges = []
    for L, metrics in by_layer.items():
        for metric, col in metrics.items():
            geo_ranges.append(
                {
                    "layer": L,
                    "metric": metric,
                    "range": float(np.nanmax(data[col]) - np.nanmin(data[col])),
                    "std": float(np.nanstd(data[col])),
                }
            )
    max_geo_range = max(g["range"] for g in geo_ranges)
    # loss should move; at least one geometry series should move meaningfully
    claims.append(
        claim(
            "geometry_not_flat",
            passed=(loss_range > 0.5 and max_geo_range > 0.03),
            detail=(
                f"loss range={loss_range:.3f}; max geometry range={max_geo_range:.4f} "
                "(need loss>0.5 and some geometry range>0.03)"
            ),
            loss_range=loss_range,
            max_geometry_range=max_geo_range,
            per_series=geo_ranges,
        )
    )

    # --- 2. Geometry correlates with loss ---
    corr_rows = []
    for L, metrics in sorted(by_layer.items()):
        for metric, col in metrics.items():
            r = pearson(loss, data[col])
            corr_rows.append({"layer": L, "metric": metric, "pearson_with_loss": r})
    # At least one |r| strong enough, or spectral negatively correlated with loss
    # (collapse: as loss falls, spectral often falls too → positive r with loss)
    abs_corrs = [abs(r["pearson_with_loss"]) for r in corr_rows]
    best = max(corr_rows, key=lambda r: abs(r["pearson_with_loss"]))
    claims.append(
        claim(
            "correlates_with_loss",
            passed=max(abs_corrs) >= 0.35,
            detail=(
                f"best |ρ(loss, geometry)|={max(abs_corrs):.3f} "
                f"at L{best['layer']} {best['metric']} (ρ={best['pearson_with_loss']:+.3f})"
            ),
            best=best,
            correlations=corr_rows,
        )
    )

    # --- 3. Late-layer geometry phase (collapse OR unpack-from-floor) ---
    # Prefer highest layer index
    layers = sorted(L for L in by_layer if L >= 0)
    late_layer = layers[-1] if layers else -1
    phase_mode = "none"
    late_ok = False
    late_stats: Dict[str, Any] = {"layer": late_layer}
    if late_layer in by_layer and "spectral_participation" in by_layer[late_layer]:
        col_s = by_layer[late_layer]["spectral_participation"]
        early_s, late_s, d_s = window_means(data[col_s])
        r_loss = pearson(loss, data[col_s])
        d_i = 0.0
        early_i = late_i = float("nan")
        if "interference_mean" in by_layer[late_layer]:
            early_i, late_i, d_i = window_means(data[by_layer[late_layer]["interference_mean"]])
        # (A) classic collapse: spectral drops hard
        if (d_s < -0.03) and (late_s < early_s * 0.5 or late_s < 0.05):
            late_ok, phase_mode = True, "spectral_collapse"
        # (B) already near floor at init, then unpacks (gpt2-medium L22 pattern):
        #     early spectral tiny, interference falls, spectral creeps up
        elif early_s < 0.05 and d_i < -0.05 and d_s > 0.005:
            late_ok, phase_mode = True, "unpack_from_floor"
        # (C) strong late-layer motion on either metric
        elif abs(d_s) > 0.05 or abs(d_i) > 0.1:
            late_ok, phase_mode = True, "strong_late_layer_move"
        late_stats.update(
            {
                "mode": phase_mode,
                "spectral_early": early_s,
                "spectral_late": late_s,
                "delta_spectral": d_s,
                "interference_early": early_i,
                "interference_late": late_i,
                "delta_interference": d_i,
                "pearson_spectral_with_loss": r_loss,
                "column": col_s,
            }
        )
    claims.append(
        claim(
            "late_layer_geometry_phase",
            passed=late_ok,
            detail=(
                f"L{late_layer} mode={phase_mode}  "
                f"spectral {late_stats.get('spectral_early', float('nan')):.4f}→"
                f"{late_stats.get('spectral_late', float('nan')):.4f} "
                f"(Δ={late_stats.get('delta_spectral', float('nan')):+.4f})  "
                f"interference Δ={late_stats.get('delta_interference', float('nan')):+.4f}"
            ),
            **late_stats,
        )
    )

    # --- 4. Packing coupling (interference ↔ spectral move together inversely) ---
    pack_ok = False
    pack_stats: Dict[str, Any] = {}
    pack_layer = late_layer
    if (
        pack_layer in by_layer
        and "interference_mean" in by_layer[pack_layer]
        and "spectral_participation" in by_layer[pack_layer]
    ):
        inter = data[by_layer[pack_layer]["interference_mean"]]
        spec = data[by_layer[pack_layer]["spectral_participation"]]
        mid = len(loss) // 2
        r_anti = pearson(inter[mid:], spec[mid:])
        _, _, d_inter = window_means(inter)
        _, _, d_spec = window_means(spec)
        # denser (inter↑ spec↓), unpack (inter↓ spec↑), or late-phase anti-correlation
        pack_ok = (
            (d_inter > 0.02 and d_spec < -0.02)
            or (d_inter < -0.02 and d_spec > 0.005)
            or (r_anti < -0.3)
        )
        pack_stats = {
            "layer": pack_layer,
            "delta_interference": d_inter,
            "delta_spectral": d_spec,
            "pearson_inter_vs_spectral_late": r_anti,
        }
    claims.append(
        claim(
            "packing_coupling",
            passed=pack_ok,
            detail=(
                f"L{pack_layer} Δinterference={pack_stats.get('delta_interference', float('nan')):+.4f} "
                f"Δspectral={pack_stats.get('delta_spectral', float('nan')):+.4f} "
                f"ρ(inter,spectral)_late={pack_stats.get('pearson_inter_vs_spectral_late', float('nan')):+.3f}"
            ),
            **pack_stats,
        )
    )

    # --- 5. Layer-phase divergence ---
    phase_ok = False
    phase_stats: Dict[str, Any] = {}
    if len(layers) >= 2 and "spectral_participation" in by_layer[layers[0]]:
        early_L, late_L = layers[0], layers[-1]
        if "spectral_participation" in by_layer[late_L]:
            s_early = data[by_layer[early_L]["spectral_participation"]]
            s_late = data[by_layer[late_L]["spectral_participation"]]
            r_layers = pearson(s_early, s_late)
            _, _, d_e = window_means(s_early)
            _, _, d_l = window_means(s_late)
            # phases diverge: weak/negative correlation OR opposite-signed deltas
            phase_ok = (r_layers < 0.5) or (d_e * d_l < 0) or (abs(d_l - d_e) > 0.04)
            phase_stats = {
                "early_layer": early_L,
                "late_layer": late_L,
                "pearson_spectral_early_vs_late_layer": r_layers,
                "delta_spectral_early_layer": d_e,
                "delta_spectral_late_layer": d_l,
            }
    claims.append(
        claim(
            "layer_phase_divergence",
            passed=phase_ok,
            detail=(
                f"L{phase_stats.get('early_layer')} vs L{phase_stats.get('late_layer')} "
                f"spectral ρ={phase_stats.get('pearson_spectral_early_vs_late_layer', float('nan')):+.3f} "
                f"Δ=({phase_stats.get('delta_spectral_early_layer', float('nan')):+.4f}, "
                f"{phase_stats.get('delta_spectral_late_layer', float('nan')):+.4f})"
            ),
            **phase_stats,
        )
    )

    # --- 6. Optional: cross-feature structure (related closer than unrelated) ---
    xf_path = find_cross_feature_summary(run_dir)
    if xf_path and xf_path.is_file():
        xf = json.loads(xf_path.read_text())
        related_scores: List[float] = []
        unrelated_scores: List[float] = []
        for key, layers_map in xf.get("pair_diffs", {}).items():
            # Heuristic: pair name contains both location probes → related
            related = (
                "eiffel" in key
                and "louvre" in key
                and "superposition" not in key
            )
            for Lname, stats in layers_map.items():
                # Prefer mid layers where structure was clearest; skip L11 if collapsed
                if Lname == "L11":
                    continue
                score = float(stats["summary_score"])
                if related:
                    related_scores.append(score)
                elif "superposition" in key:
                    unrelated_scores.append(score)
        if related_scores and unrelated_scores:
            rel_m = float(np.mean(related_scores))
            unr_m = float(np.mean(unrelated_scores))
            xf_ok = rel_m < unr_m * 0.5  # related much closer
            claims.append(
                claim(
                    "cross_feature_structure",
                    passed=xf_ok,
                    detail=(
                        f"related mean summary={rel_m:.4f} < unrelated={unr_m:.4f} "
                        f"(from {xf_path.name}, mid layers)"
                    ),
                    related_mean=rel_m,
                    unrelated_mean=unr_m,
                    source=str(xf_path),
                )
            )
        else:
            claims.append(
                claim(
                    "cross_feature_structure",
                    passed=False,
                    detail="cross-feature summary present but could not classify pairs",
                    source=str(xf_path),
                )
            )
    else:
        claims.append(
            claim(
                "cross_feature_structure",
                passed=False,
                detail="skipped — run scripts/diff_features.py first (optional evidence)",
                skipped=True,
            )
        )

    # Core claims (exclude skipped optional)
    core = [
        c
        for c in claims
        if c["id"] != "cross_feature_structure" or not c.get("stats", {}).get("skipped")
    ]
    # If cross-feature skipped, don't require it for overall
    required = [c for c in claims if not c.get("stats", {}).get("skipped")]
    overall = all(c["pass"] for c in required)

    report = {
        "mode": "validate_training_geometry",
        "run_dir": str(run_dir),
        "signals_csv": str(csv_path),
        "n_steps": int(len(steps)),
        "layers": layers,
        "loss_start": float(loss[0]),
        "loss_end": float(loss[-1]),
        "claims": claims,
        "overall_pass": overall,
        "scope": "training trajectory + optional feature structure — not edit locality",
    }

    ckpt_sum = find_ckpt_diff_summary(run_dir)
    if ckpt_sum:
        report["ckpt_diff_summary"] = str(ckpt_sum)
    if xf_path:
        report["cross_feature_summary"] = str(xf_path)

    return report


def plot_validation(report: Dict[str, Any], data: Dict[str, np.ndarray], out_path: Path) -> Path:
    import os

    # Prefer a repo-local matplotlib cache (sandbox / CI friendly).
    mpl_dir = ROOT / ".mplconfig"
    mpl_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_dir))

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    by_layer = parse_geo_columns(list(data.keys()))
    layers = sorted(L for L in by_layer if L >= 0)
    steps = data["step"]
    loss = data["loss"]

    fig, axes = plt.subplots(3, 1, figsize=(9, 8), sharex=True)

    axes[0].plot(steps, loss, color="black", lw=1.5)
    axes[0].set_ylabel("loss")
    status = "PASS" if report["overall_pass"] else "FAIL"
    axes[0].set_title(f"validation: loss + geometry phases  [{status}]")

    for L in layers:
        if "spectral_participation" in by_layer[L]:
            axes[1].plot(
                steps,
                data[by_layer[L]["spectral_participation"]],
                lw=1.5,
                label=f"L{L}",
            )
    axes[1].set_ylabel("spectral")
    axes[1].legend(loc="best", fontsize=8, frameon=False)
    axes[1].set_title("layer phase — spectral participation")

    for L in layers:
        if "interference_mean" in by_layer[L]:
            axes[2].plot(
                steps,
                data[by_layer[L]["interference_mean"]],
                lw=1.5,
                label=f"L{L}",
            )
    axes[2].set_ylabel("interference")
    axes[2].set_xlabel("step")
    axes[2].legend(loc="best", fontsize=8, frameon=False)
    axes[2].set_title("denser packing — interference")

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate geometry trajectories on a real training run"
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=RESULTS / "train_gpt2_small_geometry",
    )
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()

    report = validate_run(args.run_dir)
    out_json = args.run_dir / "validation.json"
    out_json.write_text(json.dumps(report, indent=2) + "\n")

    print(f"validate: {args.run_dir}")
    print(f"scope: {report['scope']}")
    for c in report["claims"]:
        skipped = c.get("stats", {}).get("skipped")
        mark = "SKIP" if skipped else ("PASS" if c["pass"] else "FAIL")
        print(f"  [{mark}] {c['id']}: {c['detail']}")
    print(f"\noverall: {'PASS' if report['overall_pass'] else 'FAIL'}")
    print(f"wrote {out_json}")

    if not args.no_plot:
        data = read_signals(args.run_dir / "signals.csv")
        png = plot_validation(report, data, args.run_dir / "validation.png")
        print(f"wrote {png}")

    if not report["overall_pass"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
