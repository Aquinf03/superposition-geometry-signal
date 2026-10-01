#!/usr/bin/env python3
"""Generate plain paper figures (white bg, labeled axes only).

  python experiments/paper_figures.py
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

OUT = ROOT / "paper" / "figures"

INK = "#141414"
MUTED = "#666666"
GRID = "#E0E0E0"
TEAL = "#0F766E"
ROSE = "#BE123C"
L4 = "#1E3A5F"
L8 = "#B45309"
L11 = "#166534"
LAYER_COLORS = {4: L4, 8: L8, 11: L11}


def _setup_mpl():
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    mpl.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "savefig.edgecolor": "white",
            "savefig.dpi": 300,
            "font.family": "sans-serif",
            "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
            "font.size": 10,
            "axes.edgecolor": INK,
            "axes.linewidth": 0.8,
            "axes.labelcolor": INK,
            "xtick.color": INK,
            "ytick.color": INK,
            "text.color": INK,
            "grid.color": GRID,
            "grid.linewidth": 0.5,
            "legend.frameon": False,
        }
    )
    return plt


def _read_csv(path: Path) -> Dict[str, List[float]]:
    with path.open() as f:
        rows = list(csv.DictReader(f))
    out: Dict[str, List[float]] = {}
    for col in rows[0].keys():
        vals = []
        ok = True
        for r in rows:
            raw = r.get(col)
            if raw in (None, ""):
                vals.append(float("nan"))
                continue
            try:
                vals.append(float(raw))
            except ValueError:
                ok = False
                break
        if ok:
            out[col] = vals
    return out


def _clean(ax) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, axis="y", alpha=0.7)
    ax.set_axisbelow(True)


def _save(fig, stem: str) -> List[Path]:
    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / f"{stem}.png"
    fig.savefig(p, bbox_inches="tight", pad_inches=0.15, facecolor="white", edgecolor="white")
    return [p]


def fig01_live_track(baseline_csv: Path) -> List[Path]:
    plt = _setup_mpl()
    d = _read_csv(baseline_csv)
    steps = d["step"]
    fig, axes = plt.subplots(4, 1, figsize=(7.5, 8.0), sharex=True, gridspec_kw={"hspace": 0.22})

    axes[0].plot(steps, d["loss"], color=INK, lw=1.6)
    axes[0].set_ylabel("loss")
    _clean(axes[0])

    for ax, metric, ylabel in zip(
        axes[1:],
        ["interference_mean", "spectral_participation", "coactivation_overlap"],
        ["interference", "spectral", "coactivation"],
    ):
        for L, c in LAYER_COLORS.items():
            col = f"geometry_L{L}_{metric}"
            if col in d:
                ax.plot(steps, d[col], color=c, lw=1.5, label=f"L{L}")
        ax.set_ylabel(ylabel)
        ax.legend(loc="upper right", ncol=3, fontsize=8, handlelength=1.2)
        _clean(ax)

    axes[-1].set_xlabel("step")
    fig.tight_layout()
    return _save(fig, "fig01_live_track")


def fig02_control_vs_baseline(
    base_csv: Path,
    ctrl_csv: Path,
    target: float = 0.35,
    warmup_steps: int = 25,
) -> List[Path]:
    plt = _setup_mpl()
    b = _read_csv(base_csv)
    c = _read_csv(ctrl_csv)
    fig, axes = plt.subplots(2, 1, figsize=(7.5, 5.2), sharex=True, gridspec_kw={"hspace": 0.25})

    axes[0].plot(b["step"], b["loss"], color=MUTED, lw=1.5, label="baseline")
    axes[0].plot(c["step"], c["loss"], color=ROSE, lw=1.6, label="control")
    axes[0].set_ylabel("loss")
    axes[0].legend(loc="upper right", fontsize=9)
    _clean(axes[0])

    col = "geometry_L8_interference_mean"
    axes[1].axhline(target, color=TEAL, lw=1.0, ls="--", label=f"target {target}")
    axes[1].plot(b["step"], b[col], color=MUTED, lw=1.5, label="baseline L8")
    axes[1].plot(c["step"], c[col], color=ROSE, lw=1.6, label="control L8")
    axes[1].axvspan(0, warmup_steps, color=GRID, alpha=0.5, lw=0, label="warmup")
    axes[1].set_ylabel("L8 interference")
    axes[1].set_xlabel("step")
    axes[1].legend(loc="upper right", fontsize=8, ncol=2)
    _clean(axes[1])

    fig.tight_layout()
    return _save(fig, "fig02_control_vs_baseline")


def fig03_layer_phase(baseline_csv: Path) -> List[Path]:
    plt = _setup_mpl()
    d = _read_csv(baseline_csv)
    steps = d["step"]
    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    for L, c in LAYER_COLORS.items():
        ax.plot(steps, d[f"geometry_L{L}_spectral_participation"], color=c, lw=1.6, label=f"L{L}")
    ax.set_ylabel("spectral participation")
    ax.set_xlabel("step")
    ax.legend(loc="upper right", fontsize=9)
    _clean(ax)
    fig.tight_layout()
    return _save(fig, "fig03_layer_phase")


def fig04_cross_feature(summary_path: Path) -> List[Path]:
    plt = _setup_mpl()
    s = json.loads(summary_path.read_text())
    related_key = "eiffel_located_in__vs__louvre_located_in"
    unrelated_keys = [
        "eiffel_located_in__vs__superposition_pack",
        "louvre_located_in__vs__superposition_pack",
    ]
    pairs = s["pair_diffs"]

    def mean_mid(key: str) -> float:
        block = pairs[key]
        vals = [float(block[L]["summary_score"]) for L in ("L4", "L8") if L in block]
        return sum(vals) / max(len(vals), 1)

    related = mean_mid(related_key)
    unrelated = sum(mean_mid(k) for k in unrelated_keys) / len(unrelated_keys)

    fig, ax = plt.subplots(figsize=(5.5, 3.8))
    labels = ["related", "unrelated"]
    vals = [related, unrelated]
    bars = ax.bar(labels, vals, color=[TEAL, ROSE], width=0.55)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, v, f"{v:.3f}", ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("neighborhood delta")
    ax.set_ylim(0, max(vals) * 1.2)
    _clean(ax)
    fig.tight_layout()
    return _save(fig, "fig04_cross_feature")


def table05_validation(val_path: Path) -> List[Path]:
    report = json.loads(val_path.read_text())
    claims = report.get("claims") or []
    primary_ids = [
        "downstream_loss",
        "cross_feature_baseline",
        "cross_feature_control",
        "cross_feature_preserved_under_control",
        "held_out_not_gamed",
        "held_out_not_regressed",
        "control_trajectory_checks",
    ]
    by_id = {c.get("id"): c for c in claims}
    rows = [by_id[i] for i in primary_ids if i in by_id] or claims[:8]

    OUT.mkdir(parents=True, exist_ok=True)
    md_path = OUT / "table05_validation.md"
    csv_path = OUT / "table05_validation.csv"

    lines = [
        "# Table 5. Out-of-metric validation",
        "",
        "Matched control vs baseline (seed 0). Out-of-metric validation suite.",
        "",
        "| Claim | Result | Detail |",
        "| --- | --- | --- |",
    ]
    csv_lines = ["claim,result,detail"]
    for c in rows:
        cid = str(c.get("id", "")).replace("_", " ")
        result = "PASS" if c.get("pass") else "FAIL"
        detail = str(c.get("detail", "")).replace("|", "/").replace("\n", " ")
        lines.append(f"| {cid} | **{result}** | {detail} |")
        csv_lines.append(f"\"{cid}\",{result},\"{detail}\"")

    n_pass = sum(1 for c in rows if c.get("pass"))
    lines += ["", f"**{n_pass}/{len(rows)} claims pass.**", ""]
    md_path.write_text("\n".join(lines), encoding="utf-8")
    csv_path.write_text("\n".join(csv_lines) + "\n", encoding="utf-8")
    return [md_path, csv_path]


def fig06_compare_cards(compare_path: Path) -> List[Path]:
    plt = _setup_mpl()
    rep = json.loads(compare_path.read_text())
    b, c, cmp_ = rep["baseline"], rep["control"], rep["compare"]
    fig, axes = plt.subplots(1, 3, figsize=(9.0, 3.2))

    cards = [
        ("final loss", b["loss_last"], c["loss_last"]),
        ("L8 interference", b["L8_interference_last"], c["L8_interference_last"]),
        ("|L8 - target|", cmp_["abs_err_baseline_to_target"], cmp_["abs_err_control_to_target"]),
    ]
    for ax, (title, bv, cv) in zip(axes, cards):
        ax.bar([0], [bv], color=MUTED, width=0.55, label="baseline")
        ax.bar([1], [cv], color=ROSE, width=0.55, label="control")
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["baseline", "control"])
        ax.set_ylabel(title)
        for x, v in ((0, bv), (1, cv)):
            ax.text(x, v, f"{v:.3f}", ha="center", va="bottom", fontsize=8)
        _clean(ax)
        if ax is axes[0]:
            ax.legend(loc="upper right", fontsize=8)

    fig.tight_layout()
    return _save(fig, "fig06_matched_pair")


def fig07_neighborhood_stills(diff_png: Path, unrelated_png: Path) -> List[Path]:
    plt = _setup_mpl()
    import matplotlib.image as mpimg

    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.0))
    for ax, path, tag in zip(
        axes,
        [diff_png, unrelated_png],
        ["related (Eiffel / Louvre)", "unrelated (Eiffel / superposition)"],
    ):
        ax.imshow(mpimg.imread(path))
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)
        ax.set_xlabel(tag, fontsize=9)

    fig.tight_layout()
    return _save(fig, "fig07_neighborhood_stills")


def main() -> None:
    # Primary paper figures: 500-step paper-scale matched seed-0 pair (gpt2-small).
    results = ROOT / "experiments" / "results"
    base = results / "train_gpt2_small_geometry_paper_s0"
    ctrl = results / "train_gpt2_small_geometry_paper_control_s0"
    pair = results / "paper_scale_pair_s0"
    diff_step = "00499"

    written: List[Path] = []
    written += fig01_live_track(base / "signals.csv")
    written += fig02_control_vs_baseline(
        base / "signals.csv", ctrl / "signals.csv", warmup_steps=25
    )
    written += fig03_layer_phase(base / "signals.csv")
    written += fig04_cross_feature(base / f"diff_features_step_{diff_step}" / "diff_summary.json")
    written += table05_validation(pair / "out_of_metric_validation.json")
    written += fig06_compare_cards(pair / "compare.json")
    written += fig07_neighborhood_stills(
        base / f"diff_features_step_{diff_step}" / "diff_eiffel_located_in_vs_louvre_located_in_L8.png",
        base / f"diff_features_step_{diff_step}" / "diff_eiffel_located_in_vs_superposition_pack_L8.png",
    )

    import shutil

    for src, name in [
        (base / "loss_geometry.png", "archive_baseline_dashboard.png"),
        (ctrl / "loss_geometry.png", "archive_control_dashboard.png"),
    ]:
        if src.is_file():
            dst = OUT / name
            shutil.copy2(src, dst)
            written.append(dst)

    for pdf in OUT.glob("*.pdf"):
        pdf.unlink()

    print(f"wrote {len(written)} files -> {OUT}")
    for p in written:
        print(f"  {p.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
