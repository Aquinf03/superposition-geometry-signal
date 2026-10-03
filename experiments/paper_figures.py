#!/usr/bin/env python3
"""Generate plain paper figures (white bg, labeled axes only).

Primary source: Qwen2.5-7B m★ matched seed-0 pair (Modal).

  python experiments/paper_figures.py
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

OUT = ROOT / "paper" / "figures"

INK = "#141414"
MUTED = "#666666"
GRID = "#E0E0E0"
TEAL = "#0F766E"
ROSE = "#BE123C"
# Qwen2.5-7B watch layers (early / mid / late)
L7 = "#1E3A5F"
L14 = "#B45309"
L26 = "#166534"
LAYER_COLORS = {7: L7, 14: L14, 26: L26}
CONTROL_LAYER = 14


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


def _detect_layers(d: Dict[str, List[float]]) -> List[int]:
    layers = sorted(
        {
            int(k.split("_")[1][1:])
            for k in d
            if k.startswith("geometry_L") and k.endswith("_interference_mean")
        }
    )
    return layers or list(LAYER_COLORS.keys())


def fig01_live_track(baseline_csv: Path) -> List[Path]:
    plt = _setup_mpl()
    d = _read_csv(baseline_csv)
    steps = d["step"]
    layers = _detect_layers(d)
    colors = {**LAYER_COLORS, **{L: LAYER_COLORS.get(L, INK) for L in layers}}
    fig, axes = plt.subplots(4, 1, figsize=(7.5, 8.0), sharex=True, gridspec_kw={"hspace": 0.22})

    axes[0].plot(steps, d["loss"], color=INK, lw=1.6)
    axes[0].set_ylabel("loss")
    _clean(axes[0])

    for ax, metric, ylabel in zip(
        axes[1:],
        ["interference_mean", "spectral_participation", "coactivation_overlap"],
        ["interference", "spectral", "coactivation"],
    ):
        for L in layers:
            col = f"geometry_L{L}_{metric}"
            if col in d:
                ax.plot(steps, d[col], color=colors[L], lw=1.5, label=f"L{L}")
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
    warmup_steps: int = 20,
    control_layer: int = CONTROL_LAYER,
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

    col = f"geometry_L{control_layer}_interference_mean"
    axes[1].axhline(target, color=TEAL, lw=1.0, ls="--", label=f"target {target}")
    axes[1].plot(b["step"], b[col], color=MUTED, lw=1.5, label=f"baseline L{control_layer}")
    axes[1].plot(c["step"], c[col], color=ROSE, lw=1.6, label=f"control L{control_layer}")
    axes[1].axvspan(0, warmup_steps, color=GRID, alpha=0.5, lw=0, label="warmup")
    axes[1].set_ylabel(f"L{control_layer} interference")
    axes[1].set_xlabel("step")
    axes[1].legend(loc="upper right", fontsize=8, ncol=2)
    _clean(axes[1])

    fig.tight_layout()
    return _save(fig, "fig02_control_vs_baseline")


def fig03_layer_phase(baseline_csv: Path) -> List[Path]:
    plt = _setup_mpl()
    d = _read_csv(baseline_csv)
    steps = d["step"]
    layers = _detect_layers(d)
    colors = {**LAYER_COLORS, **{L: LAYER_COLORS.get(L, INK) for L in layers}}
    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    for L in layers:
        ax.plot(
            steps,
            d[f"geometry_L{L}_spectral_participation"],
            color=colors[L],
            lw=1.6,
            label=f"L{L}",
        )
    ax.set_ylabel("spectral participation")
    ax.set_xlabel("step")
    ax.legend(loc="upper right", fontsize=9)
    _clean(ax)
    fig.tight_layout()
    return _save(fig, "fig03_layer_phase")


def _mid_layer_keys(pair_block: dict) -> List[str]:
    layers = sorted(int(str(k).lstrip("L")) for k in pair_block.keys())
    if not layers:
        return []
    late = max(layers)
    return [f"L{L}" for L in layers if L != late]


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
        keys = _mid_layer_keys(block)
        vals = [float(block[L]["summary_score"]) for L in keys if L in block]
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
        "Matched Qwen2.5-7B control vs baseline (seed 0, m★). Out-of-metric validation suite.",
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


def fig06_compare_cards(compare_path: Path, control_layer: int = CONTROL_LAYER) -> List[Path]:
    plt = _setup_mpl()
    rep = json.loads(compare_path.read_text())
    b, c, cmp_ = rep["baseline"], rep["control"], rep["compare"]
    layer = int(cmp_.get("control_layer") or b.get("control_layer") or control_layer)
    fig, axes = plt.subplots(1, 3, figsize=(9.0, 3.2))

    cards = [
        ("final loss", b["loss_last"], c["loss_last"]),
        (
            f"L{layer} interference",
            b.get("target_interference_last", b.get("L8_interference_last")),
            c.get("target_interference_last", c.get("L8_interference_last")),
        ),
        ("|m - m*|", cmp_["abs_err_baseline_to_target"], cmp_["abs_err_control_to_target"]),
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


def table06_scale(summaries: Sequence[Tuple[str, Path]]) -> List[Path]:
    """Model-scale held-out Δ table from Modal summary JSONs."""
    OUT.mkdir(parents=True, exist_ok=True)
    md_path = OUT / "table06_scale.md"
    csv_path = OUT / "table06_scale.csv"

    lines = [
        "# Table 6 — Model-scale held-out LM Δ (control − baseline)",
        "",
        "Negative Δ means control wins. Extended corpus; matched seed 0, 200 steps.",
        "",
        "| Model | Eval Δ | Pack |m−m*| base | Pack |m−m*| ctrl | Closer | Held-out win |",
        "| --- | ---: | ---: | ---: | --- | --- |",
    ]
    csv_lines = [
        "model,delta_eval_loss,abs_err_baseline,abs_err_control,closer,held_out_win"
    ]

    for label, path in summaries:
        s = json.loads(path.read_text())
        target = float(s.get("control_target", 0.35))
        layer = int(s.get("control_layer", 14))
        b = (s.get("baseline_final_geometry") or {}).get(f"L{layer}_interference_mean")
        c = (s.get("control_final_geometry") or {}).get(f"L{layer}_interference_mean")
        err_b = s.get("abs_err_baseline_to_target")
        err_c = s.get("abs_err_control_to_target")
        if err_b is None and b is not None:
            err_b = abs(float(b) - target)
        if err_c is None and c is not None:
            err_c = abs(float(c) - target)
        delta = float(s["delta_eval_loss"])
        closer = s.get("control_closer_to_target")
        if closer is None and err_b is not None and err_c is not None:
            closer = err_c < err_b
        win = bool(s.get("control_wins_vs_baseline", delta < 0))
        sign = f"{delta:+.3f}"
        lines.append(
            f"| {label} | {sign} | "
            f"{err_b:.3f} | {err_c:.3f} | "
            f"{'yes' if closer else 'no'} | {'yes' if win else 'no'} |"
        )
        csv_lines.append(
            f"\"{label}\",{delta},{err_b},{err_c},{closer},{win}"
        )

    lines += [
        "",
        "Sources: `experiments/results/modal_pull/summary_*.json`.",
        "",
    ]
    md_path.write_text("\n".join(lines), encoding="utf-8")
    csv_path.write_text("\n".join(csv_lines) + "\n", encoding="utf-8")
    return [md_path, csv_path]


def main() -> None:
    results = ROOT / "experiments" / "results" / "modal_pull"
    base = results / "train_qwen257b_geometry_mstar_s0"
    ctrl = results / "train_qwen257b_geometry_control_mstar_s0"
    pair = results / "qwen257b_mstar_pair_s0"
    # Final aligned ckpt for 200-step runs.
    diff_step = "00199"

    written: List[Path] = []
    written += fig01_live_track(base / "signals.csv")
    written += fig02_control_vs_baseline(
        base / "signals.csv",
        ctrl / "signals.csv",
        warmup_steps=20,
        control_layer=CONTROL_LAYER,
    )
    written += fig03_layer_phase(base / "signals.csv")

    xf = base / f"diff_features_step_{diff_step}" / "diff_summary.json"
    if xf.is_file():
        written += fig04_cross_feature(xf)
        related_png = (
            base
            / f"diff_features_step_{diff_step}"
            / "diff_eiffel_located_in_vs_louvre_located_in_L14.png"
        )
        unrelated_png = (
            base
            / f"diff_features_step_{diff_step}"
            / "diff_eiffel_located_in_vs_superposition_pack_L14.png"
        )
        if related_png.is_file() and unrelated_png.is_file():
            written += fig07_neighborhood_stills(related_png, unrelated_png)

    val = pair / "out_of_metric_validation.json"
    if val.is_file():
        written += table05_validation(val)

    cmp = pair / "compare.json"
    if cmp.is_file():
        written += fig06_compare_cards(cmp, control_layer=CONTROL_LAYER)
    else:
        # Build compare locally from signals if postprocess not pulled yet.
        from experiments.compare_control_vs_baseline import compare, summarize_run

        pair.mkdir(parents=True, exist_ok=True)
        payload = {
            "baseline": summarize_run(base),
            "control": summarize_run(ctrl),
            "compare": None,
            "note": "local compare from pulled signals",
        }
        payload["compare"] = compare(payload["baseline"], payload["control"])
        cmp.write_text(json.dumps(payload, indent=2) + "\n")
        written += fig06_compare_cards(cmp, control_layer=CONTROL_LAYER)

    scale_inputs = []
    s15 = results / "summary_s0.json"
    s7 = results / "summary_qwen257b_mstar_s0.json"
    if s15.is_file():
        scale_inputs.append(("Qwen2.5-1.5B", s15))
    if s7.is_file():
        scale_inputs.append(("Qwen2.5-7B (m★)", s7))
    if scale_inputs:
        written += table06_scale(scale_inputs)

    print(f"wrote {len(written)} files -> {OUT}")
    for p in written:
        print(f"  {p.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
