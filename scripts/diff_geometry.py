"""Diff superposition geometry for selected features / neighborhoods.

Modes:
  - pre_post: compare two snapshots (e.g. before vs after an edit)
  - step_vs_step: compare query geometry at step t vs step t+k (same bank)

Saves JSON under results/<run>/ and optional neighborhood bar plots.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import torch

from scripts.geometry import (
    _drop_self_neighbors,
    abs_cosine_sims,
    compute_geometry_metrics,
    topk_neighbor_indices,
)


def snapshot_neighborhood(
    query: torch.Tensor,
    bank: torch.Tensor,
    *,
    feature_id: str,
    top_k: int = 16,
    metrics: Optional[Sequence[str]] = None,
    neighbor_labels: Optional[Sequence[str]] = None,
    step: Optional[int] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Dump summary metrics + top-k neighbor |cos| for one feature vector."""
    metric_names = list(
        metrics
        or ["interference_mean", "spectral_participation", "coactivation_overlap"]
    )
    q = query.float().reshape(-1).cpu()
    b = _drop_self_neighbors(q, bank.float().cpu())
    summary = compute_geometry_metrics(q, b, top_k=top_k, metrics=metric_names)
    sims = abs_cosine_sims(q, b)
    idx = topk_neighbor_indices(q, b, top_k=top_k)
    neighbors: List[Dict[str, Any]] = []
    for rank, i in enumerate(idx.tolist()):
        label = None
        if neighbor_labels is not None and i < len(neighbor_labels):
            label = neighbor_labels[i]
        neighbors.append(
            {
                "rank": rank,
                "bank_index": i,
                "label": label,
                "abs_cos": float(sims[i].item()),
            }
        )
    snap: Dict[str, Any] = {
        "feature_id": feature_id,
        "step": step,
        "metrics": summary,
        "neighbors": neighbors,
        "query_norm": float(q.norm().item()),
        "bank_size": int(b.shape[0]),
    }
    if extra:
        snap["extra"] = extra
    return snap


def diff_snapshots(
    before: Dict[str, Any],
    after: Dict[str, Any],
    *,
    mode: str = "pre_post",
) -> Dict[str, Any]:
    """Compare two neighborhood snapshots; per-metric Δ + neighbor cos Δ."""
    keys = sorted(set(before["metrics"]) | set(after["metrics"]))
    delta_metrics = {
        k: float(after["metrics"].get(k, 0.0) - before["metrics"].get(k, 0.0)) for k in keys
    }
    before_cos = {n["bank_index"]: n["abs_cos"] for n in before.get("neighbors", [])}
    after_cos = {n["bank_index"]: n["abs_cos"] for n in after.get("neighbors", [])}
    shared = sorted(set(before_cos) & set(after_cos))
    neighbor_deltas = [
        {
            "bank_index": i,
            "abs_cos_before": before_cos[i],
            "abs_cos_after": after_cos[i],
            "delta_abs_cos": after_cos[i] - before_cos[i],
        }
        for i in shared
    ]
    # neighbors that entered / left top-k
    entered = sorted(set(after_cos) - set(before_cos))
    left = sorted(set(before_cos) - set(after_cos))

    summary_score = float(sum(abs(v) for v in delta_metrics.values()) / max(len(delta_metrics), 1))
    mean_abs_neighbor_delta = (
        float(sum(abs(n["delta_abs_cos"]) for n in neighbor_deltas) / len(neighbor_deltas))
        if neighbor_deltas
        else 0.0
    )

    return {
        "mode": mode,
        "feature_id_before": before.get("feature_id"),
        "feature_id_after": after.get("feature_id"),
        "step_before": before.get("step"),
        "step_after": after.get("step"),
        "metrics_before": before["metrics"],
        "metrics_after": after["metrics"],
        "delta_metrics": delta_metrics,
        "summary_score": summary_score,
        "mean_abs_neighbor_delta": mean_abs_neighbor_delta,
        "neighbor_deltas": neighbor_deltas,
        "neighbors_entered_topk": entered,
        "neighbors_left_topk": left,
        "before": before,
        "after": after,
    }


def diff_step_queries(
    query_t: torch.Tensor,
    query_tk: torch.Tensor,
    bank: torch.Tensor,
    *,
    feature_id: str,
    step_t: int,
    step_tk: int,
    top_k: int = 16,
    metrics: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    """Diff geometry of the same feature at step t vs t+k (fixed bank)."""
    before = snapshot_neighborhood(
        query_t, bank, feature_id=feature_id, top_k=top_k, metrics=metrics, step=step_t
    )
    after = snapshot_neighborhood(
        query_tk, bank, feature_id=feature_id, top_k=top_k, metrics=metrics, step=step_tk
    )
    return diff_snapshots(before, after, mode="step_vs_step")


def save_diff(diff: Dict[str, Any], path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(diff, indent=2) + "\n")
    return path


def plot_neighborhood_diff(diff: Dict[str, Any], out_path: Path) -> Path:
    """Bar chart: metric before/after + Δabs_cos for shared top-k neighbors."""
    import matplotlib.pyplot as plt

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    metrics = list(diff["delta_metrics"].keys())
    before_vals = [diff["metrics_before"][m] for m in metrics]
    after_vals = [diff["metrics_after"][m] for m in metrics]

    nbr = diff.get("neighbor_deltas", [])
    fig_h = 4.5 if not nbr else 7.5
    fig, axes = plt.subplots(2 if nbr else 1, 1, figsize=(9, fig_h))
    if not nbr:
        axes = [axes]

    x = range(len(metrics))
    width = 0.35
    if diff.get("mode") == "feature_a_vs_b":
        label_a = str(diff.get("feature_a") or diff.get("feature_id_before") or "A")
        label_b = str(diff.get("feature_b") or diff.get("feature_id_after") or "B")
        # Shorten @L tags for legend
        label_a = label_a.split("@")[0]
        label_b = label_b.split("@")[0]
    else:
        label_a, label_b = "before", "after"
    axes[0].bar([i - width / 2 for i in x], before_vals, width, label=label_a, color="#444")
    axes[0].bar([i + width / 2 for i in x], after_vals, width, label=label_b, color="#4C78A8")
    axes[0].set_xticks(list(x))
    axes[0].set_xticklabels(metrics, rotation=15, ha="right")
    axes[0].set_ylabel("metric")
    title_bits = f"diff [{diff.get('mode')}]"
    if diff.get("mode") == "feature_a_vs_b":
        title_bits += f"  {label_a} vs {label_b}"
    else:
        title_bits += f"  {diff.get('feature_id_before')}"
    if diff.get("layer") is not None:
        title_bits += f"  L{diff.get('layer')}"
    title_bits += f"  summary={diff.get('summary_score', 0):.3f}"
    axes[0].set_title(title_bits)
    axes[0].legend(frameon=False)

    if nbr:
        idxs = [str(n["bank_index"]) for n in nbr]
        deltas = [n["delta_abs_cos"] for n in nbr]
        colors = ["#B22222" if d > 0 else "#2E8B57" for d in deltas]
        axes[1].bar(idxs, deltas, color=colors)
        axes[1].axhline(0, color="black", lw=0.8)
        axes[1].set_xlabel("shared neighbor bank_index")
        axes[1].set_ylabel("Δ |cos|")
        axes[1].set_title("neighborhood |cos| change (shared top-k)")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


@torch.no_grad()
def extract_mlp_out_feature(model, prompt: str, layer: int, pos: int) -> torch.Tensor:
    hook = f"blocks.{layer}.hook_mlp_out"
    tokens = model.to_tokens(prompt)
    _, cache = model.run_with_cache(tokens, names_filter=lambda n: n == hook)
    return cache[hook][0, pos].detach().float().cpu()
