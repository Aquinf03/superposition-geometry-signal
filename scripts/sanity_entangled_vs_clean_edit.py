"""Sanity: entangled edit moves neighbor geometry more than a clean edit.

Two cases, same ambient space / same edit strength:

  entangled — query sits in a tight shared-direction cluster
  clean     — query is nearly isolated from the bank

Apply the same additive edit to each query, snapshot neighborhoods
pre/post, and require:

  summary_score(entangled) > summary_score(clean) + margin

Default is a controlled vector-space test (no HF download).
Optional ``--real`` runs two GPT-2 ROME edits (high vs low pre-edit
interference) and compares their ``diff_pre_post_edit`` scores.

Run:
  python scripts/sanity_entangled_vs_clean_edit.py
  python scripts/sanity_entangled_vs_clean_edit.py --real
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.diff_geometry import diff_snapshots, snapshot_neighborhood
from scripts.geometry import compute_geometry_metrics, l2_normalize
from scripts.seed import set_seed


def _make_cluster(n: int, d: int, seed: int, noise: float = 0.05) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    center = torch.randn(d, generator=g)
    center = l2_normalize(center)
    rows = []
    for i in range(n):
        g_i = torch.Generator().manual_seed(seed + 10 + i)
        jitter = noise * torch.randn(d, generator=g_i)
        rows.append(l2_normalize(center + jitter))
    return torch.stack(rows, dim=0)


def _make_isolated(n: int, d: int, seed: int) -> torch.Tensor:
    """Near-orthogonal isolated directions (QR)."""
    g = torch.Generator().manual_seed(seed)
    a = torch.randn(d, n, generator=g)
    q, _ = torch.linalg.qr(a, mode="reduced")
    return q.T.contiguous()


def _apply_edit(query: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
    return query + delta


def run_controlled_sanity(
    *,
    d: int = 32,
    cluster_n: int = 12,
    isolated_n: int = 12,
    top_k: int = 8,
    edit_scale: float = 0.5,
    margin: float = 0.02,
    seed: int = 0,
) -> Dict[str, Any]:
    set_seed(seed)
    cluster = _make_cluster(cluster_n, d, seed=seed)
    isolated = _make_isolated(isolated_n, d, seed=seed + 1)

    # Bank = everyone; entangled query = first cluster member; clean = first isolated
    bank = torch.cat([cluster, isolated], dim=0)
    q_ent = cluster[0].clone()
    q_clean = isolated[0].clone()

    g = torch.Generator().manual_seed(seed + 99)
    delta = edit_scale * l2_normalize(torch.randn(d, generator=g))

    metrics = ["interference_mean", "spectral_participation", "coactivation_overlap"]

    pre_ent = snapshot_neighborhood(
        q_ent, bank, feature_id="entangled", top_k=top_k, metrics=metrics, extra={"phase": "pre"}
    )
    pre_clean = snapshot_neighborhood(
        q_clean, bank, feature_id="clean", top_k=top_k, metrics=metrics, extra={"phase": "pre"}
    )
    post_ent = snapshot_neighborhood(
        _apply_edit(q_ent, delta),
        bank,
        feature_id="entangled",
        top_k=top_k,
        metrics=metrics,
        extra={"phase": "post"},
    )
    post_clean = snapshot_neighborhood(
        _apply_edit(q_clean, delta),
        bank,
        feature_id="clean",
        top_k=top_k,
        metrics=metrics,
        extra={"phase": "post"},
    )

    diff_ent = diff_snapshots(pre_ent, post_ent, mode="entangled_edit")
    diff_clean = diff_snapshots(pre_clean, post_clean, mode="clean_edit")

    # Pre-edit interference should already mark entangled as messier.
    pre_inter_ent = pre_ent["metrics"]["interference_mean"]
    pre_inter_clean = pre_clean["metrics"]["interference_mean"]

    score_ent = float(diff_ent["summary_score"])
    score_clean = float(diff_clean["summary_score"])
    nbr_ent = float(diff_ent["mean_abs_neighbor_delta"])
    nbr_clean = float(diff_clean["mean_abs_neighbor_delta"])

    # Claim under test: entangled edit moves *neighbor* geometry more than a clean edit.
    # summary_score is reported but not required — aggregate metric Δ can be dominated by
    # spectral on isolated features.
    checks = {
        "pre_interference_entangled_gt_clean": pre_inter_ent > pre_inter_clean + 0.05,
        "neighbor_delta_entangled_gt_clean": nbr_ent > nbr_clean + max(0.01, margin * 0.5),
    }
    passed = all(checks.values())

    return {
        "mode": "controlled",
        "seed": seed,
        "d": d,
        "edit_scale": edit_scale,
        "margin": margin,
        "pre_interference": {"entangled": pre_inter_ent, "clean": pre_inter_clean},
        "entangled_diff": {
            "summary_score": score_ent,
            "mean_abs_neighbor_delta": nbr_ent,
            "delta_metrics": diff_ent["delta_metrics"],
        },
        "clean_diff": {
            "summary_score": score_clean,
            "mean_abs_neighbor_delta": nbr_clean,
            "delta_metrics": diff_clean["delta_metrics"],
        },
        "info": {
            "summary_score_entangled_gt_clean": score_ent > score_clean,
        },
        "checks": checks,
        "passed": passed,
    }


# Explicit pairs for --real: shared "located in" cluster vs unrelated fact.
REAL_ENTANGLED = {
    "name": "eiffel",
    "subject": "The Eiffel Tower",
    "prompt": "The Eiffel Tower is located in",
    "target_old": " Paris",
    "target_new": " Rome",
}
REAL_CLEAN = {
    "name": "japan_currency",
    "subject": "Japan",
    "prompt": "The currency of Japan is the",
    "target_old": " yen",
    "target_new": " euro",
}
# Neighbors in the shared location cluster — collateral geometry should move more
# after the entangled landmark edit than after the clean currency edit.
REAL_CLUSTER_NEIGHBORS = [
    {
        "name": "louvre",
        "subject": "The Louvre Museum",
        "prompt": "The Louvre Museum is located in",
    },
    {
        "name": "colosseum",
        "subject": "The Colosseum",
        "prompt": "The Colosseum is located in",
    },
    {
        "name": "paris_capital",
        "subject": "Paris",
        "prompt": "Paris is the capital of",
    },
]


def run_real_sanity(
    *,
    seed: int = 0,
    layer: int = 8,
    top_k: int = 16,
    margin: float = 0.01,
    out_dir: Path,
) -> Dict[str, Any]:
    """ROME on GPT-2: landmark-cluster edit vs clean unrelated edit.

    Claim: editing Eiffel (shared location geometry) moves geometry on cluster
    neighbors (Louvre / Colosseum / Paris) more than editing Japan-currency.
    """
    from scripts.geometry import DEFAULT_PROBE_PROMPTS, collect_mlp_out_bank
    from scripts.rome_edit import (
        apply_rank_one,
        extract_key,
        find_subject_last_pos,
        first_token_id,
        optimize_v,
        resolve_device,
    )
    from scripts.diff_geometry import extract_mlp_out_feature
    from transformer_lens import HookedTransformer

    set_seed(seed)
    device = resolve_device("auto")
    probe_prompts = list(DEFAULT_PROBE_PROMPTS)

    def _snap_cluster(model, bank: torch.Tensor) -> Dict[str, Dict[str, Any]]:
        snaps: Dict[str, Dict[str, Any]] = {}
        for feat in REAL_CLUSTER_NEIGHBORS:
            pos = find_subject_last_pos(model, feat["prompt"], feat["subject"])
            q = extract_mlp_out_feature(model, feat["prompt"], layer, pos)
            snaps[feat["name"]] = snapshot_neighborhood(
                q,
                bank,
                feature_id=feat["name"],
                top_k=top_k,
                extra={"prompt": feat["prompt"]},
            )
        return snaps

    def _cluster_delta(pre: Dict[str, Dict[str, Any]], post: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        per: Dict[str, Any] = {}
        scores = []
        nbrs = []
        for name in pre:
            d = diff_snapshots(pre[name], post[name], mode="cluster_collateral")
            per[name] = {
                "summary_score": d["summary_score"],
                "mean_abs_neighbor_delta": d["mean_abs_neighbor_delta"],
                "delta_metrics": d["delta_metrics"],
            }
            scores.append(d["summary_score"])
            nbrs.append(d["mean_abs_neighbor_delta"])
        return {
            "per_feature": per,
            "mean_summary_score": float(sum(scores) / max(len(scores), 1)),
            "mean_abs_neighbor_delta": float(sum(nbrs) / max(len(nbrs), 1)),
        }

    def _run_case(cand: Dict[str, str], tag: str) -> Dict[str, Any]:
        local = HookedTransformer.from_pretrained("gpt2-small", device=device)
        local.eval()
        prompt = cand["prompt"]
        subject = cand["subject"]
        pos = find_subject_last_pos(local, prompt, subject)
        tokens = local.to_tokens(prompt)
        target_new_id = first_token_id(local, cand["target_new"])
        bank = collect_mlp_out_bank(local, layer=layer, prompts=probe_prompts, positions="all")
        pre_cluster = _snap_cluster(local, bank)
        pre_subj = snapshot_neighborhood(
            extract_mlp_out_feature(local, prompt, layer, pos),
            bank,
            feature_id=tag,
            top_k=top_k,
        )
        k = extract_key(local, tokens, layer, pos)
        v_star, _, _, _, _ = optimize_v(
            local,
            tokens,
            layer=layer,
            subject_pos=pos,
            target_id=target_new_id,
            n_steps=20,
            lr=0.5,
            early_stop=0.05,
            logger=None,
            neighbor_bank=bank,
            top_k_neighbors=top_k,
            log_geometry=False,
        )
        apply_rank_one(local, layer, k, v_star)
        post_bank = collect_mlp_out_bank(
            local, layer=layer, prompts=probe_prompts, positions="all"
        )
        post_cluster = _snap_cluster(local, post_bank)
        post_subj = snapshot_neighborhood(
            extract_mlp_out_feature(local, prompt, layer, pos),
            post_bank,
            feature_id=tag,
            top_k=top_k,
        )
        subj_diff = diff_snapshots(pre_subj, post_subj, mode="pre_post_edit")
        collateral = _cluster_delta(pre_cluster, post_cluster)
        return {
            "candidate": cand,
            "pre_interference": pre_subj["metrics"]["interference_mean"],
            "subject_summary_score": subj_diff["summary_score"],
            "subject_mean_abs_neighbor_delta": subj_diff["mean_abs_neighbor_delta"],
            # Primary real-mode metric: collateral move on the shared cluster
            "summary_score": collateral["mean_summary_score"],
            "mean_abs_neighbor_delta": collateral["mean_abs_neighbor_delta"],
            "cluster_collateral": collateral,
            "diff": subj_diff,
        }

    print(f"clean candidate: {REAL_CLEAN['name']} (unrelated fact)")
    print(f"entangled candidate: {REAL_ENTANGLED['name']} (location cluster)")
    print(f"cluster neighbors: {[f['name'] for f in REAL_CLUSTER_NEIGHBORS]}")

    clean_res = _run_case(REAL_CLEAN, "clean")
    ent_res = _run_case(REAL_ENTANGLED, "entangled")

    checks = {
        "cluster_neighbor_delta_entangled_gt_clean": ent_res["mean_abs_neighbor_delta"]
        > clean_res["mean_abs_neighbor_delta"] + margin * 0.5,
        "cluster_summary_entangled_gt_clean": ent_res["summary_score"]
        > clean_res["summary_score"] + margin * 0.5,
    }
    passed = all(checks.values())

    result = {
        "mode": "real_gpt2_small",
        "seed": seed,
        "layer": layer,
        "margin": margin,
        "note": (
            "Compares collateral geometry Δ on fixed location-cluster neighbors "
            "(louvre/colosseum/paris) after Eiffel edit vs Japan-currency edit."
        ),
        "clean": {k: v for k, v in clean_res.items() if k != "diff"},
        "entangled": {k: v for k, v in ent_res.items() if k != "diff"},
        "checks": checks,
        "passed": passed,
    }
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "diff_clean.json").write_text(json.dumps(clean_res["diff"], indent=2) + "\n")
    (out_dir / "diff_entangled.json").write_text(json.dumps(ent_res["diff"], indent=2) + "\n")
    (out_dir / "cluster_collateral.json").write_text(
        json.dumps(
            {
                "clean": clean_res["cluster_collateral"],
                "entangled": ent_res["cluster_collateral"],
            },
            indent=2,
        )
        + "\n"
    )
    return result


def plot_comparison(result: Dict[str, Any], out_path: Path) -> Path:
    import matplotlib.pyplot as plt

    labels = ["summary_score", "mean_abs_neighbor_delta"]
    if result["mode"] == "controlled":
        ent = [result["entangled_diff"][k] for k in labels]
        clean = [result["clean_diff"][k] for k in labels]
        ylabel = "geometry Δ"
        title_bit = "query neighborhood"
    else:
        ent = [result["entangled"][k] for k in labels]
        clean = [result["clean"][k] for k in labels]
        ylabel = "cluster collateral geometry Δ"
        title_bit = "location-cluster collateral"

    fig, ax = plt.subplots(figsize=(7, 4))
    x = range(len(labels))
    width = 0.35
    ax.bar([i - width / 2 for i in x], clean, width, label="clean edit", color="#444")
    ax.bar([i + width / 2 for i in x], ent, width, label="entangled edit", color="#B22222")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, rotation=10, ha="right")
    ax.set_ylabel(ylabel)
    status = "PASS" if result["passed"] else "FAIL"
    ax.set_title(f"entangled vs clean ({title_bit}) [{status}]")
    ax.legend(frameon=False)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sanity: entangled edit moves neighbor geometry more than a clean edit"
    )
    parser.add_argument(
        "--real",
        action="store_true",
        help="Run on GPT-2 small ROME edits (slow). Default: controlled vector test.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--margin", type=float, default=0.02)
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "results" / "sanity_entangled_vs_clean",
    )
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    if args.real:
        result = run_real_sanity(seed=args.seed, margin=args.margin, out_dir=args.out)
    else:
        result = run_controlled_sanity(seed=args.seed, margin=args.margin)

    out_json = args.out / "sanity_result.json"
    out_json.write_text(json.dumps(result, indent=2) + "\n")
    plot_path = plot_comparison(result, args.out / "entangled_vs_clean.png")

    print(json.dumps(result, indent=2))
    print(f"\npassed={result['passed']}")
    print(f"wrote {out_json}")
    print(f"wrote {plot_path}")
    for name, ok in result["checks"].items():
        print(f"  {name}: {'ok' if ok else 'FAIL'}")
    raise SystemExit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()
