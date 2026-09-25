"""Sanity check: geometry metrics rise when superposition is forced denser.

Builds two toy feature packs in the same ambient dim ``d``:

  sparse  — n_features <= d, near-orthogonal directions (little superposition)
  dense   — n_features >> d, random unit directions (forced superposition)

For each feature as query vs the rest as bank, compute metrics A/B/C and
compare sparse vs dense. Expect denser packing ⇒ higher interference_mean
and coactivation_overlap; spectral_participation should also move up.

No GPU / HF download needed.

Run (you run this):
  python scripts/tests/toy_superposition_sanity.py
  python scripts/tests/toy_superposition_sanity.py --out experiments/results/toy_superposition_sanity
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.helpers.geometry import compute_geometry_metrics
from scripts.helpers.paths import RESULTS
from scripts.helpers.seed import set_seed
from scripts.helpers.signals import SignalLogger


def near_orthogonal_features(n_features: int, d: int, seed: int) -> torch.Tensor:
    """n_features <= d axis-aligned unit vectors (no coordinate overlap)."""
    if n_features > d:
        raise ValueError("orthogonal pack requires n_features <= d")
    # Identity axes = zero coactivation overlap between distinct features.
    # Tiny signed jitter (deterministic) avoids exact-zero edge cases elsewhere.
    g = torch.Generator().manual_seed(seed)
    w = torch.eye(d)[:n_features].contiguous()
    jitter = 1e-6 * torch.randn(n_features, d, generator=g)
    w = w + jitter
    w = w / w.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    return w


def dense_superposed_features(n_features: int, d: int, seed: int) -> torch.Tensor:
    """n_features >> d random unit vectors — forced packing / interference."""
    g = torch.Generator().manual_seed(seed)
    w = torch.randn(n_features, d, generator=g)
    w = w / w.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    return w


def mean_metrics_over_features(
    features: torch.Tensor,
    top_k: int = 16,
    top_dims: int | None = None,
) -> Dict[str, float]:
    """Average geometry metrics treating each row as query, others as bank."""
    n, d = features.shape
    if top_dims is None:
        top_dims = max(2, d // 4)
    acc = {"interference_mean": 0.0, "spectral_participation": 0.0, "coactivation_overlap": 0.0}
    count = 0
    for i in range(n):
        query = features[i]
        bank = torch.cat([features[:i], features[i + 1 :]], dim=0)
        if bank.shape[0] == 0:
            continue
        m = compute_geometry_metrics(
            query,
            bank,
            top_k=min(top_k, bank.shape[0]),
            top_dims=top_dims,
        )
        for k, v in m.items():
            acc[k] += v
        count += 1
    if count == 0:
        return acc
    return {k: v / count for k, v in acc.items()}


def pairwise_mean_abs_cos(features: torch.Tensor) -> float:
    """Global mean |cos| over unique pairs (ground-truth packing density)."""
    x = features / features.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    g = (x @ x.T).abs()
    n = g.shape[0]
    if n < 2:
        return 0.0
    mask = torch.triu(torch.ones(n, n, dtype=torch.bool), diagonal=1)
    return float(g[mask].mean().item())


def run_sanity(
    *,
    d: int = 16,
    n_sparse: int = 16,
    n_dense: int = 64,
    top_k: int = 16,
    seed: int = 0,
    out_dir: Path,
) -> Dict[str, Any]:
    set_seed(seed)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    sparse = near_orthogonal_features(n_sparse, d, seed=seed)
    dense = dense_superposed_features(n_dense, d, seed=seed + 1)

    sparse_metrics = mean_metrics_over_features(sparse, top_k=top_k)
    dense_metrics = mean_metrics_over_features(dense, top_k=top_k)

    sparse_pack = pairwise_mean_abs_cos(sparse)
    dense_pack = pairwise_mean_abs_cos(dense)

    # Also log a fake "training" curve: interpolate packing density and track metrics.
    logger = SignalLogger(results_dir=out_dir.parent, run_name=out_dir.name)
    n_steps = 8
    for step in range(n_steps):
        # Gradually increase feature count in fixed d → denser packing.
        n_t = n_sparse + int((n_dense - n_sparse) * step / max(n_steps - 1, 1))
        n_t = max(n_t, 2)
        if n_t > d:
            pack = dense_superposed_features(n_t, d, seed=seed + 100 + step)
        else:
            pack = near_orthogonal_features(min(n_t, d), d, seed=seed + 100 + step)
        m = mean_metrics_over_features(pack, top_k=min(top_k, max(pack.shape[0] - 1, 1)))
        # Toy "loss": mean pairwise |cos| (packing density proxy).
        loss = pairwise_mean_abs_cos(pack)
        logger.log(step=step, loss=loss, geometry=m, n_features=int(pack.shape[0]))

    logger.write_meta(
        seed=seed,
        d=d,
        n_sparse=n_sparse,
        n_dense=n_dense,
        top_k=top_k,
        note="toy superposition sanity: sparse vs dense feature packs",
    )

    checks = {
        "pack_density_increases": dense_pack > sparse_pack + 0.05,
        "interference_increases": dense_metrics["interference_mean"]
        > sparse_metrics["interference_mean"] + 0.05,
        "coactivation_increases": dense_metrics["coactivation_overlap"]
        > sparse_metrics["coactivation_overlap"] + 0.02,
        "spectral_moves": abs(
            dense_metrics["spectral_participation"] - sparse_metrics["spectral_participation"]
        )
        > 0.02,
    }
    passed = all(checks.values())

    result: Dict[str, Any] = {
        "seed": seed,
        "d": d,
        "n_sparse": n_sparse,
        "n_dense": n_dense,
        "sparse": {
            "pairwise_mean_abs_cos": sparse_pack,
            "metrics": sparse_metrics,
        },
        "dense": {
            "pairwise_mean_abs_cos": dense_pack,
            "metrics": dense_metrics,
        },
        "delta_dense_minus_sparse": {
            k: dense_metrics[k] - sparse_metrics[k] for k in sparse_metrics
        },
        "checks": checks,
        "passed": passed,
        "signals_csv": str(logger.csv_path),
        "meta_json": str(logger.meta_path),
    }
    out_json = out_dir / "sanity_result.json"
    out_json.write_text(json.dumps(result, indent=2) + "\n")
    result["result_json"] = str(out_json)
    return result


def plot_sparse_vs_dense(result: Dict[str, Any], out_path: Path) -> Path:
    import matplotlib.pyplot as plt

    metrics = list(result["sparse"]["metrics"].keys())
    sparse_vals = [result["sparse"]["metrics"][m] for m in metrics]
    dense_vals = [result["dense"]["metrics"][m] for m in metrics]

    fig, ax = plt.subplots(figsize=(8, 4))
    x = range(len(metrics))
    width = 0.35
    ax.bar([i - width / 2 for i in x], sparse_vals, width, label="sparse (≤d)", color="#444")
    ax.bar([i + width / 2 for i in x], dense_vals, width, label="dense (≫d)", color="#4C78A8")
    ax.set_xticks(list(x))
    ax.set_xticklabels(metrics, rotation=15, ha="right")
    ax.set_ylabel("mean metric")
    status = "PASS" if result["passed"] else "FAIL"
    ax.set_title(f"toy superposition sanity [{status}]")
    ax.legend(frameon=False)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Toy superposition geometry sanity check")
    parser.add_argument("--d", type=int, default=16)
    parser.add_argument("--n-sparse", type=int, default=16)
    parser.add_argument("--n-dense", type=int, default=64)
    parser.add_argument("--top-k", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--out",
        type=Path,
        default=RESULTS / "toy_superposition_sanity",
    )
    args = parser.parse_args()

    result = run_sanity(
        d=args.d,
        n_sparse=args.n_sparse,
        n_dense=args.n_dense,
        top_k=args.top_k,
        seed=args.seed,
        out_dir=args.out,
    )
    plot_path = plot_sparse_vs_dense(result, args.out / "sparse_vs_dense.png")

    # Also plot the density sweep signals if present.
    try:
        from scripts.helpers.plot_signals import plot_signals

        plot_signals(Path(result["signals_csv"]), args.out / "loss_geometry.png")
    except Exception as exc:  # noqa: BLE001 — plot is best-effort
        print(f"plot_signals skipped: {exc}")

    print(json.dumps(result, indent=2))
    print(f"\npassed={result['passed']}")
    print(f"wrote {result['result_json']}")
    print(f"wrote {plot_path}")
    for name, ok in result["checks"].items():
        print(f"  {name}: {'ok' if ok else 'FAIL'}")
    raise SystemExit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()
