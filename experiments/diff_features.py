"""Cross-feature diff: compare feature A vs feature B at one checkpoint.

Same bank, same weights — asks how differently two features sit in the
neighborhood (related cluster vs unrelated prompt).

Example:
  python experiments/diff_features.py --config experiments/configs/train_gpt2_small_geometry.yaml
  python experiments/diff_features.py --config experiments/configs/train_gpt2_small_geometry.yaml --step 39
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.helpers.checkpoint_geometry import list_aligned_checkpoints, load_aligned_geometry
from experiments.diff_checkpoints import (
    find_ckpt_by_step,
    load_state_into,
    resolve_device,
    snapshot_features_at_ckpt,
)
from scripts.helpers.diff_geometry import diff_snapshots, plot_neighborhood_diff, save_diff
from scripts.helpers.geometry import DEFAULT_PROBE_PROMPTS
from scripts.helpers.paths import CONFIGS, resolve_under_experiments


def load_config(path: Path) -> Dict[str, Any]:
    with path.open() as f:
        return yaml.safe_load(f)


def pick_latest_step(ckpt_dir: Path) -> int:
    rows = list_aligned_checkpoints(ckpt_dir)
    if not rows:
        raise FileNotFoundError(f"no aligned checkpoints in {ckpt_dir}")
    return int(rows[-1]["step"])


def resolve_pairs(
    features: Sequence[Dict[str, Any]],
    pairs_cfg: Optional[Sequence[Any]],
    cli_pairs: Optional[Sequence[Tuple[str, str]]] = None,
) -> List[Tuple[str, str]]:
    """Return list of (feature_a, feature_b) name pairs."""
    names = [str(f.get("name") or f.get("prompt")) for f in features]
    name_set = set(names)

    if cli_pairs:
        out = []
        for a, b in cli_pairs:
            if a not in name_set or b not in name_set:
                raise ValueError(f"unknown feature in pair ({a}, {b}); have {names}")
            out.append((a, b))
        return out

    if pairs_cfg:
        out = []
        for pair in pairs_cfg:
            if not isinstance(pair, (list, tuple)) or len(pair) != 2:
                raise ValueError(f"cross_feature_pairs entry must be [A, B], got {pair!r}")
            a, b = str(pair[0]), str(pair[1])
            if a not in name_set or b not in name_set:
                raise ValueError(f"unknown feature in pair ({a}, {b}); have {names}")
            out.append((a, b))
        return out

    # Default: all unordered pairs among selected_features
    return list(itertools.combinations(names, 2))


def run_cross_feature_diff(
    *,
    run_dir: Path,
    step: int,
    model_name: str,
    device: str,
    layers: List[int],
    features: List[Dict[str, Any]],
    pairs: List[Tuple[str, str]],
    probe_prompts: List[str],
    bank_positions: str,
    top_k: int,
    metrics: List[str],
    save_plots: bool = True,
) -> Dict[str, Any]:
    ckpt_dir = run_dir / "checkpoints"
    row = find_ckpt_by_step(ckpt_dir, step)
    geo = load_aligned_geometry(Path(row["geometry_path"]))

    out_dir = run_dir / f"diff_features_step_{step:05d}"
    out_dir.mkdir(parents=True, exist_ok=True)

    from transformer_lens import HookedTransformer

    print(f"loading base model {model_name} on {device}…", flush=True)
    model = HookedTransformer.from_pretrained(model_name, device=device)
    print(f"snapshot ckpt step={step} ← {row['weight_path']}", flush=True)
    load_state_into(model, Path(row["weight_path"]), device)

    snaps = snapshot_features_at_ckpt(
        model,
        step=step,
        features=features,
        layers=layers,
        probe_prompts=probe_prompts,
        bank_positions=bank_positions,
        top_k=top_k,
        metrics=metrics,
    )

    pair_diffs: Dict[str, Any] = {}
    paths: Dict[str, str] = {}

    for fa, fb in pairs:
        key = f"{fa}__vs__{fb}"
        pair_diffs[key] = {}
        for L in layers:
            Ls = str(L)
            before = snaps[fa][Ls]
            after = snaps[fb][Ls]
            d = diff_snapshots(before, after, mode="feature_a_vs_b")
            d["feature_a"] = fa
            d["feature_b"] = fb
            d["layer"] = int(L)
            d["ckpt"] = {
                "step": step,
                "weight_path": row["weight_path"],
                "geometry_path": row["geometry_path"],
                "loss": geo.get("loss"),
            }
            safe_a = "".join(c if c.isalnum() or c in "-_" else "_" for c in fa)
            safe_b = "".join(c if c.isalnum() or c in "-_" else "_" for c in fb)
            stem = f"diff_{safe_a}_vs_{safe_b}_L{L}"
            jp = save_diff(d, out_dir / f"{stem}.json")
            paths[f"{key}@L{L}"] = str(jp)
            if save_plots:
                plot_neighborhood_diff(d, out_dir / f"{stem}.png")
            pair_diffs[key][f"L{L}"] = {
                "summary_score": d["summary_score"],
                "mean_abs_neighbor_delta": d["mean_abs_neighbor_delta"],
                "delta_metrics": d["delta_metrics"],
                "json": str(jp),
            }
            print(
                f"  {fa} vs {fb} L{L}: summary={d['summary_score']:.4f}  "
                f"nbrΔ={d['mean_abs_neighbor_delta']:.4f}  "
                f"Δmetrics={{{', '.join(f'{k}={v:+.4f}' for k, v in d['delta_metrics'].items())}}}",
                flush=True,
            )

    summary = {
        "mode": "feature_a_vs_b",
        "run_dir": str(run_dir),
        "step": step,
        "loss": geo.get("loss"),
        "layers": layers,
        "pairs": [{"a": a, "b": b} for a, b in pairs],
        "pair_diffs": pair_diffs,
        "paths": paths,
        "out_dir": str(out_dir),
    }
    summary_path = out_dir / "diff_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    summary["summary_json"] = str(summary_path)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Diff feature A vs feature B at one aligned checkpoint"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=CONFIGS / "train_gpt2_small_geometry.yaml",
    )
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument(
        "--step",
        type=int,
        default=None,
        help="Checkpoint step (default: latest aligned)",
    )
    parser.add_argument(
        "--pair",
        nargs=2,
        action="append",
        metavar=("A", "B"),
        default=None,
        help="Feature name pair (repeatable). Default: config pairs or all combinations.",
    )
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    log_cfg = cfg.get("logging", {})
    geo_cfg = cfg.get("geometry", {})
    diff_cfg = cfg.get("diff", {})

    run_dir = args.run_dir or (
        resolve_under_experiments(log_cfg.get("results_dir", "results"))
        / log_cfg["run_name"]
    )
    run_dir = Path(run_dir)
    ckpt_dir = run_dir / "checkpoints"
    step = args.step if args.step is not None else pick_latest_step(ckpt_dir)

    layers = [int(L) for L in (geo_cfg.get("layers") or [geo_cfg.get("layer", 8)])]
    metrics = list(
        geo_cfg.get(
            "metrics",
            ["interference_mean", "spectral_participation", "coactivation_overlap"],
        )
    )
    probes = list(geo_cfg.get("probe_prompts") or DEFAULT_PROBE_PROMPTS)
    row = find_ckpt_by_step(ckpt_dir, step)
    geo = load_aligned_geometry(Path(row["geometry_path"]))
    if geo.get("probe_prompts"):
        probes = list(geo["probe_prompts"])
    if geo.get("layers"):
        layers = [int(L) for L in geo["layers"]]
    bank_positions = str(
        geo.get("bank_positions") or geo_cfg.get("bank_positions", "last")
    )
    top_k = int(
        diff_cfg.get(
            "top_k_neighbors",
            geo.get("top_k_neighbors") or geo_cfg.get("top_k_neighbors", 8),
        )
    )

    features = list(diff_cfg.get("selected_features") or [])
    if not features:
        features = [
            {
                "name": "eiffel_located_in",
                "prompt": "The Eiffel Tower is located in",
                "position": "last",
            },
            {
                "name": "louvre_located_in",
                "prompt": "The Louvre Museum is located in",
                "position": "last",
            },
            {
                "name": "superposition_pack",
                "prompt": "Superposition allows neural networks to represent more features than",
                "position": "last",
            },
        ]

    cli_pairs = [tuple(p) for p in args.pair] if args.pair else None
    pairs = resolve_pairs(features, diff_cfg.get("cross_feature_pairs"), cli_pairs)

    device = resolve_device(args.device or str(cfg.get("device", "auto")))
    model_name = str(cfg.get("model", "gpt2-small"))

    print(
        f"cross-feature diff @ step={step}  run={run_dir}  layers={layers}  "
        f"pairs={pairs}",
        flush=True,
    )
    result = run_cross_feature_diff(
        run_dir=run_dir,
        step=step,
        model_name=model_name,
        device=device,
        layers=layers,
        features=features,
        pairs=pairs,
        probe_prompts=probes,
        bank_positions=bank_positions,
        top_k=top_k,
        metrics=metrics,
        save_plots=not args.no_plots,
    )
    print(
        json.dumps(
            {"summary_json": result["summary_json"], "out_dir": result["out_dir"]},
            indent=2,
        )
    )
    print(f"\nDone. wrote {result['summary_json']}")


if __name__ == "__main__":
    main()
