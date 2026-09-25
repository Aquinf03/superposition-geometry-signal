"""Diff selected-feature geometry across two aligned training checkpoints.

Loads ckpt A weights → snapshot each selected feature at each layer →
loads ckpt B → snapshot again → ``diff_snapshots`` (banks refreshed per ckpt).

Example:
  python scripts/diff_checkpoints.py --config configs/train_gpt2_small_geometry.yaml
  python scripts/diff_checkpoints.py \\
      --run-dir results/train_gpt2_small_geometry --step-a 19 --step-b 39
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.checkpoint_geometry import (
    list_aligned_checkpoints,
    load_aligned_geometry,
)
from scripts.diff_geometry import (
    diff_snapshots,
    extract_mlp_out_feature,
    plot_neighborhood_diff,
    save_diff,
    snapshot_neighborhood,
)
from scripts.geometry import DEFAULT_PROBE_PROMPTS, collect_mlp_out_banks


def resolve_device(name: str) -> str:
    if name != "auto":
        return name
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_config(path: Path) -> Dict[str, Any]:
    with path.open() as f:
        return yaml.safe_load(f)


def resolve_feature_pos(model, prompt: str, position: Any) -> int:
    tokens = model.to_tokens(prompt)
    if position == "last" or position is None:
        return int(tokens.shape[1] - 1)
    return int(position)


def find_ckpt_by_step(ckpt_dir: Path, step: int) -> Dict[str, Any]:
    rows = list_aligned_checkpoints(ckpt_dir)
    for row in rows:
        if int(row["step"]) == int(step):
            return row
    available = [r["step"] for r in rows]
    raise FileNotFoundError(
        f"no aligned checkpoint for step={step} in {ckpt_dir}; have {available}"
    )


def pick_default_steps(ckpt_dir: Path) -> Tuple[int, int]:
    rows = list_aligned_checkpoints(ckpt_dir)
    if len(rows) < 2:
        raise FileNotFoundError(
            f"need ≥2 aligned checkpoints in {ckpt_dir}; found {len(rows)}"
        )
    return int(rows[0]["step"]), int(rows[-1]["step"])


def load_state_into(model, weight_path: Path, device: str) -> int:
    blob = torch.load(weight_path, map_location=device, weights_only=False)
    model.load_state_dict(blob["model_state"])
    model.eval()
    return int(blob.get("step", -1))


@torch.no_grad()
def snapshot_features_at_ckpt(
    model,
    *,
    step: int,
    features: Sequence[Dict[str, Any]],
    layers: Sequence[int],
    probe_prompts: Sequence[str],
    bank_positions: str,
    top_k: int,
    metrics: Sequence[str],
) -> Dict[str, Dict[str, Any]]:
    """Return ``{feature_id: {layer: snapshot}}`` for the current model weights."""
    banks = collect_mlp_out_banks(
        model, layers=layers, prompts=probe_prompts, positions=bank_positions
    )
    out: Dict[str, Dict[str, Any]] = {}
    for feat in features:
        fid = str(feat.get("name") or feat.get("prompt") or "feature")
        prompt = str(feat["prompt"])
        pos = resolve_feature_pos(model, prompt, feat.get("position", "last"))
        out[fid] = {}
        for L in layers:
            query = extract_mlp_out_feature(model, prompt, int(L), pos)
            out[fid][str(L)] = snapshot_neighborhood(
                query,
                banks[int(L)],
                feature_id=f"{fid}@L{L}",
                top_k=top_k,
                metrics=metrics,
                step=step,
                extra={
                    "prompt": prompt,
                    "position": pos,
                    "layer": int(L),
                    "ckpt_step": step,
                },
            )
    return out


def diff_aligned_summary(geo_a: Dict[str, Any], geo_b: Dict[str, Any]) -> Dict[str, Any]:
    """Cheap Δ on the frozen sidecar geometry (no feature extraction)."""
    ga = geo_a.get("geometry", {})
    gb = geo_b.get("geometry", {})
    keys = sorted(set(ga) | set(gb))
    delta = {k: float(gb.get(k, 0.0) - ga.get(k, 0.0)) for k in keys}
    return {
        "mode": "ckpt_aligned_summary",
        "step_before": geo_a.get("step"),
        "step_after": geo_b.get("step"),
        "loss_before": geo_a.get("loss"),
        "loss_after": geo_b.get("loss"),
        "metrics_before": ga,
        "metrics_after": gb,
        "delta_metrics": delta,
        "summary_score": float(sum(abs(v) for v in delta.values()) / max(len(delta), 1)),
    }


def run_diff(
    *,
    run_dir: Path,
    step_a: int,
    step_b: int,
    model_name: str,
    device: str,
    layers: List[int],
    features: List[Dict[str, Any]],
    probe_prompts: List[str],
    bank_positions: str,
    top_k: int,
    metrics: List[str],
    save_plots: bool = True,
) -> Dict[str, Any]:
    ckpt_dir = run_dir / "checkpoints"
    row_a = find_ckpt_by_step(ckpt_dir, step_a)
    row_b = find_ckpt_by_step(ckpt_dir, step_b)
    geo_a = load_aligned_geometry(Path(row_a["geometry_path"]))
    geo_b = load_aligned_geometry(Path(row_b["geometry_path"]))

    out_dir = run_dir / f"diff_ckpt_{step_a:05d}_vs_{step_b:05d}"
    out_dir.mkdir(parents=True, exist_ok=True)

    aligned_summary = diff_aligned_summary(geo_a, geo_b)
    save_diff(aligned_summary, out_dir / "diff_aligned_summary.json")

    from transformer_lens import HookedTransformer

    print(f"loading base model {model_name} on {device}…", flush=True)
    model = HookedTransformer.from_pretrained(model_name, device=device)
    model.eval()

    print(f"snapshot ckpt A step={step_a} ← {row_a['weight_path']}", flush=True)
    load_state_into(model, Path(row_a["weight_path"]), device)
    snaps_a = snapshot_features_at_ckpt(
        model,
        step=step_a,
        features=features,
        layers=layers,
        probe_prompts=probe_prompts,
        bank_positions=bank_positions,
        top_k=top_k,
        metrics=metrics,
    )

    print(f"snapshot ckpt B step={step_b} ← {row_b['weight_path']}", flush=True)
    load_state_into(model, Path(row_b["weight_path"]), device)
    snaps_b = snapshot_features_at_ckpt(
        model,
        step=step_b,
        features=features,
        layers=layers,
        probe_prompts=probe_prompts,
        bank_positions=bank_positions,
        top_k=top_k,
        metrics=metrics,
    )

    feature_diffs: Dict[str, Any] = {}
    paths: Dict[str, str] = {
        "aligned_summary": str(out_dir / "diff_aligned_summary.json"),
    }

    for fid in snaps_a:
        feature_diffs[fid] = {}
        for L in snaps_a[fid]:
            before = snaps_a[fid][L]
            after = snaps_b[fid][L]
            d = diff_snapshots(before, after, mode="ckpt_a_vs_b")
            d["layer"] = int(L)
            d["feature"] = fid
            d["ckpt_a"] = {
                "step": step_a,
                "weight_path": row_a["weight_path"],
                "geometry_path": row_a["geometry_path"],
                "loss": geo_a.get("loss"),
            }
            d["ckpt_b"] = {
                "step": step_b,
                "weight_path": row_b["weight_path"],
                "geometry_path": row_b["geometry_path"],
                "loss": geo_b.get("loss"),
            }
            safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in fid)
            stem = f"diff_{safe}_L{L}"
            jp = save_diff(d, out_dir / f"{stem}.json")
            paths[f"{fid}@L{L}"] = str(jp)
            if save_plots:
                plot_neighborhood_diff(d, out_dir / f"{stem}.png")
            feature_diffs[fid][f"L{L}"] = {
                "summary_score": d["summary_score"],
                "mean_abs_neighbor_delta": d["mean_abs_neighbor_delta"],
                "delta_metrics": d["delta_metrics"],
                "json": str(jp),
            }
            print(
                f"  {fid} L{L}: summary={d['summary_score']:.4f}  "
                f"nbrΔ={d['mean_abs_neighbor_delta']:.4f}  "
                f"Δmetrics={{{', '.join(f'{k}={v:+.4f}' for k, v in d['delta_metrics'].items())}}}",
                flush=True,
            )

    summary = {
        "mode": "ckpt_a_vs_b",
        "run_dir": str(run_dir),
        "step_a": step_a,
        "step_b": step_b,
        "layers": layers,
        "features": [f.get("name") or f.get("prompt") for f in features],
        "aligned_summary_score": aligned_summary["summary_score"],
        "aligned_delta_metrics": aligned_summary["delta_metrics"],
        "feature_diffs": feature_diffs,
        "paths": paths,
        "out_dir": str(out_dir),
    }
    summary_path = out_dir / "diff_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    summary["summary_json"] = str(summary_path)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Diff selected features across two aligned training checkpoints"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "train_gpt2_small_geometry.yaml",
        help="Train/diff YAML (provides run_name, layers, selected_features)",
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=None,
        help="Override run dir (default: results/<logging.run_name>)",
    )
    parser.add_argument("--step-a", type=int, default=None, help="Earlier checkpoint step")
    parser.add_argument("--step-b", type=int, default=None, help="Later checkpoint step")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    log_cfg = cfg.get("logging", {})
    geo_cfg = cfg.get("geometry", {})
    diff_cfg = cfg.get("diff", {})

    run_dir = args.run_dir or (ROOT / log_cfg.get("results_dir", "results") / log_cfg["run_name"])
    run_dir = Path(run_dir)
    ckpt_dir = run_dir / "checkpoints"

    if args.step_a is None or args.step_b is None:
        da, db = pick_default_steps(ckpt_dir)
        step_a = args.step_a if args.step_a is not None else da
        step_b = args.step_b if args.step_b is not None else db
    else:
        step_a, step_b = args.step_a, args.step_b

    layers = [int(L) for L in (geo_cfg.get("layers") or [geo_cfg.get("layer", 8)])]
    metrics = list(
        geo_cfg.get(
            "metrics",
            ["interference_mean", "spectral_participation", "coactivation_overlap"],
        )
    )
    probes = list(geo_cfg.get("probe_prompts") or DEFAULT_PROBE_PROMPTS)
    # Prefer probe list frozen into ckpt A sidecar if present
    row_a = find_ckpt_by_step(ckpt_dir, step_a)
    geo_a = load_aligned_geometry(Path(row_a["geometry_path"]))
    if geo_a.get("probe_prompts"):
        probes = list(geo_a["probe_prompts"])
    if geo_a.get("layers"):
        layers = [int(L) for L in geo_a["layers"]]
    bank_positions = str(
        geo_a.get("bank_positions") or geo_cfg.get("bank_positions", "last")
    )
    top_k = int(
        diff_cfg.get(
            "top_k_neighbors",
            geo_a.get("top_k_neighbors") or geo_cfg.get("top_k_neighbors", 8),
        )
    )

    features = list(diff_cfg.get("selected_features") or [])
    if not features:
        # Sensible defaults from the train corpus / probes
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

    device = resolve_device(args.device or str(cfg.get("device", "auto")))
    model_name = str(cfg.get("model", "gpt2-small"))

    print(
        f"diff ckpt {step_a} → {step_b}  run={run_dir}  layers={layers}  "
        f"features={[f.get('name') for f in features]}",
        flush=True,
    )
    result = run_diff(
        run_dir=run_dir,
        step_a=step_a,
        step_b=step_b,
        model_name=model_name,
        device=device,
        layers=layers,
        features=features,
        probe_prompts=probes,
        bank_positions=bank_positions,
        top_k=top_k,
        metrics=metrics,
        save_plots=not args.no_plots,
    )
    print(json.dumps({"summary_json": result["summary_json"], "out_dir": result["out_dir"]}, indent=2))
    print(f"\nDone. wrote {result['summary_json']}")


if __name__ == "__main__":
    main()
