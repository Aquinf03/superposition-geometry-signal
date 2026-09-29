"""Thin edit-time demo: nudge one MLP ``W_out`` so geometry moves toward a target/band.

Existence proof only — not the research depth. Soft train control lives in
``train_with_geometry``; this script shows the same ``GeoControl`` knob at edit time.

Run:
  python experiments/edit_geo_nudge.py
  python experiments/edit_geo_nudge.py --config experiments/configs/edit_geo_nudge.yaml
  # or: spg edit-demo
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from spg import (
    DEFAULT_PROBE_PROMPTS,
    FeatureRelativeControl,
    GeoControl,
    collect_mlp_out_bank,
    collect_mlp_out_bank_grad,
    format_control_line,
    mean_interference_tensor,
    set_seed,
)
from scripts.helpers.paths import CONFIGS, resolve_under_experiments


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
        return yaml.safe_load(f) or {}


def watch_interference(
    model,
    layer: int,
    probes: List[str],
    *,
    positions: str,
    top_k: int,
) -> float:
    with torch.no_grad():
        bank = collect_mlp_out_bank(model, layer, probes, positions=positions)
        return float(mean_interference_tensor(bank, top_k=top_k).item())


def _prompt_vec(model, layer: int, prompt: str, *, positions: str) -> torch.Tensor:
    bank = collect_mlp_out_bank(model, layer, [prompt], positions=positions)
    return torch.nn.functional.normalize(bank.float().mean(dim=0), dim=0)


def feature_pair_scores(
    model,
    layer: int,
    related_prompts: List[str],
    unrelated_prompts: List[str],
    *,
    positions: str,
) -> tuple[float, float]:
    """Cosine distance between MLP-out vectors (lower = closer)."""
    with torch.no_grad():
        ra = _prompt_vec(model, layer, related_prompts[0], positions=positions)
        rb = _prompt_vec(
            model,
            layer,
            related_prompts[1] if len(related_prompts) > 1 else related_prompts[0],
            positions=positions,
        )
        ua = _prompt_vec(model, layer, unrelated_prompts[0], positions=positions)
        ub = _prompt_vec(
            model,
            layer,
            unrelated_prompts[1] if len(unrelated_prompts) > 1 else unrelated_prompts[0],
            positions=positions,
        )
        related = float(1.0 - torch.dot(ra, rb).item())
        unrelated = float(1.0 - torch.dot(ua, ub).item())
    return related, unrelated


def run(cfg: Dict[str, Any]) -> Dict[str, Any]:
    seed = set_seed(int(cfg.get("seed", 0)))
    device = resolve_device(str(cfg.get("device", "auto")))
    edit_cfg = cfg.get("edit") or {}
    ctrl_cfg = dict(cfg.get("control") or {})
    ctrl_cfg.setdefault("enabled", True)
    feat_cfg = cfg.get("feature_relative") or {}

    from transformer_lens import HookedTransformer

    model_name = cfg.get("model", "gpt2-small")
    model = HookedTransformer.from_pretrained(model_name, device=device)
    model.eval()

    layer = int(ctrl_cfg.get("layer", edit_cfg.get("layer", 8)))
    top_k = int(edit_cfg.get("top_k_neighbors", 8))
    positions = str(edit_cfg.get("bank_positions", "last"))
    n_steps = int(edit_cfg.get("n_steps", 12))
    lr = float(edit_cfg.get("lr", 5e-4))
    probes = list(edit_cfg.get("probe_prompts") or DEFAULT_PROBE_PROMPTS)

    control = GeoControl.from_config(ctrl_cfg)
    control.enable()
    feat_ctrl = FeatureRelativeControl.from_config(feat_cfg)

    related_prompts = list(
        edit_cfg.get("related_prompts")
        or [
            "The Eiffel Tower is located in",
            "The Louvre Museum is located in",
        ]
    )
    unrelated_prompts = list(
        edit_cfg.get("unrelated_prompts")
        or [
            "The Eiffel Tower is located in",
            "Superposition allows neural networks to represent more features than",
        ]
    )

    # Freeze everything except one MLP W_out.
    for p in model.parameters():
        p.requires_grad_(False)
    w_out = model.blocks[layer].mlp.W_out
    w_out.requires_grad_(True)
    opt = torch.optim.Adam([w_out], lr=lr)

    before = watch_interference(model, layer, probes, positions=positions, top_k=top_k)
    rel0, unrel0 = feature_pair_scores(
        model, layer, related_prompts, unrelated_prompts, positions=positions
    )
    print(
        f"edit_geo_nudge: model={model_name} device={device} seed={seed} "
        f"layer={layer} steps={n_steps} mode={control.mode}",
        flush=True,
    )
    geo0 = {f"L{layer}_interference_mean": before}
    print(
        format_control_line(step=0, loss=0.0, geometry=geo0, control=control.status(geo0)),
        flush=True,
    )
    print(f"         feature_rel: related={rel0:.4f} unrelated={unrel0:.4f}", flush=True)

    history: List[Dict[str, float]] = []
    for step in range(1, n_steps + 1):
        opt.zero_grad(set_to_none=True)
        bank = collect_mlp_out_bank_grad(model, layer, probes, positions=positions)
        metric_t = mean_interference_tensor(bank, top_k=top_k)
        pen = control.penalty_on_value(metric_t)
        if feat_ctrl.enabled:
            v = torch.nn.functional.normalize(bank.float(), dim=-1)
            if v.shape[0] >= 2:
                related_t = 1.0 - (v[0] * v[1]).sum()
                unrelated_t = 1.0 - (v[0] * v[-1]).sum()
                pen = pen + feat_ctrl.penalty_on_scores(related_t, unrelated_t)
        pen.backward()
        opt.step()

        cur = float(metric_t.detach().item())
        geo = {f"L{layer}_interference_mean": cur}
        st = control.status(geo)
        st.current = cur
        st.penalty = float(pen.detach().item())
        print(
            format_control_line(
                step=step, loss=float(pen.detach()), geometry=geo, control=st
            ),
            flush=True,
        )
        history.append({"step": float(step), "interference": cur, "penalty": st.penalty})

    after = watch_interference(model, layer, probes, positions=positions, top_k=top_k)
    rel1, unrel1 = feature_pair_scores(
        model, layer, related_prompts, unrelated_prompts, positions=positions
    )
    feat_st = feat_ctrl.status(rel1, unrel1)

    summary = {
        "model": model_name,
        "device": device,
        "seed": seed,
        "layer": layer,
        "n_steps": n_steps,
        "control": control.status({f"L{layer}_interference_mean": after}).as_dict(),
        "interference_before": before,
        "interference_after": after,
        "delta_interference": after - before,
        "related_before": rel0,
        "related_after": rel1,
        "unrelated_before": unrel0,
        "unrelated_after": unrel1,
        "feature_relative": feat_st.as_dict(),
        "history": history,
        "note": "thin edit-time demo — not the research depth",
    }

    out_dir = resolve_under_experiments(
        (cfg.get("logging") or {}).get("results_dir", "results")
    ) / str((cfg.get("logging") or {}).get("run_name", "edit_geo_nudge"))
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "summary.json"
    out_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(
        f"\nok — interference {before:.4f} → {after:.4f} (Δ={after - before:+.4f})  "
        f"related {rel0:.4f}→{rel1:.4f}  unrelated {unrel0:.4f}→{unrel1:.4f}",
        flush=True,
    )
    print(f"wrote {out_path}", flush=True)
    return summary


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "-c",
        "--config",
        type=Path,
        default=CONFIGS / "edit_geo_nudge.yaml",
        help="YAML under experiments/configs/",
    )
    args = parser.parse_args(argv)
    cfg_path = args.config
    if not cfg_path.is_file():
        # Allow running with built-in defaults when config is missing.
        cfg: Dict[str, Any] = {
            "seed": 0,
            "model": "gpt2-small",
            "device": "auto",
            "edit": {"layer": 8, "n_steps": 12, "lr": 5.0e-4},
            "control": {
                "enabled": True,
                "layer": 8,
                "metric": "interference_mean",
                "mode": "band",
                "lo": 0.30,
                "hi": 0.40,
                "target": 0.35,
                "weight": 1.0,
            },
            "feature_relative": {"enabled": False},
            "logging": {"results_dir": "results", "run_name": "edit_geo_nudge"},
        }
        print(f"config missing ({cfg_path}); using built-in defaults", flush=True)
    else:
        cfg = load_config(cfg_path)
    run(cfg)


if __name__ == "__main__":
    main()
