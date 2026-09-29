"""Out-of-metric validation for a controlled fine-tune vs watch baseline.

Does **not** score success as “hit geo_target.” Checks:

  1. downstream LM loss — control did not wreck next-token loss vs baseline
  2. cross-feature structure — related pairs still closer than unrelated
  3. held-out probes — geometry on prompts *outside* the control bank
     still moves sensibly (not pure train-bank gaming)

Also reuses core trajectory checks from ``validate_training_geometry``.

Example::

  # ensure feature diffs exist on both runs first
  python experiments/diff_features.py --config experiments/configs/train_gpt2_small_geometry.yaml
  python experiments/diff_features.py --config experiments/configs/train_gpt2_small_geometry_control.yaml

  python experiments/validate_control_geometry.py \\
      --baseline experiments/results/train_gpt2_small_geometry \\
      --control experiments/results/train_gpt2_small_geometry_control
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.validate_training_geometry import (  # noqa: E402
    claim,
    find_cross_feature_summary,
    read_signals,
    validate_run,
)
from scripts.helpers.checkpoint_geometry import list_aligned_checkpoints  # noqa: E402
from scripts.helpers.geometry import (  # noqa: E402
    DEFAULT_PROBE_PROMPTS,
    collect_mlp_out_banks,
    compute_geometry_metrics,
)
from scripts.helpers.paths import RESULTS  # noqa: E402

# Disjoint from DEFAULT_PROBE_PROMPTS (control / watch bank).
HELD_OUT_PROBE_PROMPTS: List[str] = [
    "The Nile River flows through",
    "Shakespeare wrote the play",
    "The speed of light in vacuum is",
    "DNA stands for",
    "The Pacific Ocean is larger than",
    "Newton's second law states that",
    "The chemical formula for water is",
    "The capital of Australia is",
    "Photosynthesis converts sunlight into",
    "The first element on the periodic table is",
]


def _overlap(a: Sequence[str], b: Sequence[str]) -> List[str]:
    sa = {x.strip() for x in a}
    return [x for x in b if x.strip() in sa]


def _mean_metric_on_bank(bank: torch.Tensor, *, top_k: int, metric: str) -> float:
    n = bank.shape[0]
    if n < 2:
        return float("nan")
    acc = 0.0
    for i in range(n):
        query = bank[i]
        others = torch.cat([bank[:i], bank[i + 1 :]], dim=0)
        vals = compute_geometry_metrics(
            query,
            others,
            top_k=min(top_k, others.shape[0]),
            metrics=[metric],
        )
        acc += float(vals[metric])
    return acc / n


def _load_final_model(run_dir: Path, model_name: str, device: str):
    from transformer_lens import HookedTransformer

    from experiments.diff_checkpoints import load_state_into, resolve_device

    device = resolve_device(device)
    ckpt_dir = run_dir / "checkpoints"
    rows = list_aligned_checkpoints(ckpt_dir)
    if not rows:
        # fall back to model_final.pt
        weight = run_dir / "model_final.pt"
        if not weight.is_file():
            raise FileNotFoundError(f"no checkpoints or model_final.pt in {run_dir}")
        step = -1
        weight_path = weight
    else:
        step = int(rows[-1]["step"])
        weight_path = Path(rows[-1]["weight_path"])

    model = HookedTransformer.from_pretrained(model_name, device=device)
    load_state_into(model, weight_path, device=device)
    model.eval()
    return model, device, step, weight_path


def measure_held_out(
    run_dir: Path,
    *,
    model_name: str,
    device: str,
    layer: int,
    metric: str,
    top_k: int,
    bank_positions: str,
    held_out: Sequence[str],
    train_bank: Sequence[str],
) -> Dict[str, Any]:
    overlap = _overlap(train_bank, held_out)
    if overlap:
        raise ValueError(f"held-out probes overlap train bank: {overlap[:3]}")

    model, device, step, weight_path = _load_final_model(run_dir, model_name, device)
    layers = [int(layer)]
    with torch.no_grad():
        banks_hold = collect_mlp_out_banks(
            model, layers=layers, prompts=list(held_out), positions=bank_positions
        )
        banks_train = collect_mlp_out_banks(
            model, layers=layers, prompts=list(train_bank), positions=bank_positions
        )
    hold_v = _mean_metric_on_bank(banks_hold[layer], top_k=top_k, metric=metric)
    train_v = _mean_metric_on_bank(banks_train[layer], top_k=top_k, metric=metric)
    return {
        "step": step,
        "weight_path": str(weight_path),
        "layer": layer,
        "metric": metric,
        "held_out_mean": hold_v,
        "train_bank_mean": train_v,
        "abs_gap_train_vs_heldout": abs(train_v - hold_v),
        "n_held_out": len(held_out),
        "n_train_bank": len(train_bank),
    }


def _xf_pair_means(xf_path: Path) -> Optional[Dict[str, float]]:
    xf = json.loads(xf_path.read_text())
    related: List[float] = []
    unrelated: List[float] = []
    for key, layers_map in (xf.get("pair_diffs") or {}).items():
        is_rel = "eiffel" in key and "louvre" in key and "superposition" not in key
        for Lname, stats in layers_map.items():
            if Lname in ("L11", "11"):
                continue
            score = float(stats["summary_score"])
            if is_rel:
                related.append(score)
            elif "superposition" in key:
                unrelated.append(score)
    if not related or not unrelated:
        return None
    return {
        "related_mean": float(np.mean(related)),
        "unrelated_mean": float(np.mean(unrelated)),
        "ratio_related_over_unrelated": float(np.mean(related) / max(np.mean(unrelated), 1e-12)),
    }


def validate_control_pair(
    baseline_dir: Path,
    control_dir: Path,
    *,
    device: str = "auto",
    loss_slack: float = 0.05,
    max_train_heldout_gap: float = 0.15,
    skip_held_out: bool = False,
) -> Dict[str, Any]:
    baseline_dir = Path(baseline_dir)
    control_dir = Path(control_dir)

    base_sig = read_signals(baseline_dir / "signals.csv")
    ctrl_sig = read_signals(control_dir / "signals.csv")
    base_meta = json.loads((baseline_dir / "meta.json").read_text())
    ctrl_meta = json.loads((control_dir / "meta.json").read_text())

    model_name = str(ctrl_meta.get("model") or base_meta.get("model") or "gpt2-small")
    ctrl_cfg = ctrl_meta.get("control") or {}
    layer = int(ctrl_cfg.get("layer", 8))
    metric = str(ctrl_cfg.get("metric", "interference_mean"))
    target = float(ctrl_cfg.get("target", 0.35))

    geo_cfg = ctrl_meta.get("geometry") or {}
    # frozen config may live beside meta
    frozen = control_dir / "config.frozen.yaml"
    if frozen.is_file():
        with frozen.open() as f:
            full = yaml.safe_load(f) or {}
        geo_cfg = full.get("geometry") or geo_cfg
        ctrl_cfg = full.get("control") or ctrl_cfg
        layer = int(ctrl_cfg.get("layer", layer))
        metric = str(ctrl_cfg.get("metric", metric))
        target = float(ctrl_cfg.get("target", target))

    top_k = int(geo_cfg.get("top_k_neighbors", 8))
    positions = str(geo_cfg.get("bank_positions", "last"))
    train_bank = list(geo_cfg.get("probe_prompts") or DEFAULT_PROBE_PROMPTS)

    claims: List[Dict[str, Any]] = []

    # --- 1. Downstream LM loss ---
    base_loss = float(base_sig["loss"][-1])
    ctrl_loss = float(ctrl_sig["loss"][-1])
    loss_ok = ctrl_loss <= base_loss + loss_slack
    claims.append(
        claim(
            "downstream_loss",
            passed=loss_ok,
            detail=(
                f"control final loss={ctrl_loss:.4f} vs baseline={base_loss:.4f} "
                f"(slack={loss_slack})"
            ),
            baseline_loss_last=base_loss,
            control_loss_last=ctrl_loss,
            delta=ctrl_loss - base_loss,
            slack=loss_slack,
        )
    )

    # --- 2. Cross-feature structure on both runs ---
    for label, run_dir in (("baseline", baseline_dir), ("control", control_dir)):
        xf_path = find_cross_feature_summary(run_dir)
        if not xf_path:
            claims.append(
                claim(
                    f"cross_feature_{label}",
                    passed=False,
                    detail=f"missing diff_features_step_*/diff_summary.json under {run_dir.name}",
                    skipped=False,
                )
            )
            continue
        means = _xf_pair_means(xf_path)
        if means is None:
            claims.append(
                claim(
                    f"cross_feature_{label}",
                    passed=False,
                    detail=f"could not parse related/unrelated pairs in {xf_path.name}",
                    source=str(xf_path),
                )
            )
            continue
        ok = means["related_mean"] < means["unrelated_mean"] * 0.5
        claims.append(
            claim(
                f"cross_feature_{label}",
                passed=ok,
                detail=(
                    f"{label}: related={means['related_mean']:.4f} "
                    f"< 0.5×unrelated={means['unrelated_mean']:.4f} "
                    f"({xf_path.parent.name})"
                ),
                source=str(xf_path),
                **means,
            )
        )

    # Structure preserved under control: related/unrelated ordering still holds on control
    xf_ctrl = next((c for c in claims if c["id"] == "cross_feature_control"), None)
    if xf_ctrl is not None:
        claims.append(
            claim(
                "cross_feature_preserved_under_control",
                passed=bool(xf_ctrl["pass"]),
                detail=(
                    "control still separates related vs unrelated neighborhoods"
                    if xf_ctrl["pass"]
                    else "control broke related≪unrelated structure"
                ),
            )
        )

    # --- 3. Held-out probes at final ckpt ---
    held_out_stats: Dict[str, Any] = {}
    if skip_held_out:
        claims.append(
            claim(
                "held_out_probes",
                passed=False,
                detail="skipped (--skip-held-out)",
                skipped=True,
            )
        )
    else:
        base_h = measure_held_out(
            baseline_dir,
            model_name=model_name,
            device=device,
            layer=layer,
            metric=metric,
            top_k=top_k,
            bank_positions=positions,
            held_out=HELD_OUT_PROBE_PROMPTS,
            train_bank=train_bank,
        )
        ctrl_h = measure_held_out(
            control_dir,
            model_name=model_name,
            device=device,
            layer=layer,
            metric=metric,
            top_k=top_k,
            bank_positions=positions,
            held_out=HELD_OUT_PROBE_PROMPTS,
            train_bank=train_bank,
        )
        held_out_stats = {"baseline": base_h, "control": ctrl_h}

        # (a) control should not inflate train↔held-out gap (gaming train bank only)
        gap_ok = float(ctrl_h["abs_gap_train_vs_heldout"]) <= max(
            float(base_h["abs_gap_train_vs_heldout"]) + 0.05,
            max_train_heldout_gap,
        )
        claims.append(
            claim(
                "held_out_not_gamed",
                passed=gap_ok,
                detail=(
                    f"|train−heldout| control={ctrl_h['abs_gap_train_vs_heldout']:.4f} "
                    f"baseline={base_h['abs_gap_train_vs_heldout']:.4f} "
                    f"(cap≈{max(float(base_h['abs_gap_train_vs_heldout']) + 0.05, max_train_heldout_gap):.4f})"
                ),
                baseline_gap=base_h["abs_gap_train_vs_heldout"],
                control_gap=ctrl_h["abs_gap_train_vs_heldout"],
            )
        )

        # (b) held-out metric should not regress away from target vs baseline
        #     (full generalization toward target is nicer but not required at λ=1e-2)
        err_b = abs(float(base_h["held_out_mean"]) - target)
        err_c = abs(float(ctrl_h["held_out_mean"]) - target)
        improved = err_c < err_b
        gen_ok = err_c <= err_b + 0.02  # allow tiny noise
        claims.append(
            claim(
                "held_out_not_regressed",
                passed=gen_ok,
                detail=(
                    f"held-out L{layer} {metric}: control={ctrl_h['held_out_mean']:.4f} "
                    f"(err={err_c:.4f}) vs baseline={base_h['held_out_mean']:.4f} "
                    f"(err={err_b:.4f}); target={target}; "
                    f"{'improved' if improved else 'no improvement (within slack)'}"
                ),
                baseline_held_out=base_h["held_out_mean"],
                control_held_out=ctrl_h["held_out_mean"],
                target=target,
                err_baseline=err_b,
                err_control=err_c,
                improved=improved,
            )
        )

    # --- 4. Core trajectory validation on the control run ---
    traj = validate_run(control_dir)
    traj_required = [c for c in traj["claims"] if not c.get("stats", {}).get("skipped")]
    traj_ok = all(c["pass"] for c in traj_required)
    claims.append(
        claim(
            "control_trajectory_checks",
            passed=traj_ok,
            detail=(
                f"validate_training_geometry on control: "
                f"{'PASS' if traj_ok else 'FAIL'} "
                f"({sum(1 for c in traj_required if c['pass'])}/{len(traj_required)} claims)"
            ),
            nested_overall_pass=traj.get("overall_pass"),
            nested_claims=traj.get("claims"),
        )
    )

    required = [c for c in claims if not c.get("stats", {}).get("skipped")]
    overall = all(c["pass"] for c in required)

    return {
        "mode": "validate_control_geometry",
        "baseline_dir": str(baseline_dir),
        "control_dir": str(control_dir),
        "model": model_name,
        "control_cfg": ctrl_cfg,
        "target_layer": layer,
        "target_metric": metric,
        "target_value": target,
        "held_out_probes": list(HELD_OUT_PROBE_PROMPTS),
        "held_out_stats": held_out_stats,
        "claims": claims,
        "overall_pass": overall,
        "scope": (
            "out-of-metric: downstream loss + cross-feature + held-out probes "
            "+ control trajectory — not “hit geo_target ⇒ success”"
        ),
        "trajectory_report": traj,
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--baseline",
        type=Path,
        default=RESULTS / "train_gpt2_small_geometry",
    )
    p.add_argument(
        "--control",
        type=Path,
        default=RESULTS / "train_gpt2_small_geometry_control",
    )
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--loss-slack", type=float, default=0.05)
    p.add_argument("--max-train-heldout-gap", type=float, default=0.15)
    p.add_argument("--skip-held-out", action="store_true")
    p.add_argument(
        "-o",
        "--out",
        type=Path,
        default=None,
        help="Default: <control>/../control_vs_baseline_seed0/out_of_metric_validation.json",
    )
    args = p.parse_args()

    report = validate_control_pair(
        args.baseline,
        args.control,
        device=args.device,
        loss_slack=args.loss_slack,
        max_train_heldout_gap=args.max_train_heldout_gap,
        skip_held_out=args.skip_held_out,
    )

    out = args.out
    if out is None:
        out = args.control.parent / "control_vs_baseline_seed0" / "out_of_metric_validation.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    # nested trajectory claims can be large — keep full write
    out.write_text(json.dumps(report, indent=2) + "\n")
    # also drop a copy under the control run
    (args.control / "out_of_metric_validation.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )

    print(f"validate_control: baseline={args.baseline.name}  control={args.control.name}")
    print(f"scope: {report['scope']}")
    for c in report["claims"]:
        skipped = c.get("stats", {}).get("skipped")
        mark = "SKIP" if skipped else ("PASS" if c["pass"] else "FAIL")
        print(f"  [{mark}] {c['id']}: {c['detail']}")
    print(f"\noverall: {'PASS' if report['overall_pass'] else 'FAIL'}")
    print(f"wrote {out}")

    if not report["overall_pass"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
