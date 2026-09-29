"""Multi-seed quality suite: longer train ± control → held-out LM eval.

Goal: show soft geo-control can beat matched baseline on frozen eval text,
and ideally stay closer to pretrained than a catastrophic overfit.

Example::

  python experiments/run_control_quality_suite.py
  python experiments/run_control_quality_suite.py --seeds 0 1 2 --steps 200
  python experiments/run_control_quality_suite.py --eval-only   # reuse existing runs
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.eval_control_lm import evaluate_pair
from experiments.train_with_geometry import load_config, train
from scripts.helpers.paths import CONFIGS, RESULTS, resolve_under_experiments


def _deep_update(cfg: Dict[str, Any], seed: int, run_name: str, steps: Optional[int]) -> Dict[str, Any]:
    out = copy.deepcopy(cfg)
    out["seed"] = int(seed)
    out.setdefault("logging", {})["run_name"] = run_name
    if steps is not None:
        out.setdefault("train", {})["max_steps"] = int(steps)
    return out


def _run_exists(run_dir: Path) -> bool:
    return (run_dir / "model_final.pt").is_file() or (
        run_dir / "checkpoints"
    ).is_dir() and any((run_dir / "checkpoints").glob("step_*.pt"))


def run_suite(
    *,
    seeds: List[int],
    steps: Optional[int],
    device: str,
    eval_only: bool,
    skip_pretrained: bool,
) -> Dict[str, Any]:
    base_cfg = load_config(CONFIGS / "train_gpt2_small_geometry_quality.yaml")
    ctrl_cfg = load_config(CONFIGS / "train_gpt2_small_geometry_control_quality.yaml")
    eval_corpus = resolve_under_experiments("data/eval_corpus.txt")
    train_corpus = resolve_under_experiments(
        str(base_cfg.get("train", {}).get("corpus", "data/train_corpus_extended.txt"))
    )
    model_name = str(base_cfg.get("model", "gpt2-small"))
    seq_len = int(base_cfg.get("train", {}).get("seq_len", 64))
    batch_size = int(base_cfg.get("train", {}).get("batch_size", 4))

    out_root = RESULTS / "control_quality_suite"
    out_root.mkdir(parents=True, exist_ok=True)

    per_seed: List[Dict[str, Any]] = []
    for seed in seeds:
        base_name = f"train_gpt2_small_geometry_quality_s{seed}"
        ctrl_name = f"train_gpt2_small_geometry_control_quality_s{seed}"
        base_dir = RESULTS / base_name
        ctrl_dir = RESULTS / ctrl_name

        print(f"\n=== seed={seed} ===", flush=True)
        if not eval_only:
            if _run_exists(base_dir):
                print(f"skip train baseline (exists): {base_dir}", flush=True)
            else:
                print(f"train baseline → {base_name}", flush=True)
                bcfg = _deep_update(base_cfg, seed, base_name, steps)
                if device != "auto":
                    bcfg["device"] = device
                train(
                    bcfg,
                    source_config=CONFIGS / "train_gpt2_small_geometry_quality.yaml",
                )
            if _run_exists(ctrl_dir):
                print(f"skip train control (exists): {ctrl_dir}", flush=True)
            else:
                print(f"train control → {ctrl_name}", flush=True)
                ccfg = _deep_update(ctrl_cfg, seed, ctrl_name, steps)
                if device != "auto":
                    ccfg["device"] = device
                train(
                    ccfg,
                    source_config=CONFIGS
                    / "train_gpt2_small_geometry_control_quality.yaml",
                )
        else:
            if not _run_exists(base_dir) or not _run_exists(ctrl_dir):
                raise FileNotFoundError(
                    f"eval-only missing runs for seed={seed}: {base_dir} / {ctrl_dir}"
                )

        report = evaluate_pair(
            baseline_dir=base_dir,
            control_dir=ctrl_dir,
            eval_corpus=eval_corpus,
            train_corpus=train_corpus,
            model_name=model_name,
            device=device,
            seq_len=seq_len,
            batch_size=batch_size,
            seed=seed,
            include_pretrained=not skip_pretrained,
        )
        seed_out = out_root / f"held_out_lm_eval_s{seed}.json"
        seed_out.write_text(json.dumps(report, indent=2) + "\n")
        v = report["verdict"]
        row = {
            "seed": seed,
            "baseline_dir": str(base_dir),
            "control_dir": str(ctrl_dir),
            "eval_loss_pretrained": (
                report["runs"].get("pretrained", {}).get("eval", {}).get("mean_loss")
            ),
            "eval_loss_baseline": report["runs"]["baseline"]["eval"]["mean_loss"],
            "eval_loss_control": report["runs"]["control"]["eval"]["mean_loss"],
            "eval_ppl_pretrained": (
                report["runs"].get("pretrained", {}).get("eval", {}).get("perplexity")
            ),
            "eval_ppl_baseline": report["runs"]["baseline"]["eval"]["perplexity"],
            "eval_ppl_control": report["runs"]["control"]["eval"]["perplexity"],
            "delta_eval_loss": v["delta_eval_loss_control_minus_baseline"],
            "control_wins_vs_baseline": v["control_lower_eval_loss"],
            "no_harm_vs_baseline": v["no_harm_eval_loss"],
            "eval_json": str(seed_out),
        }
        if row["eval_loss_pretrained"] is not None:
            row["control_closer_to_pretrained_than_baseline"] = abs(
                row["eval_loss_control"] - row["eval_loss_pretrained"]
            ) < abs(row["eval_loss_baseline"] - row["eval_loss_pretrained"])
            row["control_beats_pretrained"] = (
                row["eval_loss_control"] < row["eval_loss_pretrained"]
            )
        per_seed.append(row)
        print(
            f"seed={seed}  eval loss  pre={row['eval_loss_pretrained']}  "
            f"base={row['eval_loss_baseline']:.4f}  ctrl={row['eval_loss_control']:.4f}  "
            f"Δ={row['delta_eval_loss']:+.4f}  "
            f"{'WIN' if row['control_wins_vs_baseline'] else 'NO'}",
            flush=True,
        )

    deltas = [r["delta_eval_loss"] for r in per_seed]
    wins = sum(1 for r in per_seed if r["control_wins_vs_baseline"])
    summary = {
        "seeds": seeds,
        "steps": steps
        or int(base_cfg.get("train", {}).get("max_steps", 200)),
        "corpus": str(train_corpus),
        "eval_corpus": str(eval_corpus),
        "n_seeds": len(per_seed),
        "n_control_wins_vs_baseline": wins,
        "mean_delta_eval_loss": float(sum(deltas) / max(len(deltas), 1)),
        "per_seed": per_seed,
        "claim": (
            f"control beat baseline on held-out LM loss in {wins}/{len(per_seed)} seeds "
            f"(mean Δ={float(sum(deltas) / max(len(deltas), 1)):+.4f})"
        ),
    }
    if any(r.get("eval_loss_pretrained") is not None for r in per_seed):
        closer = sum(
            1 for r in per_seed if r.get("control_closer_to_pretrained_than_baseline")
        )
        beat_pre = sum(1 for r in per_seed if r.get("control_beats_pretrained"))
        summary["n_control_closer_to_pretrained_than_baseline"] = closer
        summary["n_control_beats_pretrained"] = beat_pre

    out_path = out_root / "summary.json"
    out_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"\n{summary['claim']}", flush=True)
    print(f"wrote {out_path}", flush=True)
    return summary


def main(argv: Optional[List[str]] = None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument(
        "--steps",
        type=int,
        default=None,
        help="Override max_steps (default: from quality YAML = 200)",
    )
    p.add_argument("--device", type=str, default="auto")
    p.add_argument(
        "--eval-only",
        action="store_true",
        help="Skip training; only score existing quality runs",
    )
    p.add_argument("--skip-pretrained", action="store_true")
    args = p.parse_args(argv)
    run_suite(
        seeds=list(args.seeds),
        steps=args.steps,
        device=args.device,
        eval_only=args.eval_only,
        skip_pretrained=args.skip_pretrained,
    )


if __name__ == "__main__":
    main()
