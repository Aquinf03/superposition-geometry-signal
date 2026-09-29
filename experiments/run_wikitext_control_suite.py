"""Train on WikiText-2 ± geo-control, then score WikiText / LAMBADA / HellaSwag.

  python experiments/fetch_wikitext2.py
  python experiments/run_wikitext_control_suite.py
  python experiments/run_wikitext_control_suite.py --seeds 0 --steps 400
  python experiments/run_wikitext_control_suite.py --eval-only
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

from experiments.eval_benchmarks import (
    compare_reports,
    evaluate_model,
    load_model_source,
)
from experiments.fetch_wikitext2 import fetch as fetch_wikitext2
from experiments.train_with_geometry import load_config, train
from scripts.helpers.paths import CONFIGS, RESULTS, resolve_under_experiments


def _exists(run_dir: Path) -> bool:
    if (run_dir / "model_final.pt").is_file():
        return True
    ckpt = run_dir / "checkpoints"
    return ckpt.is_dir() and any(ckpt.glob("step_*.pt"))


def run_suite(
    *,
    seeds: List[int],
    steps: Optional[int],
    device: str,
    eval_only: bool,
    max_hellaswag: int,
    max_lambada: Optional[int],
    max_wiki_blocks: Optional[int],
    skip_fetch: bool,
) -> Dict[str, Any]:
    wiki_dir = resolve_under_experiments("data/wikitext2")
    if not (wiki_dir / "wiki.train.txt").is_file():
        if skip_fetch:
            raise FileNotFoundError(f"missing {wiki_dir}/wiki.train.txt")
        print("fetching WikiText-2 …", flush=True)
        fetch_wikitext2(wiki_dir)

    base_tmpl = load_config(CONFIGS / "train_gpt2_small_wikitext.yaml")
    ctrl_tmpl = load_config(CONFIGS / "train_gpt2_small_wikitext_control.yaml")
    model_name = str(base_tmpl.get("model", "gpt2-small"))
    seq_len = int(base_tmpl.get("train", {}).get("seq_len", 128))
    batch_size = int(base_tmpl.get("train", {}).get("batch_size", 4))

    out_root = RESULTS / "benchmark_control"
    out_root.mkdir(parents=True, exist_ok=True)

    per_seed: List[Dict[str, Any]] = []
    for seed in seeds:
        base_name = f"train_gpt2_small_wikitext_s{seed}"
        ctrl_name = f"train_gpt2_small_wikitext_control_s{seed}"
        base_dir = RESULTS / base_name
        ctrl_dir = RESULTS / ctrl_name

        print(f"\n######## seed={seed} ########", flush=True)
        if not eval_only:
            for label, tmpl, name, dest in [
                ("baseline", base_tmpl, base_name, base_dir),
                ("control", ctrl_tmpl, ctrl_name, ctrl_dir),
            ]:
                if _exists(dest):
                    print(f"skip train {label}: {dest}", flush=True)
                    continue
                cfg = copy.deepcopy(tmpl)
                cfg["seed"] = int(seed)
                cfg.setdefault("logging", {})["run_name"] = name
                if steps is not None:
                    cfg.setdefault("train", {})["max_steps"] = int(steps)
                if device != "auto":
                    cfg["device"] = device
                print(f"train {label} → {name}", flush=True)
                src = (
                    CONFIGS / "train_gpt2_small_wikitext.yaml"
                    if label == "baseline"
                    else CONFIGS / "train_gpt2_small_wikitext_control.yaml"
                )
                train(cfg, source_config=src)

        reports: Dict[str, Any] = {}
        for label, run_dir in [
            ("pretrained", None),
            ("baseline", base_dir),
            ("control", ctrl_dir),
        ]:
            print(f"\n--- bench {label} (seed={seed}) ---", flush=True)
            model, dev, step, weight_path = load_model_source(
                "pretrained" if run_dir is None else "run",
                model_name=model_name,
                device=device,
                run_dir=run_dir,
            )
            metrics = evaluate_model(
                model,
                device=dev,
                wiki_valid=wiki_dir / "wiki.valid.txt",
                wiki_test=wiki_dir / "wiki.test.txt",
                seq_len=seq_len,
                batch_size=batch_size,
                max_wiki_blocks=max_wiki_blocks,
                lambada=True,
                hellaswag=True,
                max_lambada=max_lambada,
                max_hellaswag=max_hellaswag,
            )
            reports[label] = metrics
            del model

        verdict = compare_reports(reports)
        seed_report = {
            "seed": seed,
            "baseline_dir": str(base_dir),
            "control_dir": str(ctrl_dir),
            "benchmarks": reports,
            "verdict": verdict,
        }
        seed_path = out_root / f"benchmarks_s{seed}.json"
        seed_path.write_text(json.dumps(seed_report, indent=2) + "\n")
        print(f"wrote {seed_path}", flush=True)
        per_seed.append(seed_report)

    # Aggregate
    def _agg(metric_path: str, lower_better: bool) -> Dict[str, Any]:
        wins = 0
        deltas = []
        for rep in per_seed:
            v = (rep.get("verdict") or {}).get(metric_path)
            if not v:
                continue
            key = "delta_ppl" if lower_better else "delta_acc"
            deltas.append(float(v[key]))
            if v.get("control_wins"):
                wins += 1
        if not deltas:
            return {}
        return {
            "n": len(deltas),
            "n_control_wins": wins,
            "mean_delta": float(sum(deltas) / len(deltas)),
        }

    summary = {
        "seeds": seeds,
        "steps": steps or int(base_tmpl.get("train", {}).get("max_steps", 400)),
        "wiki_dir": str(wiki_dir),
        "max_hellaswag": max_hellaswag,
        "aggregates": {
            "wikitext2_test": _agg("wikitext2_test", True),
            "wikitext2_valid": _agg("wikitext2_valid", True),
            "lambada": _agg("lambada", False),
            "hellaswag": _agg("hellaswag", False),
        },
        "per_seed_paths": [
            str(out_root / f"benchmarks_s{s}.json") for s in seeds
        ],
    }
    out_path = out_root / "summary.json"
    out_path.write_text(json.dumps(summary, indent=2) + "\n")
    print("\n=== aggregate ===", flush=True)
    print(json.dumps(summary["aggregates"], indent=2), flush=True)
    print(f"wrote {out_path}", flush=True)
    return summary


def main(argv: Optional[List[str]] = None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seeds", type=int, nargs="+", default=[0])
    p.add_argument("--steps", type=int, default=None)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--eval-only", action="store_true")
    p.add_argument("--max-hellaswag", type=int, default=1000)
    p.add_argument("--max-lambada", type=int, default=None)
    p.add_argument("--max-wiki-blocks", type=int, default=None)
    p.add_argument("--skip-fetch", action="store_true")
    args = p.parse_args(argv)
    run_suite(
        seeds=list(args.seeds),
        steps=args.steps,
        device=args.device,
        eval_only=args.eval_only,
        max_hellaswag=args.max_hellaswag,
        max_lambada=args.max_lambada,
        max_wiki_blocks=args.max_wiki_blocks,
        skip_fetch=args.skip_fetch,
    )


if __name__ == "__main__":
    main()
