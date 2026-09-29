"""Held-out LM eval: does soft geo-control improve next-token quality?

Scores mean causal LM loss (+ perplexity) on a frozen eval corpus that is
**disjoint** from the train corpus. Also scores the train corpus so you can
see memorization vs generalization.

Success for \"control makes the model better\"::

  eval_loss(control) < eval_loss(baseline)   (pre-registered margin optional)

Train-loss alone does **not** count — that can be bank/geo gaming + overfit.

Example::

  python experiments/eval_control_lm.py
  python experiments/eval_control_lm.py \\
      --baseline experiments/results/train_gpt2_small_geometry \\
      --control experiments/results/train_gpt2_small_geometry_control \\
      --eval-corpus experiments/data/eval_corpus.txt
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.diff_checkpoints import resolve_device
from experiments.train_with_geometry import (
    build_token_blocks,
    causal_lm_loss,
    load_corpus_lines,
)
from experiments.validate_control_geometry import _load_final_model
from scripts.helpers.paths import RESULTS, resolve_under_experiments
from scripts.helpers.seed import set_seed


@torch.no_grad()
def score_corpus(
    model,
    lines: Sequence[str],
    *,
    seq_len: int,
    batch_size: int,
    device: str,
    seed: int = 0,
) -> Dict[str, float]:
    """Mean causal LM loss over packed blocks; perplexity = exp(loss)."""
    blocks = build_token_blocks(model, list(lines), seq_len=seq_len)
    if blocks.shape[0] == 0:
        raise ValueError("no token blocks from corpus")
    # Full pass over all blocks in order (deterministic).
    n = blocks.shape[0]
    losses: List[float] = []
    n_tok = 0
    seen = 0
    while seen < n:
        bs = min(batch_size, n - seen)
        batch = blocks[seen : seen + bs].to(device)
        seen += bs
        logits = model(batch)
        loss = causal_lm_loss(logits, batch)
        losses.append(float(loss.item()))
        n_tok += int(batch[:, 1:].numel())
    mean_loss = float(sum(losses) / max(len(losses), 1))
    return {
        "mean_loss": mean_loss,
        "perplexity": float(math.exp(min(mean_loss, 50.0))),
        "n_blocks": int(n),
        "n_tokens_scored": int(n_tok),
        "seq_len": int(seq_len),
    }


def _load_pretrained(model_name: str, device: str):
    from transformer_lens import HookedTransformer

    model = HookedTransformer.from_pretrained(model_name, device=device)
    model.eval()
    return model


def _run_label(name: str, stats: Dict[str, float]) -> str:
    return (
        f"{name:12s}  loss={stats['mean_loss']:.4f}  "
        f"ppl={stats['perplexity']:.3f}  "
        f"blocks={stats['n_blocks']}"
    )


def evaluate_pair(
    *,
    baseline_dir: Path,
    control_dir: Path,
    eval_corpus: Path,
    train_corpus: Optional[Path],
    model_name: str,
    device: str,
    seq_len: int,
    batch_size: int,
    seed: int,
    include_pretrained: bool,
) -> Dict[str, Any]:
    set_seed(seed)
    device = resolve_device(device)
    eval_lines = load_corpus_lines(eval_corpus)
    train_lines = load_corpus_lines(train_corpus) if train_corpus else None

    # Sanity: eval must not be a trivial copy of train.
    train_set = {ln.strip() for ln in (train_lines or [])}
    overlap = [ln for ln in eval_lines if ln.strip() in train_set]
    if overlap:
        raise ValueError(
            f"eval corpus overlaps train corpus ({len(overlap)} lines); "
            f"e.g. {overlap[0]!r}"
        )

    results: Dict[str, Any] = {
        "model": model_name,
        "device": device,
        "seed": seed,
        "seq_len": seq_len,
        "batch_size": batch_size,
        "eval_corpus": str(eval_corpus),
        "train_corpus": str(train_corpus) if train_corpus else None,
        "n_eval_lines": len(eval_lines),
        "baseline_dir": str(baseline_dir),
        "control_dir": str(control_dir),
        "runs": {},
    }

    specs: List[tuple[str, Optional[Path]]] = [
        ("baseline", baseline_dir),
        ("control", control_dir),
    ]
    if include_pretrained:
        specs.insert(0, ("pretrained", None))

    for label, run_dir in specs:
        if run_dir is None:
            model = _load_pretrained(model_name, device)
            step, weight_path = -1, None
        else:
            model, device, step, weight_path = _load_final_model(
                run_dir, model_name, device
            )
        entry: Dict[str, Any] = {
            "step": step,
            "weight_path": str(weight_path) if weight_path else None,
        }
        entry["eval"] = score_corpus(
            model,
            eval_lines,
            seq_len=seq_len,
            batch_size=batch_size,
            device=device,
            seed=seed,
        )
        if train_lines is not None:
            entry["train"] = score_corpus(
                model,
                train_lines,
                seq_len=seq_len,
                batch_size=batch_size,
                device=device,
                seed=seed,
            )
        results["runs"][label] = entry
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    base = results["runs"]["baseline"]["eval"]
    ctrl = results["runs"]["control"]["eval"]
    delta_loss = float(ctrl["mean_loss"] - base["mean_loss"])
    delta_ppl = float(ctrl["perplexity"] - base["perplexity"])
    # Pre-registered soft win: strictly lower eval loss.
    control_wins = delta_loss < 0.0
    # No-harm slack (same spirit as validate_control_geometry downstream_loss).
    slack = 0.05
    no_harm = delta_loss <= slack

    verdict = {
        "control_lower_eval_loss": control_wins,
        "delta_eval_loss_control_minus_baseline": delta_loss,
        "delta_eval_ppl_control_minus_baseline": delta_ppl,
        "no_harm_eval_loss": no_harm,
        "slack": slack,
        "claim": (
            "control beats baseline on held-out LM loss"
            if control_wins
            else (
                "control does not hurt held-out LM loss (within slack)"
                if no_harm
                else "control worse on held-out LM loss"
            )
        ),
    }
    if train_lines is not None:
        bt = results["runs"]["baseline"]["train"]
        ct = results["runs"]["control"]["train"]
        verdict["delta_train_loss_control_minus_baseline"] = float(
            ct["mean_loss"] - bt["mean_loss"]
        )
        # Flag: wins train, loses eval → likely overfit / not quality win.
        verdict["train_win_eval_lose"] = (
            float(ct["mean_loss"] - bt["mean_loss"]) < 0.0 and not control_wins
        )

    results["verdict"] = verdict
    return results


def main(argv: Optional[List[str]] = None) -> None:
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
    p.add_argument(
        "--eval-corpus",
        type=Path,
        default=resolve_under_experiments("data/eval_corpus.txt"),
    )
    p.add_argument(
        "--train-corpus",
        type=Path,
        default=resolve_under_experiments("data/train_corpus.txt"),
        help="Also score train corpus (memorization check). Pass empty to skip.",
    )
    p.add_argument("--no-train-corpus", action="store_true")
    p.add_argument("--model", type=str, default="gpt2-small")
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--seq-len", type=int, default=64)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--no-pretrained",
        action="store_true",
        help="Skip scoring vanilla pretrained gpt2-small",
    )
    p.add_argument(
        "-o",
        "--out",
        type=Path,
        default=None,
        help="JSON out (default: control_vs_baseline_seed0/held_out_lm_eval.json)",
    )
    args = p.parse_args(argv)

    train_corpus = None if args.no_train_corpus else args.train_corpus
    report = evaluate_pair(
        baseline_dir=args.baseline,
        control_dir=args.control,
        eval_corpus=args.eval_corpus,
        train_corpus=train_corpus,
        model_name=args.model,
        device=args.device,
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        seed=args.seed,
        include_pretrained=not args.no_pretrained,
    )

    print(
        f"eval_control_lm: baseline={args.baseline.name}  "
        f"control={args.control.name}"
    )
    print(f"eval_corpus={args.eval_corpus}  ({report['n_eval_lines']} lines)")
    print("held-out eval:")
    for label, entry in report["runs"].items():
        print(f"  {_run_label(label, entry['eval'])}")
    if train_corpus is not None:
        print("train corpus (reference):")
        for label, entry in report["runs"].items():
            if "train" in entry:
                print(f"  {_run_label(label, entry['train'])}")

    v = report["verdict"]
    flag = "WIN" if v["control_lower_eval_loss"] else ("OK" if v["no_harm_eval_loss"] else "LOSE")
    print(
        f"\n[{flag}] {v['claim']}  "
        f"(Δloss={v['delta_eval_loss_control_minus_baseline']:+.4f}, "
        f"Δppl={v['delta_eval_ppl_control_minus_baseline']:+.4f})"
    )
    if v.get("train_win_eval_lose"):
        print("  note: control better on train than on this eval split")

    out = args.out
    if out is None:
        out = args.control.parent / "control_vs_baseline_seed0" / "held_out_lm_eval.json"
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
