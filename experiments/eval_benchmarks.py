"""Standard LM benchmarks for control vs baseline (WikiText-2, LAMBADA, HellaSwag).

These are the \"is the model actually better?\" numbers — not train-bank geometry.

  python experiments/eval_benchmarks.py --model-source pretrained
  python experiments/eval_benchmarks.py \\
      --baseline experiments/results/train_gpt2_small_wikitext_s0 \\
      --control experiments/results/train_gpt2_small_wikitext_control_s0

WikiText-2 files: ``python experiments/fetch_wikitext2.py``
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.diff_checkpoints import resolve_device
from experiments.train_with_geometry import build_token_blocks, load_corpus_lines
from experiments.validate_control_geometry import _load_final_model
from scripts.helpers.paths import RESULTS, resolve_under_experiments
from scripts.helpers.seed import set_seed


def _load_pretrained(model_name: str, device: str):
    from transformer_lens import HookedTransformer

    model = HookedTransformer.from_pretrained(model_name, device=device)
    model.eval()
    return model


def load_model_source(
    source: str,
    *,
    model_name: str,
    device: str,
    run_dir: Optional[Path] = None,
):
    device = resolve_device(device)
    if source == "pretrained":
        return _load_pretrained(model_name, device), device, -1, None
    if run_dir is None:
        raise ValueError("run_dir required unless source=pretrained")
    model, device, step, weight_path = _load_final_model(run_dir, model_name, device)
    return model, device, step, weight_path


@torch.no_grad()
def wikitext_perplexity(
    model,
    lines: Sequence[str],
    *,
    seq_len: int,
    batch_size: int,
    device: str,
    max_blocks: Optional[int] = None,
) -> Dict[str, float]:
    """Token-weighted causal LM perplexity on packed WikiText blocks."""
    blocks = build_token_blocks(model, list(lines), seq_len=seq_len)
    if max_blocks is not None:
        blocks = blocks[: int(max_blocks)]
    n = int(blocks.shape[0])
    if n == 0:
        raise ValueError("empty WikiText blocks")

    total_nll = 0.0
    total_tok = 0
    seen = 0
    while seen < n:
        bs = min(batch_size, n - seen)
        batch = blocks[seen : seen + bs].to(device)
        seen += bs
        logits = model(batch)
        # [B, T-1, V] vs [B, T-1]
        nll = F.cross_entropy(
            logits[:, :-1].reshape(-1, logits.size(-1)),
            batch[:, 1:].reshape(-1),
            reduction="sum",
        )
        total_nll += float(nll.item())
        total_tok += int(batch[:, 1:].numel())

    mean_nll = total_nll / max(total_tok, 1)
    return {
        "mean_nll": mean_nll,
        "perplexity": float(math.exp(min(mean_nll, 50.0))),
        "n_tokens": int(total_tok),
        "n_blocks": n,
        "seq_len": int(seq_len),
    }


def _last_word_and_context(text: str) -> Optional[Tuple[str, str]]:
    words = text.strip().split()
    if len(words) < 2:
        return None
    target = words[-1]
    # strip trailing punctuation commonly attached to last word in LAMBADA
    context = " ".join(words[:-1])
    return context, target


@torch.no_grad()
def lambada_accuracy(
    model,
    texts: Sequence[str],
    *,
    device: str,
    max_examples: Optional[int] = None,
) -> Dict[str, float]:
    """Greedy last-word accuracy (GPT-2 BPE, leading-space target)."""
    correct = 0
    total = 0
    n = len(texts) if max_examples is None else min(len(texts), int(max_examples))
    for i in range(n):
        pair = _last_word_and_context(texts[i])
        if pair is None:
            continue
        context, target_word = pair
        # GPT-2 usually encodes mid-text words with a leading space
        target_variants = [f" {target_word}", target_word]
        ctx_ids = model.to_tokens(context, prepend_bos=True).to(device)
        ok = False
        for tv in target_variants:
            tgt_ids = model.to_tokens(tv, prepend_bos=False).to(device)
            if tgt_ids.numel() == 0:
                continue
            # Teacher-forced greedy match over target tokens
            prefix = ctx_ids
            match = True
            for t in range(tgt_ids.shape[-1]):
                logits = model(prefix)
                pred = int(logits[0, -1].argmax().item())
                want = int(tgt_ids[0, t].item())
                if pred != want:
                    match = False
                    break
                prefix = torch.cat([prefix, tgt_ids[:, t : t + 1]], dim=1)
            if match:
                ok = True
                break
        total += 1
        if ok:
            correct += 1
    acc = correct / max(total, 1)
    return {
        "accuracy": float(acc),
        "correct": int(correct),
        "n": int(total),
    }


@torch.no_grad()
def _continuation_logprob(model, context: str, ending: str, device: str) -> float:
    """Sum logprob of ``ending`` tokens given ``context`` (teacher forced)."""
    # Prefer leading space on ending for GPT-2 mid-string continuations
    endings = [ending if ending.startswith(" ") else f" {ending}", ending]
    best = None
    for end in endings:
        ctx = model.to_tokens(context, prepend_bos=True).to(device)
        tgt = model.to_tokens(end, prepend_bos=False).to(device)
        if tgt.numel() == 0:
            continue
        full = torch.cat([ctx, tgt], dim=1)
        logits = model(full)
        # positions that predict each target token
        start = ctx.shape[1] - 1
        logprobs = F.log_softmax(logits[0, start : start + tgt.shape[1]], dim=-1)
        lp = 0.0
        for i in range(tgt.shape[1]):
            lp += float(logprobs[i, int(tgt[0, i].item())].item())
        if best is None or lp > best:
            best = lp
    return float(best if best is not None else -1e9)


@torch.no_grad()
def hellaswag_accuracy(
    model,
    rows: Sequence[Dict[str, Any]],
    *,
    device: str,
    max_examples: Optional[int] = None,
) -> Dict[str, float]:
    """Zero-shot HellaSwag: pick ending with highest continuation logprob."""
    correct = 0
    total = 0
    n = len(rows) if max_examples is None else min(len(rows), int(max_examples))
    for i in range(n):
        row = rows[i]
        ctx = (row.get("ctx") or "").strip()
        endings = row.get("endings") or []
        label = int(row.get("label", -1))
        if not ctx or len(endings) < 2 or label < 0:
            continue
        scores = [
            _continuation_logprob(model, ctx, str(e), device) for e in endings
        ]
        pred = int(max(range(len(scores)), key=lambda j: scores[j]))
        total += 1
        if pred == label:
            correct += 1
    return {
        "accuracy": float(correct / max(total, 1)),
        "correct": int(correct),
        "n": int(total),
    }


def _load_lambada_texts() -> List[str]:
    from datasets import load_dataset

    ds = load_dataset("EleutherAI/lambada_openai", split="test")
    return [str(r["text"]) for r in ds if r.get("text")]


def _load_hellaswag_rows() -> List[Dict[str, Any]]:
    from datasets import load_dataset

    # Prefer cache — Hub HEAD checks often flake offline / behind DNS.
    try:
        ds = load_dataset("Rowan/hellaswag", split="validation", download_mode="reuse_cache_if_exists")
    except Exception:
        import os

        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
        ds = load_dataset("Rowan/hellaswag", split="validation")
    out = []
    for r in ds:
        lab = r.get("label", -1)
        if isinstance(lab, str) and lab.isdigit():
            label = int(lab)
        elif isinstance(lab, (int, float)):
            label = int(lab)
        else:
            label = -1
        out.append(
            {
                "ctx": r.get("ctx") or r.get("ctx_a") or "",
                "endings": list(r.get("endings") or []),
                "label": label,
            }
        )
    return out


def evaluate_model(
    model,
    *,
    device: str,
    wiki_valid: Optional[Path],
    wiki_test: Optional[Path],
    seq_len: int,
    batch_size: int,
    max_wiki_blocks: Optional[int],
    lambada: bool,
    hellaswag: bool,
    max_lambada: Optional[int],
    max_hellaswag: Optional[int],
) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    if wiki_valid is not None and wiki_valid.is_file():
        lines = load_corpus_lines(wiki_valid)
        out["wikitext2_valid"] = wikitext_perplexity(
            model,
            lines,
            seq_len=seq_len,
            batch_size=batch_size,
            device=device,
            max_blocks=max_wiki_blocks,
        )
        print(
            f"  WikiText-2 valid  ppl={out['wikitext2_valid']['perplexity']:.3f}  "
            f"nll={out['wikitext2_valid']['mean_nll']:.4f}",
            flush=True,
        )
    if wiki_test is not None and wiki_test.is_file():
        lines = load_corpus_lines(wiki_test)
        out["wikitext2_test"] = wikitext_perplexity(
            model,
            lines,
            seq_len=seq_len,
            batch_size=batch_size,
            device=device,
            max_blocks=max_wiki_blocks,
        )
        print(
            f"  WikiText-2 test   ppl={out['wikitext2_test']['perplexity']:.3f}  "
            f"nll={out['wikitext2_test']['mean_nll']:.4f}",
            flush=True,
        )
    if lambada:
        texts = _load_lambada_texts()
        out["lambada"] = lambada_accuracy(
            model, texts, device=device, max_examples=max_lambada
        )
        print(
            f"  LAMBADA           acc={out['lambada']['accuracy']:.4f}  "
            f"({out['lambada']['correct']}/{out['lambada']['n']})",
            flush=True,
        )
    if hellaswag:
        rows = _load_hellaswag_rows()
        out["hellaswag"] = hellaswag_accuracy(
            model, rows, device=device, max_examples=max_hellaswag
        )
        print(
            f"  HellaSwag         acc={out['hellaswag']['accuracy']:.4f}  "
            f"({out['hellaswag']['correct']}/{out['hellaswag']['n']})",
            flush=True,
        )
    return out


def compare_reports(reports: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """Derive win/lose vs baseline on each benchmark."""
    if "baseline" not in reports or "control" not in reports:
        return {}
    b, c = reports["baseline"], reports["control"]
    verdict: Dict[str, Any] = {}

    def ppl_key(split: str) -> Optional[str]:
        return split if split in b and split in c else None

    for split in ("wikitext2_valid", "wikitext2_test"):
        if ppl_key(split):
            bp = float(b[split]["perplexity"])
            cp = float(c[split]["perplexity"])
            verdict[split] = {
                "baseline_ppl": bp,
                "control_ppl": cp,
                "delta_ppl": cp - bp,
                "control_wins": cp < bp,
            }
    for task in ("lambada", "hellaswag"):
        if task in b and task in c:
            ba = float(b[task]["accuracy"])
            ca = float(c[task]["accuracy"])
            verdict[task] = {
                "baseline_acc": ba,
                "control_acc": ca,
                "delta_acc": ca - ba,
                "control_wins": ca > ba,
            }

    if "pretrained" in reports and "wikitext2_test" in reports["pretrained"]:
        pre = float(reports["pretrained"]["wikitext2_test"]["perplexity"])
        if "wikitext2_test" in verdict:
            verdict["wikitext2_test"]["pretrained_ppl"] = pre
            verdict["wikitext2_test"]["control_beats_pretrained"] = (
                verdict["wikitext2_test"]["control_ppl"] < pre
            )
            verdict["wikitext2_test"]["baseline_beats_pretrained"] = (
                verdict["wikitext2_test"]["baseline_ppl"] < pre
            )
    return verdict


def main(argv: Optional[List[str]] = None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", type=str, default="gpt2-small")
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--seq-len", type=int, default=128)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument(
        "--wiki-dir",
        type=Path,
        default=resolve_under_experiments("data/wikitext2"),
    )
    p.add_argument("--skip-wiki", action="store_true")
    p.add_argument("--skip-lambada", action="store_true")
    p.add_argument("--skip-hellaswag", action="store_true")
    p.add_argument(
        "--max-wiki-blocks",
        type=int,
        default=None,
        help="Cap WikiText blocks (smoke). Default: full split.",
    )
    p.add_argument("--max-lambada", type=int, default=None)
    p.add_argument(
        "--max-hellaswag",
        type=int,
        default=1000,
        help="HellaSwag val subset (default 1000; full ~10k is slow)",
    )
    p.add_argument(
        "--model-source",
        choices=["pretrained", "run"],
        default=None,
        help="Score a single model (pretrained or --run-dir)",
    )
    p.add_argument("--run-dir", type=Path, default=None)
    p.add_argument("--baseline", type=Path, default=None)
    p.add_argument("--control", type=Path, default=None)
    p.add_argument(
        "--include-pretrained",
        action="store_true",
        help="Also score vanilla gpt2-small when comparing pair",
    )
    p.add_argument("-o", "--out", type=Path, default=None)
    args = p.parse_args(argv)

    set_seed(args.seed)
    wiki_valid = None if args.skip_wiki else args.wiki_dir / "wiki.valid.txt"
    wiki_test = None if args.skip_wiki else args.wiki_dir / "wiki.test.txt"
    if not args.skip_wiki and (wiki_valid is None or not wiki_valid.is_file()):
        raise FileNotFoundError(
            f"missing {wiki_valid}; run: python experiments/fetch_wikitext2.py"
        )

    sources: List[Tuple[str, Optional[Path]]] = []
    if args.model_source == "pretrained":
        sources = [("pretrained", None)]
    elif args.model_source == "run":
        if args.run_dir is None:
            raise SystemExit("--run-dir required with --model-source run")
        sources = [("run", args.run_dir)]
    else:
        # pair mode
        if args.baseline is None or args.control is None:
            # defaults: wikitext quality runs if present
            args.baseline = args.baseline or (
                RESULTS / "train_gpt2_small_wikitext_s0"
            )
            args.control = args.control or (
                RESULTS / "train_gpt2_small_wikitext_control_s0"
            )
        if args.include_pretrained:
            sources.append(("pretrained", None))
        sources.append(("baseline", args.baseline))
        sources.append(("control", args.control))

    reports: Dict[str, Any] = {
        "model": args.model,
        "device": resolve_device(args.device),
        "seq_len": args.seq_len,
        "benchmarks": {},
    }

    for label, run_dir in sources:
        print(f"\n=== {label} ===", flush=True)
        model, device, step, weight_path = load_model_source(
            "pretrained" if run_dir is None else "run",
            model_name=args.model,
            device=args.device,
            run_dir=run_dir,
        )
        metrics = evaluate_model(
            model,
            device=device,
            wiki_valid=wiki_valid,
            wiki_test=wiki_test,
            seq_len=args.seq_len,
            batch_size=args.batch_size,
            max_wiki_blocks=args.max_wiki_blocks,
            lambada=not args.skip_lambada,
            hellaswag=not args.skip_hellaswag,
            max_lambada=args.max_lambada,
            max_hellaswag=args.max_hellaswag,
        )
        reports["benchmarks"][label] = {
            "step": step,
            "weight_path": str(weight_path) if weight_path else None,
            "run_dir": str(run_dir) if run_dir else None,
            **metrics,
        }
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    bench_only = {
        k: {kk: vv for kk, vv in v.items() if kk not in ("step", "weight_path", "run_dir")}
        for k, v in reports["benchmarks"].items()
    }
    reports["verdict"] = compare_reports(bench_only)

    if reports["verdict"]:
        print("\n=== verdict (control vs baseline) ===", flush=True)
        for k, v in reports["verdict"].items():
            print(f"  {k}: {v}", flush=True)

    out = args.out
    if out is None:
        out = RESULTS / "benchmark_control" / "benchmarks.json"
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(reports, indent=2) + "\n")
    print(f"\nwrote {out}", flush=True)


if __name__ == "__main__":
    main()
