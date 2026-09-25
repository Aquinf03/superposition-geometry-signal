"""Hero run: fine-tune a small LM and print live loss | geometry every N steps.

Uses TransformerLens GPT-2 small + a local text corpus (no edit algorithms).

Example live line (multi-layer):
  step=12  loss=0.41  |  geometry L4: interference=… spectral=… coact=…  |  L8: …  |  L11: …

Run (you run this):
  python scripts/train_with_geometry.py --config configs/train_gpt2_small_geometry.yaml
  python scripts/plot_signals.py --csv results/train_gpt2_small_geometry/signals.csv
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

from scripts.geometry import (
    DEFAULT_PROBE_PROMPTS,
    collect_mlp_out_banks,
    compute_geometry_metrics,
)
from scripts.run_artifacts import save_run_artifacts
from scripts.seed import set_seed
from scripts.signals import SignalLogger


def resolve_device(name: str) -> str:
    if name != "auto":
        return name
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def resolve_layers(geo_cfg: Dict[str, Any]) -> List[int]:
    """Prefer ``layers: [...]``; fall back to single ``layer``."""
    if geo_cfg.get("layers"):
        return [int(L) for L in geo_cfg["layers"]]
    return [int(geo_cfg.get("layer", 8))]


def load_config(path: Path) -> Dict[str, Any]:
    with path.open() as f:
        return yaml.safe_load(f)


def load_corpus_lines(path: Path) -> List[str]:
    text = path.read_text(encoding="utf-8")
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if not lines:
        raise ValueError(f"empty corpus: {path}")
    return lines


def build_token_blocks(model, lines: List[str], seq_len: int) -> torch.Tensor:
    """Pack corpus into contiguous token blocks of shape [n_blocks, seq_len]."""
    ids: List[int] = []
    eos = int(model.tokenizer.eos_token_id)
    for line in lines:
        tok = model.to_tokens(line, prepend_bos=False)[0].tolist()
        ids.extend(int(i) for i in tok)
        ids.append(eos)

    if len(ids) < seq_len:
        reps = (seq_len // max(len(ids), 1)) + 2
        ids = ids * reps

    n_blocks = len(ids) // seq_len
    ids = ids[: n_blocks * seq_len]
    return torch.tensor(ids, dtype=torch.long).view(n_blocks, seq_len)


class BlockLoader:
    def __init__(self, blocks: torch.Tensor, batch_size: int, seed: int = 0) -> None:
        self.blocks = blocks
        self.batch_size = batch_size
        self.g = torch.Generator().manual_seed(seed)
        self._perm = torch.randperm(blocks.shape[0], generator=self.g)
        self._i = 0

    def next(self) -> torch.Tensor:
        n = self.blocks.shape[0]
        if self._i + self.batch_size > n:
            self._perm = torch.randperm(n, generator=self.g)
            self._i = 0
        idx = self._perm[self._i : self._i + self.batch_size]
        self._i += self.batch_size
        return self.blocks[idx]


def _mean_geometry_on_bank(
    bank: torch.Tensor, *, top_k: int, metrics: List[str]
) -> Dict[str, float]:
    acc = {m: 0.0 for m in metrics}
    n = bank.shape[0]
    if n < 2:
        return {m: 0.0 for m in metrics}
    for i in range(n):
        query = bank[i]
        others = torch.cat([bank[:i], bank[i + 1 :]], dim=0)
        vals = compute_geometry_metrics(
            query, others, top_k=min(top_k, others.shape[0]), metrics=metrics
        )
        for m in metrics:
            acc[m] += vals[m]
    return {m: acc[m] / n for m in metrics}


@torch.no_grad()
def measure_geometry_layers(
    model,
    *,
    layers: List[int],
    probe_prompts: List[str],
    positions: str,
    top_k: int,
    metrics: List[str],
) -> Dict[str, float]:
    """Mean geometry per layer; keys are ``L{layer}_{metric}`` for the logger."""
    was_training = model.training
    model.eval()
    banks = collect_mlp_out_banks(
        model, layers=layers, prompts=probe_prompts, positions=positions
    )
    out: Dict[str, float] = {}
    for L in layers:
        vals = _mean_geometry_on_bank(banks[L], top_k=top_k, metrics=metrics)
        for m, v in vals.items():
            out[f"L{L}_{m}"] = v
    if was_training:
        model.train()
    return out


def causal_lm_loss(logits: torch.Tensor, tokens: torch.Tensor) -> torch.Tensor:
    # logits: [B, T, V], tokens: [B, T]
    return torch.nn.functional.cross_entropy(
        logits[:, :-1].reshape(-1, logits.size(-1)),
        tokens[:, 1:].reshape(-1),
    )


def train(cfg: Dict[str, Any], source_config: Optional[Path] = None) -> Dict[str, Any]:
    seed = set_seed(int(cfg.get("seed", 0)))
    device = resolve_device(str(cfg.get("device", "auto")))
    train_cfg = cfg["train"]
    geo_cfg = cfg.get("geometry", {})
    log_cfg = cfg.get("logging", {})

    from transformer_lens import HookedTransformer

    model_name = cfg.get("model", "gpt2-small")
    model = HookedTransformer.from_pretrained(model_name, device=device)
    model.train()

    corpus_path = ROOT / train_cfg["corpus"]
    lines = load_corpus_lines(corpus_path)
    seq_len = int(train_cfg.get("seq_len", 64))
    batch_size = int(train_cfg.get("batch_size", 4))
    max_steps = int(train_cfg.get("max_steps", 40))
    lr = float(train_cfg.get("lr", 5e-5))
    weight_decay = float(train_cfg.get("weight_decay", 0.01))
    grad_clip = float(train_cfg.get("grad_clip", 1.0))

    blocks = build_token_blocks(model, lines, seq_len=seq_len)
    loader = BlockLoader(blocks, batch_size=batch_size, seed=seed)

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    layers = resolve_layers(geo_cfg)
    every_n = int(geo_cfg.get("every_n_steps", 1))
    top_k = int(geo_cfg.get("top_k_neighbors", 8))
    positions = str(geo_cfg.get("bank_positions", "last"))
    metrics = list(
        geo_cfg.get(
            "metrics",
            ["interference_mean", "spectral_participation", "coactivation_overlap"],
        )
    )
    probes = list(geo_cfg.get("probe_prompts") or DEFAULT_PROBE_PROMPTS)
    save_every = int(log_cfg.get("save_every_n_steps", 0) or 0)

    logger = SignalLogger(
        results_dir=log_cfg.get("results_dir", "results"),
        run_name=log_cfg.get("run_name"),
        live_print=bool(log_cfg.get("live_print", True)),
    )
    artifacts = save_run_artifacts(
        logger.root,
        cfg,
        seed,
        source_config=source_config,
        extra_meta={
            "model": model_name,
            "device": device,
            "hero": "train_with_geometry",
            "layers": layers,
        },
    )
    logger.write_meta(
        seed=seed,
        model=model_name,
        device=device,
        train=train_cfg,
        geometry=geo_cfg,
        layers=layers,
        n_blocks=int(blocks.shape[0]),
        artifacts=artifacts,
        note="hero: fine-tune with live loss | multi-layer geometry",
    )

    ckpt_dir = logger.root / "checkpoints"
    if save_every > 0:
        ckpt_dir.mkdir(parents=True, exist_ok=True)

    last_loss = None
    last_geo: Dict[str, float] = {}

    for step in range(max_steps):
        batch = loader.next().to(device)
        logits = model(batch)
        loss = causal_lm_loss(logits, batch)
        loss.backward()
        if grad_clip and grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        opt.step()
        opt.zero_grad(set_to_none=True)

        last_loss = float(loss.item())

        if step % max(every_n, 1) == 0:
            last_geo = measure_geometry_layers(
                model,
                layers=layers,
                probe_prompts=probes,
                positions=positions,
                top_k=top_k,
                metrics=metrics,
            )
            logger.log(step=step, loss=last_loss, geometry=last_geo)

        if save_every > 0 and (step + 1) % save_every == 0:
            ckpt_path = ckpt_dir / f"step_{step:05d}.pt"
            torch.save({"step": step, "model_state": model.state_dict(), "seed": seed}, ckpt_path)

    # Final checkpoint
    final_path = logger.root / "model_final.pt"
    torch.save({"step": max_steps - 1, "model_state": model.state_dict(), "seed": seed}, final_path)

    result = {
        "seed": seed,
        "model": model_name,
        "device": device,
        "max_steps": max_steps,
        "layers": layers,
        "final_loss": last_loss,
        "final_geometry": last_geo,
        "signals_csv": str(logger.csv_path),
        "meta_json": str(logger.meta_path),
        "artifacts": artifacts,
        "model_final": str(final_path),
    }
    out = logger.root / "train_result.json"
    out.write_text(json.dumps(result, indent=2) + "\n")
    result["result_json"] = str(out)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune small LM with live loss|geometry")
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "train_gpt2_small_geometry.yaml",
    )
    args = parser.parse_args()
    cfg = load_config(args.config)
    result = train(cfg, source_config=args.config)
    print(json.dumps(result, indent=2))
    print("\nDone (hero training run).")
    print(f"final_loss={result['final_loss']:.4f}")
    print(f"signals={result['signals_csv']}")
    print(f"wrote {result['result_json']}")
    print("Plot with:")
    print(f"  python scripts/plot_signals.py --csv {result['signals_csv']}")


if __name__ == "__main__":
    main()
