"""Minimal ROME-style rank-one factual edit (TransformerLens).

Simplified baseline (matches the Aquin experimental editor sketch):
  1) key k  = MLP post-activation at the last subject token
  2) value v = optimize MLP output so next-token = target_new
  3) W_out ← W_out + outer(v - W_out @ k, k) / ||k||^2

No covariance cache (full ROME C^{-1}) and no causal-trace layer search yet.
Default model: gpt2-small. Swap model/layer in the YAML for Pythia later.

Run (you run this):
  python scripts/rome_edit.py --config configs/rome_gpt2_small.yaml
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.geometry import (
    DEFAULT_PROBE_PROMPTS,
    collect_mlp_out_bank,
    compute_geometry_metrics,
)
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


def load_config(path: Path) -> Dict[str, Any]:
    with path.open() as f:
        return yaml.safe_load(f)


def find_subject_last_pos(model, prompt: str, subject: str) -> int:
    """Index of the last token of ``subject`` inside ``prompt``."""
    prompt_toks = model.to_str_tokens(prompt)
    subject_toks = model.to_str_tokens(subject)
    # Prefer subject as it appears in the prompt (leading-space tokenization).
    n, m = len(prompt_toks), len(subject_toks)
    for start in range(n - m, -1, -1):
        if prompt_toks[start : start + m] == subject_toks:
            return start + m - 1
    # Fallback: search without relying on exact subject tokenization.
    subj_ids = model.to_tokens(subject, prepend_bos=False)[0].tolist()
    prompt_ids = model.to_tokens(prompt, prepend_bos=True)[0].tolist()
    m = len(subj_ids)
    for start in range(len(prompt_ids) - m, -1, -1):
        if prompt_ids[start : start + m] == subj_ids:
            return start + m - 1
    raise ValueError(f"Could not locate subject={subject!r} inside prompt={prompt!r}")


def first_token_id(model, text: str) -> int:
    ids = model.to_tokens(text, prepend_bos=False)[0]
    if ids.numel() == 0:
        raise ValueError(f"Empty tokenization for {text!r}")
    return int(ids[0].item())


@torch.no_grad()
def next_token_probs(model, prompt: str, token_ids: List[int]) -> Dict[int, float]:
    toks = model.to_tokens(prompt)
    logits = model(toks)[0, -1]
    probs = torch.softmax(logits.float(), dim=-1)
    return {tid: float(probs[tid].item()) for tid in token_ids}


def extract_key(model, tokens: torch.Tensor, layer: int, subject_pos: int) -> torch.Tensor:
    hook_name = f"blocks.{layer}.mlp.hook_post"
    _, cache = model.run_with_cache(tokens, names_filter=lambda n: n == hook_name)
    k = cache[hook_name][0, subject_pos].detach().clone()
    return k


def optimize_v(
    model,
    tokens: torch.Tensor,
    layer: int,
    subject_pos: int,
    target_id: int,
    n_steps: int,
    lr: float,
    early_stop: float,
    logger: Optional[SignalLogger] = None,
    neighbor_bank: Optional[torch.Tensor] = None,
    geometry_metrics: Optional[List[str]] = None,
    top_k_neighbors: int = 16,
    log_every_n: int = 1,
    log_geometry: bool = True,
) -> Tuple[torch.Tensor, List[float]]:
    """Optimize MLP output at subject_pos so the final token predicts target_id.

    Each step logs ``loss`` beside real ``geometry_*`` metrics of the current
    MLP-out vector (v + delta) against ``neighbor_bank``.
    """
    hook_out = f"blocks.{layer}.hook_mlp_out"
    with torch.no_grad():
        _, cache = model.run_with_cache(tokens, names_filter=lambda n: n == hook_out)
        v = cache[hook_out][0, subject_pos].detach().clone()

    delta = torch.zeros_like(v, requires_grad=True)
    opt = torch.optim.Adam([delta], lr=lr)
    losses: List[float] = []
    metric_names = geometry_metrics or [
        "interference_mean",
        "spectral_participation",
        "coactivation_overlap",
    ]

    def add_delta(act: torch.Tensor, hook):
        act = act.clone()
        act[0, subject_pos] = act[0, subject_pos] + delta
        return act

    for step in range(n_steps):
        opt.zero_grad()
        logits = model.run_with_hooks(tokens, fwd_hooks=[(hook_out, add_delta)])
        loss = torch.nn.functional.cross_entropy(
            logits[0, -1].unsqueeze(0),
            torch.tensor([target_id], device=logits.device),
        )
        loss.backward()
        opt.step()
        loss_f = float(loss.item())
        losses.append(loss_f)

        if logger is not None and (step % max(log_every_n, 1) == 0):
            geometry: Dict[str, float] = {}
            if log_geometry and neighbor_bank is not None and neighbor_bank.numel() > 0:
                query = (v + delta.detach()).float().cpu()
                geometry = compute_geometry_metrics(
                    query,
                    neighbor_bank,
                    top_k=top_k_neighbors,
                    metrics=metric_names,
                )
            logger.log(step=step, loss=loss_f, geometry=geometry)

        if loss_f < early_stop:
            break

    return (v + delta.detach()).clone(), losses


def apply_rank_one(model, layer: int, k: torch.Tensor, v_star: torch.Tensor) -> float:
    """W ← W + outer(v* - W k, k) / ||k||^2  on MLP W_out (d_model, d_mlp)."""
    W = model.blocks[layer].mlp.W_out  # [d_model, d_mlp]
    k = k.to(W.device, W.dtype)
    v_star = v_star.to(W.device, W.dtype)
    with torch.no_grad():
        v_old = W @ k
        residual = v_star - v_old
        denom = torch.dot(k, k).clamp_min(1e-8)
        delta_W = torch.outer(residual, k) / denom
        W.add_(delta_W)
        return float(delta_W.norm().item())


def generate_continuation(model, prompt: str, max_new_tokens: int = 8) -> str:
    toks = model.to_tokens(prompt)
    out = model.generate(toks, max_new_tokens=max_new_tokens, do_sample=False, verbose=False)
    return model.to_string(out[0])


def run_edit(cfg: Dict[str, Any]) -> Dict[str, Any]:
    seed = set_seed(int(cfg.get("seed", 0)))
    device = resolve_device(str(cfg.get("device", "auto")))
    edit = cfg["edit"]
    log_cfg = cfg.get("logging", {})

    from transformer_lens import HookedTransformer

    model_name = cfg.get("model", "gpt2-small")
    model = HookedTransformer.from_pretrained(model_name, device=device)
    model.eval()

    prompt = edit["prompt"]
    subject = edit["subject"]
    layer = int(edit["layer"])
    target_old_id = first_token_id(model, edit["target_old"])
    target_new_id = first_token_id(model, edit["target_new"])
    subject_pos = find_subject_last_pos(model, prompt, subject)
    tokens = model.to_tokens(prompt)

    before = next_token_probs(model, prompt, [target_old_id, target_new_id])
    before_text = generate_continuation(model, prompt)

    logger = SignalLogger(
        results_dir=log_cfg.get("results_dir", "results"),
        run_name=log_cfg.get("run_name"),
    )

    diff_cfg = cfg.get("diff", {})
    top_k = int(diff_cfg.get("top_k_neighbors", log_cfg.get("top_k_neighbors", 16)))
    metric_names = list(
        log_cfg.get(
            "geometry_metrics",
            ["interference_mean", "spectral_participation", "coactivation_overlap"],
        )
    )
    probe_prompts = list(log_cfg.get("probe_prompts", DEFAULT_PROBE_PROMPTS))
    # Include the edit prompt so the bank covers the local context.
    if prompt not in probe_prompts:
        probe_prompts = [prompt] + probe_prompts

    neighbor_bank = collect_mlp_out_bank(
        model,
        layer=layer,
        prompts=probe_prompts,
        positions=str(log_cfg.get("bank_positions", "all")),
    )

    logger.write_meta(
        seed=seed,
        model=model_name,
        device=device,
        edit=edit,
        subject_pos=subject_pos,
        target_old_id=target_old_id,
        target_new_id=target_new_id,
        geometry_metrics=metric_names,
        top_k_neighbors=top_k,
        neighbor_bank_size=int(neighbor_bank.shape[0]),
        neighbor_bank_dim=int(neighbor_bank.shape[1]),
        probe_prompts=probe_prompts,
        note="track: loss + real geometry_* each v-step against MLP-out neighbor bank",
    )

    k = extract_key(model, tokens, layer, subject_pos)
    do_log = bool(log_cfg.get("log_loss", True) or log_cfg.get("log_geometry", True))
    v_star, v_losses = optimize_v(
        model,
        tokens,
        layer=layer,
        subject_pos=subject_pos,
        target_id=target_new_id,
        n_steps=int(edit.get("v_steps", 25)),
        lr=float(edit.get("v_lr", 0.5)),
        early_stop=float(edit.get("early_stop_loss", 0.05)),
        logger=logger if do_log else None,
        neighbor_bank=neighbor_bank,
        geometry_metrics=metric_names,
        top_k_neighbors=top_k,
        log_every_n=int(log_cfg.get("every_n_steps", 1)),
        log_geometry=bool(log_cfg.get("log_geometry", True)),
    )
    delta_norm = apply_rank_one(model, layer, k, v_star)

    after = next_token_probs(model, prompt, [target_old_id, target_new_id])
    after_text = generate_continuation(model, prompt)

    result = {
        "seed": seed,
        "model": model_name,
        "device": device,
        "layer": layer,
        "subject": subject,
        "prompt": prompt,
        "subject_pos": subject_pos,
        "target_old": edit["target_old"],
        "target_new": edit["target_new"],
        "before": {
            "p_old": before[target_old_id],
            "p_new": before[target_new_id],
            "continuation": before_text,
        },
        "after": {
            "p_old": after[target_old_id],
            "p_new": after[target_new_id],
            "continuation": after_text,
        },
        "v_final_loss": v_losses[-1] if v_losses else None,
        "v_steps_ran": len(v_losses),
        "delta_W_norm": delta_norm,
        "success": after[target_new_id] > before[target_new_id] and after[target_new_id] > after[target_old_id],
        "signals_csv": str(logger.csv_path),
        "meta_json": str(logger.meta_path),
    }

    out_path = logger.root / "rome_result.json"
    out_path.write_text(json.dumps(result, indent=2) + "\n")
    result["result_json"] = str(out_path)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one simplified ROME edit")
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "rome_gpt2_small.yaml",
        help="Path to YAML config",
    )
    args = parser.parse_args()
    cfg = load_config(args.config)
    result = run_edit(cfg)
    print(json.dumps(result, indent=2))
    print("\nDone.")
    print(f"success={result['success']}  p_new {result['before']['p_new']:.4f} → {result['after']['p_new']:.4f}")
    print(f"wrote {result['result_json']}")


if __name__ == "__main__":
    main()
