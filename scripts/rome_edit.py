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

from scripts.diff_geometry import (
    diff_snapshots,
    diff_step_queries,
    extract_mlp_out_feature,
    plot_neighborhood_diff,
    save_diff,
    snapshot_neighborhood,
)
from scripts.eval_edit import (
    capture_activation_fingerprint,
    capture_ripple_state,
    run_edit_evals,
)
from scripts.geometry import (
    DEFAULT_PROBE_PROMPTS,
    collect_mlp_out_bank,
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
) -> Tuple[torch.Tensor, List[float], torch.Tensor, torch.Tensor, int]:
    """Optimize MLP output at subject_pos so the final token predicts target_id.

    Each step logs ``loss`` beside real ``geometry_*`` metrics of the current
    MLP-out vector (v + delta) against ``neighbor_bank``.

    Returns ``(v_star, losses, query_step0, query_last, last_step)``.
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
    query_step0: Optional[torch.Tensor] = None
    query_last: Optional[torch.Tensor] = None
    last_step = -1

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

        query = (v + delta.detach()).float().cpu()
        if query_step0 is None:
            query_step0 = query.clone()
        query_last = query.clone()
        last_step = step

        if logger is not None and (step % max(log_every_n, 1) == 0):
            geometry: Dict[str, float] = {}
            if log_geometry and neighbor_bank is not None and neighbor_bank.numel() > 0:
                geometry = compute_geometry_metrics(
                    query,
                    neighbor_bank,
                    top_k=top_k_neighbors,
                    metrics=metric_names,
                )
            logger.log(step=step, loss=loss_f, geometry=geometry)

        if loss_f < early_stop:
            break

    assert query_step0 is not None and query_last is not None
    return (v + delta.detach()).clone(), losses, query_step0, query_last, last_step


def apply_rank_one(model, layer: int, k: torch.Tensor, v_star: torch.Tensor) -> float:
    """Rank-one write so MLP maps key ``k`` → value ``v_star``.

    TransformerLens ``W_out`` layout varies by version:
      - [d_model, d_mlp] → out = W @ k
      - [d_mlp, d_model] → out = k @ W
    """
    W = model.blocks[layer].mlp.W_out
    k = k.to(device=W.device, dtype=W.dtype).reshape(-1)
    v_star = v_star.to(device=W.device, dtype=W.dtype).reshape(-1)
    with torch.no_grad():
        if W.shape[1] == k.numel() and W.shape[0] == v_star.numel():
            # W: [d_model, d_mlp]
            v_old = W @ k
            residual = v_star - v_old
            denom = torch.dot(k, k).clamp_min(1e-8)
            delta_W = torch.outer(residual, k) / denom
        elif W.shape[0] == k.numel() and W.shape[1] == v_star.numel():
            # W: [d_mlp, d_model]
            v_old = k @ W
            residual = v_star - v_old
            denom = torch.dot(k, k).clamp_min(1e-8)
            delta_W = torch.outer(k, residual) / denom
        else:
            raise ValueError(
                f"W_out shape {tuple(W.shape)} incompatible with "
                f"k={tuple(k.shape)} v={tuple(v_star.shape)}"
            )
        W.add_(delta_W)
        return float(delta_W.norm().item())


def generate_continuation(model, prompt: str, max_new_tokens: int = 8) -> str:
    toks = model.to_tokens(prompt)
    out = model.generate(toks, max_new_tokens=max_new_tokens, do_sample=False, verbose=False)
    return model.to_string(out[0])


def run_edit(cfg: Dict[str, Any], source_config: Optional[Path] = None) -> Dict[str, Any]:
    seed = set_seed(int(cfg.get("seed", 0)))
    device = resolve_device(str(cfg.get("device", "auto")))
    edit = cfg["edit"]
    log_cfg = cfg.get("logging", {})
    eval_cfg = cfg.get("eval", {})

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
        live_print=bool(log_cfg.get("live_print", True)),
    )
    artifact_paths = save_run_artifacts(
        logger.root,
        cfg,
        seed,
        source_config=source_config,
        extra_meta={"model": model_name, "device": device},
    )

    diff_cfg = cfg.get("diff", {})
    top_k = int(diff_cfg.get("top_k_neighbors", log_cfg.get("top_k_neighbors", 16)))
    metric_names = list(
        log_cfg.get(
            "geometry_metrics",
            ["interference_mean", "spectral_participation", "coactivation_overlap"],
        )
    )
    probe_prompts = list(log_cfg.get("probe_prompts") or DEFAULT_PROBE_PROMPTS)
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
        artifacts=artifact_paths,
        note="track + diff + eval (success/paraphrase/ripple/activation) + frozen config/seed",
    )

    # Pre-edit behavioral / activation baselines for eval.
    ripple_before = capture_ripple_state(model, probes=eval_cfg.get("ripple_probes"))
    fingerprint_before = capture_activation_fingerprint(
        model,
        probes=eval_cfg.get("fingerprint_probes"),
        n_layers=int(eval_cfg.get("fingerprint_n_layers", 8)),
    )

    k = extract_key(model, tokens, layer, subject_pos)
    do_log = bool(log_cfg.get("log_loss", True) or log_cfg.get("log_geometry", True))
    diff_enabled = bool(diff_cfg.get("enabled", True))
    diff_modes = set(diff_cfg.get("modes", ["pre_post_edit", "step_vs_step"]))
    save_plot = bool(diff_cfg.get("save_neighborhood_plot", True))

    # Pre-edit snapshot of the selected edit-subject feature (for pre_post diff).
    pre_query = extract_mlp_out_feature(model, prompt, layer, subject_pos)
    pre_snap = snapshot_neighborhood(
        pre_query,
        neighbor_bank,
        feature_id="edit_subject",
        top_k=top_k,
        metrics=metric_names,
        step=None,
        extra={"prompt": prompt, "subject_pos": subject_pos, "phase": "pre_edit"},
    )

    # Optional extra selected features (probe prompts, last token).
    extra_features = list(diff_cfg.get("selected_features", []))
    pre_extra_snaps: Dict[str, Dict[str, Any]] = {}
    for feat in extra_features:
        fid = str(feat.get("name", feat.get("prompt", "feature")))
        fprompt = str(feat.get("prompt"))
        fpos = feat.get("position", "last")
        ftoks = model.to_tokens(fprompt)
        pos_i = int(ftoks.shape[1] - 1) if fpos == "last" else int(fpos)
        fq = extract_mlp_out_feature(model, fprompt, layer, pos_i)
        pre_extra_snaps[fid] = snapshot_neighborhood(
            fq,
            neighbor_bank,
            feature_id=fid,
            top_k=top_k,
            metrics=metric_names,
            extra={"prompt": fprompt, "position": pos_i, "phase": "pre_edit"},
        )

    v_star, v_losses, q0, q_last, last_step = optimize_v(
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

    diff_paths: Dict[str, str] = {}
    if diff_enabled:
        # step_vs_step: geometry of write direction at v-step 0 vs last (fixed pre-edit bank)
        if "step_vs_step" in diff_modes:
            step_diff = diff_step_queries(
                q0,
                q_last,
                neighbor_bank,
                feature_id="edit_subject_v_opt",
                step_t=0,
                step_tk=last_step,
                top_k=top_k,
                metrics=metric_names,
            )
            p = save_diff(step_diff, logger.root / "diff_step_vs_step.json")
            diff_paths["step_vs_step"] = str(p)
            if save_plot:
                plot_neighborhood_diff(step_diff, logger.root / "diff_step_vs_step.png")

        # pre_post_edit: re-collect bank post-edit; snapshot subject feature again
        if "pre_post_edit" in diff_modes:
            post_bank = collect_mlp_out_bank(
                model,
                layer=layer,
                prompts=probe_prompts,
                positions=str(log_cfg.get("bank_positions", "all")),
            )
            post_query = extract_mlp_out_feature(model, prompt, layer, subject_pos)
            # Compare using each side's own bank (geometry in that weight state).
            post_snap = snapshot_neighborhood(
                post_query,
                post_bank,
                feature_id="edit_subject",
                top_k=top_k,
                metrics=metric_names,
                extra={"prompt": prompt, "subject_pos": subject_pos, "phase": "post_edit"},
            )
            pre_post = diff_snapshots(pre_snap, post_snap, mode="pre_post_edit")
            p = save_diff(pre_post, logger.root / "diff_pre_post_edit.json")
            diff_paths["pre_post_edit"] = str(p)
            if save_plot:
                plot_neighborhood_diff(pre_post, logger.root / "diff_pre_post_edit.png")

            # Extra selected features pre/post
            for fid, pre_s in pre_extra_snaps.items():
                feat = next(f for f in extra_features if str(f.get("name", f.get("prompt"))) == fid)
                fprompt = str(feat.get("prompt"))
                fpos = feat.get("position", "last")
                ftoks = model.to_tokens(fprompt)
                pos_i = int(ftoks.shape[1] - 1) if fpos == "last" else int(fpos)
                fq = extract_mlp_out_feature(model, fprompt, layer, pos_i)
                post_s = snapshot_neighborhood(
                    fq,
                    post_bank,
                    feature_id=fid,
                    top_k=top_k,
                    metrics=metric_names,
                    extra={"prompt": fprompt, "position": pos_i, "phase": "post_edit"},
                )
                d = diff_snapshots(pre_s, post_s, mode="pre_post_edit")
                safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in fid)
                dp = save_diff(d, logger.root / f"diff_pre_post_{safe}.json")
                diff_paths[f"pre_post_{safe}"] = str(dp)
                if save_plot:
                    plot_neighborhood_diff(d, logger.root / f"diff_pre_post_{safe}.png")

    evals = run_edit_evals(
        model,
        subject=subject,
        prompt=prompt,
        target_old=edit["target_old"],
        target_new=edit["target_new"],
        p_old_before=before[target_old_id],
        p_new_before=before[target_new_id],
        p_old_after=after[target_old_id],
        p_new_after=after[target_new_id],
        ripple_before=ripple_before,
        fingerprint_before=fingerprint_before,
        paraphrase_templates=eval_cfg.get("paraphrase_templates"),
        fingerprint_probes=eval_cfg.get("fingerprint_probes"),
    )
    eval_path = logger.root / "eval.json"
    # Drop non-JSON tensors from any accidental leakage — evals are already plain.
    eval_path.write_text(json.dumps(evals, indent=2) + "\n")

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
        "success": evals["edit_success"]["success"],
        "eval": {
            "edit_success": evals["edit_success"]["success"],
            "paraphrase_mean_p_new": evals["paraphrase"]["mean_p_new"],
            "paraphrase_pass": evals["paraphrase"]["pass_mean_p_new_ge_0_10"],
            "ripple_mean_kl": evals["ripple"]["mean_kl"],
            "ripple_pass": evals["ripple"]["pass_mean_kl_lt_0_05"],
            "activation_mean_cosine": evals["activation_cosine"]["mean_cosine"],
            "activation_pass": evals["activation_cosine"]["pass_mean_cosine_ge_0_92"],
        },
        "signals_csv": str(logger.csv_path),
        "meta_json": str(logger.meta_path),
        "eval_json": str(eval_path),
        "artifacts": artifact_paths,
        "diffs": diff_paths,
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
    result = run_edit(cfg, source_config=args.config)
    print(json.dumps(result, indent=2))
    print("\nDone.")
    print(f"success={result['success']}  p_new {result['before']['p_new']:.4f} → {result['after']['p_new']:.4f}")
    ev = result.get("eval", {})
    print(
        "eval: "
        f"paraphrase_mean_p_new={ev.get('paraphrase_mean_p_new', 0):.4f} "
        f"ripple_mean_kl={ev.get('ripple_mean_kl', 0):.4f} "
        f"activation_cos={ev.get('activation_mean_cosine', 0):.4f}"
    )
    print(f"wrote {result['result_json']}")
    print(f"eval_json={result.get('eval_json')}")
    print(f"frozen_config={result.get('artifacts', {}).get('config_frozen')}")
    print(f"seed_txt={result.get('artifacts', {}).get('seed_txt')}")
    if result.get("diffs"):
        print("diffs:")
        for name, path in result["diffs"].items():
            print(f"  {name}: {path}")


if __name__ == "__main__":
    main()
