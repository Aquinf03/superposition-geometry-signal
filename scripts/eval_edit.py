"""Post-edit behavioral + representation evals for the ROME baseline.

Logged every run:
  - edit_success: target_new dominates target_old on the edit prompt
  - paraphrase: mean p(target_new) over rephrase templates
  - ripple: how much unrelated probe next-token dists move (mean KL + top-token shift)
  - activation_cosine: mean residual-stream cosine pre vs post on fingerprint probes
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F


DEFAULT_PARAPHRASE_TEMPLATES = [
    "{subject} is located in",
    "{subject} is in",
    "You can find {subject} in",
    "{subject} stands in",
    "{subject} can be found in",
    "People visit {subject} in",
    "The location of {subject} is",
]

DEFAULT_RIPPLE_PROBES = [
    "Paris is the capital of",
    "The Louvre Museum is located in",
    "The Colosseum is located in",
    "Water boils at",
    "The president of the United States is",
    "The currency of Japan is",
    "Berlin is the capital of",
    "Tokyo is the capital of",
]

DEFAULT_FINGERPRINT_PROBES = [
    "The Eiffel Tower is located in",
    "Paris is the capital of",
    "List three colors:",
    "What is 12 multiplied by 8?",
    "The president of the United States is",
]


def first_token_id(model, text: str) -> int:
    ids = model.to_tokens(text, prepend_bos=False)[0]
    if ids.numel() == 0:
        raise ValueError(f"Empty tokenization for {text!r}")
    return int(ids[0].item())


@torch.no_grad()
def next_token_prob(model, prompt: str, token_id: int) -> float:
    toks = model.to_tokens(prompt)
    logits = model(toks)[0, -1].float()
    probs = torch.softmax(logits, dim=-1)
    return float(probs[token_id].item())


@torch.no_grad()
def next_token_log_probs(model, prompt: str) -> torch.Tensor:
    toks = model.to_tokens(prompt)
    logits = model(toks)[0, -1].float()
    return F.log_softmax(logits, dim=-1)


@torch.no_grad()
def next_token_top(model, prompt: str) -> Tuple[int, float]:
    toks = model.to_tokens(prompt)
    logits = model(toks)[0, -1].float()
    probs = torch.softmax(logits, dim=-1)
    tid = int(torch.argmax(probs).item())
    return tid, float(probs[tid].item())


def edit_success_metrics(
    p_old_before: float,
    p_new_before: float,
    p_old_after: float,
    p_new_after: float,
) -> Dict[str, Any]:
    success = p_new_after > p_new_before and p_new_after > p_old_after
    return {
        "success": bool(success),
        "p_old_before": p_old_before,
        "p_new_before": p_new_before,
        "p_old_after": p_old_after,
        "p_new_after": p_new_after,
        "delta_p_new": p_new_after - p_new_before,
        "delta_p_old": p_old_after - p_old_before,
    }


@torch.no_grad()
def paraphrase_metrics(
    model,
    subject: str,
    target_new: str,
    target_old: str,
    templates: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    templates = list(templates or DEFAULT_PARAPHRASE_TEMPLATES)
    new_id = first_token_id(model, target_new)
    old_id = first_token_id(model, target_old)
    rows: List[Dict[str, Any]] = []
    for tmpl in templates:
        prompt = tmpl.format(subject=subject)
        p_new = next_token_prob(model, prompt, new_id)
        p_old = next_token_prob(model, prompt, old_id)
        rows.append({"prompt": prompt, "p_new": p_new, "p_old": p_old})
    mean_p_new = float(sum(r["p_new"] for r in rows) / max(len(rows), 1))
    mean_p_old = float(sum(r["p_old"] for r in rows) / max(len(rows), 1))
    return {
        "n": len(rows),
        "mean_p_new": mean_p_new,
        "mean_p_old": mean_p_old,
        "pass_mean_p_new_ge_0_10": mean_p_new >= 0.10,
        "per_prompt": rows,
    }


@torch.no_grad()
def capture_ripple_state(
    model,
    probes: Optional[Sequence[str]] = None,
) -> Dict[str, Dict[str, Any]]:
    """Store next-token log-probs + top token for each ripple probe."""
    probes = list(probes or DEFAULT_RIPPLE_PROBES)
    state: Dict[str, Dict[str, Any]] = {}
    for prompt in probes:
        logp = next_token_log_probs(model, prompt).cpu()
        top_id, top_p = next_token_top(model, prompt)
        state[prompt] = {
            "log_probs": logp,
            "top_id": top_id,
            "top_p": top_p,
            "top_str": model.to_string(torch.tensor([top_id])),
        }
    return state


@torch.no_grad()
def ripple_metrics(
    before: Dict[str, Dict[str, Any]],
    after_model,
) -> Dict[str, Any]:
    """Compare post-edit next-token dists to pre-edit captures."""
    rows: List[Dict[str, Any]] = []
    kls: List[float] = []
    top_shifts = 0
    for prompt, pre in before.items():
        logp_after = next_token_log_probs(after_model, prompt).cpu()
        logp_before = pre["log_probs"]
        # KL(before || after) on next-token categorical
        p_before = logp_before.exp()
        kl = float(torch.sum(p_before * (logp_before - logp_after)).item())
        kls.append(kl)
        top_id_after, top_p_after = next_token_top(after_model, prompt)
        shifted = int(top_id_after != pre["top_id"])
        top_shifts += shifted
        rows.append(
            {
                "prompt": prompt,
                "kl_before_after": kl,
                "top_before": pre["top_str"],
                "top_after": after_model.to_string(torch.tensor([top_id_after])),
                "top_shifted": bool(shifted),
                "top_p_before": pre["top_p"],
                "top_p_after": top_p_after,
            }
        )
    mean_kl = float(sum(kls) / max(len(kls), 1))
    return {
        "n": len(rows),
        "mean_kl": mean_kl,
        "frac_top_shifted": float(top_shifts / max(len(rows), 1)),
        "pass_mean_kl_lt_0_05": mean_kl < 0.05,
        "per_prompt": rows,
    }


@torch.no_grad()
def capture_activation_fingerprint(
    model,
    probes: Optional[Sequence[str]] = None,
    n_layers: int = 8,
) -> Dict[str, torch.Tensor]:
    """Residual stream at last token for evenly spaced layers × probes."""
    probes = list(probes or DEFAULT_FINGERPRINT_PROBES)
    n_total = model.cfg.n_layers
    # sample up to n_layers indices across depth
    if n_total <= n_layers:
        layer_ids = list(range(n_total))
    else:
        layer_ids = [int(round(i * (n_total - 1) / (n_layers - 1))) for i in range(n_layers)]
    hooks = [f"blocks.{L}.hook_resid_post" for L in layer_ids]
    out: Dict[str, torch.Tensor] = {}
    for prompt in probes:
        toks = model.to_tokens(prompt)
        _, cache = model.run_with_cache(toks, names_filter=lambda n: n in hooks)
        for L, name in zip(layer_ids, hooks):
            key = f"{prompt}||L{L}"
            out[key] = cache[name][0, -1].detach().float().cpu()
    return out


@torch.no_grad()
def activation_cosine_metrics(
    before: Dict[str, torch.Tensor],
    after: Dict[str, torch.Tensor],
) -> Dict[str, Any]:
    cosines: List[float] = []
    rows: List[Dict[str, Any]] = []
    for key in sorted(set(before) & set(after)):
        a = before[key]
        b = after[key]
        cos = float(torch.nn.functional.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item())
        cosines.append(cos)
        rows.append({"site": key, "cosine": cos})
    mean_cos = float(sum(cosines) / max(len(cosines), 1))
    return {
        "n": len(rows),
        "mean_cosine": mean_cos,
        "min_cosine": float(min(cosines) if cosines else 0.0),
        "pass_mean_cosine_ge_0_92": mean_cos >= 0.92,
        "per_site": rows,
    }


def run_edit_evals(
    model,
    *,
    subject: str,
    prompt: str,
    target_old: str,
    target_new: str,
    p_old_before: float,
    p_new_before: float,
    p_old_after: float,
    p_new_after: float,
    ripple_before: Dict[str, Dict[str, Any]],
    fingerprint_before: Dict[str, torch.Tensor],
    paraphrase_templates: Optional[Sequence[str]] = None,
    fingerprint_probes: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    """Compute the full eval bundle after the edit has been applied."""
    success = edit_success_metrics(p_old_before, p_new_before, p_old_after, p_new_after)
    paraphrase = paraphrase_metrics(
        model, subject, target_new, target_old, templates=paraphrase_templates
    )
    ripple = ripple_metrics(ripple_before, model)
    fingerprint_after = capture_activation_fingerprint(model, probes=fingerprint_probes)
    activation = activation_cosine_metrics(fingerprint_before, fingerprint_after)
    return {
        "edit_success": success,
        "paraphrase": paraphrase,
        "ripple": ripple,
        "activation_cosine": activation,
    }
