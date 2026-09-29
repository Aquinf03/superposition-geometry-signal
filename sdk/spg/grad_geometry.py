"""Differentiable geometry bits for soft control.

Watch metrics in ``spg.geometry`` are ``@torch.no_grad`` floats. Control needs a
tensor path so ``λ · (metric − target)²`` can backprop into the model.
"""

from __future__ import annotations

from typing import List, Sequence

import torch

EPS = 1e-8


def l2_normalize(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    return x / x.norm(dim=dim, keepdim=True).clamp_min(EPS)


def mean_interference_tensor(bank: torch.Tensor, top_k: int = 8) -> torch.Tensor:
    """Mean top-k |cos| interference over a bank ``[n, d]`` (keeps grad)."""
    b = bank.float()
    if b.ndim != 2:
        raise ValueError(f"bank must be [n, d], got {tuple(b.shape)}")
    n = b.shape[0]
    if n < 2:
        return b.new_zeros(())
    bn = l2_normalize(b, dim=-1)
    sims = (bn @ bn.T).abs()
    # Mask self-similarity
    sims = sims - torch.eye(n, device=sims.device, dtype=sims.dtype) * 10.0
    k = min(int(top_k), n - 1)
    vals = torch.topk(sims, k, dim=-1).values
    return vals.mean()


def collect_mlp_out_bank_grad(
    model,
    layer: int,
    prompts: Sequence[str],
    positions: str = "last",
) -> torch.Tensor:
    """Gather ``hook_mlp_out`` rows **with** autograd (device tensors, no detach)."""
    if positions not in ("last", "all"):
        raise ValueError(f"positions must be 'all' or 'last', got {positions!r}")
    hook_name = f"blocks.{int(layer)}.hook_mlp_out"
    rows: List[torch.Tensor] = []

    def _hook(act: torch.Tensor, hook) -> torch.Tensor:
        # act: [batch, seq, d]
        if positions == "last":
            rows.append(act[0, -1])
        else:
            rows.append(act[0].reshape(-1, act.shape[-1]))
        return act

    for prompt in prompts:
        tokens = model.to_tokens(prompt)
        model.run_with_hooks(tokens, fwd_hooks=[(hook_name, _hook)])

    if not rows:
        raise ValueError(f"empty grad bank at layer {layer}")
    # "all" may append multi-row tensors
    flat: List[torch.Tensor] = []
    for r in rows:
        if r.ndim == 1:
            flat.append(r)
        else:
            flat.extend(list(r))
    return torch.stack(flat, dim=0)
