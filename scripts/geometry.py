"""Local superposition-geometry metrics for the track signal.

Metric A — interference_mean:
    mean |cos| between the query vector and its top-k neighbors in a direction bank.

Metric B — spectral_participation:
    participation ratio of the query's energy across eigenvectors of the neighbor
    covariance (frame-style). Normalized to [0, 1]: higher ⇒ more spread / less
    localized (more superposition-like).

Metric C — coactivation_overlap:
    mean Jaccard overlap of top-|activation| dimensions between the query and
    each of its top-k cosine neighbors.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence

import torch


EPS = 1e-8


def _as_2d(bank: torch.Tensor) -> torch.Tensor:
    if bank.ndim == 1:
        return bank.unsqueeze(0)
    if bank.ndim != 2:
        raise ValueError(f"bank must be [n, d], got {tuple(bank.shape)}")
    return bank


def l2_normalize(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    return x / x.norm(dim=dim, keepdim=True).clamp_min(EPS)


@torch.no_grad()
def abs_cosine_sims(query: torch.Tensor, bank: torch.Tensor) -> torch.Tensor:
    """|cos| of query [d] against each row of bank [n, d]."""
    q = l2_normalize(query.float().reshape(-1))
    b = l2_normalize(_as_2d(bank.float()), dim=-1)
    return (b @ q).abs()


@torch.no_grad()
def _drop_self_neighbors(
    query: torch.Tensor,
    bank: torch.Tensor,
    self_cos: float = 0.999,
) -> torch.Tensor:
    """Remove bank rows that are essentially the query (avoids trivial top-1)."""
    b = _as_2d(bank.float())
    if b.shape[0] == 0:
        return b
    sims = abs_cosine_sims(query, b)
    keep = sims < self_cos
    if bool(keep.any()):
        return b[keep]
    return b


@torch.no_grad()
def topk_neighbor_indices(query: torch.Tensor, bank: torch.Tensor, top_k: int) -> torch.Tensor:
    sims = abs_cosine_sims(query, bank)
    k = min(int(top_k), int(sims.numel()))
    if k <= 0:
        return torch.empty(0, dtype=torch.long, device=sims.device)
    return torch.topk(sims, k).indices


@torch.no_grad()
def interference_mean(query: torch.Tensor, bank: torch.Tensor, top_k: int = 16) -> float:
    """Metric A: mean |cos| to top-k neighbors."""
    sims = abs_cosine_sims(query, bank)
    k = min(int(top_k), int(sims.numel()))
    if k <= 0:
        return 0.0
    return float(torch.topk(sims, k).values.mean().item())


@torch.no_grad()
def spectral_participation(query: torch.Tensor, bank: torch.Tensor, max_rank: int = 64) -> float:
    """Metric B: normalized participation ratio of query energy in neighbor covariance.

    Builds C = B^T B / n in activation space, takes leading eigenspace (capped),
    measures energy of query on those eigenvectors, returns
    ((sum e)^2 / sum e^2 - 1) / (r - 1) clipped to [0, 1] when r > 1.
    0 ⇒ energy in one mode; 1 ⇒ uniform across modes.
    """
    b = _as_2d(bank.float())
    n, d = b.shape
    if n < 2 or d < 1:
        return 0.0
    q = query.float().reshape(-1)
    if q.numel() != d:
        raise ValueError(f"query dim {q.numel()} != bank dim {d}")

    # Covariance in feature space; use low-rank via Gram when n << d.
    rank_cap = min(max_rank, n, d)
    # eig of B B^T (n x n), map to feature-space energies via B^T u
    gram = (b @ b.T) / max(n, 1)
    evals, evecs = torch.linalg.eigh(gram)
    # ascending → take largest
    evals = evals.flip(0)[:rank_cap]
    evecs = evecs.flip(1)[:, :rank_cap]
    # feature-space directions proportional to B^T u; energy of q along them
    feat = b.T @ evecs  # [d, r]
    feat = l2_normalize(feat, dim=0)
    coeffs = feat.T @ l2_normalize(q)
    energy = coeffs.pow(2)
    total = float(energy.sum().item())
    if total < EPS:
        return 0.0
    energy = energy / total
    r = int(energy.numel())
    if r == 1:
        return 0.0
    pr = float((energy.sum() ** 2 / energy.pow(2).sum()).item())  # in [1, r]
    return float(max(0.0, min(1.0, (pr - 1.0) / (r - 1.0))))


@torch.no_grad()
def coactivation_overlap(
    query: torch.Tensor,
    bank: torch.Tensor,
    top_k: int = 16,
    top_dims: int = 32,
) -> float:
    """Metric C: mean Jaccard overlap of top-|act| dims vs top-k neighbors."""
    b = _as_2d(bank.float())
    q = query.float().reshape(-1)
    if b.shape[0] == 0:
        return 0.0
    # Never select the full ambient dim — that forces Jaccard=1 for all dense vectors.
    m = min(int(top_dims), max(1, q.numel() // 4), q.numel() - 1) if q.numel() > 1 else 1
    m = max(1, m)
    sims = abs_cosine_sims(q, b)
    k = min(int(top_k), int(sims.numel()))
    if k <= 0:
        return 0.0
    nbr_idx = torch.topk(sims, k).indices
    q_set = set(torch.topk(q.abs(), m).indices.tolist())
    overlaps: List[float] = []
    for i in nbr_idx.tolist():
        n_set = set(torch.topk(b[i].abs(), m).indices.tolist())
        union = q_set | n_set
        if not union:
            overlaps.append(0.0)
        else:
            overlaps.append(len(q_set & n_set) / len(union))
    return float(sum(overlaps) / len(overlaps))


@torch.no_grad()
def compute_geometry_metrics(
    query: torch.Tensor,
    bank: torch.Tensor,
    top_k: int = 16,
    metrics: Optional[Sequence[str]] = None,
    top_dims: int = 32,
) -> Dict[str, float]:
    """Compute selected geometry metrics for one query against a neighbor bank."""
    wanted = list(metrics) if metrics is not None else [
        "interference_mean",
        "spectral_participation",
        "coactivation_overlap",
    ]
    bank = _drop_self_neighbors(query, bank)
    out: Dict[str, float] = {}
    for name in wanted:
        if name == "interference_mean":
            out[name] = interference_mean(query, bank, top_k=top_k)
        elif name == "spectral_participation":
            out[name] = spectral_participation(query, bank)
        elif name == "coactivation_overlap":
            out[name] = coactivation_overlap(query, bank, top_k=top_k, top_dims=top_dims)
        else:
            raise ValueError(f"Unknown geometry metric: {name}")
    return out


DEFAULT_PROBE_PROMPTS: List[str] = [
    "The Eiffel Tower is located in",
    "The Louvre Museum is located in",
    "The Great Wall of China is located in",
    "Mount Everest is located in",
    "The Colosseum is located in",
    "Paris is the capital of",
    "Rome is the capital of",
    "Berlin is the capital of",
    "Tokyo is the capital of",
    "The president of the United States is",
    "The currency of Japan is",
    "Water boils at",
]


def _rows_from_acts(acts: torch.Tensor, positions: str) -> List[torch.Tensor]:
    """Flatten one [seq, d] activation tensor into bank rows."""
    if positions == "last":
        return [acts[-1].detach().float().cpu()]
    if positions == "all":
        return list(acts.detach().float().cpu())
    raise ValueError(f"positions must be 'all' or 'last', got {positions!r}")


@torch.no_grad()
def collect_mlp_out_banks(
    model,
    layers: Sequence[int],
    prompts: Iterable[str],
    positions: str = "all",
) -> Dict[int, torch.Tensor]:
    """Gather ``hook_mlp_out`` banks for many layers in one forward per prompt.

    Returns ``{layer: [n, d_model]}``.
    """
    layer_list = [int(L) for L in layers]
    if not layer_list:
        raise ValueError("layers must be non-empty")
    hooks = {L: f"blocks.{L}.hook_mlp_out" for L in layer_list}
    wanted = set(hooks.values())
    rows: Dict[int, List[torch.Tensor]] = {L: [] for L in layer_list}
    for prompt in prompts:
        tokens = model.to_tokens(prompt)
        _, cache = model.run_with_cache(tokens, names_filter=lambda n: n in wanted)
        for L, hook in hooks.items():
            rows[L].extend(_rows_from_acts(cache[hook][0], positions))
    out: Dict[int, torch.Tensor] = {}
    for L, rs in rows.items():
        if not rs:
            raise ValueError(f"empty neighbor bank at layer {L}")
        out[L] = torch.stack(rs, dim=0)
    return out


@torch.no_grad()
def collect_mlp_out_bank(
    model,
    layer: int,
    prompts: Iterable[str],
    positions: str = "all",
) -> torch.Tensor:
    """Gather ``hook_mlp_out`` vectors at ``layer`` into a [n, d_model] bank.

    positions:
      - \"all\": every token position
      - \"last\": final token only
    """
    return collect_mlp_out_banks(model, [layer], prompts, positions=positions)[int(layer)]
