# Geometry metrics (track signal)

Logged every step beside `loss` as `geometry_*` columns in `results/<run>/signals.csv`.

Query vector (ROME baseline): current MLP output at the edit subject position, `v + δ`, during `v` optimization.

Neighbor bank: `hook_mlp_out` activations at the edit layer over a probe prompt set (all token positions by default).

## Metric A — `interference_mean`

Mean absolute cosine similarity between the query and its **top-k** neighbors in the bank.

High ⇒ query sits close to many other directions (more local interference).

## Metric B — `spectral_participation`

Participation of the query’s energy across eigenvectors of the neighbor covariance (Gram / frame-style).

Normalized to **[0, 1]**:

- **0** → energy concentrated in one local mode (more localized)
- **1** → energy spread across modes (more superposition-like)

## Metric C — `coactivation_overlap`

For each of the top-k cosine neighbors, take the Jaccard overlap of the top-|activation| dimensions of query vs neighbor; average.

High ⇒ query and neighbors light up overlapping coordinates.

## Config knobs

See `configs/default.yaml` / `configs/rome_gpt2_small.yaml`:

- `logging.geometry_metrics`
- `logging.every_n_steps`
- `diff.top_k_neighbors` (also used as top-k for A/C)
- `logging.probe_prompts` (optional override)
- `logging.bank_positions`: `all` | `last`

## Plot

```bash
python scripts/plot_signals.py --csv results/rome_gpt2_small_baseline/signals.csv
```
