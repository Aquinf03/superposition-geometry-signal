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

Uses at most ~`d/4` top dimensions (never the full ambient dim), so dense vectors do not trivially score 1.0.

## Config knobs

Hero train config (`configs/train_gpt2_small_geometry.yaml`):

- `geometry.layers`: e.g. `[4, 8, 11]` — track A/B/C at each MLP-out layer (CSV: `geometry_L{n}_*`)
- `geometry.layer`: single-layer fallback if `layers` omitted
- `geometry.every_n_steps`
- `geometry.top_k_neighbors`
- `geometry.bank_positions`: `all` | `last`
- `geometry.probe_prompts` (optional override)

ROME / default configs also use:

- `logging.geometry_metrics`
- `logging.every_n_steps`
- `diff.top_k_neighbors` (also used as top-k for A/C)
- `logging.probe_prompts` (optional override)
- `logging.bank_positions`: `all` | `last`

## Plot

```bash
python scripts/plot_signals.py --csv results/train_gpt2_small_geometry/signals.csv
```

Multi-layer CSVs plot one panel per metric with a line per layer.

## Checkpoint-aligned freezes

On the hero train run, every weight save also writes a geometry sidecar:

- `results/<run>/checkpoints/step_00019.pt`
- `results/<run>/checkpoints/step_00019.geometry.json` — same step, probes, layers, metrics
- `results/<run>/checkpoints/manifest.json` — index for ckpt A vs B diffs

`model_final.pt` + `model_final.geometry.json` at the run root are aliases of the last step.

## Diff (selected features / neighborhoods)

`scripts/diff_geometry.py` snapshots a feature’s summary metrics + top-k neighbor `|cos|`, then diffs two snapshots.

| Mode | File | Meaning |
| --- | --- | --- |
| `step_vs_step` | `diff_step_vs_step.json/.png` | same write-direction at v-step 0 vs last (fixed pre-edit bank) |
| `pre_post_edit` | `diff_pre_post_edit.json/.png` | edit-subject MLP-out before vs after rank-one (banks refreshed) |
| extra features | `diff_pre_post_<name>.json/.png` | from `diff.selected_features` in YAML |

JSON fields: `delta_metrics`, `summary_score`, `mean_abs_neighbor_delta`, `neighbor_deltas`, entered/left top-k.

## Toy sanity check

```bash
python scripts/toy_superposition_sanity.py
```

Compares near-orthogonal features (`n ≤ d`) vs forced packing (`n ≫ d`) in the same ambient dim. Expect denser ⇒ higher `interference_mean` / `coactivation_overlap`, and `spectral_participation` moves. Writes `results/toy_superposition_sanity/` (`sanity_result.json`, `sparse_vs_dense.png`, `signals.csv`).
