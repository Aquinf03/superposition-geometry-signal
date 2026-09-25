# Geometry metrics (track signal)

Logged every step beside `loss` as `geometry_*` columns in `experiments/results/<run>/signals.csv`.

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

Hero train config (`experiments/configs/train_gpt2_small_geometry.yaml`):

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
python -m scripts.helpers.plot_signals --csv experiments/results/train_gpt2_small_geometry/signals.csv
```

Multi-layer CSVs plot one panel per metric with a line per layer.

## Checkpoint-aligned freezes

On the hero train run, every weight save also writes a geometry sidecar:

- `experiments/results/<run>/checkpoints/step_00019.pt`
- `experiments/results/<run>/checkpoints/step_00019.geometry.json` — same step, probes, layers, metrics
- `experiments/results/<run>/checkpoints/manifest.json` — index for ckpt A vs B diffs

`model_final.pt` + `model_final.geometry.json` at the run root are aliases of the last step.

## Diff across checkpoints (selected features)

```bash
python experiments/diff_checkpoints.py --config experiments/configs/train_gpt2_small_geometry.yaml
# or pick steps explicitly:
python experiments/diff_checkpoints.py --config experiments/configs/train_gpt2_small_geometry.yaml --step-a 19 --step-b 39
```

Writes under `experiments/results/<run>/diff_ckpt_XXXXX_vs_YYYYY/`:

| File | Role |
| --- | --- |
| `diff_aligned_summary.json` | Δ of frozen sidecar geometry (no reload) |
| `diff_<feature>_L{n}.json/.png` | Per-feature neighborhood diff at layer n |
| `diff_summary.json` | Index of all feature×layer scores |

## Cross-feature diff (feature A vs B)

Same checkpoint, shared neighbor bank — compare how two features sit:

```bash
python experiments/diff_features.py --config experiments/configs/train_gpt2_small_geometry.yaml
# optional: --step 39  --pair eiffel_located_in louvre_located_in
```

Writes under `experiments/results/<run>/diff_features_step_XXXXX/`. Related pairs (eiffel vs louvre) should usually score closer than unrelated (eiffel vs superposition).

## Validate on a real training run

```bash
python experiments/validate_training_geometry.py --run-dir experiments/results/train_gpt2_small_geometry
```

Checks (training only — not edit locality):

1. geometry not flat while loss moves  
2. geometry correlates with loss  
3. late-layer geometry phase (spectral collapse **or** unpack-from-floor)  
4. packing coupling (interference ↔ spectral move inversely)  
5. layer-phase divergence  
6. optional: related features closer than unrelated (`diff_features`)

Writes `validation.json` + `validation.png`. Exit code 1 on FAIL.

When the signal is flat or lies, see **`failure_notes.md`**.

## Scaled model (gpt2-medium)

Same track + diff + validate pipeline; swap the YAML:

```bash
python experiments/train_with_geometry.py --config experiments/configs/train_gpt2_medium_geometry.yaml
# … then plot / diff_checkpoints / diff_features / validate on experiments/results/train_gpt2_medium_geometry
```

Layers `[6, 12, 22]`, batch 2, geometry every 2 steps — still no edit depth.

## Diff (selected features / neighborhoods)

`scripts/helpers/diff_geometry.py` snapshots a feature’s summary metrics + top-k neighbor `|cos|`, then diffs two snapshots.

| Mode | File | Meaning |
| --- | --- | --- |
| `step_vs_step` | `diff_step_vs_step.json/.png` | same write-direction at v-step 0 vs last (fixed pre-edit bank) |
| `pre_post_edit` | `diff_pre_post_edit.json/.png` | edit-subject MLP-out before vs after rank-one (banks refreshed) |
| extra features | `diff_pre_post_<name>.json/.png` | from `diff.selected_features` in YAML |

JSON fields: `delta_metrics`, `summary_score`, `mean_abs_neighbor_delta`, `neighbor_deltas`, entered/left top-k.

## Toy sanity check

```bash
python scripts/tests/toy_superposition_sanity.py
```

Compares near-orthogonal features (`n ≤ d`) vs forced packing (`n ≫ d`) in the same ambient dim. Expect denser ⇒ higher `interference_mean` / `coactivation_overlap`, and `spectral_participation` moves. Writes `experiments/results/toy_superposition_sanity/` (`sanity_result.json`, `sparse_vs_dense.png`, `signals.csv`).
