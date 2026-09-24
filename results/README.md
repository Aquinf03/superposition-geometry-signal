# Results

Each run writes under `results/<run_name>/`:

| Artifact | Role |
| --- | --- |
| `signals.csv` | **Track** — `step`, `loss`, `geometry_interference_mean`, `geometry_spectral_participation`, `geometry_coactivation_overlap` |
| `loss_geometry.png` | **Track** plot — from `scripts/plot_signals.py` |
| `meta.json` | seed, model, config snapshot, neighbor bank size |
| `*_diff.json` / figures | **Diff** — selected feature neighborhoods pre/post or step vs step |
| `rome_result.json` | **Act** — edit success + probs (ROME baseline) |

Produced by `scripts/signals.py` and experiment entrypoints.
