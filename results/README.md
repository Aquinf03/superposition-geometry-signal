# Results

Each run writes under `results/<run_name>/`:

| Artifact | Role |
| --- | --- |
| `signals.csv` | **Track** — `step`, `loss`, `geometry_*` |
| `loss_geometry.png` | **Track** plot — from `scripts/plot_signals.py` |
| `meta.json` | seed, model, config snapshot, neighbor bank size |
| `diff_step_vs_step.json/.png` | **Diff** — v-step 0 vs last |
| `diff_pre_post_edit.json/.png` | **Diff** — edit subject pre vs post rank-one |
| `diff_pre_post_<feature>.*` | **Diff** — extra selected features |
| `rome_result.json` | **Act** — edit success + probs + diff paths |

Produced by `scripts/signals.py`, `scripts/diff_geometry.py`, and `scripts/rome_edit.py`.
