# Results

Each run should write under `results/<run_name>/`:

- `signals.csv` — columns: `step`, `loss`, `geometry_*` (geometry beside loss)
- `meta.json` — seed, model, config snapshot
- figures (loss+geometry curves, edit tables)

Produced by `scripts/signals.py` and experiment entrypoints.
