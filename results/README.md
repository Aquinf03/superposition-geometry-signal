# Results

Each run writes under `results/<run_name>/`:

| Artifact | Role |
| --- | --- |
| `signals.csv` / `loss_geometry.png` | **Track** |
| `diff_*.json/.png` | **Diff** |
| `eval.json` | **Eval** — edit success, paraphrase, ripple KL, activation cosine |
| `config.frozen.yaml` | Exact config used (includes seed) |
| `config.source.yaml` | Copy of the YAML you passed on the CLI |
| `seed.txt` | Seed only |
| `run_manifest.json` | Seed + timestamps + config paths |
| `rome_result.json` | **Act** summary + paths to everything above |
| `meta.json` | Logger meta (model, bank size, etc.) |
