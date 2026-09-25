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
| `rome_result.json` | Thin edit demo summary |
| `train_result.json` | **Hero** training summary + final loss/geometry |
| `model_final.pt` / `model_final.geometry.json` | Final weights + aligned geometry freeze |
| `checkpoints/step_XXXXX.pt` | Weight snapshot |
| `checkpoints/step_XXXXX.geometry.json` | Geometry frozen at the same step (aligned) |
| `checkpoints/manifest.json` | Index of aligned weight↔geometry pairs |
| `diff_ckpt_*_vs_*/` | Ckpt A vs B feature diffs + plots |
| `diff_features_step_*/` | Feature A vs B diffs at one ckpt |
| `validation.json` / `validation.png` | Training-geometry validation report |
| `meta.json` | Logger meta (model, bank size, etc.) |
