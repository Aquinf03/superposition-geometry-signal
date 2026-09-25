# Experiments

Runnable pipelines + their configs, data, and results.

```
experiments/
  configs/     # run YAML
  data/        # corpora / fixtures
  results/     # signals, diffs, plots, checkpoints
  *.py         # train / diff / validate / demos
```

Library code: `scripts/helpers/`. Sanities: `scripts/tests/`.

| File | Role |
| --- | --- |
| `train_with_geometry.py` | **Hero:** fine-tune + live `loss \| geometry` |
| `diff_checkpoints.py` | Ckpt A vs B on selected features |
| `diff_features.py` | Feature A vs B at one checkpoint |
| `validate_training_geometry.py` | Validate collapse / packing / layer phase |
| `rome_edit.py` | Thin edit demo only |
| `eval_edit.py` | Edit eval helpers (used by ROME demo) |

```bash
# Hero (gpt2-small)
python experiments/train_with_geometry.py --config experiments/configs/train_gpt2_small_geometry.yaml
python -m scripts.helpers.plot_signals --csv experiments/results/train_gpt2_small_geometry/signals.csv

# Scaled (gpt2-medium)
python experiments/train_with_geometry.py --config experiments/configs/train_gpt2_medium_geometry.yaml
python -m scripts.helpers.plot_signals --csv experiments/results/train_gpt2_medium_geometry/signals.csv
python experiments/diff_checkpoints.py --config experiments/configs/train_gpt2_medium_geometry.yaml
python experiments/diff_features.py --config experiments/configs/train_gpt2_medium_geometry.yaml
python experiments/validate_training_geometry.py --run-dir experiments/results/train_gpt2_medium_geometry

# Thin edit demo
python experiments/rome_edit.py --config experiments/configs/rome_gpt2_small.yaml
```
