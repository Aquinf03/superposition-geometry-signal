# Scripts

SDK-facing layout:

```
scripts/
  helpers/   # library: geometry, signals, diff, plot, seed, paths
  tests/     # sanities + env check
experiments/
  configs/   # run YAML
  data/      # corpora / fixtures
  results/   # outputs
  *.py       # train / diff / validate / demos
```

| Path | Role |
| --- | --- |
| `helpers/geometry.py` | Metrics A/B/C + MLP-out banks |
| `helpers/signals.py` | Live `loss \| geometry` logger |
| `helpers/paths.py` | `experiments/{configs,data,results}` resolvers |
| `helpers/checkpoint_geometry.py` | Aligned weight↔geometry freezes |
| `helpers/diff_geometry.py` | Neighborhood snapshot + diff |
| `helpers/plot_signals.py` | Plot loss beside geometry |
| `helpers/seed.py` / `run_artifacts.py` | Seed + frozen run config |
| `tests/check_env.py` | Verify torch / transformer_lens |
| `tests/toy_superposition_sanity.py` | Sparse vs dense packs |
| `tests/sanity_entangled_vs_clean_edit.py` | Controlled neighbor-Δ sanity |

```bash
python scripts/tests/check_env.py
python scripts/tests/toy_superposition_sanity.py
python -m scripts.helpers.plot_signals --csv experiments/results/train_gpt2_small_geometry/signals.csv
```

See `experiments/README.md` for training / diff / validate commands.
