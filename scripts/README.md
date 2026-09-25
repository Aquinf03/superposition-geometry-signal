# Scripts

```
sdk/spg/       # real package (pip install -e .)
scripts/
  helpers/     # thin shims → spg (experiments keep working)
  tests/       # sanities + env check
experiments/   # configs / data / results + runners
```

```bash
pip install -e .
spg plot experiments/results/train_gpt2_small_geometry/signals.csv
spg track --config experiments/configs/train_gpt2_small_geometry.yaml
spg diff --config experiments/configs/train_gpt2_small_geometry.yaml
spg diff --features --config experiments/configs/train_gpt2_small_geometry.yaml
```

See `sdk/README.md` and `experiments/README.md`.
