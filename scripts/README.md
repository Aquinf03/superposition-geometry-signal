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
python -c "from spg import Tracker, Diff, plot; print(Tracker, Diff, plot)"
python scripts/tests/check_env.py
python -m spg.plot_signals --csv experiments/results/train_gpt2_small_geometry/signals.csv
```

See `sdk/README.md` and `experiments/README.md`.
