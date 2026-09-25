# spg — Superposition Geometry

Live **geometry beside loss**: track every step, diff selected features, plot the curves.

```bash
pip install -e .
```

```python
from spg import Tracker, Diff, plot
# …
```

## CLI

```bash
spg plot experiments/results/train_gpt2_small_geometry/signals.csv
spg track --config experiments/configs/train_gpt2_small_geometry.yaml
spg diff --config experiments/configs/train_gpt2_small_geometry.yaml
spg diff --features --config experiments/configs/train_gpt2_small_geometry.yaml
```

| Command | Role |
| --- | --- |
| `spg track` | fine-tune + live `loss \| geometry` |
| `spg diff` | ckpt A vs B (default) or `--features` |
| `spg plot` | loss + geometry curves from `signals.csv` |

Also: `python -m spg …`.

Package source: `sdk/spg/`. Experiments import the same code via `scripts/helpers/` shims.
