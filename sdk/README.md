# spg — Superposition Geometry

Live **geometry beside loss**: track every step, diff selected features, plot the curves, optionally **control** packing via a soft regularizer.

**Start here:** [`QUICKSTART.md`](../QUICKSTART.md) (<20 lines).

```bash
pip install -e .
```

```python
from spg import Tracker, Diff, plot, GeoControl
# …
```

## CLI

```bash
spg plot experiments/results/train_gpt2_small_geometry/signals.csv
spg track --config experiments/configs/train_gpt2_small_geometry.yaml
spg track --config experiments/configs/train_gpt2_small_geometry_control.yaml
spg track --config experiments/configs/train_gpt2_small_geometry_control_band.yaml
spg diff --config experiments/configs/train_gpt2_small_geometry.yaml
spg diff --features --config experiments/configs/train_gpt2_small_geometry.yaml
spg view --run-dir experiments/results/train_gpt2_small_geometry --open
spg demo
spg edit-demo
```

| Command | Role |
| --- | --- |
| `spg track` | fine-tune + live `loss \| geometry` (+ optional `control:`) |
| `spg diff` | ckpt A vs B (default) or `--features` |
| `spg plot` | loss + geometry curves from `signals.csv` |
| `spg view` | interactive HTML neighborhood scrubber |
| `spg demo` | toy point / band / feature-relative (no model) |
| `spg edit-demo` | thin edit-time geo nudge (one MLP `W_out`) |

Also: `python -m spg …`.

Live lines use a small ANSI palette (teal = geometry, rose = control). Auto on TTY;
off with `NO_COLOR` or `SPG_COLOR=0`; force on with `SPG_COLOR=1`.

Package source: `sdk/spg/`. Soft control: `GeoControl` / `FeatureRelativeControl` / `grad_geometry` (off by default). Spec: [`CONTROL_GEO.md`](../CONTROL_GEO.md).
