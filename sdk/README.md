# spg — Superposition Geometry

Live **geometry beside loss**: track every step, diff selected features, plot the curves.

```bash
pip install -e .
```

```python
from spg import Tracker, Diff, plot

tracker = Tracker(results_dir="runs", run_name="demo", live_print=True)
tracker.log(
    step=0,
    loss=1.23,
    geometry={
        "interference_mean": 0.41,
        "spectral_participation": 0.18,
        "coactivation_overlap": 0.09,
    },
)
plot(tracker.csv_path)
```

| API | Role |
| --- | --- |
| `Tracker` | append `step`, `loss`, `geometry_*` (+ live stdout) |
| `Diff` | neighborhood snapshot / compare / save / plot |
| `plot` | loss + geometry curves from `signals.csv` |

Package source: `sdk/spg/`. Experiments and helpers import the same code via shims.
