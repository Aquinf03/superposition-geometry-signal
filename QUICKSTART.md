# Quickstart (<20 lines)

Superposition geometry beside loss — track, diff, plot.

```bash
pip install -e .
```

```python
from spg import Tracker, Diff, plot
import torch

t = Tracker(results_dir="runs", run_name="quickstart", live_print=True)
q0, q1 = torch.randn(32), torch.randn(32)
bank = torch.randn(64, 32)
for step, (loss, q) in enumerate([(1.2, q0), (0.8, q1)]):
    snap = Diff.snapshot(q, bank, feature_id="demo", step=step)
    t.log(step=step, loss=loss, geometry=snap["metrics"])
d = Diff.compare(Diff.snapshot(q0, bank, feature_id="t0"), Diff.snapshot(q1, bank, feature_id="t1"))
print("Δ summary", round(d["summary_score"], 4))
plot(t.csv_path)  # → runs/quickstart/loss_geometry.png
```

That’s it: **Tracker** logs live `loss | geometry`, **Diff** compares neighborhoods, **plot** draws the curves. Soft control: `from spg import GeoControl` (opt-in; see `CONTROL_GEO.md`).

```bash
spg plot runs/quickstart/signals.csv
spg view --run-dir experiments/results/train_gpt2_small_geometry --open
spg control-demo
spg edit-demo
```

Real training: `spg track --config experiments/configs/train_gpt2_small_geometry.yaml`  
Control: `spg track --config experiments/configs/train_gpt2_small_geometry_control.yaml`  
Band: `spg track --config experiments/configs/train_gpt2_small_geometry_control_band.yaml`  
Caveats: `failure_notes.md` · Control: `CONTROL_GEO.md`
