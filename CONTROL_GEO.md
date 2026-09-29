# Controlling geometry

**Status:** promoted into stable **`spg`** (`from spg import GeoControl`).  
**Today:** geometry is *watched* beside loss **and** can be *set* via an opt-in soft regularizer.  
Checklist: `TODO.md` → Paper.

## Idea

Loss has a knob: minimize it.  
Geometry should get a knob too: push packing / interference / spectral phase toward a target while the model trains (or during a thin edit).

Same cadence as loss. Same live line. Extra term (or constraint) on A/B/C — or on selected-feature neighborhood structure.

```text
step=12  loss=0.41  |  geometry: interference=0.22  spectral=0.31  coact=0.09  |  geo_target: …
```

## What “set geo” means

| Mode | Mechanism | Use |
| --- | --- | --- |
| Soft regularizer (point) | `λ · (metric − target)²` on the train loss | discourage runaway interference / encourage mid-layer structure |
| Target band | hinge outside `[lo, hi]` (`mode: band`) | stabilize packing while fine-tuning |
| Feature-relative | `FeatureRelativeControl`: shrink related-pair Δ, keep unrelated Δ above a margin | structure the tangle, not just scalars |
| Edit-time (thin) | `spg edit-demo` — nudge one MLP `W_out` toward a target/band | existence proof only — not the research depth |

Start with **one layer + one metric + one λ**. Multi-objective geo soup later.

## Safety rails

1. Always log **loss and geo** live — control must not hide the watch dashboard.
2. Freeze probe bank + layers in the run config (same as aligned checkpoints).
3. Validate with **out-of-metric** checks (`validate_control_geometry.py`) — not “geo went to target ⇒ success.”
4. Abort / warn if geo is flat or saturated before enabling the regularizer.
5. **Watch warmup:** `control.warmup_steps` logs geometry with λ=0, then `GeoControl.gate_enable` must pass before the regularizer arms.
6. Control **off by default** (`control.enabled: false`).

## API

```python
from spg import GeoControl, FeatureRelativeControl, format_control_line

# Point target
control = GeoControl(
    layer=8, metric="interference_mean", target=0.35, weight=1e-2, enabled=True
)

# Band [lo, hi] — zero penalty inside, hinge outside
band = GeoControl(
    layer=8, metric="interference_mean", mode="band", lo=0.30, hi=0.40,
    weight=1e-2, enabled=True,
)

# Feature-relative (Diff-style scores: lower = closer)
feat = FeatureRelativeControl(
    weight_related=1e-2, weight_unrelated=1e-2, margin=0.15, enabled=True
)
# pen = feat.penalty_on_scores(related_t, unrelated_t)

# differentiable path in the train loop:
# metric_t = mean_interference_tensor(bank_grad, top_k=…)
# loss = lm_loss + control.penalty_on_value(metric_t)
print(format_control_line(step=t, loss=float(lm_loss), geometry=geometry, control=control.status(geometry)))
```

```bash
spg demo
spg track --config experiments/configs/train_gpt2_small_geometry_control.yaml
spg track --config experiments/configs/train_gpt2_small_geometry_control_band.yaml
spg edit-demo   # thin edit-time nudge
python experiments/validate_control_geometry.py
```

## Matched pair (seed 0)

See `experiments/results/control_vs_baseline_seed0/` — compare.json + out_of_metric_validation.json.

```bash
spg track --config experiments/configs/train_gpt2_small_geometry_control.yaml
python experiments/validate_control_geometry.py
```

Caveat: train-bank L8 moved toward target; held-out probes / quality claims need care (`failure_notes.md` §9).

## Paper one-liner

> We first treat superposition geometry as a live companion to loss; we then show it can be softly controlled as a training regularizer without collapsing the watch signal — validated out-of-metric (loss, cross-feature, held-out probes), not by hitting the geo target alone.
