# Controlling geometry (future)

**Status:** parked until the track + diff + SDK path ships.  
**Today:** geometry is *watched* beside loss (`Tracker` / `Diff` / `plot`).  
**Next act:** geometry is *set* — a controllable training signal, not only a dashboard.

## Idea

Loss has a knob: minimize it.  
Geometry should get a knob too: push packing / interference / spectral phase toward a target while the model trains (or during a thin edit).

Same cadence as loss. Same live line. Extra term (or constraint) on A/B/C — or on selected-feature neighborhood structure.

```text
step=12  loss=0.41  |  geometry: interference=0.22  spectral=0.31  coact=0.09  |  geo_target: …
```

## Why wait

We already know the signal can be flat or misleading (`failure_notes.md`):

- wrong layer / post-collapse washout
- bank- and probe-dependent neighborhoods
- tiny-corpus overfit looks like “structure”
- late-layer phase can be collapse *or* unpack-from-floor

If we optimize geometry before the watch path is solid, we will game the metric and call it science.

## What “set geo” could mean

| Mode | Mechanism | Use |
| --- | --- | --- |
| Soft regularizer | add `λ · geo_penalty` to the train loss | discourage runaway interference / encourage mid-layer structure |
| Target band | keep metric(s) inside `[lo, hi]` via hinge / barrier | stabilize packing while fine-tuning |
| Feature-relative | shrink related-pair Δ, grow unrelated-pair Δ | structure the tangle, not just scalars |
| Edit-time (thin) | one-liner demo: nudge weights so geometry moves toward a target | existence proof only — not the research depth |

Start with **one layer + one metric + one λ**. Multi-objective geo soup later.

## Safety rails (required if we build this)

1. Always log **loss and geo** live — control must not hide the watch dashboard.
2. Freeze probe bank + layers in the run config (same as aligned checkpoints).
3. Validate with **out-of-metric** checks (downstream loss, cross-feature structure, held-out probes) — not “geo went to target ⇒ success.”
4. Abort / warn if geo is flat or saturated before enabling the regularizer.
5. Keep control **off by default** in `spg`; opt-in API only.

## Sketch API (do not implement yet)

```python
from spg import Tracker, GeoControl  # future

tracker = Tracker(...)
control = GeoControl(
    layers=[8],
    metric="interference_mean",
    target=0.35,
    weight=1e-3,
)

loss = lm_loss + control.penalty(geometry)
tracker.log(step=t, loss=loss, geometry=geometry, geo_control=control.status())
```

## When to pick this up

After:

- [x] live track + multi-layer + ckpt/feature diffs + validation
- [x] `spg` package (`Tracker` / `Diff` / `plot`)
- [ ] CLI + quickstart
- [ ] paper figures from the *watch* story

Then: one controlled fine-tune on the hero corpus vs baseline, same validation suite, failure notes updated for “gamed geometry.”

## Paper one-liner (later)

> We first treat superposition geometry as a live companion to loss; we then show it can be softly controlled as a training regularizer without collapsing the watch signal.
