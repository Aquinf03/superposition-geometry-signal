# Superposition Geometry as a Training Signal

## What this is about

Models pack more features than they have dimensions. That sharing is *superposition*. Its shape — angles, interference, who overlaps whom — is *superposition geometry*.

**This project treats geometry like loss:** a live metric you track every step, and diff for selected features as training runs.

```text
step=12  loss=0.41  |  geometry: interference=0.22  spectral=0.31  coact=0.09
```

Weight editing is only a thin demo that geometry moves when internals change — **not** the research depth.

## Why it matters

| Signal | Tells you |
| --- | --- |
| Loss | how wrong |
| Geometry (track) | how tangled |
| Feature diff | what changed in the tangle |

Watching only loss misses how representation geometry is shifting while the model learns. Geometry beside loss makes that visible live.

## Core claim

**Superposition geometry belongs beside the loss curve as a training signal you track and diff.**

See `THESIS.md`.

## What we will do

1. Cheap local geometry metrics (interference / spectral / coactivation).
2. **Track** them live beside loss on real training runs.
3. **Diff** selected features across steps / checkpoints.
4. Validate the signal on real models (not toy-primary).
5. Ship an SDK (+ optional 3D) and a conference paper.

## Success looks like

- Live `loss | geometry` on a real training run.
- Feature diffs that show entanglement changing over the run.
- Outsiders can use the SDK in <20 lines.
- Paper evidence matches that story.

## Link to prior work

Toy superposition, spectral geometry, interference weights — see `REFERENCES.md`.  
Gap: treat geometry as a **live training metric beside loss**, not only a post-hoc diagnostic.

## Repo layout

```
configs/     # run YAML (seed + logging)
data/        # small fixtures
paper/       # LaTeX
results/     # signals, diffs, plots
scripts/     # track / diff / plots / thin demos
```

## Setup

Python **≥ 3.10**. PyTorch + TransformerLens (+ HF).

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python scripts/check_env.py
```

### Thin demo (optional — not the hero)

```bash
python scripts/rome_edit.py --config configs/rome_gpt2_small.yaml
python scripts/plot_signals.py --csv results/rome_gpt2_small_baseline/signals.csv
```

Hero next: a real training loop with live `loss | geometry`.
