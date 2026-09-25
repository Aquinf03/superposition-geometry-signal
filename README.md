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

See `THESIS.md`. Caveats: `failure_notes.md`. Future (after watch ships): `CONTROL_GEO.md`.

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
experiments/
  configs/     # run YAML
  data/        # corpora / fixtures
  results/     # signals, diffs, plots
  *.py         # train / diff / validate / demos
paper/         # LaTeX
sdk/spg/       # Superposition Geometry package (pip install -e .)
scripts/
  helpers/     # shims → spg
  tests/       # sanities + env check
```

## Setup

Python **≥ 3.10**. PyTorch + TransformerLens (+ HF).

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .          # spg SDK (Tracker / Diff / plot)
python scripts/tests/check_env.py
```

### Hero — train with live geometry

```bash
# gpt2-small (primary)
python experiments/train_with_geometry.py --config experiments/configs/train_gpt2_small_geometry.yaml
python -m scripts.helpers.plot_signals --csv experiments/results/train_gpt2_small_geometry/signals.csv

# gpt2-medium (scaled live example — same track+diff, no edits)
python experiments/train_with_geometry.py --config experiments/configs/train_gpt2_medium_geometry.yaml
python -m scripts.helpers.plot_signals --csv experiments/results/train_gpt2_medium_geometry/signals.csv
python experiments/diff_checkpoints.py --config experiments/configs/train_gpt2_medium_geometry.yaml
python experiments/diff_features.py --config experiments/configs/train_gpt2_medium_geometry.yaml
python experiments/validate_training_geometry.py --run-dir experiments/results/train_gpt2_medium_geometry
```

### Thin demo (optional — not the research depth)

```bash
python experiments/rome_edit.py --config experiments/configs/rome_gpt2_small.yaml
python -m scripts.helpers.plot_signals --csv experiments/results/rome_gpt2_small_baseline/signals.csv
```
