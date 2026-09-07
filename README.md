# Superposition Geometry as a Training Signal

## What this is about

Models pack more features than they have dimensions. Features share space. That sharing is *superposition*. The shape of that sharing — angles, clusters, who interferes with whom — is *superposition geometry*.

Loss is the curve everyone watches. This project puts **superposition geometry beside that curve** as a live signal: track how tangled features are while the model learns or while you edit, then act on that signal.

## Why it matters

Loss tells you *how wrong*. Geometry tells you *how tangled*. Watching only loss misses interference that makes weight edits brittle — change one fact, nearby facts move.

If geometry sits next to loss, you can see interference rise, then:

- prefer cleaner directions
- constrain the update so it stays in a less-interfering subspace
- penalize moves that disturb high-interference neighbors

Goal: a geometry curve that is as readable as loss, and edits that stick under paraphrase without wrecking locality.

## Core claim

**Superposition geometry belongs beside the loss curve as a training signal — and that signal can drive more precise weight edits.**

## What we will do

1. Define cheap local geometry metrics (angles, interference, spectral localization).
2. Log them **beside loss / edit objective** every step (same dashboard, two curves).
3. Use the signal at edit time (choose layer / constrain update / soft geometry penalty).
4. Compare against standard ROME on hold, generalization, and locality.
5. Show when the geometry signal helps, when it does not, and why.

## Success looks like

- Geometry curve moves meaningfully beside loss / edit objective.
- Geometry-aware edits beat standard edits on locality / ripple without losing edit success or paraphrase generalization.
- Small model first; honest failure cases included.

## Link to prior work

Builds on Aquin’s experimental weight editor (ROME + validation + locality benchmarks) and recent feature-geometry / interference-weight research. See `REFERENCES.md`. The new piece: treat geometry as a **live signal beside loss**, then use it — not only inspect it after the fact.

See also `THESIS.md` for the one-sentence claim and scope freeze.

## Repo layout

```
configs/     # default + per-run YAML (seed + signal logging)
data/        # datasets / fact triples
paper/       # LaTeX / workshop draft
results/     # run outputs, curves, figures, tables
scripts/     # experiments, metrics, signal logging, evals
```

## Setup

Python **≥ 3.10**. Stack: PyTorch + TransformerLens (+ HuggingFace `transformers` as fallback).

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python scripts/check_env.py
```

Seed control: call `set_seed` from `scripts/seed.py` (or `seed_from_config`) at the top of every run. Default seed + signal-logging knobs live in `configs/default.yaml`. Log loss and geometry together via `scripts/signals.py`.
