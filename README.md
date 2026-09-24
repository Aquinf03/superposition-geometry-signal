# Superposition Geometry as a Training Signal

## What this is about

Models pack more features than they have dimensions. Features share space. That sharing is *superposition*. The shape of that sharing — angles, clusters, who interferes with whom — is *superposition geometry*.

This project does three things:

1. **Track** — log superposition geometry **beside the loss curve** as training / editing runs.
2. **Diff** — compare that geometry for **selected features** (before vs after, step vs step, target vs neighbors).
3. **Act** — use track+diff as a signal for cleaner weight edits (first application).

## Why it matters

| Signal | Tells you |
| --- | --- |
| Loss | how wrong |
| Geometry (track) | how tangled |
| Feature diff | what changed in the tangle |

Watching only loss misses interference that makes weight edits brittle — change one fact, nearby facts move. Track+diff makes that interference visible around the features you care about.

Then you can:

- prefer cleaner directions
- constrain the update so it stays in a less-interfering subspace
- penalize moves that disturb high-interference neighbors

## Core claim

**Superposition geometry is a training signal you track beside loss and diff over selected features — usable to drive more precise weight edits.**

See `THESIS.md` for the one-sentence claim and scope freeze.

## What we will do

1. Define cheap local geometry metrics (angles, interference, spectral localization).
2. **Track:** log them beside loss / edit objective every step.
3. **Diff:** selected features / neighborhoods across steps or pre/post edit.
4. **Act:** geometry-aware edit (choose layer / constrain update / soft geometry penalty).
5. Compare against standard ROME on hold, generalization, and locality.
6. Show when track+diff helps, when it does not, and why.

## Success looks like

- Geometry curve moves meaningfully beside loss / edit objective (track).
- Feature diffs show which neighbors got more / less entangled (diff).
- Geometry-aware edits beat standard edits on locality / ripple without losing edit success or paraphrase generalization (act).
- Small model first; honest failure cases included.

## Link to prior work

Builds on Aquin’s experimental weight editor (ROME + validation + locality benchmarks) and recent feature-geometry / interference-weight research. See `REFERENCES.md`.

**Gap:** track geometry beside loss, diff it for selected features, then act — not only inspect interference after the fact.

## Repo layout

```
configs/     # default + per-run YAML (seed + track/diff logging)
data/        # datasets / fact triples
paper/       # LaTeX / workshop draft
results/     # signals (track), feature diffs, figures, tables
scripts/     # track logger, metrics, diff, ROME baseline, evals
```

## Setup

Python **≥ 3.10**. Stack: PyTorch + TransformerLens (+ HuggingFace `transformers` as fallback).

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python scripts/check_env.py
```

Seed control: `scripts/seed.py`. Track logger: `scripts/signals.py` (`step`, `loss`, `geometry_*`). Defaults in `configs/default.yaml`.

### ROME baseline (act target; track placeholders for now)

```bash
python scripts/rome_edit.py --config configs/rome_gpt2_small.yaml
```

Edits `The Eiffel Tower is located in` → `Rome` on GPT-2 small (layer 8). Writes `results/<run>/rome_result.json` and `signals.csv` with **real** `geometry_*` beside `loss` each `v` step (see `docs/metrics.md`).

```bash
python scripts/plot_signals.py --csv results/rome_gpt2_small_baseline/signals.csv
```
