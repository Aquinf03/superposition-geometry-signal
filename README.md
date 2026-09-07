# Superposition Geometry as a Training Signal

## What this is about

Models pack more features than they have dimensions. Features share space. That sharing is *superposition*. The shape of that sharing — angles, clusters, who interferes with whom — is *superposition geometry*.

Most work only *looks* at that geometry after training. This project asks: can we *use* it as a signal while editing or training?

## Why it matters

Weight edits (ROME-style rank-one updates) often break nearby facts. The edit hits a direction that is entangled with other features. Side effects show up as ripple failures and locality failures.

If we can measure local geometry around an edit target, we can:

- prefer cleaner directions
- constrain the update so it stays in a less-interfering subspace
- penalize moves that disturb high-interference neighbors

Goal: edits that stick under paraphrase and leave the rest of the model more stable.

## Core claim

**Superposition geometry is not only a diagnostic. It can be a control signal for more precise weight edits.**

## What we will do

1. Measure local geometry around edit targets (angles, interference, spectral localization).
2. Feed that signal into the edit (choose layer / constrain update / add a geometry penalty).
3. Compare against standard ROME on hold, generalization, and locality.
4. Show when geometry helps, when it does not, and why.

## Success looks like

Geometry-aware edits beat standard edits on locality / ripple metrics without losing edit success or paraphrase generalization — on a small model first, then scaled carefully.

## Link to prior work

Builds on Aquin’s experimental weight editor (ROME + validation + locality benchmarks) and recent feature-geometry / interference-weight research. The new piece is using geometry *during* the edit, not only after.

## Repo layout

```
data/        # datasets / fact triples
paper/       # LaTeX / workshop draft
results/     # run outputs, figures, tables
scripts/     # experiments + evals
```
