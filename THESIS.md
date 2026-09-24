# Thesis

**Track superposition geometry beside the loss curve as training runs, then diff that geometry for selected features (across steps, checkpoints, or edits) so interference is visible and actionable — first use: cleaner factual weight edits.**

## Intuition

Two moves:

1. **Track** — while training (or while optimizing an edit), log how tangled features are, same cadence as loss.
2. **Diff** — pick features (or an edit target + its neighbors) and compare their superposition geometry: before vs after, step t vs step t+k, clean vs entangled.

Loss = how wrong. Geometry = how shared / interfering. Diff = what changed in that sharing.

## Scope freeze

**In for v1:**
- small model
- track geometry **next to loss** every step / every edit
- **diff** selected features / neighborhoods (pre vs post edit; optionally across training steps)
- factual weight edits as the first place we *act* on the signal
- show geometry track+diff predicts or reduces locality damage vs standard ROME

**Out for v1 (follow-up if time):** full train-time geometry regularizer as the main claim, large models, non-factual edits as primary results.
