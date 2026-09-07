# Thesis

**Superposition geometry belongs beside the loss curve: track local feature interference as a first-class training signal, and use that signal to drive cleaner weight updates (edits) without sacrificing task loss.**

## Intuition

Loss tells you *how wrong* the model is. Geometry tells you *how tangled* the features are while it learns / while you edit. Plot them on the same dashboard. Act on the geometry signal the same way you already act on loss.

## Scope freeze

**In for v1:**
- small model
- log geometry metrics **next to loss** (or next to the edit objective) every step / every edit
- factual weight edits as the first place we *use* the signal (constrain / select the update)
- show that the geometry signal predicts or reduces locality damage vs standard ROME

**Out for v1 (follow-up if time):** full train-time geometry regularizer as the main claim, large models, non-factual edits as primary results.
