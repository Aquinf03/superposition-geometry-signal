# Thesis

**Superposition geometry is a live training metric beside loss: track it every step, diff it for selected features as the run transforms — so you can watch entanglement the same way you watch loss.**

## Intuition

Loss = how wrong.  
Geometry = how tangled.  

Same dashboard. Same cadence. Diff = what changed in the tangle for features you pick.

## Scope

**In (the product):**
- live `loss | geometry: …` on real runs
- track + diff on trained models
- validate the signal moves when geometry actually changes
- SDK + optional 3D so anyone can watch it
- conference paper on geometry-as-training-signal

**Also shipping:**
- one simple edit example showing geometry moves when weights change

**Out of scope for this paper:**
- multi-variant ROME / edit-algorithm research as the spine
