# Failure notes — reading the geometry signal

Product docs: [`docs/documentation/caveats/`](docs/documentation/caveats/).

Notes from the GPT-2 small hero run (`experiments/results/train_gpt2_small_geometry/`) plus the toy / controlled sanities. Use them to read the live dashboard.

## Flat geometry (signal silent)

| Situation | What you see | Why |
| --- | --- | --- |
| Wrong / inactive layer | Curves barely twitch while loss falls | Early layers can stay nearly constant; mid/late move more on this corpus |
| Too-coarse cadence | Sparse points look constant | `every_n_steps` large → miss fast phase changes |
| Tiny / saturated bank | Metrics stuck near floor/ceiling | Few probes, or `top_k` / `top_dims` degenerate; coact was capped for full-dim Jaccard=1 |
| Already-collapsed regime | Late run: spectral ≈ 0 and stays there | After L11 collapse (~step 15+), spectral has little left to say |
| No weight change | Geometry frozen | Eval-only / zero LR / logging without stepping |

**Check:** `validate_training_geometry.py` claim `geometry_not_flat`. If FAIL, widen layers, probes, or step budget before trusting diffs.

## How the signal moves

### 1. Overfit on a tiny corpus looks like “structure”

Hero run: loss `4.58 → 0.03` on ~50 lines. L11 spectral collapses and interference climbs — packing change on a memorizing fine-tune. The gpt2-medium scaled run uses the same corpus, so read it the same way: geometry tracked the trajectory.

### 2. Late-layer collapse hides feature structure

Cross-feature at step 39:

- L4/L8: related (eiffel↔louvre) ≪ unrelated (vs superposition) — good
- L11: all pairs look similarly close (~0.03)

After spectral collapse, **feature A vs B gaps shrink**. Prefer mid layers for structure; treat collapsed late layers as a phase marker.

### 3. Positive ρ(loss, spectral) is collapse, not “more superposition”

On gpt2-small, L11 spectral ρ with loss ≈ **+0.90**: as loss falls, spectral falls. That is **collapse / localization of energy**, not rising superposition.

On gpt2-medium, L22 can start **already near the spectral floor** (~0.01) with very high interference; fine-tuning then *unpacks* slightly (interference↓, spectral↑). Same claim family (“late-layer phase”), opposite arrow — do not hard-code “collapse only.”

### 4. Bank / probe choice biases the story

Metrics are always **relative to the probe bank**. A bank of location prompts makes location features look coherent; an unrelated prompt looks far. Change the bank → change the narrative. Freeze probes in the aligned `.geometry.json` and reuse them for ckpt diffs.

### 5. Neighbor index is only stable if the bank is ordered the same way

Ckpt A vs B and feature A vs B compare `bank_index` under a fixed probe list + `bank_positions`. If you rebuild with different prompts or `all` vs `last`, “neighbors entered/left top-k” is not comparable.

### 6. Device / numerics

TransformerLens warns that **MPS may be silently wrong** on some PyTorch builds. If a plot looks absurd, re-run on CPU before writing it into the paper. Deprecation warnings on `HookedTransformer.from_pretrained` are noise for now.

### 7. Edit locality and training geometry

ROME / weight-edit demos move geometry because weights changed. Pair them with `validate_training_geometry.py` on fine-tune trajectories for the training-signal story.

### 8. Coactivation near 1.0

If coact sits at ~1 with huge `top_dims`, you are measuring ambient overlap, not sparse sharing. The implementation caps top dims (~`d/4`); still watch for saturation.

### 9. Soft geo-control: train bank and held-out

Matched pair (`control_vs_baseline_seed0`, λ=1e-2 on L8 interference→0.35):

- **Train-bank / live L8** moved closer to target than baseline (compare.json).
- **Held-out probes** (disjoint prompts) stayed within slack rather than chasing the same target.
- Cross-feature related≪unrelated and downstream LM loss **PASS**ed under control.
- **Held-out LM eval** (`eval_control_lm.py`, `data/eval_corpus.txt`): control slightly beat baseline on eval loss (Δ≈−0.019). Both sit above pretrained gpt2-small on this short fine-tune; multi-seed / longer data sharpens the quality read.

**Check:** `python experiments/validate_control_geometry.py` + `python experiments/eval_control_lm.py`.

## How to read a run

1. Loss + multi-layer plot first (`python -m scripts.helpers.plot_signals`).
2. Ask which **layer phase** moved (collapse vs mild drift).
3. Diff **selected features** at mid layers; treat collapsed late layers as context.
4. Run `validate_training_geometry.py`.
5. If control is on: run `validate_control_geometry.py` and `eval_control_lm.py`; read held-out claims with the train-bank move.
6. Keep bank, layer, and collapse phase in the write-up.

## Paper-facing one-liner

> Geometry is a **live companion to loss**, validated when it moves with collapse, packing, and layer phase across frozen probe banks.
