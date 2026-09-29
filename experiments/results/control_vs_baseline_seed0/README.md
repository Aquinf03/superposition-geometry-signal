# Control vs baseline (seed 0)

Matched hero fine-tunes: same corpus, seed, model, steps, LR.

| Run | Config | Control |
| --- | --- | --- |
| `../train_gpt2_small_geometry` | `train_gpt2_small_geometry.yaml` | off |
| `../train_gpt2_small_geometry_control` | `train_gpt2_small_geometry_control.yaml` | L8 interference → 0.35, λ=1e-2, warmup=5 |

```bash
python experiments/compare_control_vs_baseline.py
python experiments/diff_features.py --config experiments/configs/train_gpt2_small_geometry.yaml
python experiments/diff_features.py --config experiments/configs/train_gpt2_small_geometry_control.yaml
python experiments/validate_control_geometry.py
python experiments/eval_control_lm.py   # held-out LM loss / PPL (quality claim)
```

| Artifact | What it answers |
| --- | --- |
| `compare.json` | Did control move L8 closer to target? |
| `out_of_metric_validation.json` | Loss / cross-feature / held-out probes OK? |
| `held_out_lm_eval.json` | Does control beat baseline on **frozen eval text**? |

**Held-out LM (seed 0):** control slightly lower eval loss than baseline (Δ≈−0.019). Both are worse than pretrained gpt2-small — the 40-step toy fine-tune overfits `train_corpus.txt`. Treat the Δ as a directional signal, not a strong quality claim until multi-seed / longer runs.
