# Scripts

Pipeline verbs: **track** → **diff** → **act**.

| File | Role |
| --- | --- |
| `seed.py` | Deterministic seeds for every run |
| `check_env.py` | Verify torch / transformers / transformer_lens |
| `geometry.py` | **Track metrics** A/B/C + MLP-out neighbor bank |
| `signals.py` | **Track logger:** `step`, `loss`, `geometry_*` → `results/` |
| `plot_signals.py` | Plot loss beside geometry curves |
| `rome_edit.py` | **Act:** ROME baseline; tracks real geometry each `v` step |

```bash
python scripts/rome_edit.py --config configs/rome_gpt2_small.yaml
python scripts/plot_signals.py --csv results/rome_gpt2_small_baseline/signals.csv
```

See `docs/metrics.md` for metric definitions.
