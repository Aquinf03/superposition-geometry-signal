# Scripts

Pipeline verbs: **track** → **diff** → **act**.

| File | Role |
| --- | --- |
| `seed.py` | Deterministic seeds for every run |
| `check_env.py` | Verify torch / transformers / transformer_lens |
| `geometry.py` | **Track metrics** A/B/C + MLP-out neighbor bank |
| `signals.py` | **Track logger:** `step`, `loss`, `geometry_*` → `results/` |
| `plot_signals.py` | Plot loss beside geometry curves |
| `diff_geometry.py` | **Diff:** neighborhood snapshots, pre/post + step vs step, plots |
| `rome_edit.py` | **Act:** ROME baseline; tracks + diffs automatically |

```bash
python scripts/rome_edit.py --config configs/rome_gpt2_small.yaml
python scripts/plot_signals.py --csv results/rome_gpt2_small_baseline/signals.csv
```

Outputs under `results/<run>/`:
- `signals.csv`, `loss_geometry.png` — track
- `diff_pre_post_edit.json/.png`, `diff_step_vs_step.json/.png` — diff
- `diff_pre_post_<feature>.json/.png` — extra selected features
- `rome_result.json` — act

See `docs/metrics.md` for metric definitions.
