# Scripts

Pipeline: **train** (hero) → **track** → **diff** → thin **act** demo.

| File | Role |
| --- | --- |
| `seed.py` | Deterministic seeds for every run |
| `check_env.py` | Verify torch / transformers / transformer_lens |
| `geometry.py` | **Track metrics** A/B/C + MLP-out neighbor bank |
| `signals.py` | **Track logger:** `step`, `loss`, `geometry_*` → `results/` |
| `plot_signals.py` | Plot loss beside geometry curves |
| `train_with_geometry.py` | **Hero:** fine-tune + live `loss \| geometry` |
| `diff_geometry.py` | **Diff:** neighborhood snapshots, pre/post + step vs step |
| `eval_edit.py` | **Eval:** success / paraphrase / ripple / activation cosine |
| `run_artifacts.py` | Freeze `config` + `seed` into the run dir |
| `rome_edit.py` | Thin edit demo only |
| `toy_superposition_sanity.py` | Sanity: sparse vs dense packs → metrics must move |
| `sanity_entangled_vs_clean_edit.py` | Controlled neighbor-Δ sanity |

```bash
# Hero
python scripts/train_with_geometry.py --config configs/train_gpt2_small_geometry.yaml
python scripts/plot_signals.py --csv results/train_gpt2_small_geometry/signals.csv

# Thin edit demo
python scripts/rome_edit.py --config configs/rome_gpt2_small.yaml
python scripts/toy_superposition_sanity.py
python scripts/sanity_entangled_vs_clean_edit.py
```

Per-run outputs under `results/<run>/`:
- `signals.csv`, `loss_geometry.png`
- `train_result.json` (hero) or `rome_result.json` (demo)
- `config.frozen.yaml`, `seed.txt`, `run_manifest.json`
