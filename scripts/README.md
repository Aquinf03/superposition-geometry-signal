# Scripts

Pipeline verbs: **track** → **diff** → **act** (+ **eval** / frozen config).

| File | Role |
| --- | --- |
| `seed.py` | Deterministic seeds for every run |
| `check_env.py` | Verify torch / transformers / transformer_lens |
| `geometry.py` | **Track metrics** A/B/C + MLP-out neighbor bank |
| `signals.py` | **Track logger:** `step`, `loss`, `geometry_*` → `results/` |
| `plot_signals.py` | Plot loss beside geometry curves |
| `diff_geometry.py` | **Diff:** neighborhood snapshots, pre/post + step vs step |
| `eval_edit.py` | **Eval:** success / paraphrase / ripple / activation cosine |
| `run_artifacts.py` | Freeze `config` + `seed` into the run dir |
| `rome_edit.py` | **Act:** ROME baseline; track + diff + eval + artifacts |
| `toy_superposition_sanity.py` | Sanity: sparse vs dense packs → metrics must move |
| `sanity_entangled_vs_clean_edit.py` | Sanity: entangled edit ⇒ larger neighbor geometry Δ than clean |

```bash
python scripts/rome_edit.py --config configs/rome_gpt2_small.yaml
python scripts/plot_signals.py --csv results/rome_gpt2_small_baseline/signals.csv
python scripts/toy_superposition_sanity.py
python scripts/sanity_entangled_vs_clean_edit.py
# optional real-model version:
python scripts/sanity_entangled_vs_clean_edit.py --real
```

Per-run outputs under `results/<run>/`:
- `signals.csv`, `loss_geometry.png`
- `diff_*.json/.png`
- `eval.json` — success, paraphrase, ripple, activation cosine
- `config.frozen.yaml`, `config.source.yaml`, `seed.txt`, `run_manifest.json`
- `rome_result.json`
