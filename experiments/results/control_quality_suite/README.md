# Control quality suite

Longer matched fine-tunes (200 steps, `train_corpus_extended.txt`, lr=2e-5) × seeds 0/1/2, then held-out LM eval vs pretrained.

```bash
python experiments/run_control_quality_suite.py --seeds 0 1 2
```

See `summary.json` for the multi-seed table. Per-seed: `held_out_lm_eval_s{N}.json`.
