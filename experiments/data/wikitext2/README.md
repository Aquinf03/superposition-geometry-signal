# WikiText-2 (gitignored large dumps)

```bash
python experiments/fetch_wikitext2.py
```

Writes `wiki.{train,valid,test}.txt` here. Used by:

- `train_gpt2_small_wikitext*.yaml` — fine-tune ± geo-control
- `eval_benchmarks.py` / `run_wikitext_control_suite.py` — WikiText PPL + LAMBADA + HellaSwag
