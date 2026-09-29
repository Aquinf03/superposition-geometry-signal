# WikiText-2 ± geo-control benchmarks (seed 0)

400-step gpt2-small fine-tunes on WikiText-2 train, then:

- WikiText-2 valid/test perplexity
- LAMBADA (full test)
- HellaSwag (1000 val)

```bash
python experiments/fetch_wikitext2.py
python experiments/run_wikitext_control_suite.py --seeds 0
```

See `summary.json` / `benchmarks_s0.json`.
