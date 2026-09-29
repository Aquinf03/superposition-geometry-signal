# Data

Corpora and fixtures for experiments (paths in YAML are relative to `experiments/`).

| File | Role |
| --- | --- |
| `train_corpus.txt` | Local fine-tune corpus for the hero training loop |
| `train_corpus_extended.txt` | Larger local corpus (optional longer runs) |
| `eval_corpus.txt` | Held-out text for optional LM eval |
| `wikitext2/` | WikiText-2 raw (optional; `python experiments/fetch_wikitext2.py`) |
| `eiffel_rome.yaml` | Thin edit demo fixture |

Keep large raw dumps out of git (`experiments/data/raw/`, `experiments/data/processed/` are gitignored).
