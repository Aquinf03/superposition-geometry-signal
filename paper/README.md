# Paper

Source only. No PDF builds in this repo.

- **LaTeX:** [`main.tex`](main.tex) + [`refs.bib`](refs.bib)
- **Figures:** [`figures/`](figures/) (PNG + table markdown/CSV). Regenerate with `python experiments/paper_figures.py`.

Primary evidence: **Qwen2.5-7B** matched m★ pair (seed 0, 200 steps) under
`experiments/results/modal_pull/`. Scale table compares 1.5B vs 7B held-out Δ.

Thesis: geometry beside loss (live) then soft control, validated out-of-metric.
See root `THESIS.md` and [`docs/documentation/control/`](../docs/documentation/control/).
