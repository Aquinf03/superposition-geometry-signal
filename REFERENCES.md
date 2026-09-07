# Related work

Core citations for the paper. Grouped by thread.

## Weight editing

- Meng, Bau, Andonian, Belinkov. *Locating and Editing Factual Associations in GPT* (ROME). NeurIPS 2022. https://rome.baulab.info/ · https://arxiv.org/abs/2202.05262
- Meng, Sharma, Andonian, Belinkov, Bau. *Mass-Editing Memory in a Transformer* (MEMIT). ICLR 2023. https://memit.baulab.info/ · https://arxiv.org/abs/2210.07229
- Geva, Schuster, Berant, Levy. *Transformer Feed-Forward Layers Are Key-Value Memories*. EMNLP 2021. https://arxiv.org/abs/2012.14913

## Toy superposition / feature geometry

- Elhage et al. *Toy Models of Superposition*. Transformer Circuits, 2022. https://transformer-circuits.pub/2022/toy_model/index.html
- Bricken et al. *Towards Monosemanticity: Decomposing Language Models With Dictionary Learning*. Transformer Circuits, 2023. https://transformer-circuits.pub/2023/monosemantic-features/index.html

## Spectral geometry

- Ivanov, Oozeer, Raval, Pejovic, Upadhyay, Abdullah. *Spectral Superposition: A Theory of Feature Geometry*. arXiv:2602.02224, 2026. https://arxiv.org/abs/2602.02224

## Interference weights

- Olah, Turner, Conerly. *A Toy Model of Interference Weights*. Transformer Circuits, 2025. https://transformer-circuits.pub/2025/interference-weights/index.html
- *Characterizing interference weights in a tiny language model*. Transformer Circuits, 2026. https://transformer-circuits.pub/2026/interference_effectiveness_helpfulness/index.html

## How this paper sits

Editing work (ROME/MEMIT) shows facts can be rewritten with rank-one / multi-layer updates, but locality still fails when targets share space. Superposition + spectral geometry explain *why* features interfere. Interference-weight work shows many weight connections are compromise, not computation.

**Gap we fill:** treat local superposition geometry as a **live training signal beside loss** (same dashboard, second curve), then *use* that signal to constrain edits — not only as a post-hoc diagnostic of interference.
