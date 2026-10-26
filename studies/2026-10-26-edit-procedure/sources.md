# Source check

Checked on 26 October 2026. The links below are primary sources. The checks establish relevant prior work. They do not establish that this project is the first to test these concerns.

- Fred Zhang and Neel Nanda, [Towards Best Practices of Activation Patching in Language Models: Metrics and Methods](https://arxiv.org/abs/2309.16042v2), version 2, 2024. It tests how patching choices and scores affect conclusions. Our donor and score controls are a small diagnostic extension.
- Stefan Heimersheim and Neel Nanda, [How to use and interpret activation patching](https://arxiv.org/abs/2404.15255v1), 2024. It explains limits on circuit evidence from patching and pitfalls in metric choice. Our report must separate an edit result from a correct variable explanation.
- Atticus Geiger and colleagues, [Finding Alignments Between Interpretable Causal Variables and Distributed Neural Representations](https://proceedings.mlr.press/v236/geiger24a.html), 2024. It tests distributed alignments between high-level variables and network states. Our fixed coordinate masks do not exhaust the possible alignments.
- Dan Braun and colleagues, [Identifying Functionally Important Features with End-to-End Sparse Dictionary Learning](https://arxiv.org/abs/2405.12241v2), 2024. It distinguishes activation reconstruction from downstream function. Our constrained code search uses a known synthetic readout and does not train its proposed method.
- Jimmy Lei Ba, Jamie Ryan Kiros, and Geoffrey Hinton, [Layer Normalization](https://arxiv.org/abs/1607.06450v1), 2016. It defines shared normalization statistics within a layer. Our frozen-statistics comparison tests a known change to that rule; it does not discover the dependence on mean and scale.

Internal sources are the fixed repository assets in `input-manifest.json`, the [24-study review](../../docs/REVIEW_PAPER.md), and the [research plan](../../docs/RESEARCH_PLAN.md). The old studies remain unchanged. The latest task-ability study adds a separate completed paper but provides no new training for this experiment.
