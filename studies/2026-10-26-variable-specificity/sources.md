# Source check

Primary work was checked on 26 October 2026. The study is a controlled extension, not a claim of a new alignment method.

- Atticus Geiger and colleagues, [Finding Alignments Between Interpretable Causal Variables and Distributed Neural Representations](https://proceedings.mlr.press/v236/geiger24a.html), 2024. It uses learned distributed alignments rather than requiring one variable to occupy one neuron set. Our learned three-direction replacement is a small task-specific comparison within this established approach.
- Fred Zhang and Neel Nanda, [Towards Best Practices of Activation Patching in Language Models: Metrics and Methods](https://arxiv.org/abs/2309.16042v2), 2024. Its evidence on method and metric choices supports keeping donor, target, and scoring rules explicit. It does not supply a success guarantee for this experiment.
- Stefan Heimersheim and Neel Nanda, [How to use and interpret activation patching](https://arxiv.org/abs/2404.15255v1), 2024. Its interpretation limits apply to this study. A successful edit on a finite synthetic task is narrower than a complete semantic explanation of a real language model.

Internal sources are the original Study 1 training recipe, the [roadmap](../../docs/RESEARCH_PLAN.md), the completed [Experiment 2A](../2026-10-26-edit-procedure/paper.md), and [Experiment 2B](../2026-10-26-search-procedure/paper.md). The input manifest fixes exact source files. The graph branch remains blocked by Experiment 1. Experiment 2B's failed normal test is not used here. Fixed practical behavior gates are not population significance tests.
