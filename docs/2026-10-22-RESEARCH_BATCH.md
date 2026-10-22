# Research batch: nine complete papers and one partial model study

**Date:** 22 October 2026. **Status:** Research stopped at the required usage reserve.

The request was for ten more papers. This batch has **nine complete papers**. The planned tenth study, a continuation of Experiment 1, is incomplete and is not counted as a complete paper.

The nine papers contain formal analyses, constructed-model tests, or a statistical simulation. None tests a trained language model. Each states its assumptions, close prior work, code, results, checks, and limits. These are new repository artifacts, not claims of nine new general principles. AI review is not human peer review.

| Paper | Main result | Evidence and limit |
| --- | --- | --- |
| [Donor choice can change the meaning of a hidden-state test](../studies/2026-10-22-donor-support/paper.md) | Models with identical ordinary answers can give different answers after a replacement. | 312 exact constructed cases; no trained model. |
| [Single-component tests leave many small mechanisms unresolved](../studies/2026-10-22-coalition-identifiability/paper.md) | 152 of 166 mechanisms share their singleton-test evidence with another mechanism. | Complete four-input monotone Boolean census; not a rate in real networks. |
| [Two representation edits can conflict](../studies/2026-10-22-edit-order/paper.md) | Overlapping projection edits often depend on order. Common donors do not always remove the conflict. | 3,645 exact rational cases and an algebraic condition. |
| [A circuit can resist further pruning and still be larger than necessary](../studies/2026-10-22-greedy-circuit-search/paper.md) | A locally minimal result can contain six components when one is sufficient. | 17 constructed circuits, 1,016 subset checks, and 14,550 deletion orders. |
| [Perfect reconstruction can leave a coordinate edit undetermined](../studies/2026-10-22-decoder-edit-ambiguity/paper.md) | Equal decoded inputs need not define equal coordinate-edit outputs. | 9,600 exact cases and a null-space condition; no trained encoder. |
| [Opposite output directions can hide the same useful task edit](../studies/2026-10-22-context-effect-cancellation/paper.md) | A nuisance effect ranked above a useful edit in six of ten fully correct constructed models under the fixed signed-mean score. | Known context and task labels; no unsupervised discovery claim. |
| [A local edit can spread through layer normalization](../studies/2026-10-22-layernorm-spillover/paper.md) | Every unedited normalized coordinate changed in all 2,048 fixed cases. | Random constructed states and an exact finite-edit decomposition. |
| [Larger logit gains need not give larger probability gains](../studies/2026-10-22-metric-rankings/paper.md) | 704 of 7,100 strict logit rankings reversed under probability gain. | A fixed binary-output grid; not a language-model failure rate. |
| [Selecting the largest noisy effect can create false evidence](../studies/2026-10-22-selection-null/paper.md) | Reusing selection data sharply increased false positives under the declared zero-effect model. | 180,000 simulation trials with analytic references; known Gaussian variance. |

## The incomplete tenth study

The [task-ability continuation report](../studies/2026-10-22-capability-completion/paper.md) records six complete graph models, including one reused model. All six passed their task and output-replay checks. Only two of the required four final models completed. The next final model has partial weights and 22,028 recorded updates. The last final graph model and the other task branches did not start.

The first fresh timing estimate failed. One fixed revision then passed with the same fourfold allowance and four-hour limit. Both records remain. The scientific run stopped at the usage reserve after 317.07 seconds of combined Experiment 1 process time, including the older attempt. No extra credit, reset, rented compute, or paid model API was used for these experiments.

The first [public plan commit](https://github.com/gavenren/mechanistic-interpretability-research/tree/1de96267c03027226b8ae33b10c23483256d49f6) preceded the revised timing check. The [final execution lock](https://github.com/gavenren/mechanistic-interpretability-research/tree/88e8df7689b65fbeef221423c115cd1b1c198f22) preceded all nine formal or simulation runs and the model continuation. The [nine complete papers and results](https://github.com/gavenren/mechanistic-interpretability-research/tree/f211b1031576fee4e81b5e907ef6ffdac559d17a) were published together.

The repository now has 24 complete research studies: the earlier 15 and these nine. The review paper and partial execution reports are separate. Automatic research remains paused. A later model continuation needs a new usage check and an explicit plan for the interrupted final seed.
