# Mechanistic Interpretability Research

This repository studies how neural networks produce their answers. Each completed study includes a paper, executable methods, results, and clear limits.

## Research standard

The work is AI-assisted. AI review is not human peer review. A new experiment is not proof that its idea is new. We check related work and state the specific contribution of each study.

Read the [research protocol](RESEARCH_PROTOCOL.md) for the publication rules and the [research state](docs/RESEARCH_STATE.md) for current work.

## Papers

| Date | Paper | Main result |
| --- | --- | --- |
| 13 October 2026 | [When an input should not change the answer](studies/2026-10-13-inactive-variable-audit/paper.md) | Two neuron-selection rules chose the same sets in eight small models. This negative result did not validate those sets: substantial intervention errors remained. |
| 13 October 2026 | [Rare-event weighting does not reliably isolate one causal feature](studies/2026-10-13-rare-feature-sae/paper.md) | Giving rare cases more weight was worse for the selected feature in five of six main comparisons. The full learned code carried more useful change than the selected feature. |
| 13 October 2026 | [The same subspace can specify different interventions](studies/2026-10-13-basis-transport/paper.md) | Selected coordinate patches changed answers correctly on 43.73% of test pairs after rotations. The correctly transformed intervention kept 100% accuracy in this constructed task. |

Each completed study includes reproduction instructions, saved model weights, full result tables, and an independent AI calculation review. These claims apply to the tested small models. None has recorded human peer review.

## Use and rights

Papers cite external sources. External papers and datasets retain their own terms. No repository-wide licence has been selected. Public access alone does not grant a general reuse licence.
