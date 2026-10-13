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
| 13 October 2026 | [Testing whether patch-interaction corrections transfer to new prompts](studies/2026-10-13-patch-calibration-transfer/paper.md) | A correction fitted on one set of prompts gave no clear benefit on new words and sentence forms in Pythia-14M. Mean prediction error increased from 0.030472 to 0.031135 logit units. |

Each completed study includes reproduction instructions, full result tables, and an independent AI calculation review. The first three studies include their trained model weights. The fourth provides a fixed public model revision and file checks. It also records a failed planned numerical check and the added check that explained it. These claims apply to the tested small models. None has recorded human peer review.

## Use and rights

Papers cite external sources. External papers and datasets retain their own terms. No repository-wide licence has been selected. Public access alone does not grant a general reuse licence.
