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
| 13 October 2026 | [How one attention head selects and copies a supplied value](studies/2026-10-13-copy-routing/paper.md) | Changing the route or the values in a selected head each reversed the mean preference between two supplied answers. Applying both changes largely restored it. This is a bounded replication of a known copying mechanism. |
| 13 October 2026 | [Selecting a patch site for a later output effect](studies/2026-10-13-patch-memory/paper.md) | The layer with the largest immediate change differed from the layer with the largest change after eight fixed tokens. The later choice transferred to new words, but the result does not show better answers. |

Each completed study includes reproduction instructions, full result tables, and an independent AI calculation review. The first three studies include their trained model weights. The last three provide a fixed public model revision and file checks. The fourth records a failed planned numerical check and the added check that explained it. These claims apply to the tested small models. None has recorded human peer review.

## Use and rights

Papers cite external sources. External papers and datasets retain their own terms. No repository-wide licence has been selected. Public access alone does not grant a general reuse licence.
