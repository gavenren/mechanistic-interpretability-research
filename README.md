# Mechanistic Interpretability Research

This repository studies how neural networks produce their answers. Each completed study includes a paper, executable methods, results, and clear limits.

## Research standard

The papers are AI-authored. AI review is not human peer review. A new experiment is not proof that its idea is new. We check related work and state the specific contribution of each study.

Start with the [reading guide](docs/READING_GUIDE.md) for the main terms and the limits of each score. Read the [research protocol](RESEARCH_PROTOCOL.md) for the publication rules and the [research state](docs/RESEARCH_STATE.md) for current work.

## Papers

| Date | Paper | Main result |
| --- | --- | --- |
| 13 October 2026 | [When an input should not change the answer](studies/2026-10-13-inactive-variable-audit/paper.md) | Two neuron-selection rules chose the same sets in eight small models. This negative result did not validate those sets: substantial intervention errors remained. |
| 13 October 2026 | [Rare-event weighting does not reliably isolate one causal feature](studies/2026-10-13-rare-feature-sae/paper.md) | Giving rare cases more weight was worse for the selected feature in five of six main comparisons. The full learned code carried more useful change than the selected feature. |
| 13 October 2026 | [The same subspace can specify different interventions](studies/2026-10-13-basis-transport/paper.md) | Selected coordinate patches changed answers correctly on 43.73% of test pairs after rotations. The correctly transformed intervention kept 100% accuracy in this constructed task. |
| 13 October 2026 | [Testing whether patch-interaction corrections transfer to new prompts](studies/2026-10-13-patch-calibration-transfer/paper.md) | A correction fitted on one set of prompts gave no clear benefit on new words and sentence forms in Pythia-14M. Mean prediction error increased from 0.030472 to 0.031135 logit units. |
| 13 October 2026 | [How one attention head selects and copies a supplied value](studies/2026-10-13-copy-routing/paper.md) | Changing the route or the values in a selected head each reversed the mean preference between two supplied answers. Applying both changes largely restored it. This is a bounded replication of a known copying mechanism. |
| 13 October 2026 | [Selecting a patch site for a later output effect](studies/2026-10-13-patch-memory/paper.md) | The layer with the largest immediate change differed from the layer with the largest change after eight fixed tokens. The later choice transferred to new words, but the result does not show better answers. |
| 13 October 2026 | [A joint test of position spacing and access to extra tokens](studies/2026-10-13-position-access/paper.md) | Each single edit sharply reduced the binding score. Both together matched the short-input control. The normal long input had a lower mean margin but more correct top predictions. |
| 13 October 2026 | [Same text, different token splits](studies/2026-10-13-segmentation-transfer/paper.md) | With different source/query splits, only 1 of 1,024 requested values was the top prediction. A positive head effect remained, but the fixed capability rule failed. |
| 13 October 2026 | [Tiny order effects after balanced retraining](studies/2026-10-13-training-order-paths/paper.md) | The main internal-path difference was −0.00274 percentage points. All 48 final models and all recorded final patches kept the correct class. |
| 13 October 2026 | [A small, uncertain gain from training state edits on one future step](studies/2026-10-13-register-write-order/paper.md) | The added loss increased the full copy/write test score by 0.21 percentage points. The interval includes no gain, and three of four models failed the fixed task-ability rule. |

Each completed study includes reproduction instructions, full result tables, and an independent AI calculation review. Studies 1–3, 9, and 10 include their trained model weights. Studies 4–8 provide a fixed public model revision and file checks. Study 4 records a failed planned numerical check and the added check that explained it. Study 8 retains a failed added probability check and an exact replay of the declared float32 calculation. These claims apply to the tested small models. None has recorded human peer review.

## Use and rights

Papers cite external sources. External papers and datasets retain their own terms. No repository-wide licence has been selected. Public access alone does not grant a general reuse licence.
