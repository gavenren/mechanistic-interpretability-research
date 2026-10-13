# Research state

Updated: 13 October 2026.

## Completed studies

1. [When an input should not change the answer](../studies/2026-10-13-inactive-variable-audit/paper.md). Eight trained selector networks. Both selection rules chose the same masks; substantial intervention errors remained. Plan lock: `ff9d78c15aa6e97b4819e18f4253b60fd165b50c`. The complete study was verified on GitHub at `9adf1a41e0648c5125a20306ae4cfa2ea3b443b5`.
2. [Rare-event weighting does not reliably isolate one causal feature](../studies/2026-10-13-rare-feature-sae/paper.md). All 72 planned sparse autoencoders completed. In the primary condition, the mean benefit of balancing was 0.0241, but five of six paired differences were negative. The complete code carried more useful change than the selected latent. Plan lock: `75ab86efd6159a4965b8461a534a0a29592d98ac`. The full evidence package and independent AI calculation review were verified on GitHub at `905ac21f3b6f9469851f84db580a202752c6a268`.

3. [The same subspace can specify different interventions](../studies/2026-10-13-basis-transport/paper.md). Eight trained networks, 26 coordinate systems, and all 70 rank-four coordinate masks. Rotated coordinate masks had mean test TV error 0.5759 and 43.73% class accuracy. Exact operator transport kept 100% class accuracy. Euclidean projection onto the same image subspace gave a different intervention after non-orthogonal coordinate changes. Plan lock: `d4ca99ae08121070facd5ff755b6ca4b164dc304`. The complete study was verified on GitHub at `1bc2620b5921277c10e7a45dab95c3c6f7ec35cb`.
4. [Testing whether patch-interaction corrections transfer to new prompts](../studies/2026-10-13-patch-calibration-transfer/paper.md). One fixed Pythia-14M checkpoint, 256 directed copy-task cases, and two numerical conditions. The primary correction increased mean absolute error from 0.030472 to 0.031135 logit units. The fixed-case resampling interval included zero. Plan lock: `4dc5db20c261f9057d08a4ad2c7cc9548dcba1ba`. The complete study was verified on GitHub at `0d8888bb432e2ca3682da7d214e263e76e06b114`.

The first three are controlled synthetic studies. The fourth tests a pretrained language model on fixed artificial copy tasks. Their broad concerns have substantial prior work. None establishes a new general principle or a failure rate across language models. No human peer review is recorded.

## Current state

All four studies have complete experiments, papers, saved results, and independent AI calculation reviews. The fourth study has a recorded methods deviation. Its planned float32 whole-block delta check failed in 1,385 of 4,608 patched cases. A later whole-block reconstruction preserved the model's original addition order and matched all saved target logits and baseline activations exactly in both numerical conditions. The first failed check, its source code, and its tolerance remain in the record. The added check supports the saved outputs; it does not convert the planned failure into a pass. Verify the current remote files before treating the latest publication as confirmed. Do not repeat completed experiments to seek better results.

A later run should select a distinct intervention question after a primary-source check. The fourth study does not support use of one fixed correction across prompt changes. It also shows why an algebraically equal audit can differ in float32. Do not divide its transfer, normalization, or precision checks into extra papers. Broader task coverage, other checkpoints, and stronger learned correction rules remain untested. A new study needs a new testable claim, not just another parameter sweep.

## Current research batch

The user requested continued research until 20% of usage remains. Two distinct questions are under design review: persistence of a one-time internal patch through cached model state, and the role of attention routing in conflicting copy tasks. No outcome data have been collected for these candidates. Each needs a separate locked plan and independent review.

## Continuation rules

Check usage and the current remote repository first. Keep a 20% reserve in every core usage window. The existing repeat schedule is every four hours, subject to available usage and the app scheduler. Do not buy credits, redeem resets, use paid model APIs, or rent compute without a new instruction.

The ignored `.local-state/` directory holds publication receipts. If a receipt is missing, check the current GitHub tree and file contents. A local file or an upload attempt does not prove publication.

Before a later study, review current primary work and choose a distinct, falsifiable question. Record a plan before collecting new test results. Changes only to seeds, mask sizes, loss weights, or donor conditions are extensions, not automatically new papers. Preserve null results. A repeat run may produce no paper if the contribution is insufficient.
