# Versioned analysis plan: the inactive-variable audit

Date: 13 October 2026. This is a versioned analysis plan, not an externally registered study. The code and this plan must be committed before model training or test evaluation. The run records that commit and verifies both files against it.

## Question and bounded claim

Can a neuron set selected to change an answer also preserve the answer when the proposed variable is inactive? At the same neuron budget, compare selection for active-variable repair with selection that also checks the inactive variable. This is a new controlled experiment, not a new discovery of patching sensitivity, causal abstraction, or objective misranking.

The primary prediction is that balanced selection reduces held-out worst-group counterfactual error. An improvement on the selection data is true by construction and is not confirmation. Publish negative findings and all model seeds. Do not change the metric after looking at held-out results.

## Task and training

Inputs are (a,b,s,n), with a,b in {0,1,2,3}, s in {0,1}, and n in {0,1,2,3,4,5}. The answer is a if s=0, otherwise b. All 192 states receive equal weight. All are used for ordinary task training. The held-out test is over intervention pairs, not over ordinary inputs.

Use a one-hot input with 16 coordinates and an MLP with widths 16–12–24–4. Both hidden layers use tanh. The output uses softmax. Initialize weights independently from a zero-mean normal distribution with standard deviation 1/sqrt(fan-in). Initialize biases to zero. Use NumPy default_rng seeds 0–7, float64, full-batch Adam, 4,000 steps, learning rate 0.02, beta1=0.9, beta2=0.999, epsilon=1e-8, and no weight decay or learning-rate change. Record losses every 100 steps and at the first and last step.

An intact accuracy of at least 99% is the primary eligibility gate. Run and retain all eight seeds regardless of this gate. Do not extend training, replace failed seeds, or search for better hyperparameters. At least six eligible seeds are required for a confirmatory summary. If fewer pass, report gate failure. A failure remains useful evidence about this protocol.

## Interventions and fixed split

Patch first-layer hidden neuron subsets and recompute the full downstream network. The candidate variable is a. For receiver x=(a,b,s,n) and donor x'=(a',b',s',n'), the exact counterfactual input is (a',b,s,n).

The primary target is the intact model probability vector on this counterfactual input. A semantic accuracy measure against the task label is secondary. This distinction avoids treating intact-model errors as patching errors. In the inactive group, the task label must stay fixed, but the intact model's probability vector can change. Therefore primary inactive error is model-patching error, not a test of exact output-distribution invariance. Also report TV between intact counterfactual and intact receiver outputs, and TV between patched and intact receiver outputs. The same-a control below requires the exact receiver distribution.

Use every ordered pair with a' != a: 27,648 pairs. Define eight strata from receiver selector, donor-selector match, and donor-b match. Within each stratum, sort the SHA-256 hashes of the text `inactive-variable-audit-v1:<receiver-id>:<donor-id>`. The first half is discovery; the second half is test. IDs follow lexicographic input enumeration. The paired count is even in every stratum. Save the full split ledger. No pair occurs in both sets. Ordinary states occur in both by design. Means use natural pair frequencies, not equal weights for the four donor strata: donor b matches with probability 1/4, donor selector matches with probability 1/2, and each receiver selector has probability 1/2. Publish the strata separately.

The matched-donor sensitivity analysis uses the subset with b'=b, s'=s, and n'=n. It keeps the primary split assignment; each subset has both active and inactive cases. Select masks again within matched discovery pairs. Its held-out scores use only matched pairs already assigned to the primary test pool; no discovery pair is reused as a held-out pair. Do not pool matched and unrestricted results. Same-a pairs (9,216, including identity pairs) form a separate negative-control set. They are never used for selection, and their exact target is the intact receiver distribution.

## Candidate family and scores

Enumerate all coordinate masks of sizes 1, 2, and 3: 12+66+220=298. The primary size is 3. Use total variation error: half the sum of absolute differences between patched and counterfactual output probabilities. It lies between 0 and 1 and has no clean–corrupt normalization denominator.

Repair-only selection minimizes discovery active-group mean error. Balanced selection minimizes the larger of discovery active-group and inactive-group mean errors. Break ties within an absolute 1e-12 tolerance by lexicographic neuron tuple. Keep both masks fixed for test evaluation.

Use 20 fixed-seed random masks per size, or all masks if fewer than 20 exist. Sample without replacement using seed 9000+k. The same index list applies to all model seeds and donor regimes. Publish all random-mask outcomes and their average; do not select the best. The empty-mask baseline is reported separately and is not a same-size representation. Complete-layer patching is a code control, not a candidate at these budgets.

## Outputs and inference

For every mask and split, record mean error, active error, inactive error, all eight donor-stratum errors, semantic accuracy, and donor-label copying rate. The latter is descriptive; matching a donor answer is not evidence of a valid intervention on a.

The primary outcome is the seed-level difference in held-out worst-group error at k=3: repair-only minus balanced. Positive values favour balanced selection. Report all seed values, eligible mean, median, range, and a 95% percentile paired bootstrap interval over eligible model seeds. Use 10,000 bootstrap draws and RNG seed 20261013. This interval describes variation over the chosen random initializations; it does not cover other tasks or architectures. Do not treat individual pairs as independent model replicates.

Secondary results include sizes 1 and 2, donor strata, matched-donor selection, same-a control errors for selected masks, selected-mask overlap, and the active/inactive tradeoff. Give all candidate scores, so a reader can inspect the frontier. Any best-test-mask statistic is an explicitly labelled oracle and must not be used as a selected method. This runner does not compute that oracle.

## Correctness checks

Before training, run finite-difference checks on 24 randomly selected parameter coordinates from a small fixed batch. Require relative/absolute scaled error below 1e-6. For an untrained network, verify empty, identity, and full first-layer patches. Full patching must equal the intact donor; identity and empty patches must equal the receiver. Require maximum probability error below 1e-12. Verify the input-level counterfactual cache against direct execution. Verify a factorized reference selector for every primary pair, including the no-change inactive condition. Check all pair counts, identities, split disjointness, and stratum balance.

Repeat patch-execution controls for every trained model. Record all control outcomes. A failed control stops the run. This protects result validity; it does not license changes to the scientific protocol without a new version.

## Resource and reproducibility rules

Use one BLAS thread and one process. Evaluate masks sequentially, using vectorized pair batches. Save weights, all candidate scores, selected-mask scores, same-a controls, intact predictions, losses, environment versions, runtime, file hashes, and this protocol's lock commit. The reproduction command is `python code/run.py --run --lock-commit <commit>`. `python code/run.py --check` runs correctness checks without training.

No data downloads, paid APIs, external model checkpoints, or GPUs are required. Do not split masks, seeds, or donor variants into separate papers.

## Limits and closest prior work

The task semantics are known. The learned internal circuit is not known merely because the task is synthetic. Coordinate masks may fail to represent a distributed variable. The balanced rule has access to a specified high-level hypothesis; it is not an unsupervised general-purpose circuit discovery algorithm. Pair holding does not establish generalization to new task inputs. A narrow donor pool may yield different conclusions. Causal variables need not occupy single neurons.

This study extends established interchange-intervention evaluation. Cite [Geiger et al. (2022)](https://proceedings.mlr.press/v162/geiger22a.html), [Zhang and Nanda (2023/2024)](https://arxiv.org/abs/2309.16042), [Heimersheim and Nanda (2024)](https://arxiv.org/abs/2404.15255), [InterpBench](https://arxiv.org/abs/2407.14494), and [MIB](https://arxiv.org/abs/2504.13151). Current close work includes [Bucketing the Good Apples](https://arxiv.org/abs/2605.02234), [The Curse of Multiple Mediators](https://arxiv.org/abs/2606.27510), and [Objective-Level Recovery Gaps](https://arxiv.org/abs/2610.02098). Do not claim the first negative-control, stratified, or misranking study. A targeted search cannot prove novelty.
