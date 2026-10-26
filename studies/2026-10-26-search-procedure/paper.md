# What a successful search does and does not establish

**Date:** 26 October 2026. **Type:** Completed diagnostic study, Experiment 2B.

**Research and writing:** AI-assisted work for this repository. A separate AI audit is complete. No human peer review or full independent model reproduction is recorded.

## Abstract

We tested component search, score selection, and statistical selection in eight saved selector models. All 32 greedy deletion orders found the exact minimum in every model. This was a weak test: seven models allowed no deletion from the selected eight-neuron panel. All 40 score-specific size-four choices failed the final fidelity rule. In a constructed zero-mean reference, reusing selection data gave mean false-positive rates from 21.16% to 57.19% as the candidate count increased. Independent validation reduced the mean rates to about 5%, but failed the fixed evidence rule in two of 24 conditions. All numerical checks passed. The results show why search success, a large score, and adequate statistical evidence must be assessed separately.

## 1. The question

A search can return its best candidate without establishing the claim attached to it. A greedy method can succeed on an easy search table. A score can reward a change that does not preserve task behavior. A statistical test can appear positive because the same noisy data selected its candidate.

These concerns are established. [Conmy and colleagues](https://arxiv.org/abs/2304.14997v4) study automated circuit discovery. [Zhang and Nanda](https://arxiv.org/abs/2309.16042v2) examine patching methods and scores. [Heimersheim and Nanda](https://arxiv.org/abs/2404.15255v1) explain limits on interpreting patching results. This study adds a bounded comparison on this repository's trained selector models. It does not introduce a new general search method.

The [plan and code](https://github.com/gavenren/mechanistic-interpretability-research/commit/f4817fe87b9759ee9cca25162a714b302c0686bf) were public before new model outputs. [Passed preparation](https://github.com/gavenren/mechanistic-interpretability-research/commit/05d0047233ce6fb99716a94f6f32a4a90f7776dd) preceded development calibration. The [panels, score choices, reference banks, and raw-file hashes](https://github.com/gavenren/mechanistic-interpretability-research/commit/5d05c4bdefc2f67986c89cae86bf1b25cb95f923) were public before final tests.

## 2. A small conditional search problem

We reused the eight models and fixed receiver/request split from Experiment 2A. For each block, the two donors either matched the receiver's other inputs or changed all three of those inputs. Each model had 768 development pairs and 768 test pairs. The target in this study was the receiver's intact answer. The donor supplied replacement values; its requested label was not the answer target.

Development single-neuron sensitivity selected eight of each model's 12 first-layer neurons. We then evaluated all 256 retained subsets. Neurons outside the panel kept their receiver values. An empty panel therefore still contained four intact receiver neurons elsewhere in the layer. A minimum in this table is not a minimum complete network.

A subset passed only if it preserved at least 99% of intact answers and had mean probability distance at most 0.01. The distance is half the sum of absolute probability differences, on a zero-to-one scale. Exhaustive search and every greedy decision used this same rule.

## 3. Greedy success on a restrictive table

All 256 greedy runs—32 orders in each of eight models—reached an exact smallest passing subset. No greedy endpoint permitted a further single deletion.

| Models | Exact minimum retained panel size | Number of passing subsets |
| --- | ---: | ---: |
| 0–6 | 8 | 1 per model |
| 7 | 7 | 2 |

Each model had one minimum solution. In seven models, only the full eight-neuron panel passed. In the eighth, one seven-neuron subset and the full panel passed.

Thus, the fixed minimum-size criterion passed, but the table gave greedy search little opportunity to make a wrong choice. It would be incorrect to use this result as broad evidence that greedy pruning finds minimum circuits. The minimum is conditional on these panels, donors, cases, outside neurons, and fidelity limits.

## 4. Different scores choose different behavior

From the 70 subsets of size four, development data selected one candidate for each of five scores. We then evaluated those choices on final test data. Some choices were the same subset, so the 40 choices are not 40 distinct circuits.

| Development selection score | Mean final answer preservation |
| --- | ---: |
| Signed change in class-zero probability | 68.59% |
| Absolute change in class-zero probability | 71.60% |
| Required-answer margin gain | 80.58% |
| Required-answer probability gain | 83.48% |
| Answer accuracy | 83.01% |

All 40 choices failed the final fidelity rule. The rule also required probability distance at most 0.01; answer preservation alone was not enough.

The differences reflect different objectives. A large change in one class probability need not preserve the requested task answer. A large gain from the empty-panel reference need not restore the intact model. The table does not establish that one score is best for every task. Its purpose is to show the final behavior associated with each fixed choice.

## 5. Selection creates false positives when data are reused

We constructed a zero-mean reference from development effects only. Each model contributed a bank with 768 rows and 254 candidate columns. We subtracted each column's mean and then its residual mean. Exactly constant columns became exactly zero. The reference is artificial; it is not a claim that the real effects were zero.

For each model, we tested candidate counts of 8, 32, and 128. Each condition had 2,000 trials. A trial drew 64 complete rows for selection and another 64 rows independently for validation. Complete-row draws preserved the dependence between candidate effects. The selected candidate had the largest standardized selection mean.

We compared a one-sided normal test using the selection data, the same test using independent validation data, and a Bonferroni-adjusted threshold using selection data. All used the known bank variance. The normal law was an approximation to the finite bank, not an exact sampling law. Zero-variance candidates could not reject.

The table gives mean rejection rates across eight models. Every reference mean was zero by construction, so each rejection was a false positive in this simulation.

| Number of candidates | Reused data | Independent validation | Bonferroni threshold |
| --- | ---: | ---: | ---: |
| 8 | 21.16% | 5.39% | 3.78% |
| 32 | 41.48% | 5.09% | 3.78% |
| 128 | 57.19% | 5.41% | 2.91% |

Reused-data testing became much less reliable as the candidate count increased. Independent validation removed most of that increase in the mean rates. However, the primary rule required more than a mean close to 5%.

## 6. The primary statistical rule failed

The fixed acceptance rule required the upper endpoint of a two-sided 95% Wilson interval to be at most 7% in every condition. Independent validation passed 22 of 24 conditions. These two failed:

| Model | Candidates | Rejections | Observed rate | Required upper bound |
| --- | ---: | ---: | ---: | ---: |
| 3 | 128 | 130 of 2,000 | 6.50% | 7.6660% |
| 4 | 32 | 118 of 2,000 | 5.90% | 7.0197% |

Both observed rates were below 7%. The result does not establish that their true rates exceed 7%. It shows that the declared evidence requirement was not met. Requiring a bound in all 24 conditions is also stricter than requiring the pooled mean to pass. These are separate per-condition intervals, not simultaneous 95% coverage.

All 24 Bonferroni conditions met the same descriptive upper-bound check. We do not replace the failed primary method with that comparison after seeing the results. Nor does a multiple-testing threshold make an inaccurate normal approximation exact.

This failure applies to the specified normal test, empirical banks, and evidence rule. It does not show that independent validation is generally invalid. Later positive claims in this program must not use this failed rule without a new method and an evidence requirement fixed before new results.

## 7. What is new, and what remains limited

The study adds complete trained-model subset tables to the repository's earlier constructed examples. It connects development choices to final behavior and tests statistical selection with candidate dependence taken from measured development effects. It also preserves a failed primary statistical check instead of selecting a successful comparison after the test.

The source models and task states were already studied. The two donor contexts are restricted. The subset minimum is conditional. The nearly unprunable tables give limited evidence about harder searches. The simulation measures false positives under its constructed reference; it does not measure power or error rates for real model explanations. No new general principle or priority claim follows.

The next experiment tests a different question: whether a three-direction edit can change one specified variable while preserving unrelated behavior. It will use fixed practical accuracy and probability-distance criteria. Those criteria do not use the failed normal test from this study. Passing a practical criterion would still need a bounded interpretation.

## 8. Checks, costs, and evidence

The run saved 3,145,728 subset-output rows across development and test, 73,728 single-neuron development rows, and all 48,000 simulation trials. All fixed numerical checks passed. The largest recorded subset replay difference was `1.42e-14`; the largest statistical replay difference was `5.33e-15`. Both were below `1e-10`.

The separate audit checked saved tables, choices, banks, draws, decisions, and intervals. It did not repeat training or model forwards. Its first attempt stopped on an incorrect saved-file name. The failed code and receipt remain public. Correcting that lookup allowed the audit to pass without changing scientific data, methods, or tolerances. AI review is not human peer review.

The two scientific phases together are charged 22.28 seconds, including their closing allowances. Bank checks, audit checks, and archive checks add one conservative 120-second allowance. The total Experiment 2B charge is 142.28 seconds against a 7,200-second cap. No new model training or language tokens were used. The [compute record](analysis/compute-accounting.json) carries prior charges forward once.

Read the [fixed plan](analysis-plan.md), [subset method](SUBSETS_METHOD.md), [selection method](SELECTION_METHOD.md), [source notes](sources.md), [plan review](PLAN_REVIEW.md), and [result review](RESULT_REVIEW.md). The [development manifest](calibration/manifest.json) and [final manifest](results/manifest.json) fix all raw file bytes. Thirteen [result bundles](artifacts/manifest.json) preserve all 593 development and final branch files. The status files and source manifests are also published directly.

Restore the raw files from the repository root:

```sh
.venv/bin/python studies/2026-10-26-search-procedure/analysis/result_archive.py unpack
```

The archive tool checks every bundle before extraction. A fresh restore, an identical repeat restore, and rejection of a conflicting file all passed. The archive check adds no scientific model output.
