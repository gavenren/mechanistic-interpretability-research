# Donor choice can change the meaning of a hidden-state test

**Date:** 22 October 2026.  
**Evidence:** Exact constructed-model study. No neural network was trained.  
**Review:** AI-authored, with a [separate source review](ROOT_REVIEW.md). No human peer review is recorded.

## Main finding

Six constructed models gave the same correct answer on every ordinary input. Their answers after a hidden-state edit were different. The difference came from states that never occur in ordinary operation.

Donor choice also changed the question. A donor selected to have the same label preserved every answer. That result did not show that the edited coordinate was unnecessary: the selection rule retained the signal being tested. An edit that changed all parts of the signal gave the intended opposite answer in every model. Replacing only one part did not.

The study adds an exact, fully recorded test case for these distinctions. It does not establish how often they occur in trained networks.

## Question and prior work

Activation patching replaces part of a model's internal state and measures the output. The replacement is part of the experiment. It can preserve information, remove information, or create a state outside the ordinary input distribution.

Sensitivity to patching choices is established prior work. [Zhang and Nanda](https://arxiv.org/abs/2309.16042v2) studied differences caused by corruption methods and scores. [Heimersheim and Nanda](https://arxiv.org/abs/2404.15255v1) explain why donor selection determines which behavior a patch tests. This paper gives a small exact benchmark; it does not claim to discover those broad concerns. The [source record](sources.json) gives the checked versions.

The earlier repository study on basis transport changed coordinates while preserving an intended edit. Here the coordinates stay fixed. The model's ordinary behavior also stays fixed. Its behavior outside the ordinary state set can vary.

## Construction and method

Let `s` be a signal and `n` a nuisance value. Each is independently -1 or 1, with equal probability. The label is `s`. The hidden state is

`h = (2+s, n, s*n)`.

Define the output score as

`F_beta(h) = (h1-2) + beta * (h3-(h1-2)*h2)`.

A score of zero or more predicts +1. Use the six fixed beta values 0, 1/2, 1, 2, 4, and 8. On every ordinary state, the bracketed term is zero. Thus every model has score `s` and 100% intact accuracy. This is a direct substitution proof, not a result of training.

We compared seven edits. Zero and mean replacement set `h1` to 0 or 2. Marginal donors come from all four inputs. Nuisance-matched donors have the same `n`. Label-matched donors have the same `s`. Opposite-label donors have signal `-s`. These donor methods copy only `h1`. The coherent opposite-label edit also changes `h3` to `-s*n`.

The last edit produces the complete state for input `(-s,n)`. Substitution then gives score `-s` for every beta. It is a known correct reference with access to the construction. It is not a learned intervention method.

## Results

The calculation saved 312 exact input-and-replacement rows and 42 summary rows. All six models had 100% intact accuracy.

| Beta | Original accuracy after zero replacement | Original accuracy after marginal replacement | Intended donor accuracy after one-coordinate opposite-label edit |
| --- | ---: | ---: | ---: |
| 0 | 50% | 50% | 100% |
| 1/2 | 50% | 62.5% | 75% |
| 1 | 75% | 75% | 50% |
| 2 | 50% | 75% | 50% |
| 4 | 50% | 75% | 50% |
| 8 | 50% | 75% | 50% |

Mean replacement kept 50% original accuracy in every setting. Label-matched replacement kept 100%. Nuisance-matched replacement gave the same scores as marginal replacement in this independent-sign construction. The coherent opposite-label edit gave 100% intended donor accuracy in all six settings. Its original-label accuracy was zero, as required by that changed task.

Zero and mean replacement always left the ordinary hidden-state set. Marginal and nuisance-matched donors left it in half the cases. A one-coordinate opposite-label edit always left it. Label-matched and coherent edits stayed in it.

Mean absolute score change also depended on the method. At beta 8 it was 16 after zero replacement and 8 after mean replacement, although both had 50% original accuracy. Thus the answer score and the class decision provide different information. All exact values are in [summary.csv](results/summary.csv).

## Checks, limits, and next step

An expanded polynomial reproduced every raw score. A saved-row check independently recalculated probabilities, support membership, both accuracy measures, and score changes. All checks passed. The run took 0.134 seconds with one Python thread, below its 300-second cap. The [plan](analysis-plan.md) and code were fixed in [public commit 88e8df7](https://github.com/gavenren/mechanistic-interpretability-research/tree/88e8df7689b65fbeef221423c115cd1b1c198f22) before execution.

The model is deliberately constructed. Its off-support term makes the issue visible. The four ordinary states are fully enumerated, so no sampling interval is needed. Their frequencies are not rates for neural networks. Label matching uses the answer and must not be presented as evidence that a signal is dispensable.

A useful next test would define valid donor pairs in a trained model before measuring effects. It should report both the intended changed answer and unwanted changes. Correct task performance before an edit does not determine the meaning of that edit.

The [README](README.md) gives the command and environment. The [metrics](results/metrics.json), raw rows, and [file hashes](results/manifest.json) preserve the complete calculation.
