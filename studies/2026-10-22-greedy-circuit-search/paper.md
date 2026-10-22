# A circuit can resist further pruning and still be larger than necessary

**22 October 2026 · Formal and computational research note · AI-authored**

## Main finding

Greedy deletion can return a faithful circuit from which no single component can be removed, although a smaller faithful circuit exists. In one fixed family, the smallest circuit had one component. Depending on deletion order, the algorithm retained up to six components. Every retained circuit still gave the correct answer on the complete task.

We checked 1,016 retained subsets and 14,550 deletion orders across 17 constructed circuits. All exact checks passed. These are small specified Boolean networks. No network was trained, and the counts do not estimate failure rates in language models.

## Question and prior work

A circuit is *faithful* here if it preserves every answer in the specified task. It is *inclusion-minimal* if removing any one remaining component breaks that requirement. It has *minimum size* if no faithful circuit contains fewer components. These are different properties.

The general problem is established. Conmy and colleagues show that ACDC can retain one input to an OR gate while removing another. Their Appendix M explains why the deletion order matters. Our study does not introduce that observation or reproduce the full ACDC algorithm. It adds an exact size comparison, a simple formula, and a complete order audit for a fixed small panel. [Conmy et al., version 4, Appendix M](https://arxiv.org/html/2304.14997v4).

This question also differs from asking whether a component is necessary or sufficient. We hold the faithfulness requirement fixed and measure whether one search procedure finds its smallest solution.

## Model and method

The task input is one bit, x. Each hidden component copies x. A route computes the AND of its components. The output computes the OR of all routes. Removing a component sets its value to zero. With every component present, the answer is x.

A retained subset is faithful when it answers correctly for both possible inputs. Thus the task is tested in full. The construction also has a threshold-network implementation: an AND is `ReLU(sum(inputs) - route_size + 1)`, and an OR is `1 - ReLU(1 - sum(route_outputs))` on these binary inputs.

Five cases contain two routes: the singleton {0}, and the longer route {1,…,k}, for k from two to six. Twelve additional six-component cases use the fixed generator and seed in the [plan](plan.md). Superset routes are removed before testing. Every generated case remains in the record.

For each circuit, exhaustive search checks all retained subsets and finds the smallest faithful size. Greedy search starts with every component. It visits each component once and removes it only when the current circuit stays faithful. We run every possible visit order. There is no choice based on a final result.

## Why the size gap occurs

In the two-route family, the singleton is a faithful circuit, so the minimum size is one.

If component 0 is visited first, the longer route is still intact. Deleting 0 preserves the answer. Every later deletion would then break the only intact route. The final size is k.

If another component is visited first, deleting it breaks the longer route but preserves the singleton. Component 0 must then remain. Every unused part of the longer route can be removed. The final size is one.

Exactly one of the k+1 components appears first in a uniformly chosen order. Therefore the fraction of orders that retain k components is `1/(k+1)`. The mean size is `1 + (k-1)/(k+1)`. This argument was fixed before calculation.

| Longer route size k | Orders checked | Minimum size | Largest greedy size | Fraction above minimum |
| --- | ---: | ---: | ---: | ---: |
| 2 | 6 | 1 | 2 | 1/3 |
| 3 | 24 | 1 | 3 | 1/4 |
| 4 | 120 | 1 | 4 | 1/5 |
| 5 | 720 | 1 | 5 | 1/6 |
| 6 | 5,040 | 1 | 6 | 1/7 |

The exact calculation matched every formula. Eleven of the twelve additional fixed cases also had at least one order above the minimum. Their largest excess was two components. The remaining case reached the minimum in every order. These case counts describe this generator only.

## Checks, cost, and limits

A separate calculation used integer masks and threshold-gate equations. It reproduced every subset output, deletion endpoint, minimum size, endpoint count, and formula. Every greedy endpoint was faithful and inclusion-minimal. The full circuit passed, and the empty circuit failed the positive input.

The run, including its audit, took 0.92 seconds with Python 3.12.14. Peak process memory was 31,031,296 bytes. The plan and code were public at commit `88e8df7689b65fbeef221423c115cd1b1c198f22` before execution. [Raw results and audit](results/summary.json); [reproduction instructions](README.md).

The result does not show how often large trained models have this structure. Exhaustive search becomes expensive as the component count grows. Repeating several greedy orders can find a smaller circuit, but does not by itself certify a global minimum. Researchers should report faithfulness, inability to remove a single component, and minimum-size evidence as separate claims.

The [source review](PRE_RUN_REVIEW.md) was performed by separate AI agents. The equation audit shares the declared task assumptions. No human peer review or external replication is recorded.
