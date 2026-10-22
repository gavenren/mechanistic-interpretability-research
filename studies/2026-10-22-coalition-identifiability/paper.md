# Single-component tests leave many small mechanisms unresolved

**Date:** 22 October 2026.  
**Evidence:** Exact census of constructed Boolean functions. No neural network was trained.  
**Review:** AI-authored, with a [separate source review](ROOT_REVIEW.md). No human peer review is recorded.

## Main finding

We tested every nonconstant monotone Boolean function of four labeled inputs. There were 166 functions. For each function, we recorded all four single-component restoration tests and all four single-component removal tests.

These eight outcomes identified only 14 functions uniquely within this class. The remaining 152 shared their outcomes with at least one other function. The largest group contained 64 functions. For all 64, no single restoration turned the output on and no single removal turned it off.

This is an exact statement about a small, defined class. It is not a claim that 152 out of 166 real neural circuits are unresolved by such tests.

## Question and prior work

A component can be useful without being necessary on its own. Several components can also be required together. These are established concerns in circuit analysis. [Heimersheim and Nanda](https://arxiv.org/abs/2404.15255v1) illustrate them with AND and OR gates. [Wang and colleagues](https://arxiv.org/abs/2211.00593v1) use faithfulness, completeness, and minimality to assess a language-model circuit. The [Hydra study](https://arxiv.org/abs/2307.15771v1) reports compensating computation after model ablation.

Our contribution is a complete small-domain census of the information in eight fixed queries. It extends a few illustrative gates to every function in the declared class. It does not claim a new principle of redundancy, a new counting theorem, or evidence of adaptive repair. The [source record](sources.json) gives the checked versions.

This question differs from donor selection. Here every replacement is fixed at zero or one. The unknown is how groups of components determine the output.

## Domain and method

Each of four inputs is either zero or one. A function is monotone if turning an input from zero to one never turns the output from one to zero. We enumerated all 65,536 possible truth tables and retained those with this property, output zero at `0000`, and output one at `1111`. The input labels remain fixed; functions related by a permutation count separately.

The clean state is `1111`. A single removal sets one coordinate to zero. The corrupt state is `0000`. A single restoration sets one coordinate to one. We stored the eight answers as a signature and grouped functions with identical signatures.

We also tested every one of the 16 component sets. A sufficient set S produces output one when exactly S is active. A disabling set D produces output zero when D is removed from the clean state. A set is minimal if none of its proper subsets has the same property. Minimal does not mean that its size is the smallest among all qualifying sets.

Each monotone function can be written as an OR over its minimal sufficient sets, with an AND inside each set. This supplies a Boolean circuit representation. It does not identify a unique internal implementation.

## Results and exact relation

| Measure | Result |
| --- | ---: |
| Retained functions | 166 |
| Different eight-test signatures | 35 |
| Functions uniquely identified by the signature | 14 |
| Functions sharing a signature | 152 |
| Largest signature group | 64 |
| Functions with no sufficient singleton | 113 |
| Functions with no disabling singleton | 113 |
| Functions where both singleton tests detect nothing | 64 |

The last two sets of 113 functions overlap. They must not be added. All coalition results are recorded in 2,656 rows, with each function's minimal sets saved separately.

A disabling set must intersect every minimal sufficient set. If it misses one, that sufficient set remains active and the output stays one. Conversely, if removal leaves output one, the remaining active set contains a minimal sufficient set. That set was not intersected. These two directions prove the relation for monotone functions.

This classical relation explains why sufficient and disabling coalitions need not be the same. It also gives a calculation check. The code verified the relation for all 2,656 function-and-set pairs. A second generator built functions from all antichains of nonempty input subsets. An antichain contains no pair where one set contains the other. It produced exactly the same function set as truth-table filtering. Finally, every saved truth table was reconstructed from its saved minimal sufficient sets.

## Limits and research use

All checks passed. The calculation took 0.210 seconds with one Python thread, below its 300-second cap. The [plan](analysis-plan.md) and code were fixed in [public commit 88e8df7](https://github.com/gavenren/mechanistic-interpretability-research/tree/88e8df7689b65fbeef221423c115cd1b1c198f22) before execution. The finite census has no sampling uncertainty. Its chosen domain still limits its meaning.

Real circuits need not be Boolean or monotone. Their activations may be dependent, their replacement values may be invalid, and downstream computation may adapt after removal. A truth table also does not reveal a unique network topology. None of these features is measured here.

The result supports a practical test requirement: record what singleton evidence leaves unresolved before naming a complete mechanism. A next study could choose group interventions that distinguish the remaining candidate functions. Its query rule and candidate class should be fixed before testing. This paper does not propose or evaluate such a search algorithm.

The [README](README.md) gives the reproduction command. The [function records](results/functions.json), [signature groups](results/signature_groups.json), [metrics](results/metrics.json), and [hashes](results/manifest.json) preserve the complete evidence.
