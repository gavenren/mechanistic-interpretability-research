# Two representation edits can conflict even when each is well defined

**Date:** 22 October 2026.  
**Evidence:** Exact algebra and a fixed rational grid. No neural network was trained.  
**Review:** AI-authored, with a [separate source review](ROOT_REVIEW.md). No human peer review is recorded.

## Main finding

Applying two projection edits in opposite orders often gave different states in the fixed test grid. There were two distinct causes. Nonorthogonal directions could interfere even with the same donor. Identical directions could conflict when the donors requested different values.

A joint solve met both requests whenever they were consistent. This did not make it a safe or meaningful model edit. In some cases the joint state was far from the original state. Algebraic agreement is one check; it is not a semantic explanation.

## Question and prior work

[Distributed Alignment Search](https://proceedings.mlr.press/v236/geiger24a.html) established interventions in distributed representation coordinates. [Heimersheim and Nanda](https://arxiv.org/abs/2404.15255v1) explain the need to specify and interpret patching operations carefully. This study adds a small exact diagnostic for composing two such operations. It does not claim a new projection theorem or a new learned editing method. Checked sources are listed in [sources.json](sources.json).

The repository's earlier basis-transport study asks whether a coordinate change preserves an edit. Here coordinates are fixed. The question is what happens when two edits are applied to one vector. There is no training update or model time step between them.

## Operation and formal result

Let x be a state, a a donor, and u a unit vector. The edit

`T_u^a(x) = x + u*u^T*(a-x)`

replaces the component along u and retains the perpendicular component. Write `P=u*u^T` and `Q=v*v^T`. Compare the order P then Q with Q then P, using donors a and b.

Direct expansion gives the state difference

`(QP-PQ)x - QPa + PQb`.

For the same donor d, this reduces to `(QP-PQ)(x-d)`. For fixed donors, agreement for every base x therefore requires both `QP=PQ` and `QPa=PQb`. The first condition concerns directions. The second concerns donor requests. One condition alone is insufficient.

Orthogonal directions have `PQ=QP=0`, so the orders agree for all donors. Identical directions commute, but different requested values on that direction cannot both be retained. The last edit determines that value.

## Fixed test and results

We fixed u=(1,0). The five v directions were (1,0), (0,1), (3/5,4/5), (4/5,3/5), and (5/13,12/13). Every base and donor came from the nine-point grid {-1,0,1} squared. Each direction pair had 729 base-and-donor triples, including 81 with the same donor. All calculations used exact fractions.

| Second direction | Order changes, out of 729 | Same-donor order changes, out of 81 | Inconsistent joint requests |
| --- | ---: | ---: | ---: |
| (1,0) | 486 | 0 | 486 |
| (0,1) | 0 | 0 | 0 |
| (3/5,4/5) | 702 | 72 | 0 |
| (4/5,3/5) | 702 | 72 | 0 |
| (5/13,12/13) | 702 | 72 | 0 |

For each nonorthogonal independent pair, the second edit violated the first requested projection in 648 of 729 cases. The second request itself always held. Thus checking only the last edit would miss many conflicts with the first.

We also solved both requested projections at once. Independent directions give two independent linear equations in two dimensions and therefore one solution. Identical directions give a solution only when their target projections agree. The solver returned the same result when the constraint order was reversed.

The largest squared change from the original state was `232/9` for direction (4/5,3/5). The largest value for the orthogonal pair was 8. These are maxima on the fixed grid, not general bounds. The larger change shows why a joint solution requires an additional check for acceptable state change.

## Checks and limits

The explicit composition matched the derived difference on all 3,645 rows. No-op and repeated-edit controls passed exactly. Every last-request check passed. Every consistent joint solution satisfied both projections. A separate scalar calculation replayed the first-then-second state from the saved rows. No numerical tolerance was needed for these rational calculations.

The run took 1.672 seconds with one Python thread, below its 300-second cap. The [plan](analysis-plan.md) and code were fixed in [public commit 88e8df7](https://github.com/gavenren/mechanistic-interpretability-research/tree/88e8df7689b65fbeef221423c115cd1b1c198f22) before execution. The general identity follows from algebra; the grid counts apply only to the listed states and directions.

A trained representation need not give these directions separate meanings. A state that meets two projection targets can lie outside normal model behavior. A consistent solution can also change other meaningful features. This study tests none of those semantic conditions and gives no real-model conflict rate.

A useful next test would fix two learned edits before evaluation, apply both orders, and check both intended answers and preserved behavior. It should report conflicting requests separately from failed implementations. The [README](README.md), [raw data](results/raw.csv), [summary](results/summary.json), and [hashes](results/manifest.json) support reproduction of this constructed test.
