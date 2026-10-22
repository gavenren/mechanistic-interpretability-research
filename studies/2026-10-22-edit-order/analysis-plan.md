# Fixed plan: when two representation edits depend on order

Date: 22 October 2026. Status: plan before results. This is exact linear algebra on constructed two-dimensional states, with no trained network.

## Question and contribution

When do two donor-projection edits give the same state in either order? When can one joint edit meet both requests?

[Distributed Alignment Search](https://proceedings.mlr.press/v236/geiger24a.html) established interventions in distributed representation coordinates. [Heimersheim and Nanda](https://arxiv.org/abs/2404.15255v1) discuss how to interpret patching operations. This study does not claim that projection algebra or the use of distributed edits is new. Its contribution is an exact diagnostic benchmark that separates geometric overlap from donor conflict.

The earlier basis-transport study tests whether the same edit is preserved under coordinate changes. This study keeps coordinates fixed and composes two edits. It also differs from training-order and temporal-memory studies: order here means two operations on one stored vector, with no intervening model step.

## Fixed operation and domain

For a unit vector u and donor a, define `T_u^a(x)=x+u*u^T*(a-x)`. This replaces the u projection and preserves the orthogonal component. Let P and Q be the corresponding projection matrices. Compare `T_Q^b(T_P^a(x))` with `T_P^a(T_Q^b(x))`.

Fix u=(1,0). Use v=(1,0), (0,1), (3/5,4/5), (4/5,3/5), and (5/13,12/13). All are exact rational unit vectors. Each base and each of the two donors comes from the nine-point grid {-1,0,1} squared. Enumerate all 729 triples for each vector pair. Separate the 81 same-donor triples per vector pair from different-donor triples.

Record both sequential states, their difference, squared difference, and the first requested projection's residual after the second edit. The second requested projection must always be satisfied. Also solve both projection constraints jointly. If the directions are independent, solve the two-by-two linear system exactly. If they are the same, reject conflicting targets and use the closest projection when they agree. Save whether the joint constraints are consistent and the joint solution when one exists.

## Formal analysis and checks

The sequential difference is `(QP-PQ)x-QPa+PQb`. With a common donor d it becomes `(QP-PQ)(x-d)`. For fixed donors, order agreement for every base requires `QP=PQ` and `QPa=PQb`. Orthogonal directions meet both conditions for all donors. Identical directions can still conflict if their donor projections differ. These are direct algebraic consequences, not a new theorem claim.

Check the difference formula against both directly composed edits on every row. Check no-op behavior, idempotence of each single edit, and satisfaction of the last projection. Compare joint solutions after reversing the constraint order. Replay saved rows with separate scalar equations. All checks use exact rational arithmetic.

Report all vector pairs, same-donor and full-grid order changes, first-constraint failures, consistent and inconsistent joint requests, maximum squared order difference, and maximum squared joint change. Do not use random-model language, p-values, or population rates. A joint solution may be far from the original state; satisfying algebraic constraints does not establish a valid semantic edit.

## Resources and outputs

Use one local Python process, with a 300-second cap including exact checks. The three intervention studies together have a 900-second cap. No training, GPU, download, or paid API is required. The root runs scientific work in sequence. Check the private usage permit and deadline throughout. Stop and save a failure record if any exact or resource check fails. No post-result grid changes are allowed.

The public lock must contain the plan, code, source record, and shared guard before execution. Save raw CSV, summary JSON, metrics, time, plan hash, and all file hashes. Require a separate AI review. No human peer review is recorded.
