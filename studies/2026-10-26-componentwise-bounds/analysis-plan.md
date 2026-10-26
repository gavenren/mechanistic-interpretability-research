# Experiment 5: can a more detailed error bound save useful work?

Date: 26 October 2026. Initial state: a fixed design and code preparation. No saved scientific model has been evaluated for this study. No timing benchmark has run.

## Question and scope

Study 15 used a global contraction bound to check the signs of internal-removal effects. This study compares that bound with one fixed candidate that retains error information for each state coordinate. The question is whether the extra bound work gives reliable results and a useful reduction in total time.

Use the four saved Study 15 models, seeds 401, 409, 419, and 421. Use the same 120 images and all 32 removals per model: 3,840 effects per model and 15,360 effects in total. Keep the saved class pairs, input order, coarse stopping rule, intervention, and numerical target. Do not select cases from their effects. Do not train development models or scientific models in this study. Develop and check the code on dummy cases only.

This is one study. Timing repeats, model seeds, arithmetic controls, and two comparison arms are not separate papers. A valid negative result can complete the study. A stopped or incomplete run cannot be counted as a completed comparison.

## Public lock and fixed target

Publish the plan, code, source notes, input manifest, separate plan review, and resource controls before loading the saved models for new calculations. Verify all remote bytes. Then run dummy-only preparation and publish its passed record before scientific work. The input manifest must fix the saved weights, inputs, panel rows, class pairs, original method, and previous compute record. No source placeholder may remain at the public lock.

The state equation is

```text
F(z) = ReLU(W z + B x + b).
```

All stored binary64 weights and inputs are treated as exact real values. The target is the unique real equilibrium of those equations. An ordinary floating-point iterate is a candidate state, not the target itself.

Removal of coordinate j replaces its update equation with `z_j = 0` permanently. The other equations and the readout remain fixed. Solve the changed system again. Do not use a one-step deletion or change the intervention meaning.

For each image, retain Study 15's fixed original winner and runner-up. Let `c` be the exact difference of their readout weight rows. The signed effect is

```text
effect = c.T (z_original* − z_removed*).
```

A positive effect means removal reduces that fixed margin. A negative effect means removal increases it. These classes need not be the correct digit labels. The common readout biases cancel in the effect. Do not choose a new class pair after removal.

## The fixed candidate bound

Let `M = abs(W)`. Let `q` be an outward upper bound on the largest row sum of M. Require `q < 1`. Let h be a candidate state. Compute a nonnegative vector r that bounds each coordinate of `abs(F(h) − h)` from above. Let g be a nonnegative outward upper bound on `abs(c)`.

Use exactly eight series terms:

```text
v_0 = g
v_(t+1) = M.T v_t
w_8 = v_0 + v_1 + ... + v_7

B(h) = w_8.T r + sum(v_8) × max(r) / (1 − q).
```

Use the same conservative M for original and removed systems. Use each system's own map and residual vector. Do not take a minimum with the old bound, change the number of terms, use a fitted local derivative, or tune the method on the fixed panel.

For the original and removed candidate states, compute an outward interval C for `c.T (h_original − h_removed)`. The effect interval is

```text
[C_lower − B(h_original) − B(h_removed),
 C_upper + B(h_original) + B(h_removed)],
```

with outward rounding at every step.

### Why the bound is valid in real arithmetic

Let `e = abs(z* − h)`, coordinate by coordinate. ReLU is 1-Lipschitz, so

```text
e ≤ M e + r.
```

The row-sum condition gives a convergent nonnegative matrix series. Therefore

```text
e ≤ (I − M)^−1 r = sum(t≥0) M^t r.
```

The margin error is at most `g.T e`. Moving each matrix power to g gives the terms `v_t.T r`. After the first eight terms, the remaining sum is

```text
v_8.T (I − M)^−1 r
    ≤ sum(v_8) × max(r) / (1 − q).
```

Permanent removal makes one row of the map's coordinatewise Lipschitz matrix zero. That matrix is at most M, so the same argument remains valid. The two state errors add to bound the signed effect error.

This uses established contraction and matrix-series arguments. It is not a new theorem. In exact arithmetic, with common g and q, this expression is no larger than the corresponding global bound. Outward arithmetic can add different excess widths. Report measured widths; do not assume every computed candidate interval is narrower.

## Rounding requirements

Keep Study 15's numerical trust assumptions: correct IEEE-754 binary64 elementary operations, correct `nextafter`, gradual underflow, and no unaccounted fused operations. This is not a formal verification of the library or processor.

The residual must be a vector. Do not replace it with its largest entry until the tail term. Enclose the input products, state map, ReLU, removed equation, and subtraction from h. Compute c by outward subtraction of the two readout rows; taking the absolute value of an ordinary rounded difference is insufficient.

Round each nonnegative product and each addition upward when constructing v, w, dot products, and tail terms. Compute a downward lower bound for `1 − q`, check that it is positive, and round division upward. Use outward endpoints for the center and final interval. One outward adjustment after a general matrix product or sum is not a justified enclosure. Stop on nonfinite values, invalid endpoint order, an invalid denominator, or an unproved contraction.

The recurrence is `M.T v` for a column vector. For a row vector it is `v @ M`. A nonsymmetric dummy case must check this orientation.

## Original-bound comparison and refinement

The global arm uses Study 15's published bound and the same exact-real effect target. Each timed arm loads its own inputs and constructs its own bounds. It recomputes ordinary coarse states from zero using Study 15's map arithmetic. Each state stops at an ordinary infinity-norm residual at most 0.001, with a 512-update cap. Count final residual evaluations as map calls. A coarse solve that misses its cap is a numerical stop, not an omitted case.

Both arms refine their original and removed states together. One paired update advances both states. Duplicate the original state per effect, as in Study 15, and count that work. At every check, use this priority:

1. The interval strictly excludes zero: record its sign.
2. Otherwise, the interval is contained in `[-1e-7, +1e-7]`: record a small-magnitude result.
3. Otherwise, 128 extra paired updates have been used: record an unresolved result.

Freeze each effect when it reaches its first stop. Save every checked interval and its state-update count. A small-magnitude result does not prove an exact zero. A sign has priority even when its magnitude is small. Successive valid intervals must intersect; disjoint intervals stop the attempt.

## Independent reference and reliability rule

Use a separate implementation of the global bound for the reference. It must not import the candidate arithmetic. Start it from separate copies of the same coarse states. Refine each reference until its outward interval width is at most `1e-10`, or until 1,024 extra paired updates. Keep the same exact-real target, class pair, and removal equation.

Compare every retained global and candidate history interval with the final independent reference for that effect. Do this in every timed arm, not only after a promising result.

- A sufficiently narrow reference wholly inside the tested interval passes the containment check.
- Overlap without containment fails this conservative acceptance check. It does not prove the tested interval is mathematically wrong.
- A reference still wider than `1e-10` at the cap is unresolved. It cannot pass the reliability requirement, even if it overlaps or is contained.
- Disjoint enclosures or opposite certified signs are a numerical failure. Preserve the evidence and stop. At least one claimed enclosure would then be inconsistent with the shared target.

Do not enlarge a candidate interval, alter a threshold, or silently refine the reference beyond the fixed rule after a failure. A finite containment test is not a proof of universal soundness. The mathematical argument and checked arithmetic are also required.

Reliability success requires every effect and retained history interval to pass, with no unresolved reference or contradicted sign. A non-containment gate failure can remain a completed negative result. Preserve its counts separately from disjoint intervals and reference-cap cases.

## Timing and practical success

Run five paired repetitions for each model. Each pair includes one global arm and one candidate arm. Alternate which arm runs first. The first pair starts with the global arm for even model index and the candidate arm for odd model index, in the fixed model order above.

Use identical hardware, thread counts, input files, and stop rules. Do not run another numerical job in parallel. Do not carry computed bounds, states, or reference results across arms or repetitions. Any reuse within one arm must be fixed in the code before the lock and constructed inside its timer.

Start each arm's timer before input loading. End it only after bound construction, coarse solves, refinement, independent reference calculation, saving, reopening, validation, content hashing, and any duplicate-file removal. Include these costs even if they exceed the solver work. Report source checks and shared orchestration separately and include them in the stage charge.

Retain the first complete raw artifact for each arm and model. A later repeat may remove redundant numeric files only after exact array comparisons with that retained artifact. Keep its hashes, timing, work counts, and comparison receipt. A changed result must retain its full artifact and fail the repeat check. Preserve all failed and partial evidence.

For each model, compare the median of its five complete candidate-arm times with the median of its five complete global-arm times. Practical success requires the candidate median to be at most 75% of the global median in every model. It also requires reliable bounds and resolution of every effect to a strict sign or the same small-magnitude criterion. An unresolved effect or incomplete repetition prevents success. Report paired times, both medians, time ratios, state updates, map calls and rows, bound calls, reference work, memory, and saved-file costs. Do not substitute state-update savings for measured end-to-end savings.

## Dummy checks, limits, and reporting

Before saved-model calculations, check exact rational positive, negative, zero, very small, and unresolved examples where feasible. Include a nonsymmetric triangular matrix, W equal to zero, a high-contraction case that reaches its cap, a no-change operation, permanent removal, and separate global-reference checks. Verify each claimed enclosure against the exact target. Exercise all stop outcomes and the conservative containment rules. Preparation must time the real branch shapes sufficiently to project all 40 timed arms and their independent references with a fixed safety factor of four.

Use local CPU only, two compute threads, one inter-operation thread, at most 4 GiB peak memory in preparation, at most 16 GiB in science, and at least 20 GiB free disk. No training, model download, paid API, rented hardware, or language-model forward is allowed.

The stage cap is 7,200 seconds, including failures, every repetition, all reference work, result audit, and archive checks. Retain a single 120-second post-run reserve. The program preparation cap remains 3,600 seconds and the total roadmap cap remains 183,600 seconds. The root controller must carry the latest verified prior charges forward once. A 120-second pre-lock check charge and measured dummy preparation use the preparation budget. Reject a projection that does not fit the remaining stage and program limits. A proposed 200-paper target creates no new compute allowance.

The supervisor must check live usage at least once per minute. A private atomic permit lasts at most 120 seconds. Stop at 20% remaining in any reported core window, an unavailable meter, an invalid permit, or any resource limit. Keep partial records. Do not resume automatically or retry for a preferred result.

Report this as a fixed efficiency test on an already known panel. Study 15's five sign reversals are prior results, not new discoveries here. Distinguish a mathematical enclosure, passed numerical checks, conservative reference acceptance, and useful timing savings. Publish the plan, exact code and inputs, raw evidence, all failures, a separate AI review, a complete compute record, and a plain-English paper. Verify remote bytes before recording completion. AI review is not human peer review.
