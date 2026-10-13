# Independent result review

13 October 2026. Separate AI review. No human peer review is claimed.

**Status: saved-weight numerical replay and final report review passed.**

## Independent replay

The [audit code](../code/independent_audit.py) imports none of the study's runner, model, protocol, or interval modules. It rebuilds the ordinary solver, outward interval calculations, effect calculation, stop decisions, and summary calculations. It makes exact comparisons with zero tolerance. The [result receipt](independent-result-audit.json) records a pass.

The audit checks the public lock and source hashes, complete result manifest, input hashes and fixed panel, initial parameters, final weights, training orders, and 4,800 loss records. It checks the saved finite-parameter assertions but does not recreate intermediate training weights. **Training was not repeated.**

All 53,748 saved state rows match exactly: 7,188 full-test states, 15,840 coarse panel states, and 30,720 final paired states. Both intact and edited refinement branches are replayed. The first acceptable ordinary state is checked against the fixed residual rule. The saved logits, residuals, update counts, class pairs, and class enclosures also match.

All 102,578 refinement-history rows match exactly, including every bound component and stop status. The audit verifies the priority of sign, then small magnitude, then cap. It verifies that completed pairs stay fixed and successive intervals intersect. All 15,360 effect rows match, including zero cases, fixed bins, sign reversals, and primary numerators and denominators.

All 9,975 saved controls, per-model summaries, overall summary, and work counters match. The replay uses 5,454 ordinary map calls on 706,131 rows. It checks 177 bound requests covering 109,766 request rows, corresponding to 294 validated map passes and 212,344 map rows. It also checks 64 input-affine enclosures on 7,668 rows. The duplicated intact refinement work is included. There are 87,218 extra paired updates.

The result audit records 1,450,661 comparisons and takes 9.03 seconds on this host. Its largest compared array error is zero. The separate [rational arithmetic receipt](independent-interval-checks.json) has 1,512 passed checks. This audit verifies that receipt's interval and checker source hashes. The exact rational tests are separate evidence from the saved-model replay. Neither check formally verifies the CPU or NumPy implementation.

## Interpretation checks

The equal-model primary mean is 0.18659238. It is the fraction of nonzero ordinary effects whose initial validated interval includes zero, averaged over the four models. It is not a wrong-sign rate. The primary model counts are 627/3,019, 664/3,258, 468/3,033, and 608/3,367. All zero ordinary effects remain in the complete tables and the all-effect unresolved count.

Refinement gives 5,982 positive, 5,793 negative, and 3,585 small-magnitude effects. No pair reaches the cap unresolved. Only five ordinary signs are later contradicted by an interval that excludes zero. The small-magnitude classification does not establish exact zero. A positive small interval is classified by sign first, as required.

All four full-test accuracy gates pass. The first-update readout already has high accuracy. Recurrence changes some decisions, but this does not establish useful reasoning or a semantic mechanism. The evaluated object is the unique real equilibrium of the saved binary-valued parameters. Training used a finite 32-update unrolling. The paper must keep these two objects distinct.

The finite panel, one layer width, one edit, one tolerance pair, conservative global norm, and trusted arithmetic assumptions limit the result. The study applies standard contraction bounds. It does not establish a new theorem or general rate of explanation failure.

## Final paper and report checks

I read the full paper, README, and report code. I independently checked all 37 report rows and 400 cells against saved results. All values match exactly. I also checked the five reversal records, the 902 small-magnitude cases that began with nonzero estimates, and all 7,188 established top classes. These report checks made no model calls.

I viewed the final four-panel figure. It shows the task accuracy, primary denominator, final sign or small-magnitude outcomes, and pooled extra-update counts clearly. Its small-magnitude label does not imply exact zero. Panel D includes the 10,310 cases with no extra update.

The abstract and results separate the 18.66% unresolved-bound rate from the five proved sign reversals. The paper states the conservative-bound limit, exact-real target, finite-unroll training, global contraction assumption, and trusted arithmetic assumptions. It does not assign the sign reversals solely to floating-point rounding. It reports high first-update accuracy and makes no semantic-mechanism or general population claim.

No computation or result-claim issue remains within this review scope. The review is complete. The audit code and receipt are fixed. Training was not repeated, and this is not human peer review.
