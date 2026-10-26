# Review of the preparation stop

Date: 2026-10-26. This is a separate AI review, not human peer review.

**Decision: the saved resource stop follows the fixed plan. Do not start the saved-model experiment or repeat preparation to obtain a passing estimate.** The roadmap states: “A stopped resource check pauses all new research.” Publish the preparation record and stop report, and keep new research paused.

This review read the saved preparation status, branch record, manifest, controller, bound method, and roadmap. It made no model calculation, benchmark, numerical replay, or additional test. The manifest was inspected; this review did not repeat file hashing.

## Reason for the stop

Preparation ran under public plan lock `d4e22d59dbfc065443b98e8a90fcead7f9f64e22`. Its final status is `stopped`, with the reason `Conservative projection exceeds stage cap`.

The branch record reports a passed dummy-control check. This includes the transpose-direction case, exact rational effects, zero weights, a no-change edit, the sign-first rule, and the unresolved update caps. The recorded triangular radius is 0.2500000000000008, which encloses the exact target 1/4. These controls support the implementation on the tested dummy cases. They do not establish accuracy or speed on the saved scientific models.

The saved arm estimates and fixed formula agree:

| Item | Seconds |
| --- | ---: |
| One global arm | 85.942328187 |
| One candidate arm | 85.290123727 |
| `4 × [20 × (global + candidate) + 60]` | 13,938.596153 |
| Shared setup allowance | 120.000000 |
| Reserved result audit and archive checks | 120.000000 |
| Total projected stage cost | **14,178.596153** |
| Fixed scientific-stage cap | **7,200.000000** |

The total estimate is about 3.9385 hours. It exceeds the two-hour cap by 6,978.596153 seconds. The controller correctly rejected it. The 3,600-second `stage_cap_seconds` field in the preparation status names the preparation cap. It is not the scientific-stage cap used for this admission check.

This is a conservative full-cap estimate. It is not an observed runtime for the scientific panel and does not prove that the panel would take 3.9385 hours. The fixed admission rule requires the estimate to fit, so that uncertainty does not permit a run.

## Charges and scientific status

| Item | Seconds |
| --- | ---: |
| Previous cumulative preparation | 453.149483500 |
| Fixed pre-lock check charge | 120.000000000 |
| Measured preparation work | 2.772861000 |
| Closing receipt allowance | 2.000000000 |
| This preparation attempt, including closing allowance | 4.772861000 |
| New cumulative preparation | **577.922344500** |
| Cumulative scientific work, unchanged | **1,568.821297250** |

The sums agree with the saved receipt. The projected cost is not an actual compute charge. Preparation remains below its cumulative one-hour cap; the stop occurred because the proposed scientific stage failed its own two-hour admission limit.

The branch record states `dummy_only: true`. No saved-model result was produced, and the token count is zero. Stage 5 remains unfinished. It adds no completed paper. The program count remains 4 of the requested 200 completed papers, with 196 remaining. The passed dummy checks must not be presented as a completed scientific study, a speed improvement, or evidence that all scientific intervals pass.

Keep the failed preparation record, the fixed method, and the source files. Do not change the safety factor, caps, panel, repetitions, reference work, or method to turn this attempt into a pass. Any later proposal requires a separate explicit decision; this stop does not authorize a retry or work on another research branch.
