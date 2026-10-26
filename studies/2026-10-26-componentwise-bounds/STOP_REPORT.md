# Experiment 5 stopped at its compute check

**Date:** 26 October 2026. **Status:** Preparation finished; scientific comparison did not start. This is an unfinished study, not a completed research paper.

The dummy numerical checks passed. The fixed conservative cost estimate did not pass. It projected **14,178.60 seconds, or 3.94 hours**, against the unchanged **7,200-second, two-hour stage cap**. The runner stopped before using any saved scientific model.

## What was checked

The [plan and code](https://github.com/gavenren/mechanistic-interpretability-research/commit/d4e22d59dbfc065443b98e8a90fcead7f9f64e22) were public and their bytes were verified before preparation. The [separate plan review](PLAN_REVIEW.md) found no open blocker for preparation. Nineteen controller tests and one added interruption test passed before the lock. They used test stubs, not scientific models.

Preparation checked exact positive, negative, zero, small, and unchanged effects. It checked a nonsymmetric matrix, the sign-first rule, a zero-weight case, and a slow case that remained unresolved at the fixed update limits. These controls passed. They are implementation checks on constructed data. They are not results for the four saved study models.

The cost check used random dummy arrays with the full intended shapes. It measured coarse state updates, paired updates, both bounds, the independent reference bound, coefficient construction, guard checks, and full-history saving and replay. Each operation used the slowest of three measured passes where the method specified repeated timing.

The fixed projection allowed the full coarse, refinement, and reference update caps in all 40 timed arms, then applied a factor of four. It included 20 global-bound arms and 20 candidate-bound arms. This conservative projection was fixed before the timing results. It is not a measured runtime for the unstarted experiment.

## Why the run stopped

| Cost component | Projected seconds |
| --- | ---: |
| Full branch, including fixed loading and summary allowances and the factor of four | 13,938.60 |
| Shared setup and source-check allowance | 120.00 |
| Reserved scientific audit and archive checks | 120.00 |
| Total projected stage | 14,178.60 |
| Allowed stage cap | 7,200.00 |

The independent reference is a large part of the projection. Each arm allows up to 1,025 reference-bound evaluations and 1,024 paired reference updates. The reference-bound pass took at most 0.06583 seconds in the fixed three-pass dummy measurement. The estimate intentionally does not assume that the saved models will stop early.

Actual model behavior could require much less work. That possibility does not authorize this run under its fixed admission rule. No threshold, repetition count, reference cap, or safety factor was reduced after the estimate. No new timing attempt was used to obtain a pass.

The [roadmap](../../docs/RESEARCH_PLAN.md) says that a stopped resource check pauses all new research. The research schedule was therefore paused. The 20% account reserve remains a separate stop rule. This stop was caused by the compute projection, not by low account usage.

## Cost and preserved evidence

Preparation itself took 2.772861 measured seconds, plus its two-second closing allowance. Its charge is 4.772861 seconds. Together with the fixed 120-second pre-lock allowance, this stage adds 124.772861 seconds to preparation. Cumulative preparation is **577.922344500 seconds** against its 3,600-second cap.

Cumulative scientific charges remain **1,568.821297250 seconds**. Experiment 5 adds no scientific run charge. Its 120-second scientific postcheck reserve remains unused. About **50.4037 hours** remain under the whole 51-hour roadmap cap, but that remaining time does not enlarge this experiment's two-hour cap. The [compute record](analysis/compute-accounting.json) keeps these amounts separate. Account usage is not measured in these CPU-time charges.

The [preparation status](preparation/status.json), [dummy measurements](preparation/bounds/bounds-preparation.json), and [file manifest](preparation/manifest.json) preserve the failure. The [stop review](STOP_REVIEW.md) checks these records and the reason to stop. No trained-model effect, timing comparison, reliability result, or useful speed gain can be claimed from this preparation run.

## What remains unfinished

The four-model, 15,360-effect comparison has not run. Neither its reliability requirement nor its 25% complete-cost saving requirement has been tested. The study adds no completed paper to the 200-paper count.

Any later attempt needs a reviewed execution decision that resolves this cost failure before scientific work. It must preserve this record and all cumulative charges. A smaller or faster design would need its own explicit justification and fixed checks. An automatic retry, a hidden reduction of the test, or a larger unrecorded budget is not permitted.

This report and code were produced with AI assistance. Separate AI review is not human peer review.
