# Independent AI result review

Date: 13 October 2026. Status: review complete. No issue remains open within this review.

This is a separate AI review. It is not human peer review.

## Independent replay

The reviewer wrote [independent_audit.py](../code/independent_audit.py) without importing the study runner. The replay uses explicit GRU gate equations instead of PyTorch's `nn.GRU` call. It applies each low-rank edit through its full projection matrix. This changes both the recurrent implementation and the order of the edit matrix multiplications.

The audit passed with the original 1e-10 numerical limit. It checked all 86,016 test contracts and replayed all 6,881,280 saved logit values. The maximum replay error was 2.13e-14. All replayed class choices matched. The self-edit and complete-donor continuation controls had zero error in the independent replay.

The audit also checked:

- All 1,792 histories, their labels, their generation rules, their separation, and the fixed training schedules.
- The full contract grid and all five target states for every route.
- All per-contract scores, route summaries, donor-B groups, ability measures, and primary calculations.
- The complete set of 528 saved numerical controls and their fixed bounds.
- All 97 result file hashes, all five public-lock files, 24 model checkpoints, and 36 edit checkpoints.
- All 14,000 training records, update counts, and recorded loss combinations.

The primary mean difference and fixed seed-bootstrap interval match the saved summary. The audit took 13.86 seconds. The full receipt is [independent-audit.json](independent-audit.json).

The audit did **not** repeat training. Its checkpoint and training-record checks do not independently reproduce the optimization path. The replay verifies the saved models' reported test behavior and the calculations made from that behavior.

## Interpretation checks

The transition-minus-immediate primary differences are +0.013671875, +0.004231771, +0.006184896, and −0.015625 in the fixed seed order. Their mean is +0.002115885, or about +0.212 percentage points. The descriptive interval is approximately −1.017 to +1.131 percentage points. The preset direction rule passes because the mean is positive and three of four differences are positive. This is not a significance test or clear evidence of a reliable improvement.

Only seed 17027 passes every declared 99% ability requirement. Seeds 17011, 17021, and 17033 fail the factual first-step requirement. These models remain in the results, as required by the plan. A high edit score cannot by itself establish successful semantic-state editing for a model that fails the declared ability rule.

The same-state routes use factual targets. They test whether a change in history with the same register values preserves the required outputs. Their scores must stay separate from the scores that request a changed A value. The natural full-state route is a capability reference. It is not a low-rank intervention success.

The reviewer read the full paper and README, checked the report script, checked all 168 report-table rows against the audited summaries, and viewed the figure. The paper retains the small mean gain, uncertainty, failed ability rules, first-step errors, and extra computation cost. The reported immediate-output decrease and copy-then-write increase match the saved outputs. The figure shows all four seeds and states that its left axis starts at 86%.

The README gives a repeat-audit command with a fresh output path. The paper states that the audit did not repeat training. The interpretation stays within the scope of the fixed test. No claim or numerical issue remains open.
