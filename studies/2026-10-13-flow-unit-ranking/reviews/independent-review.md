# Independent result review

13 October 2026. This is a separate AI review. It is not human peer review.

**Status: numerical replay and final report review passed.**

## Replay scope

The [audit code](../code/independent_audit.py) does not import the scientific runner. It rebuilds the four coupling steps from saved weights. It uses an explicit analytic Jacobian recurrence rather than the runner's autograd control. It retains every model, removal condition, calibration point, test point, and latent input.

The [receipt](independent-audit.json) records a pass. It checks the public lock, source and result hashes, fixed input arrays, initial parameters, all checkpoint shapes and finite values, 16,000 training records, and the saved control limits. It does not repeat training.

All 12,681,216 saved density terms match exactly. The audit regenerates all 50,724,864 output coordinate values and uses them to recalculate every quadrant count, transition count, squared displacement, and crossing count. Only selected test coordinates were archived as raw points; all of those also match exactly. This does not imply that every generated coordinate was in the original archive.

All 1,032 score rows match. The calibration-only selectors, tie rules, full rankings, Kendall correlations, fit gates, and primary bootstrap values match. The independent analytic Jacobian error is at most 3.11e-15 on 33,024 points. The largest round-trip coordinate error is 1.55e-14. Both are below the fixed 1e-9 limit. Every edited scalar and untouched hidden value passes the exact local check.

The audit makes 328,638 comparisons. It uses 62,436 mappings and 63,439,104 input examples. It takes 67.29 seconds on this host. These counts exclude training because the audit does not repeat training.

## Interpretation checks

All four models fail the fixed excess-NLL fit gate. Their values are 0.22566–0.23822 against the fixed limit of 0.20. All four pass the separate quadrant-frequency gate. The experiment therefore does not support a result about a panel of well-fitted densities. No seed may be removed or trained further to change this result.

The mean held-out selector difference is 0.0181961 in four-bin total variation, with a descriptive four-seed interval of [0.0041275, 0.0371246]. Three model values are positive. One is zero because the two rules select the same unit. A larger change is not an improvement in generated data quality. The reference rule also uses calibration generation measurements that the likelihood rule does not use.

The four-bin score can miss movement within a quadrant and can cancel moves in opposite directions. The full displacement and transition tables are therefore part of the result. The effect is limited to this finite model, task, edit, and selector comparison. It is not evidence that likelihood has no information about generation, that a selected unit has a semantic meaning, or that the general metric distinction is new.

## Final paper and report checks

I read the final paper, README, and report code. I viewed the four-panel figure. I separately checked all 1,077 rows in the five report tables against the saved results and recalculated the table summaries. All 24,958 checked cells match within 1e-12. This report check made no model calls.

The abstract, main fit table, figure, and conclusion state that all four density-fit gates failed. The paper limits the positive selector contrast to these imperfectly fitted models. It does not claim better samples, a new flow method, a complete semantic interpretation, or a general failure of likelihood rankings. It reports the moderate positive ranking correlations and the likelihood selector's larger likelihood damage.

The paper keeps the calibration and test split, extra selector cost, four-model replication unit, and descriptive interval clear. It distinguishes all saved density terms and selected generated coordinates from other coordinates that can be replayed. The audit scope and the fact that training was not repeated are stated. The figure has readable labels and shows the failed fit rule with the selector results.

No computation or result-claim issue remains within this review scope. The review is complete. This remains an AI-authored and AI-reviewed finite experiment, with no human peer review.
