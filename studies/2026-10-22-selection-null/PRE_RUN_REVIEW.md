# Separate AI source review

Date: 22 October 2026. The reviewer read the plan, runner, support code, and audit before execution. No simulation was run for this review. The reviewer did not write the study code. No human peer review is recorded.

The simulated sample means have the declared variance and within-split correlation. Selection uses calibration values only. Validation draws are separate. Thus the selected validation value retains the stated marginal Gaussian law under this construction. The independent-candidate formula `1-.95^K` is correct when rho is zero. The Bonferroni bound follows from a union bound and does not require independent candidate errors.

The grid has 36 cells and 5,000 trials per cell. Saved selected values and exact seed regeneration permit a check of the chosen maximum. The scalar audit recomputes selection, rejection counts, Wilson intervals, and the analytic reference. The simulation tolerance is a finite-run check, not a test about real circuits.

One correction was requested: add an explicit finite-value check for the saved calibration and selected validation arrays. The plan promises this check. The root must also verify the public lock and pass the same absolute `MI_STAGE_DEADLINE` to the runner and audit, because each support-module import otherwise starts a new 120-second allowance.

There are no true circuit effects or trained models in this study. A high selected calibration score is selection noise by construction. The report must not turn this constructed error rate into an estimate for mechanistic interpretability research.

## Correction check before lock

The reviewer checked the revised files. Explicit finite-value checks are present. The revised storage keeps selected calibration values, selected indices, and selected validation values. The audit regenerates the same fixed arrays and checks each saved selection. This is seed regeneration, not a new set of trials. The support module now verifies the public lock and takes an exclusive compute lease. The root will supply a common deadline. The reported source issues are resolved.
