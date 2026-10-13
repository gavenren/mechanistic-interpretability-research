# Review of the ordered research plan

**Date:** 13 October 2026  
**Status:** Complete. No unresolved issue within this review scope.  
**Reviewer:** A separate AI agent in the same project. This is not external review or human peer review.

## Scope

I read the full [research plan](../RESEARCH_PLAN.md) and [compute budget](../RESEARCH_COMPUTE_BUDGET.csv). I checked the order of work, dependencies, success rules, development/test separation, model counts, stop rules, and compute assumptions. I compared the plan with the existing review and extracted timing evidence from saved study records.

This was a planning review. I did not train or call models, run an experiment, repeat numerical results, or conduct a new literature search. The proposed experiments still need fixed execution plans, exact inputs, code, and controls before any run. This review does not approve execution or establish that the proposed models will pass.

## Main checks

- The plan tests task ability before dependent mechanism claims. Failed seeds remain in the record. Failed branches stop without blocking unrelated work.
- The success rules separate intact ability, intended changes, unwanted changes, and relative improvement. The thresholds are stated as proposed rules, not accepted field standards.
- Recipe selection uses development results. Final task and intervention results cannot select recipes, models, or numerical limits. The plan discloses the previous use of the digit dataset and the graph task's training coverage.
- Transfer tests distinguish selection of a method in a new model from reuse of the same head index. Future tests separate imposed continuations from free generation.
- Numerical tests retain a precise target and require a valid derivation and checked arithmetic. An unresolved bound is not called a wrong effect. Reproduction is separate from a positive scientific result.
- The budget table sums to 8.25–32.5 elapsed CPU hours, with a 47-hour total cap if every branch runs. These are planning figures. The two language stages each include a separate 16-million-forwarded-token ceiling. The assumed processing rates are explicitly unmeasured and must be replaced by dummy timing before launch.
- The 16 GB memory and 20 GB disk figures are allowances, not measured peaks. The plan separates process hours from core-hours, and excludes coding, human review, downloads, and assistant usage from machine-time estimates.

## Measured timing check

The five timing rows in the plan match the saved records, after rounding:

| Study | Scientific run source | Separate replay source |
| --- | --- | --- |
| 10 | `studies/2026-10-13-register-write-order/results/run-metadata.json`: 32.871 seconds | `reviews/independent-audit.json` in that study: 13.857 seconds |
| 11 | `studies/2026-10-13-shift-ablation/results-float64/summary.json`: 184.844 seconds | `reviews/independent-audit.json` in that study: 123.150 seconds |
| 14 | `studies/2026-10-13-flow-unit-ranking/results/manifest.json`: 91.706 seconds | `reviews/independent-audit.json` in that study: 67.288 seconds |
| 15 | `studies/2026-10-13-equilibrium-sign-audit/results/run-metadata.json`: 19.805 seconds | `reviews/independent-result-audit.json` in that study: 9.033 seconds |
| 7 | `studies/2026-10-13-position-access/results/run-metadata.json`: 181.201 seconds | `reviews/independent-audit.json` in that study: 90.579 seconds |

These are recorded local times, not new benchmarks. The plan correctly states that the image-shift follow-up reused five trained checkpoints. The runs include work beyond training. Saved-weight replays do not repeat training. Earlier failures and development checks can add time.

## Corrections resolved

1. The 40-model count now applies only to Stage 1. The plan also counts eight selector models, at most four numerical-development models, and sixteen repeat runs. Stage 6 now fixes exactly two fresh seeds per panel. The maximum is 68 task/development model training runs; edit fitting and replay add work within the stated budgets.
2. Register task ability now requires 99% accuracy in every operation-order/horizon group.
3. If neither generated-points recipe meets the development region-error limit, that branch stops. Both failed recipes remain; no third recipe is added.
4. The numerical reference must be independently validated. It has a fixed width target and update cap. Insufficient reference precision leaves a case unresolved. Non-containment alone is not proof of an invalid candidate bound. Containment remains a conservative acceptance check.

All four points are resolved. No positive result, adequate task performance, runtime guarantee, or external participation is promised.

## Reviewed versions

The plan has SHA-256 `b70b4176678e53da69699f60adbfe19357c3a09d71379dee1775bccc48a41876`.

The budget table has SHA-256 `a6606ee0e86ca0a751916f5420315e852501da6eda82f18a51c99393bc12eabc`.

Publication and all future experiment results require separate checks.
