# Check of the updated research plan

**Date:** 22 October 2026.

**Status:** Planning and file checks complete. No unresolved material issue within this scope.

**Review:** Two separate AI agents checked the plan. One checked methods and coverage. The other checked the saved continuation state and resource rules. The primary agent applied the corrections and checked the document files. This is not human peer review or external replication.

## Scope and evidence

The [plan](../RESEARCH_PLAN.md) and [budget](../RESEARCH_COMPUTE_BUDGET.csv) update the roadmap for all 24 completed studies. Source state was fixed at repository commit `802bacf849f073c34c838c33876bf2f121ff06d3`. The check used the updated full review, the 22 October continuation report and saved records, the earlier fixed task methods, and the existing plan. Official Pythia-70M and SmolLM2-135M model cards were checked for identity and size on 22 October. They provide no timing or task-success guarantee for this plan.

No experiment, benchmark, training run, model download, or numerical-result replay was performed for this update. Saved code and metadata were inspected without running their scientific jobs. The [original plan review](RESEARCH_PLAN_REVIEW.md) remains the record for the earlier plan.

## Main checks

- Every completed study, 1 through 24, has a stated use or a stated deferral. New mathematical reference tests are not presented as trained-language-model results.
- Preparation and the unfinished Experiment 1 come first. IDs are preserved for the older experiments. The cheaper numerical test runs before the larger language stages.
- The graph panel remains unfinished. Recipe A and complete seeds are retained. The proposed full restart uses the same final seed 8103, original initialization, and 32,768-update schedule. It requires a new public execution record before training. Partial files contain no optimizer or random-generator state, so the plan does not call this an exact resume.
- The two partial attempts remain charged. No failed seed is replaced. A second automatic retry is not permitted. The original four-hour scientific cap and fourfold allowance remain.
- Success rules distinguish task ability, intended effects, unwanted effects, exact-search limits, calibration, useful savings, and reproduction. A negative finding can be reported without claiming a successful explanation.
- Search optimality is limited to the chosen eight components and fixed cases. Selection and validation draws come independently from a fixed centered development bank. This null test does not estimate real-circuit error rates or statistical power.
- Estimates, measured history, and hard caps are separate. No external participation, scientific success, or runtime is promised.

## Corrections applied

1. The draft register rule treated horizon 1 as if both operations were present. It now preserves the nine fixed groups: horizon 0 after the 12-operation history; copy-only and write-only at horizon 1; both orders at horizons 2, 8, and 32. The future-use proposal also uses valid one-operation groups at horizon 1.
2. The selection reference now explicitly uses development effects only. Selection and validation samples are independent, while candidate dependence is preserved within each draw.
3. The model budget is described as training runs, including repeated original seeds. It is not described as 68 independent confirmations.

## Arithmetic and file checks

The CSV has nine budget rows in run order `0, 1, 2A, 2B, 2C, 5, 3, 4, 6`.

| Quantity | Checked value |
| --- | ---: |
| Additional estimated CPU time | 9.75–35.5 hours |
| Sum of cumulative stage caps | 51 hours |
| Prior preparation and scientific time charged | 332.672555751 seconds |
| Sum of remaining stage caps | 50.907590956736 hours |
| Combined forwarded-language-token cap | 33,000,000 |

The preparation charge matches the saved revised preparation record. The scientific charge matches the saved Experiment 1 status. Each remaining cap equals its cumulative cap minus prior charged time. All proposed upper estimates fit their own remaining caps, but execution still requires fresh dummy timing and the stated safety factor.

All local Markdown links in the changed documents were checked after this record was written. The README and research state now describe the updated plan. The original study files, scientific results, and partial attempts are unchanged. GitHub commit and file checks are performed after publication; this record alone does not prove publication.

## Checked versions

Plan SHA-256: `9eb3a895a77bd5f8bc18795c13c12689955eded4ced89e2fc3c8615ab5402490`.

Budget SHA-256: `275e89bde1887363d5b0ee0484d0fd91b6b4cb1c61b874e42fae49c8919be27d`.

This check does not authorize compute beyond the stop rules or establish scientific validity, priority, human review, or external replication.
