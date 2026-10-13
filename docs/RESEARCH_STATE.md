# Research state

Updated: 13 October 2026.

## Current study

`studies/2026-10-13-inactive-variable-audit/`

State: completed experiment, paper, and independent AI calculation review. The full evidence package is included in this repository version.

Question: Does a neuron set selected only for changing an answer also preserve the answer when the proposed variable is inactive?

Design: a small trained selector network; exact enumeration of fixed-size neuron sets; separate selection and test intervention pairs; eight fixed training seeds; active and inactive groups; matched-donor checks; explicit negative controls.

The contribution is a controlled variable-localization experiment. Causal abstraction tests, objective sensitivity, and context distortion have substantial prior work. Do not claim to invent these ideas.

The plan and code were fixed in remote commit `ff9d78c15aa6e97b4819e18f4253b60fd165b50c` before training. All eight seeds reached 100% ordinary task accuracy. Both selection rules chose the same masks in all tested sizes and donor regimes. The primary difference is zero. Selected masks still had substantial intervention error. The paper preserves this negative result.

The independent audit recalculated all selected non-random masks from saved weights. All checks passed, with a maximum numerical difference of 2.78e-16. No human peer review is recorded.

Paper: [When an input should not change the answer](../studies/2026-10-13-inactive-variable-audit/paper.md).

On continuation, verify the remote paper and complete file package before treating publication as confirmed. The local `.local-state/` receipt records the verified remote commit after upload. If it is missing, verify the current GitHub tree directly; do not repeat the completed experiment without a reason.

## Future work

The next run must check usage and current repository state first. Keep a 10% reserve. The repeat schedule is every four hours, subject to available usage and the app scheduler.

Next research action: review current primary work on whether proposed variable representations remain valid after changes in representation basis or irrelevant context. Select a distinct, falsifiable question only after the literature check. Record a plan before collecting new test results. Do not claim this topic is novel in advance.

Changing only the mask size, seed count, or score in the first experiment is an extension of that study, not automatically a new paper. An explicit inactive-error constraint could be a useful extension, but it has not been tested. A later paper must have a distinct contribution. A repeat run may produce no paper if the evidence is insufficient.
