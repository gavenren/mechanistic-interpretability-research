# Review before execution

Date: 26 October 2026. Review type: separate AI code and method check. No human peer review occurred.

The reviewer read the new plan, source record, instructions, runner, and tests. It compared them with the fixed task methods and prior results. It did not run a model experiment.

## Findings and corrections

The restart rule, reuse of six complete graph models, fixed Recipe A, task order, seeds, success rules, and maximum count match the published roadmap. At most 34 new model trainings remain, for 40 complete distinct models. The two earlier partial attempts remain separate.

The review found missing output-shape and precision checks in graph preparation. These checks were restored. It also found that the old permit reader accepted some invalid values. The new wrapper requires a true boolean, a finite numeric expiry, and no more than 120 seconds of validity. Private permit errors use fixed text. Deadline errors keep their correct cause.

The final time record now includes the main result writes and full hash scan. Each stage charges one second for its closing status and manifest writes. A measured closing-write overrun records a stop. The measured time and allowance are separate fields. Preparation also charges a conservative 60 seconds for pre-lock checks, including repeated small tests. This is not a measured test duration.

## Checks completed

The implementer checked the fixed source bytes and both old result manifests. They matched. The main agent ran all 11 focused tests with the repository Python environment. All passed; the test runner reported 0.153 seconds. Syntax and change-format checks passed.

The tests use mock jobs. They cover selection, counts, task order, no repeated completed jobs, no replacement of old partial files, denied and invalid permits, prior time charges, saved records, and a forced stop without retry. They do not train models or establish scientific results.

The separate reviewer read the final corrections and found no remaining blocking issue. Publish these files before preparation. Publish a passed preparation record before scientific execution. A later result review must check the actual outputs and report.
