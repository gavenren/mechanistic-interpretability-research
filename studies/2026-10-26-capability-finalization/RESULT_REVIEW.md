# Result and report review

Date: 26 October 2026. Separate AI review. No human peer review.

The scientific runner finished all fixed tasks. Its saved record contains 32 jobs and 40 models. Six graph models were reused and 34 model trainings were new. Both old partial attempts remain unchanged. All saved-weight replay checks passed.

The separate checker recalculated scores from raw outputs. It checked all 40 models, recipe choices, training traces, schedules, task order, final decisions, supporting register scores, and all three result manifests. It passed. Its output is [result-audit/audit.json](analysis/result-audit/audit.json). It did not repeat training or model forward calculations. Its measured work before the closing receipt was 0.6684 seconds. A conservative 60 seconds is charged.

The reviewer checked the paper against these records. The displayed numbers and model counts agree. Final passes are graph 3/4, register 4/4, digits 0/8, and generated points 4/4. The full graph and digit panels fail. The graph explanation test remains blocked. Register endpoint success must not hide complete-checkpoint accuracy as low as 93.75% at horizon 32. Three register models have at least one such group below 95%.

The review requested two text corrections. The paper now says that no complete training run was independently repeated as a separate reproduction test. This avoids ignoring the two explicit restarts of partial trajectories. The reproduction guide now explains the environment and private state directory needed in a fresh checkout.

The archive check verified four bundles containing all 218 new raw task files. A fresh temporary extraction restored every file. A second extraction added no files. A conflicting file was rejected. The measured check time was 1.7250 seconds before its receipt; a conservative 60 seconds is charged. The [archive check](analysis/archive-check.json) records this result.

The total Experiment 1 charge is 1,091.639085751 seconds, including prior work and both post-run allowances. Preparation is charged separately. These are conservative budget charges, not claims that every allowance was used.

No scientific rerun or change to a result was needed after review. The paper states the finite-task, shared-data, prior-test-exposure, small-panel, and replay limits. The results support two task panels under the declared rules. They do not establish an internal mechanism or a general training method.

Remote publication remains a separate check. Count this study in the program only after the paper, code, direct results, archive bundles, and reviews are verified on GitHub.
