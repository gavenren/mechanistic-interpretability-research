# Separate AI source review

Date: 22 October 2026. This review read the plan, runner, support code, and audit before execution. It did not run the experiment. The reviewer did not write this study's code. No human peer review is recorded.

The normalization decomposition is correct: substituting the changed centered vector gives the coordinate term, old-vector scale term, and shared-mean term. The Jacobian has the correct epsilon dependence through the scale. The fixed grid has 2,048 cases. Constant-shift invariance is valid for both epsilon values. PyTorch and the scalar sum provide separate calculation checks.

Two corrections were requested before the public lock. The plan should say that relative linear error divides by the norm of the output change. The current wording, 'entire output norm,' is ambiguous. The audit should also recompute the changed-coordinate fraction and the Jacobian error if these values appear in the report. It currently checks the off-coordinate energy measure.

The shared support module starts a fresh 120-second deadline in each process. The root must set one absolute `MI_STAGE_DEADLINE` across the run and audit to enforce the combined cap. The root must also check the code and plan against the public lock before launch; the support code does not itself take a lock argument.

The construction is a useful operator test. Its random vectors are not measured neural activations. The source review found no basis for a claim of a new normalization theorem or a real-model frequency.

## Correction check before lock

The reviewer checked the revised files. The plan now names the output-change norm. The audit now recomputes the changed-coordinate fraction, its aggregate count, and the Jacobian relative error. The support module now checks the public lock and takes an exclusive compute lease. The root will pass one absolute deadline across both processes. The reported source issues are resolved.
