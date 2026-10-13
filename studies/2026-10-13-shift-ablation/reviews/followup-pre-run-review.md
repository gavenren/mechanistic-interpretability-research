# Independent AI review of the post-failure addendum

Status: ready for a new public lock. Reviewed 13 October 2026. No follow-up scientific model was trained or evaluated during this review. This is an AI review, not a human review.

The reviewed addendum is `followup-plan.md`, SHA-256 `520ff8607095dc3eb09432fce6d351611a418ef4dca00d06ebc0996a818c1981`. The reviewed runner is `code/run_float64.py`, SHA-256 `9fcfccc20ae75a6dd00c12bf303137f64ab7ba27c3e38aaf9080501653a2cb71`. The matching dummy receipt and logs are in `reviews/followup-dummy-preflight/`.

The original float32 run remains failed and incomplete. The addendum states this in its opening, method, interpretation, and output rules. It defines a separate float64 evaluation of all sixteen originally planned models. It does not raise the original tolerance or resume the failed runner.

I compared the original and new source trees without running a model. The data construction, shifts, augmentation, architecture, initialization, optimizer, schedule generator, training function, and score function have identical Python abstract syntax trees. The static receipt is `followup-static-check.json`. It records the earlier runner hash; the only later code change is an added final time check after end-of-run reuse verification. The reviewed final runner includes that check.

I ran the read-only checkpoint-reuse verifier. It made zero model calls. All five checkpoints match the original public files and exact hashes. The full original manifest matches. All 1,500 original training records are present, with 300 ordered updates for each reused model. This includes seed 2 direct, whose training completed before evaluation failed. Batch sizes, global update IDs, finite losses, and the three shared schedules match. The new runner copies these records and checkpoints unchanged. The remaining eleven models use the original float32 training procedure, then save float32 checkpoints before conversion for evaluation.

The dtype boundary is correct. Training inputs and model creation stay in float32. Evaluation converts saved weights, the fixed filter, and test inputs to float64. A new model is created for each architecture and seed, so conversion of one model does not alter the next model's training dtype. Float64 is a separate numerical computation, not a claim about exact float32 behavior.

The new numerical limits are fixed before the follow-up: 1e-10 for float64 logit and feature symmetry, zero for no-op and local edit checks, and 1e-12 for probability normalization and the triangle bound. All original images, channels, shifts, paired seeds, metrics, and subgroup rules remain. The paper must keep the post-failure label even if this follow-up passes.

I checked the new counts: 3,300 new updates and 420,530 training examples; 20,640 evaluation calls and 2,472,672 evaluation examples; 23,940 total full calls and 2,893,202 forwarded examples; 45,632 controls; sixteen final model summaries. Reused training is recorded separately. The source-check, hard 1,200-second limit, expected-count checks, partial-output records, and failure manifest are present.

All 1,197 final dummy controls passed. The dummy run uses float32 training, float64 evaluation, real batch sizes, both architectures, and all fixed control types. Its two-fold time estimate is 406.5 seconds, below the 1,200-second limit. This remains an estimate. No scientific tolerance was changed during this review.

No blocker remains. The addendum and runner must be fixed in a new public commit before the follow-up runs. After completion, independent verification will check complete saved arrays, all score arithmetic, checkpoint provenance, training records, controls, and an independent intervention replay. That replay will not repeat training.
