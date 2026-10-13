# Independent result review

Status: the independent result audit and final claim review passed. Ready for publication. Reviewed on 13 October 2026. This is an AI review, not a human review.

## Audit scope and result

The independent code does not import either scientific runner. It constructs the network separately, uses explicit source indices for image shifts, and applies the channel edit in a ReLU output hook. It uses the saved float32 model weights converted to float64. It does not repeat training.

The audit replayed all 16 models, all 12,938,400 saved class-score values, and every model control. All saved values matched exactly. It replayed 20,640 full model calls and 2,472,672 examples. All 45,632 control records matched within their fixed limits; the largest difference from a recorded error was 2.7756e-17, in score arithmetic. The audit took 123.15 seconds.

The audit independently rebuilt every model summary, digit and channel result, shift result, correct-at-all-shifts subgroup, and paired seed bootstrap. All matched. It checked every result-file hash before and after replay. It also checked all eight training schedules, all 3,300 new and 1,500 reused training records, and all five reused model files. The reused files matched the bytes in the public failed-run commit. The initial parameter hashes matched the reconstruction from the fixed seeds. Training records and saved weights do not prove a second training run would produce the same weights; training was not repeated.

The result manifest is `13cb0f754494f42190ed340a558cf07a8c6b00d03ee8909e8d121363b0c73987`. The audit receipt is [independent-audit.json](independent-audit.json). The executed code SHA-256 was `d3c9c07f6d7e2a4fabdd9ea63e9e5005c5dbbe504a9d2d997465257a972eafc3`. After the audit, the receipt path field was changed to repository-relative form. The [current code](../code/independent_audit.py) makes only that output-format change; its SHA-256 is `210df82afc6ae21e7efb37d810b81f4137704e473e8b9f91781dcc7d37569ca2`. All numerical values are unchanged. The receipt retains both hashes and the path-change note. The original receipt is retained in local staging.

## Required interpretation

The original float32 run remains incomplete and failed. This audit does not replace that outcome. The float64 evaluation was fixed in a new public plan after the failure. Its original training method, seed panel, data, and score rules were retained. It is a post-failure analysis within the same paper.

The positive paired difference measures greater stability of a specified channel-removal effect under image shifts. Low classification accuracy must be central. The result does not establish a useful digit-recognition mechanism, a general improvement in causal explanation, or a result for other training designs. A larger mean channel-removal effect in the filtered condition rules out only a uniformly weaker average effect as a complete account. It does not establish equal confidence, equal competence, or matched per-example effects.

The interval describes variation across eight seeds for the fixed split and training design. It does not include uncertainty from the data choice, architecture choice, or the post-failure decision to use float64 evaluation.

## Final paper and figure review

The reviewer read the complete paper and README, checked the report code, verified all 378 descriptive table rows against the audited results, and viewed the figure. The paper and figure lead with low task accuracy. They retain the original failure and identify the float64 result as a post-failure analysis. The effect-size limit, unequal subgroup membership, seed-interval scope, training provenance, and independent audit scope are clear.

Two wording edits were made: the last line now identifies the work as AI-authored, and the record refers to the public locks without an incorrect count. The audit path-format code also accepts the separate reproduction worktree described in the README. No numerical computation or result changed. No claim blocker remains. This review, the audit code, and the public receipt are now fixed.
