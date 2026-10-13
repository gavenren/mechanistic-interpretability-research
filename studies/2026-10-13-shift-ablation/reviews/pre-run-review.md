# Independent AI review before the run

Status: ready for a public lock. Reviewed 13 October 2026. The reviewer did not train a study model or inspect scientific test outputs. This is an AI review, not human review.

## Reviewed files

- `analysis-plan.md`, SHA-256 `15cd02bd82c815bd63aff7a5a19d9c411bf6e6d45627e56b98452018c727fdae`.
- `code/run.py`, SHA-256 `24932faca1ef30914b101cc2fd58f944ee45e86097da61ebde98a9c2a0c1864d`.
- Both original UCI data files, the data record, sources, requirements, and fetch code.
- `reviews/dummy-preflight/dummy-preflight.json` (v5), which matches the exact plan and runner above. Earlier dummy receipts remain. The final change labels aggregate probability checks with their whole-model scope.

## Checks

The original data split is fixed. I independently read the two numeric files and verified their stated hashes, shapes, and label counts. There are 3,823 training rows and 1,797 test rows. Every class is present. No exact 64-pixel image is in both files. This does not establish independence between all images from the same writer.

The two architectures have the same parameter creation order. Each pair starts from one seed, has the same initial parameter bytes, and uses the same saved row permutations and image shifts. Shifts are assigned to the ordered training positions as the plan states. Both models get ten full epochs. There is no checkpoint selection or test-based tuning.

The intervention removes one complete first-ReLU channel before downsampling. It checks the removed channel and all untouched channels at that site. All eight channels, five main shifts, and all test images enter the saved logit arrays. Channel identity is compared within each model, not assumed to have equal meaning across architectures.

The primary arithmetic matches the plan. It is a half-L1 change between signed effect vectors, with range 0 to 2. It is not TV between two probability distributions. I independently compared a scalar implementation with the runner on fixed random dummy logits. The absolute difference was `9.99e-16`; the receipt is `metric-preflight.json`. That receipt records an earlier runner hash. I checked that the score formula is unchanged in the final runner. The final dummy run also exercises the complete score summary, probability normalization, and triangle bound.

The seed pairing and whole-seed bootstrap are correct. The correct-under-all-shifts subgroup is secondary and can differ across models. All images remain in the primary estimate. The plan states that lower D can arise from weaker effects, lower confidence, or output stability. It does not claim that lower D alone proves better mechanistic understanding.

I independently checked the expected totals: 4,800 updates, 25,440 full model calls, 3,084,352 forwarded examples, and 45,632 scientific control rows. The runner checks these totals, the final time limit, and the locked file bytes. It retains partial logits, completed model files, current model weights, a failure record, and a manifest if a run fails. A successful run also has a manifest.

## Issues and fixes

1. The first time estimate scaled by call count. Dummy evaluation batches have fewer images on average than scientific batches. The author changed the estimate to exact forwarded-example ratios before the two-fold safety factor. The final estimate is 565.0 seconds. It is an estimate, not a promise; the hard 1,200-second cap remains.
2. The plan first described separate parameter hashes, while the code used one aggregate hash. The plan now states the exact aggregate method.
3. The author added final count checks, a final time check, failure manifests, and partial-result storage. These are present in the reviewed runner.
4. The final dummy run tests real batch sizes 128 and 111 for training, and 128 and 5 for evaluation. Its local no-op, zero-channel, and untouched-channel errors are zero. Its largest first-feature shift error is `8.35e-7`; largest four-pixel logit error is `3.36e-7`. These are below the fixed limits. No tolerance was increased.

No pre-run blocker remains. Do not change the plan or scientific runner after the public lock. The results still need a separate saved-logit and model replay audit. This review does not validate results that do not yet exist.
