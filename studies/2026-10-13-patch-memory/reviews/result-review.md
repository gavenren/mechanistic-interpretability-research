# Independent result review

Date: 13 October 2026. Reviewer: a separate AI agent. No human peer review has occurred.

## Status and scope

The numerical audit passed. The [audit program](../code/independent_audit.py) does not import the experiment runner. The [receipt](independent-audit.json) records its file hash, the locked experiment commit, the checked rows, and the maximum errors.

The audit checked:

- All result-file hashes and the plan and runner bytes at lock commit `512fdba309df32aa3a98db59c65ec5d5d383635f`.
- The fixed text, token IDs, word pairs, suffix, split, and complete case grid.
- All 3,456 score rows, 576 baseline rows, complete group summaries, calibration choices, test differences, and negligible-value labels.
- All recorded logit, hook, and exact cache checks.
- Every saved post-block vector through a separate model replay.
- Every baseline and patched full prefix, with no key-value cache, at all nine horizons.

The separate replay used 4,096 forwards. Its maximum TV difference from the saved cached run was `5.612142695010647e-13`. The maximum target-logit difference was `2.1104451519704526e-11`. Both are below the fixed `1e-8` bound. Saved post-block vectors, summary arithmetic, and selector arithmetic matched exactly. Full-prefix future logits after a final-block edit matched the unmodified logits exactly.

The runner's full-vocabulary cache-versus-full controls cover every case and patch. The separate audit recalculates every full-vocabulary TV score and checks saved target logits. It does not store a second full-vocabulary tensor archive.

## Interpretation

The two fixed selectors choose different blocks on calibration data. On the held-out cases, the delayed selector gives a larger mean output change after eight forced tokens. This supports the stated comparison on this fixed set.

The result does not show better answers, useful steering, semantic memory, or slower memory decay. A larger edit can produce a larger later effect. Suffix content also changes with the horizon. The final-block zero-future result is a known architecture control. The four held-out word pairs do not support a broad population claim.

## Limits of this review

I read the paper and README and inspected the figure. The capability counts and edit norms agree with the saved rows. The figure clearly states its separate vertical scales. The paper reports all selected group effects and the non-monotone horizon example. It distinguishes later output change from correct copying. No scientific wording blocker was found.

This review covers the fixed checkpoint, common suffix, and stated arithmetic condition. Double arithmetic still uses shared float32 rotary tables. A separate AI implementation is useful evidence, but it is not human replication or peer review.
