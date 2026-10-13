# Independent result review

Date: 13 October 2026. Reviewer: a separate AI agent. No human peer review has occurred.

## Status and scope

The numerical audit passed. The [audit program](../code/independent_audit.py) does not import the experiment runner. The [receipt](independent-audit.json) records its file hash, the locked experiment commit, the checked rows, and the maximum errors.

The audit checked:

- All result-file hashes and the plan and runner bytes at lock commit `512fdba309df32aa3a98db59c65ec5d5d383635f`.
- The tokenizer pool, shuffled groups, exact prompt tokens, binding assignments, query pairing, and calibration/test split.
- All 768 baseline rows and 9,216 intervention rows, including signs and the five effect formulas.
- Every head summary and group mean, the calibration head choice, all 10,000 group bootstrap draws, and the fixed primary conjunction.
- The saved control bounds and unchanged-output checks.
- Independent model replay of all baseline attention rows and target logits, plus all 1,152 test intervention examples for the selected head.

All checked values matched exactly. The model replay used 60 batched forwards. It did not repeat the intervention model runs for the other 23 heads. Their complete saved tables were checked by arithmetic only.

## Interpretation

The selected head is layer 3, head 1, with zero-based indices. The positive route drop, value drop, and joint rescue pass the fixed primary conjunction. The swapped-binding control is important: these effects depend on the supplied pair assignments. They are not explained by a fixed key–word association alone.

The joint edit is close to the baseline on the mean score. The interval for its difference from baseline includes zero. This is not a test of equivalence. Do not claim exact restoration, an exclusive circuit, or a pure semantic variable. The route edit is artificial and the exchanged values are contextual vectors. The result is a bounded replication in one small checkpoint and one constructed task.

## Limits of this review

I read the paper and README and inspected the figure. The reported test counts, gap scores, and primary effect values agree with the saved tables. The paper states the full-vocabulary capability limit, the non-equivalence limit, the large descriptive effect of another head, and the restricted model replay scope. No scientific wording blocker was found.

This review does not certify scientific novelty, generalization to other models, or ordinary-language performance. A separate AI implementation is useful evidence, but it is not human replication or peer review.
