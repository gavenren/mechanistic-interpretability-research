# Independent result and claim review

Date: 13 October 2026. This is a separate AI review. It is not human peer review.

Status: complete within the scope below. No blocking issue remains.

## Computation

The [audit code](../code/independent_audit.py) does not import the scientific runner. It rebuilds the complete graph labels, all named obstruction groups, exact labelled degrees, graph IDs, masks, and training orders. It loads the final saved weights into a separate full-forward network. A first-ReLU hook applies the zero mask. This differs from the runner's cached continuation.

All 262,144 intact natural-graph logits and all 1,213,632 edited logits match exactly. The audit rebuilds all 64,800 degree-group score rows, unit rankings, selected masks, ability gates, and the fixed seed bootstrap. The largest score or norm difference is 1.78e-15, below 1e-10. It checks all 32,768 training records and the saved schedules. It does not repeat training or prove the training process by a second run.

The [receipt](independent-audit.json) records 66,091 comparisons, 3,256 full model calls, 1,475,776 examples, and 10.98 seconds. It pins the result manifest and audit code hashes. All 1,188 saved controls meet the fixed limit.

## Claims and figure

I read the paper, README, and report generator. I checked the tables and viewed the complete figure. The primary endpoint is the selected-minus-random joint-mask score within each model. The paper reports all four models and keeps the failed ability requirements central. It states that every natural graph was used for training; only the intervention context is held out.

The full-layer mean score of 1.977 and the Hall-3 false acceptance change from 6.34% to 84.72% agree with the saved results and the stated equal weights. The paper does not treat these weighted rates as independent observations. It explains why a constant output can have positive selectivity. It also reports the change to control decisions and the unmatched activation-removal magnitudes.

The result does not validate a Hall-rule component. It shows a relative effect under the specified unit selection and removal procedure. The paper states this limit and does not claim a unique graph algorithm, faithful semantic units, or unseen-graph prediction. The final audit text states its full replay scope and its lack of repeated training. The README uses a fresh path for repeat audit output.

## Limits

This review checks the finite saved computation and the paper's claims. Four models, many related graph labels, a narrow intervention context, and failed task ability limit the inference. The audit does not establish novelty, semantic ground truth, independent training replication, or human review.
