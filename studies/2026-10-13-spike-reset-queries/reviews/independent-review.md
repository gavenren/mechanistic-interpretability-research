# Independent result review

13 October 2026. This is a separate AI review. It is not human peer review.

**Status: computation and final report review passed.**

## Replay scope

The [audit](../code/independent_audit.py) does not import the scientific runner. It reconstructs the hard spiking recurrence from the saved weights. It uses a manual reverse gradient recurrence, with no autograd, to check the scientific runner's leaf derivatives and forward tangent checks. The derivative code was first checked against the saved nonzero dummy witness. Both dummy gradients matched exactly.

The full result [receipt](independent-audit.json) records a pass. It checks the public lock, result manifest, raw inputs, fixed panel, model initialization hashes, final weight hashes, and schedules. The audit reads all 4,800 training records and checks their counts, batch sizes, update order, and finite losses. It **does not repeat training**.

All 71,880 full-test logits and 360,000 panel logits match exactly. All 27,648,000 saved binary spike/message trace values match exactly. The baseline and both leaf-query voltage traces also match exactly. These cover every baseline, singleton deletion, joint deletion, and self-replacement condition saved by the study.

The manual reverse gradient recurrence agrees within 5.55e-17, below the fixed 1e-10 limit. It retains all future reset paths and differs only in the selected event's immediate reset path. The audit verifies the ranking and tie rules from both saved and independently rebuilt predictions. All selected units and four-unit sets match.

The audit reconstructs all 30,720 event rows, 1,920 regret rows, 2,880 joint rows, and 480 case rows. The exact effects, primary scores, model means, ability gates, sign summaries, and fixed bootstrap interval match. All 1,688 saved numerical controls satisfy their fixed limits. The audit uses 360 full forward calls on 43,188 examples and takes 3.91 seconds. It records 37,371 comparisons.

## Interpretation checks

All four full-test ability gates pass. The primary matched-selection benefit is 0.01227375 margin units, with descriptive interval [0.01007888, 0.01492220]. All four model-level primary scores are positive. However, seed 317 has a negative message-query gain. The primary score can therefore hide a directional loss, as the plan predicted.

The two selectors agree on 78.75% of model–image cases. Their joint four-event effects also have directional exceptions: seed 313 loses under node deletion, and seed 331 loses under message deletion. The paper must retain these limits and the complete two-by-two table. Joint effects are measured directly and are not an exhaustive joint-search optimum.

The finite oracle is the best single deletion among 32 candidates under the stated margin and query. It is not semantic ground truth. The primary contrast cancels the oracle maxima and tests relative selection benefit. Neither passing task accuracy nor a positive primary score establishes complete explanation accuracy.

## Final report check

The final paper and README preserve the one message-query loss, both four-event losses, the absolute regret table, and the limits of the finite panel. The report states that 415 of 480 cases have zero primary gain. Its sign-error table uses pooled eligible counts and does not claim improved sign accuracy. Margin units are clearly separated from probability and accuracy points.

I separately recalculated all 64 report rows from the saved result tables and model arrays. All 228 numeric report cells matched within 1e-12. This check used no model calls. I read the full report code, paper, and README and viewed the four-panel figure. The labels, displayed values, and caption agree with the saved results. The figure uses separate scales for accuracy, matching gain, regret, and joint effects.

No claim or computation issue remains within this review scope. This is an AI review of a fixed small experiment. It is not human peer review or an independent repeat of training.
