# Review of the completed Experiment 2C results

Date: 26 October 2026. Result: the paper agrees with the saved result records after the small wording changes listed below. All three full-panel explanation claims fail. The calculation and saved-result checks pass.

This is an AI result review. It is not human peer review or a full independent reproduction. The reviewer read the paper, audit record, phase status records, compute record, transport plan and review, and archive check receipts. The reviewer did not run another experiment, audit, model forward, training update, or numerical test for this review.

## Findings and claim limits

All eight fixed models passed ordinary task accuracy at 100%. All eight seeds remain in the report. Method 1 has eight choices. Method 2 has seven; seed 91001 has no eligible choice. Method 3 has six; seeds 91001 and 91002 have no eligible choice. Missing choices remain failures of the full-panel claim.

No available method-model choice passes every final requirement. Each of the three eight-model explanation claims therefore fails. The learned-method improvement claim also fails. Its primary eight-model mean gain and secondary eight-model mean gain are both null. The paper correctly keeps those results absent.

The reported means agree with the audit tables. Learned edits have 83.11% active-case accuracy among the six available models. All six improve over Method 1, with a conditional mean gain of 41.04 percentage points. This describes only those six comparisons. It does not replace the missing primary result or the fixed 95% requirement for every model. All six learned edits fail both the active-case accuracy and active-case probability-distance requirements. Three also fail at least one final preservation limit. Selection eligibility did not guarantee final eligibility.

The controls support the paper's narrow explanation. The known coherent input change achieves the reference behavior. The no-edit control scores 0% on active changed-value answers and 100% on inactive answers, for a pooled score of 50%. Full donor replacement and fixed random masks do not meet the explanation rule. These controls do not establish a general result about large language models.

The paper states the main limits. All ordinary task inputs were seen in training. Intervention pairs share underlying states. Eight seeds from one small synthetic task do not estimate a production-model failure rate. Failure of this fixed three-direction method does not prove that the variable is absent, that no useful rotation exists, or that another layer, loss, edit size, or schedule must fail.

## Audit scope

The saved-result audit passed. Its receipt records 27,549 checks, 226 raw output files, and 490 file-hash checks. Its maximum arithmetic difference is `5.995204332975845e-15`, below the fixed `1e-10` tolerance. The root agent reports 1.711446 seconds of measured run time. The receipt records no closing-write overrun. This work and the syntax checks fit within the fixed 50-second audit allocation.

The audit independently checks the data split, ordinary-answer results, saved logits and probabilities, selected and final output metrics, candidate eligibility, minimum-score choices, practical success rules, and manifests. It imports neither study code nor Torch and performs no model forward. Its per-model table preserves missing choices. Its available-model means are clearly marked as descriptive.

The audit does not reproduce model training, optimizer gradients, rotation matrix exponentials, or model outputs for unselected candidates. For those candidates, it checks the saved metric tables and selection logic. Orthogonality and saved loss accounting are checked; they do not independently validate every training update. The audit confirms consistency of the saved evidence. It is not a second full experiment.

## Publication and transport history

The local publication history and phase locks record this order: plan and code, passed preparation, all eight ordinary models, complete edit calibration with fixed choices, then final tests. The final result status names commit `b5f94da233f6b77e13415f1dc071c6cc1feb4faf` as its lock. The root agent remains responsible for verifying the final paper and result files on GitHub after publication.

The full calibration ZIP exceeded the browser upload limit. A reviewed amendment split its exact bytes into two parts in a separate transport folder. It did not change scientific code, selected edits, data, scores, limits, or tolerances. The first proposed layout put parts inside the calibration folder. Review found that this would fail the original exact-membership check. The first wrapper, test, and receipt remain in the record. The corrected mock test called the original calibration verifier and passed. No final model test ran under the first layout.

The six final result archives preserve 218 branch files. The archive receipt records a successful fresh restore, an identical repeat restore with no added files, and rejection of a conflicting file. These local checks do not by themselves prove final publication. Public byte checks and the final GitHub commit must supply that evidence.

## Compute accounting

The recorded phase charges are 9.830011458001536 seconds for ordinary training, 36.906813291996514 seconds for calibration, and 16.573985957998957 seconds for final tests. The one 120-second post-run reserve remains intact: 20 seconds for the frozen archive, 10 for both transport check attempts, 50 for the result audit, and 40 for final archive checks. No second reserve was added.

The resulting Experiment 2C charge is **183.310810707997 seconds**, below its 7,200-second limit. Cumulative preparation is **453.1494834999976 seconds**, below its separate 3,600-second limit. Cumulative scientific charge, including earlier studies once, is 1568.8212972499932 seconds. These are charged amounts, including conservative allowances; they are not claims that every allowance was fully spent.

## Small paper changes made during review

The root agent clarified the no-edit control sentence during this review. The reviewer limited the zero-inference-token claim to the scientific runs. The paper now states that AI-assisted coding and writing used account usage. It also states the separate cumulative preparation charge. The audit description now says “490 file-hash checks,” without implying 490 distinct files. Missing spaces after several commas were corrected. No result, threshold, method, or primary null value changed.
