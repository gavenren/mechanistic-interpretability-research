# Separate AI check of the saved results

Date: 26 October 2026.

The saved-result audit passed. The main statistical validation rule failed. These are separate results. The failed rule must not support later positive claims in this research program.

This check used a separate AI agent. It is not human peer review. No model forward pass, training, new search output, or new simulation trial was produced.

## Evidence checked

The audit checked all 295 final result files, seven preparation files, and 300 calibration files against their manifests. It also checked the 298 entries in the inner calibration manifest, all 19 source hashes, and the public calibration files at commit `5d05c4bdefc2f67986c89cae86bf1b25cb95f923`. The public bank archive had exactly the eight required members. Their hashes matched the raw-file record.

The audit recalculated summaries from all 3,145,728 saved development and final subset output rows. It checked the fixed row split, subset order, probability normalization, pass decisions, all greedy steps, minimum sizes, context groups, development panel choices, and development score choices. It checked that the same choices were used on final data. It rebuilt each statistical bank from its saved development effects.

For all 24 statistical conditions, the audit checked the recorded draws against their fixed seeds, the candidate order, the winner selected from recorded candidate means, both selected means from their recorded bank rows, standardized scores, rejection flags, table counts, Wilson bounds, and final rule decisions. It did not repeat the full set of candidate-mean calculations. The fixed scientific runner already replayed those calculations using row-frequency counts. The audit checked those recorded replay results.

The largest recorded subset replay error was `1.4210854715202004e-14`. The largest recorded statistical replay error was `5.329070518200751e-15`. Both were below the fixed `1e-10` limit. The audit did not repeat model arithmetic. Its probability checks use saved logits and outputs.

## Main results

Every greedy order reached the exact smallest passing subset in every model. For models 0 through 6, only all eight panel neurons passed. For model 7, masks 127 and 255 passed; mask 127 keeps seven neurons. Each model had one smallest passing subset. There was no larger greedy endpoint. This is limited evidence for greedy search: almost no deletion was allowed by this test. It does not show that greedy search generally finds a minimum.

All 40 score-specific size-four candidates failed the final fidelity rule. These are five choices for each of eight models; some choices are the same subset. The score choices produced these mean final answer-preservation rates:

| Development selection score | Mean final answer preservation |
|---|---:|
| Signed class-zero probability change | 68.59% |
| Absolute class-zero probability change | 71.60% |
| Target margin gain | 80.58% |
| Target probability gain | 83.48% |
| Answer accuracy | 83.01% |

These averages describe this small panel. Different scores have different objectives. The table does not prove that one score is best for other models or tasks. The fidelity rule also requires mean probability distance at most 0.01; answer preservation alone is not sufficient.

The statistical test used a constructed zero-mean bank. The following percentages are means across eight models, with 2,000 trials per model and candidate count.

| Candidate count | Reused selection data | Independent validation | Bonferroni threshold |
|---|---:|---:|---:|
| 8 | 21.16% | 5.39% | 3.78% |
| 32 | 41.48% | 5.09% | 3.78% |
| 128 | 57.19% | 5.41% | 2.91% |

Independent validation passed 22 of the 24 fixed conditions. Both failing conditions had observed rejection rates below 7%, but their required upper confidence bounds exceeded 7%:

| Model | Candidates | Rejections | Observed rate | Upper Wilson bound |
|---|---:|---:|---:|---:|
| 3 | 128 | 130 of 2,000 | 6.50% | 7.6660% |
| 4 | 32 | 118 of 2,000 | 5.90% | 7.0197% |

Thus the required overall rule failed. This does not establish that the true rejection rates exceed 7%. It shows that the declared evidence requirement was not met. All 24 Bonferroni conditions met the same descriptive upper-bound check. That finding does not replace the failed primary rule or authorize a change after results.

## Audit correction and compute

The first audit attempt stopped because its code used the wrong name for the saved single-neuron file. Its manifests and source checks had passed. Its elapsed time was 0.493767 seconds. The failed source is saved in `analysis/audit-attempt1.py`; its receipt is saved in `analysis/audit-attempt-1.json`.

The corrected audit uses `single-deletions.npz`. No scientific source, output, method, or tolerance changed. The corrected check passed in 1.523444 seconds before its closing receipt. Including one second for each closing receipt, both attempts used 4.017211 seconds. Both share one conservative 60-second audit charge. Do not add the failed receipt's stated allowance a second time. The corrected receipt records this accounting rule.

The audit used one CPU process, two compute threads, a 4 GiB peak-memory limit, the shared compute lock, and the live usage permit. Its recorded peak memory was 158,810,112 bytes. Bank archive preparation and final archive checks have separate charges assigned by the parent agent. This review does not certify final archive publication or remote completion.

## Limits and next use

The results concern eight previously studied small models, a selected eight-neuron panel, two donor contexts, and fixed cases. The other four first-layer neurons remained active. The minimum is conditional on this procedure. The statistical bank was centered by construction. Its rejection rates are not measured error rates for real model explanations. The test uses a normal approximation and separate per-condition intervals; it does not measure power or simultaneous coverage.

There is no numerical blocker for reporting this completed negative validation result. There is a scientific blocker for later positive claims that require the failed independent-validation rule. A later study must state a suitable new method and evidence requirement before its results. It must not silently change this study's threshold or promote the Bonferroni comparison to the original primary result.

## Paper check

The reviewer checked `paper.md` and `analysis/compute-accounting.json` after the result audit. The displayed result counts, percentages, bounds, numerical errors, and rounding match the audited evidence. The paper states the failed primary rule and the limited test of greedy search. The stage charge is 22.281584 seconds for both scientific phases plus the shared 120-second allowance, for 142.281584 seconds in total. This allowance is divided into 20 seconds for the bank archive, 60 seconds for both audit attempts, and 40 seconds for final archive checks. Prior science and preparation charges are carried forward once. The parent agent reported that 13 archives preserved 593 branch files and passed fresh restoration, repeat restoration, and conflict rejection in 5.222 seconds within the 40-second charge. The reviewer did not repeat the archive check. No factual correction was required. Final remote publication must still be verified separately.
