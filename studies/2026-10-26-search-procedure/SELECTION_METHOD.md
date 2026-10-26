# Selection tests: fixed method for Experiment 2B

This is one branch of Experiment 2B. It is not a separate paper. Publish this method and code before the scientific run. Publish and verify the development calibration bank before the selection tests start. No final test output can enter that bank.

## Question

When many possible edits have no effect on average, how often does selection make one appear effective?

We test three rules on a constructed reference with zero mean effect. We compare a test that reuses the selection data, a test with independent validation data, and a correction for the number of candidates. This is a calibration test. It is not a claim that the real edits have zero effect. It does not measure how often a test finds a real effect, or how often a real model explanation is wrong.

## Published development bank

Use `bank-seed-0.npz` through `bank-seed-7.npz` from the published calibration folder. Record the SHA-256 hash of each file before any trial. Check that hash again when loading the file. The parent runner must verify these hashes against the public calibration record.

Each file has 768 development rows and 254 candidate columns. It contains these arrays:

| Array | Meaning |
|---|---|
| `effects` | Float64 probability gains, shape 768 by 254 |
| `centered_effects` | The constructed zero-mean reference, same shape |
| `mean` | The original column means, before centering |
| `variance` | The known reference variance, using denominator 768 |
| `subset_ids` | The retained-feature bit masks, in fixed column order |
| `row_ids` | Positions in the full Experiment 2A donor ledger |
| `rows` | The nine donor-ledger columns, followed by donor context |

A probability gain is the candidate's probability for the intact receiver's correct task label, minus the empty-panel probability for that same label. The bank uses only the fixed development split and two fixed donor contexts. The final test split is excluded. The selector branch defines and publishes the bank. This selection branch verifies the development flag and ledger IDs before use.

There are 254 candidates: the nonempty, nonfull subsets of the eight fixed components. Sort mask IDs 1 through 254 by the hexadecimal SHA-256 digest of `search-procedure-v1:subset:{mask_id}`. A hash tie goes to the lower mask ID. Use the first 8, first 32, and first 128 columns. Do not sort them by observed effects. Each model supplies three conditions, for 24 conditions in total.

For each column, subtract the ordinary float64 mean of its 768 effects. Subtract the residual mean once more. If every original effect in a column is exactly equal, set the centered column to exactly zero. Its variance is exactly zero. These rules prevent a constant rounding residual from becoming a large standardized score.

Use the mean of the squared centered values as the known variance. Before trials, require the absolute residual column mean to be at most `1e-12` times its standard deviation in each column with positive variance. This is a numerical check. It does not remove or replace a column with a small variance. Stop if the check fails. Do not silently drop a candidate or change its variance.

## Fixed trial law

Run 2,000 trials in each of the 24 conditions. Draw 64 row positions with replacement for selection. Draw another 64 row positions with replacement for validation. Use independent random generators for the two draws. A draw selects whole rows: every candidate in that draw uses the same row positions. This retains the dependence between candidate effects in the bank.

For candidate `j`, let `s_j` be the square root of its known bank variance. Its selection score is:

```text
Z_j = sqrt(64) × selection sample mean_j / s_j
```

If its variance is zero, its score is zero and it cannot reject the null. Select the candidate with the largest score. A tie goes to the first column in the fixed hash order. Keep that candidate for all three tests. Do not select a different candidate after validation.

Use a one-sided nominal 5% test. The ordinary threshold is the 95th percentile of the standard normal distribution, approximately 1.644854. Reject only when the score is strictly greater than the threshold. The three methods are:

| Method | Score and threshold |
|---|---|
| Reused data | Test the selected candidate's selection score against the ordinary threshold |
| Independent validation | Compute its score on the independent validation draw, then use the ordinary threshold |
| Bonferroni correction | Test its selection score against the normal percentile `1 − 0.05/K`, where `K` is 8, 32, or 128 |

The known bank variance is used for every score. No sample variance is estimated from the 64 draws. This is a normal approximation for the finite empirical bank. It is not an exact normal test for that bank. Dependence between candidates, small samples, and non-normal row values can affect calibration. The simulation measures this limitation. The Bonferroni normal threshold does not make an inaccurate single-candidate normal approximation exact.

Selection and validation draws can contain some of the same row positions by chance. They still come from independent generators. Disallowing overlap would change the fixed sampling law.

## Seeds and draw order

Use NumPy's explicit `PCG64` generator. For model index `m` and candidate count `K`, the selection seed is `2026102600 + 10000 × m + K`. The validation seed is the selection seed plus 5000. Draw a complete 2,000-by-64 array of integer row positions with `integers(0, 768, dtype=int64)`. Save those positions as uint16 without changing their values. Process trials in groups of 32 after the draws are fixed.

| Model | Selection seeds for K = 8, 32, 128 | Validation seeds for K = 8, 32, 128 |
|---|---|---|
| 0 | 2026102608, 2026102632, 2026102728 | 2026107608, 2026107632, 2026107728 |
| 1 | 2026112608, 2026112632, 2026112728 | 2026117608, 2026117632, 2026117728 |
| 2 | 2026122608, 2026122632, 2026122728 | 2026127608, 2026127632, 2026127728 |
| 3 | 2026132608, 2026132632, 2026132728 | 2026137608, 2026137632, 2026137728 |
| 4 | 2026142608, 2026142632, 2026142728 | 2026147608, 2026147632, 2026147728 |
| 5 | 2026152608, 2026152632, 2026152728 | 2026157608, 2026157632, 2026157728 |
| 6 | 2026162608, 2026162632, 2026162728 | 2026167608, 2026167632, 2026167728 |
| 7 | 2026172608, 2026172632, 2026172728 | 2026177608, 2026177632, 2026177728 |

These are pseudorandom simulations with fixed seeds. They are not independently collected observations from a new model population.

## Main success rule

For each condition, count the independent-validation rejections among 2,000 trials. Report the rejection fraction and the upper endpoint of a two-sided 95% Wilson interval. Use the standard normal 97.5th percentile in that interval.

The independent-validation rule passes only if that upper endpoint is at most 7% in **every one of the 24 conditions**. The nominal test level remains 5%. The wider 7% gate allows simulation error. These are per-condition intervals, not a simultaneous 95% statement about all 24 conditions.

Report the reused-data and Bonferroni results in every condition, even if they fail. Complete the fixed comparison after a calibration failure unless a numerical or resource stop applies. Do not tune the law, threshold, number of trials, or candidates after results. If the main rule fails, mark it unsuitable for later positive claims in this program. Passing this gate still does not replace Experiment 2C's task criteria.

## Controls and saved-file replay

Preparation uses hand-made means and variances to check standardized scores, one-sided decisions, and the first-column tie rule. Zero-variance candidates must never reject. Check the normal tail probability at each threshold. Under an exact standard normal law, the ordinary tail is 0.05 and `K` times each Bonferroni tail is 0.05. These analytic identities check the code. They do not prove that the empirical bank is normal. Check the Wilson formula at zero and 2,000 rejections.

The scientific output folder must be empty. Save one compressed archive per condition. Close it, reopen it, and require every saved array and data type to match the intended array exactly. Regenerate the draw positions from their fixed seeds.

For independent numerical replay, convert each trial's row positions into row-frequency counts. Compute its selection means by multiplying those counts by the bank, then divide by 64. Compute its validation mean from the selected column and its validation row counts. This uses a different summation order from the indexed mean used for the original trial. Compare all candidate selection means, the selected estimates, standardized scores, and rejection flags. Numerical values must agree within absolute error `1e-10`. Rejection flags must agree exactly. Stop on failure.

Replay the fixed tie rule on the saved candidate means with a separate explicit maximum calculation. A different sum order can change a last-bit tie; it must not change which recorded values define the original choice. All original candidate means must still pass the independent numerical check. No tolerance is added to the statistical rejection threshold.

Require finite values throughout. Stop for a changed bank hash, invalid bank shape or identities, a centering failure, a failed control, or a resource limit. Do not increase tolerances after results.

## Compute and evidence

No model training or model forward pass is part of this branch. Use the parent runner's two-thread CPU setting. Stay under the preparation limit of 4 GiB of peak memory. The branch shares Experiment 2B's two-hour cap with the subset-search work. The parent also enforces the program cap and the 20% remaining-usage stop rule.

Preparation times one complete synthetic condition with 128 candidates and 2,000 trials, including compression, reloading, and independent replay. Its random bank has 768 rows and uses seed 2026102627. It contains no learned-model values. Estimate the scientific branch as `4 × (24 × measured synthetic-condition seconds + 30)`. The factor of four covers variation. The extra 30 seconds covers bank loading and summaries. Preparation is charged to the shared preparation limit. Scientific work is charged to Experiment 2B. Do not run if the estimate cannot fit the remaining cap.

Save all trial row positions, all candidate selection means, selected column and subset IDs, selected selection and validation means, both selected scores, known variances, and all three rejection flags. Save the 24 sets of seed and replay checks. Save a 72-row results table: eight models, three candidate counts, and three methods. Save the source hashes and the main gate result. This is about 30 MB of compressed trial evidence, depending on the values; it does not require large model files.

Report each condition, including failures. A summary must distinguish a passed numerical check from a passed calibration gate. The final claim is limited to this constructed reference, these saved models, these development effects, and this fixed sampling law.
