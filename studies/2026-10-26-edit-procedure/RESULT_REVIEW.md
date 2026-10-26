# Separate review of the saved results

**Date:** 26 October 2026. **Decision:** The saved-result audit passed. No blocking error was found in the checked results. This is a separate AI review, not human peer review.

The public plan record is `f4174fe862cd1c660d0bc1db5eddfe873d7e3a56`. The scientific execution record uses `352f1551dde4be89c0e603402a54b7adc534b604`.

## What was checked

The [audit code](analysis/audit.py) read the saved arrays and [wrote a receipt](analysis/audit.json). It checked all 1,170 result files and 11 preparation files against their manifests. It also checked all 31 repository source hashes and matched the preparation and scientific runtime records.

The audit checked all 411,648 selector rows, the saved edit coordinates, each batch's replay record, no-change controls, and every reported group mean. It checked all 288 sparse-code cases, nonnegative coefficients, sparsity, reconstruction limits, witness flags, readout values, and summary spans. It checked all 32 normalization prompts, 2,048 main/replay check records, saved normalization formulas, and all 1,024 between-rule comparisons, including zero edits.

The source run used 4,128 scientific language-model forwards and 33,024 scientific tokens. Preparation used another 512 tokens. The cumulative count is 33,536. The audit added no model-forward tokens.

Measured audit time before its closing receipt was **1.1974 seconds**. Peak process memory was 66,617,344 bytes. The separate conservative charge is **60 seconds**, within the reserved post-run allowance. No training, trained-model forward, or candidate search was repeated. Locked code and plans were not changed.

## Selector results

The table gives means across the eight fixed models on the test blocks. Each model has equal weight. “Active” means that the answer uses `a`. “Inactive” means that the answer uses `b`. Answer changes are unwanted in the three rows marked “yes.”

| Requested value of `a` | Task state | Correct required answer | Answer changed | Change is unwanted |
| --- | --- | ---: | ---: | --- |
| Different | Active | 42.81% | 61.15% | No; correctness still matters |
| Different | Inactive | 90.17% | 9.83% | Yes |
| Same | Active | 82.20% | 17.80% | Yes |
| Same | Inactive | 86.92% | 13.08% | Yes |

The original three-neuron masks do not reliably act as a replacement for `a` across these donor contexts. Even when `a` stays the same, a donor can change the answer. This does not rule out a better mask, a different internal representation, or a narrower valid donor rule.

| Relationship between the two requests | Task state | Answer differs between edit orders |
| --- | --- | ---: |
| Different requested values | Active | 54.92% |
| Different requested values | Inactive | 16.54% |
| Same requested value | Active | 25.07% |
| Same requested value | Inactive | 9.86% |

Both edits write to the same frozen mask. The last write sets those coordinates, as required by the operator. Equal requested values can still give different coordinate vectors because donor contexts differ. The output differences do not establish a new composition rule for two separate semantic variables.

These figures describe the fixed diagnostic panel. Dependent rows are not independent model replications. They are not estimates of a population failure rate.

## Sparse-code results

| Fixed split | Cases | Distinct valid alternative found | Same-vector witness found | Mean readout span | Largest readout span |
| --- | ---: | ---: | ---: | ---: | ---: |
| Development | 96 | 96 | 0 | 0.006629 | 0.055147 |
| Test | 192 | 192 | 0 | 0.009762 | 0.152699 |

The readout uses a zero-to-one probability scale. The mean test span is about 0.98 percentage points; the largest is about 15.27 points. Model-level mean test spans range from 0.002827 to 0.034057. The largest effects are not typical of every model.

The constructed alternatives satisfy the declared reconstruction allowance and sparsity limits. They are not evidence that the encoder produces those codes. No same-vector witness was found by this finite search. That is not a uniqueness proof. The cases contain equal numbers of both rare-bit values, so the table does not estimate behavior at the original 1% event rate.

## Normalization results

Each row covers 256 prompt/site cases. The score gap is the largest absolute difference among the eight fixed token scores in one case. The table then reports its median and maximum across cases. These are score differences, not probability differences.

| Edit size | Median selected-score gap | Largest selected-score gap | Mean normalization-vector gap, Euclidean length |
| --- | ---: | ---: | ---: |
| 0.1 | 0.013514 | 0.149417 | 0.236908 |
| 0.5 | 0.147835 | 2.416556 | 2.301457 |
| 1.0 | 1.367561 | 12.748755 | 9.151215 |

Recomputing normalization and keeping its original statistics give different results for the same edit. Larger edits have larger median gaps in this panel. This does not establish copying ability or identify a semantic variable.

The largest checked zero-edit difference among saved coordinates and scores was about `5.54e-12`. The largest recorded full-logit replay error was about `1.79e-11`. Both are below the fixed `1e-8` tolerance.

Full vocabulary logits were not saved. This audit checked their recorded replay maxima. It independently recalculated the saved normalization vectors and eight-score effects. The original replay shares the downstream transformer code. It is not an independent implementation of the full model.

## Publication limit

The evidence supports one completed diagnostic study with three branches. It does not support three separate discoveries. Publication still requires complete evidence archives, a plain-English paper with these limits, and verified remote file bytes. This audit does not by itself establish successful GitHub publication.

## Paper check

The separate reviewer checked `paper.md` and `analysis/compute-accounting.json` after the paper was written. The displayed task values, percentages, case counts, token count, and rounding match the checked records. The 151.589816833-second Stage 2A charge includes both 60-second post-run allowances. Prior work and the 205.138880917-second preparation charge are included once in the roadmap balance. The paper keeps the donor, finite-search, dependent-case, and shared-transformer limits. No factual correction or new scientific run is required by this review. Remote publication still needs its own verification.
