# Exact subset search and fixed score choices

This branch extends the constructed tests in Studies 17, 19, 21, and 23 to the eight saved selector models. It tests a fixed conditional pruning problem. It does not claim a new search principle or a complete circuit for the task. No model is trained.

## Sources and fixed cases

Use Stage 2A's selector source at public commit `352f1551dde4be89c0e603402a54b7adc534b604`. The runner checks that source against the public Git file. It then uses the source's fixed hashes for the original Study 1 model files, input files, selected-mask record, and original code. `INPUT_FILES` in `code/subsets.py` lists all 14 source paths. The original three-neuron masks remain part of the source record; this stage selects a separate eight-neuron search panel using development data only.

Use the same receiver/request block split as Stage 2A. For each of its 768 blocks, retain two donors: the matching context, and the context where `b` increases by one modulo four, `s` flips, and `n` increases by one modulo six. The donor's `a` value is the block's requested value. This gives 1,536 pairs per model: 768 development and 768 test pairs. Row identifiers are positions in the full Stage 2A donor ledger. There are no new pair splits chosen from model results.

Unlike Stage 2A's variable-change test, this branch asks whether the edited model preserves the **receiver's intact behavior**. The donor is a fixed replacement source. Its label is not the required answer here. Both same-value and different-value donor cases remain in the comparison.

## Publish development choices before final tests

The `calibrate` phase computes development interventions only. Ordinary intact outputs for the 192 task inputs support the existing task-ability check and numerical reference. These inputs were all in Study 1 training; they are not fresh test inputs. No final test-pair intervention output is calculated in calibration.

For each model, replace each of the 12 first-layer neurons with its development donor value. Rank neurons by mean TV from the intact receiver output, largest first. Exact score ties use neuron number. Freeze the first eight neurons in that order. These are panel positions zero through seven.

For each of the 256 retained subsets, retain the receiver values at its selected panel positions. Replace the other panel positions with donor values. Keep all four neurons outside the panel at their receiver values. Subset identifiers are integers zero through 255; their eight binary digits specify retained positions. The full panel must reproduce the intact receiver output. The empty subset still retains four receiver neurons. It is not an empty network.

Save every development subset's logits and probabilities. Use the 70 subsets of size four to select one candidate under each of five scores:

1. Mean signed change in class-zero probability.
2. Mean absolute change in class-zero probability.
3. Mean required-answer margin gain.
4. Mean required-answer probability gain.
5. Task-answer accuracy.

Changes are relative to the empty-panel intervention. The required answer is the receiver's known task label. For the margin, the comparison class is `(required_label+1) mod 4`. Select the highest development score; exact ties use the smaller subset identifier. No test result selects a candidate or changes a score.

Publish `calibration.json`, all eight frozen panel/choice records, bank files, and the complete development file hashes before the final run. The main controller verifies the public lock. The module also checks every calibration file against `calibration-manifest.json` before and after final testing.

## Exact search and greedy search

Evaluate all 256 subsets on the fixed test pairs. A subset passes only if it keeps at least 99% of intact receiver answers and has mean TV at most 0.01. Apply the same rule to exhaustive search and every greedy decision. TV is half the sum of absolute probability differences, on a zero-to-one scale.

Report every passing subset and every smallest passing subset. Start each greedy run with the full panel. Visit each panel position once. Remove it only if the resulting subset passes. Use 32 fixed visit orders: ascending, descending, and the 30 remaining permutations with the smallest SHA-256 values of `search-procedure-v1:order:comma-separated-positions`.

Compare each greedy result's size with the exact minimum. The minimum-size claim passes only if all 32 orders find that minimum in every model. Record larger results without replacing seeds or orders. Also check whether a further single deletion is possible at the end. A nonmonotone network can make an earlier rejected deletion possible later; do not assume one pass gives an inclusion-minimal result.

The comparison uses the same test table for every method. It is a finite algorithm diagnostic, not an independent test of a selected circuit. Its minimum is conditional on these eight panel neurons, these pairs, this donor rule, the four retained outside neurons, and the stated tolerance rules. Subset tables include all individual, paired, and full-group interventions.

## Score results and development bank

Report all five scores, TV, and answer accuracy by same-value/different-value request, active/inactive receiver, and donor context. Keep development and test records separate. Evaluate the five development-selected four-neuron candidates on the test pairs without changing them. A ranking difference describes different objectives; it does not establish that one score is universally correct.

For the statistical branch, use all 254 nonempty, nonfull subset candidates. Sort them by SHA-256 of `search-procedure-v1:subset:mask_id`. The statistical branch uses the first 8, 32, and 128 columns, without selection from their values.

For each development row and subset, the raw effect is the candidate's probability of the receiver's task label minus the empty-panel probability of that label. Center each column by subtracting its development mean. Subtract the floating-point residual column mean once more. If the raw column is exactly constant, set its centered column to exact zero. Its variance is also zero. Population variance uses denominator 768.

Each `bank-seed-N.npz` contains `effects` and `centered_effects`, both shaped `(768,254)`; `mean`; `variance`; the ordered `subset_ids`; `row_ids`; and `rows`. The ten row columns are receiver, donor, Stage 2A required-input identifier, split, same-a flag, active flag, b-match flag, s-match flag, n-match flag, and context. Context zero is matched; context one changes the three other inputs. The Stage 2A required-input identifier is retained for provenance. It is not this branch's answer target. `calibration.json` records each bank's SHA-256 value.

## Replay and stop rules

Use float64 and a fixed maximum absolute tolerance of `1e-10`. Each model must keep at least 99% ordinary task accuracy. Stop on a failed source check, nonfinite value, shape error, no-change control, or replay. Do not enlarge the tolerance after failure.

The main readout uses matrix multiplication. Every saved subset block is reopened and replayed using columnwise arithmetic. Four fixed entries in every block also use a scalar implementation with `math.fsum`, `math.tanh`, and `math.exp`. Both logits and probabilities must match. The single-neuron development files are reopened and checked as well. Scalar reductions must reproduce gate decisions exactly. Greedy paths are replayed with sets instead of bit operations. These checks repeat the declared calculation; they do not repeat training.

The module accepts `guard()`, supplied by the main controller, before every bounded computation block. Completed blocks remain if execution stops. Output folders must be absent or empty. Exceptions propagate to the controller. Substantive search failures remain results; they do not cause seed or order replacement.

## Compute and interfaces

The entry points are `prepare(out, guard)`, `calibrate(out, guard)`, and `run(out, guard, calibration_path)`. Calibration output is expected at `calibration/subsets`. Final execution receives that exact directory. The controller must publish the calibration and enforce the shared two-hour Stage 2B cap before final tests. It must include preparation, both scientific phases, the statistical branch, saving, and replay under their applicable budgets.

There are **3,145,728 subset-output rows** across development and test, plus **73,728 single-neuron development rows**. Each row has four logits and four probabilities. Model activations are cached once per model and phase. Search and score analysis use saved outputs, without new model passes. Planning allowances are 15–60 CPU minutes, below 512 MiB of working memory, and about 250 MiB of uncompressed model/bank data. These are estimates, not measurements.

Preparation uses fixed random inputs and random weights with seed 2026102602. It opens no trained weights for computation. It times one 12-neuron dummy panel check and one 16-subset dummy block, including saving and replay. The projection is four times eight panel checks, 256 scientific subset blocks across both phases, and a fixed 120-second allowance for summaries, banks, locks, and receipts. The statistical branch adds its own estimate. The controller must stop if the combined estimate does not fit.

Charge a conservative ten seconds for the module's pre-lock syntax and small mock checks. No scientific model output or preparation benchmark was run while writing this method.
