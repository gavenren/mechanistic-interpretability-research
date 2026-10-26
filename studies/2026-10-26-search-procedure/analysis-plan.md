# Experiment 2B: does the search result support the claim?

Date: 26 October 2026. Initial state: no new model outputs, timing benchmark, calibration, or simulation has run.

## Question and scope

This study asks whether a search procedure and its chosen score support the claim made from the largest effect. It compares exhaustive subset search, one-pass greedy deletion, fixed score choices, and tests after statistical selection. All branches belong to one study. They are not separate papers.

Experiment 2A is complete. This stage reuses its eight-model selector panel and fixed block split. The models and original task states have already been studied. This is a diagnostic extension, not fresh confirmation or a claim of a new search principle. The [research plan](../../docs/RESEARCH_PLAN.md) defines the stage. The source commit is `32d2cf65cc9343c7d6f9a1fe1d2a9dcffd50d56e`. The input manifest fixes exact source files.

## Fixed order

1. Publish this plan, code, methods, source notes, and separate review. Verify the remote bytes.
2. Run dummy-only preparation. Its fourfold combined projection must fit the two-hour stage cap. Publish and verify passed preparation before any trained-model intervention output.
3. Run development calibration. Use only development intervention pairs to choose the eight-neuron panels and five score-specific candidates. Save all development outputs and the centered reference banks. Publish the complete file hashes, panel choices, candidate choices, and exact bank files before final tests.
4. Evaluate all 256 subsets on final test pairs. Compare exhaustive and greedy search. Evaluate the development-selected candidates without changing them.
5. Run the fixed selection simulation from the published development banks. No final test output can enter these banks.
6. Audit the saved results, write the plain-English paper, and publish all evidence. Count the study only after complete remote verification.

The [subset method](SUBSETS_METHOD.md) and [selection method](SELECTION_METHOD.md) are part of this fixed plan. They define all rows, arithmetic, random seeds, scores, thresholds, and output files. No choice may change after outputs to obtain a pass.

## Search and score tests

Use two fixed donor contexts for each receiver/request block. There are 768 development and 768 test pairs per model. Unlike 2A's variable-change test, the required answer here is the receiver's intact answer. The donor supplies replacement values. It does not define the answer target.

For each model, choose eight of its 12 first-layer neurons by development single-neuron sensitivity. Evaluate all 256 retained subsets. The four neurons outside the panel keep their receiver values. Thus an empty panel is not an empty network.

A subset passes only if it preserves at least 99% of intact answers and has mean zero-to-one probability distance at most 0.01. Apply exactly this rule to exhaustive enumeration and all 32 fixed greedy orders. Report every smallest passing subset, larger greedy results, and any further possible deletion after a greedy pass. The minimum-size claim passes only if every order reaches the exact minimum in every model. Failure is a valid result. It does not show that the model has no explanation.

The same final table supports all search comparisons. This is an exact finite algorithm diagnostic, not an independent test of the returned circuit. Its minimum is conditional on the chosen panel, outside neurons, donors, and cases.

Use development data to choose one of the 70 size-four subsets under each of five declared scores. Report all selected candidates on final test data. Keep context groups separate. Different rankings can reflect different objectives. Do not treat one measure as universally correct.

## Statistical selection test

Each model's bank contains 768 development rows and 254 candidate columns. Center each column twice to remove its residual mean. Make exactly constant columns exactly zero. Check the residual mean relative to each nonzero standard deviation. Do not drop a column that fails the numerical check.

For each of eight models and each candidate count 8, 32, and 128, run 2,000 trials. Each estimate uses 64 row draws with replacement. Selection and validation draws are independent. Each draw uses complete rows, retaining dependence between candidates. Fix the candidate order by a hash, not by observed effects. The total is 48,000 trials in 24 conditions.

Compare reused-data testing, separate validation, and a Bonferroni-adjusted normal threshold. All use known bank variances. The one-sided nominal test level is 5%. A zero-variance candidate cannot reject. This is a normal approximation for a constructed empirical reference. The correction does not make an inaccurate underlying approximation exact.

The primary separate-validation rule passes only if the upper endpoint of its two-sided 95% Wilson interval is at most 7% in every condition. Report all methods even when that gate fails. These are per-condition intervals, not simultaneous coverage. Complete valid measurements after a substantive gate failure. Stop using a failed validation rule for later positive claims. Passing does not measure power, a population circuit-error rate, or the stronger task criteria in 2C.

## Calculation checks

All model calculations use float64. The fixed maximum absolute tolerance is `1e-10`. Main subset calculations use matrix multiplication. Every saved block is reopened and checked with columnwise arithmetic, with additional scalar checks at fixed entries. No-change controls and gate decisions must agree. The statistical replay uses row-frequency counts, a different summation order, saved means, regenerated draw indices, and exact rejection-flag checks. Do not add a tolerance to a statistical threshold.

All eight models must retain at least 99% ordinary task accuracy. Stop the whole attempt on a failed numerical or ability check, invalid source, nonfinite value, changed calibration file, or resource stop. Keep complete blocks and publish partial status. Do not retry a failed panel or alter a tolerance silently.

## Compute limits

Use local CPU only, two compute threads, and one inter-operation thread. Preparation must remain below 4 GiB peak process memory. Scientific work must remain below 16 GiB. Keep at least 20 GiB free disk. No training, new model downloads, paid APIs, rented hardware, or language-model forwards are allowed.

The cumulative preparation cap is 3,600 seconds. The previous charge is 205.13888091699482 seconds. Add a fixed 120-second allowance for all current pre-lock syntax, mock, and review checks. Then measure dummy preparation from source checks through final receipts. Do not reset the cap between stages.

The Experiment 2B cap is 7,200 seconds across development calibration, final model tests, simulation, failures, replay, and saving. Keep a 120-second reserve for post-run audit and archive checks, including bank archive preparation. The final run deducts calibration's actual charge from the same stage cap. Preparation projects all scientific work with a factor of four, plus 120 seconds for shared setup and 120 seconds for the reserved checks. Launch only if this combined projection fits.

The roadmap cap stays at 183,600 seconds. Prior scientific charges total 1,243.228902584 seconds, with exact prior values in the source compute record. No unused stage allowance creates a new budget. Count each attempt and each check once.

The supervisor must check live usage at least once a minute during computation. A private atomic permit lasts no more than 120 seconds. Stop when any reported core window has 20% or less remaining, the meter is unavailable, or the permit is invalid. The code checks between bounded batches; a one-second signal also checks between native operations. Native operations can delay that signal. Record any overrun. Do not leave a runner without active supervision. Pause scheduled compute at a usage stop.

## Completion and reporting

A completed paper requires all fixed measurements, code, raw evidence, source hashes, a separate AI check, a compute record, and a plain-English report. Numerical success and substantive success must be separate. A completed negative result can qualify. Partial attempts, branches, seeds, and documentation do not count as extra papers.

Publish exact raw files directly or in checked archives. Verify remote bytes before updating the ledger. State that AI review is not human peer review. State that the study reuses a small diagnostic panel and does not establish a general search method or failure rate.
