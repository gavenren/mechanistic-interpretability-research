# Experiment 2A: specify the full edit procedure

Date: 26 October 2026. Status at plan creation: no new scientific outputs.

## Question and scope

Can an edit result change when we keep the intended request fixed but change its execution rule? We test donor choice, two writes to the same coordinates, allowed sparse codes, and normalization. These are three branches of one diagnostic study. They do not count as separate papers.

This is the next stage in [the research plan](../../docs/RESEARCH_PLAN.md). It uses saved models from Studies 1, 2, and 4–8. These models have already informed this project. They are a diagnostic panel, not fresh confirmation. The broad concerns have prior work. The contribution is a fixed test of these procedures on these assets, with complete saved evidence and separate arithmetic checks. We make no claim of a new general method.

The baseline source commit is `168d952dee2fbb5a6a73c4c9c01bd32cd8bc52c5`. The input manifest fixes repository asset hashes. The normalization module also fixes the cached model files and their revision. No new model training is allowed.

## Fixed branches and order

1. **Selector donors and edit order.** Use all eight Study 1 models and their original repair-only three-neuron masks. Use all 192 task states. Hold the receiver and requested task change fixed across 48 donor contexts. Compare donor replacements, unchanged states, zero replacements, and coherent input changes. Then test both orders of two replacements on one mask. Check both requests after the second edit. The [selector method](SELECTOR_METHOD.md) fixes all 411,648 rows, the split, measures, and tolerances. Last-write replacement is a known property of this operator; it is not a discovered rule for composing two semantic variables.
2. **Allowed alternative sparse codes.** Use all six reconstruction-only dictionaries at rare-event rate 0.01 and amplitude 0.25. Keep the input, error budget, nonnegative weights, and at-most-four active features fixed. Search a declared finite candidate family for different readout values. The [code method](CODES_METHOD.md) fixes all source paths, input cases, supports, candidate rules, and checks. A valid alternative is a witness. Absence from this search is not a uniqueness proof. A constructed code need not be an encoder output.
3. **Normalization rule.** Use the pinned Pythia-14M model. Test 32 fixed prompts, eight sites, and three edit sizes. Compare recomputed normalization with fixed original normalization statistics. The [normalization method](NORMALIZATION_METHOD.md) fixes inputs, scores, model files, arithmetic, replay, and token counts. Report numerical spread in eight declared token scores and complete normalization vectors. Do not infer a semantic or copying mechanism from this test.

The branches run in this order in one process. No scientific jobs run in parallel. A numerical or resource failure stops the entire attempt. Save completed evidence. Do not silently retry a failed branch, change a tolerance, drop a model, or replace a test case.

## What counts as success

Every identity or no-change check must pass its fixed tolerance. Separate arithmetic must reproduce every measured effect in the declared scope. Saved files must pass hash and shape checks. Every branch must finish all fixed cases. Any finite, valid disagreement can be a successful diagnostic result. A positive effect is not required.

The selector branch also requires at least 99% ordinary task accuracy in each model. Failure means this attempted diagnostic panel is incomplete. The other branches do not claim a demonstrated task mechanism. Their limits remain in the report. Numerical agreement supports calculation accuracy only.

Report all cases and separate development and test summaries. Do not tune from either split. Do not pool dependent rows as independent model replications. Do not report a population failure rate or a significance claim from this fixed panel. Keep exact numerical tolerances in the branch notes. Do not change them after results.

## Compute and stop rules

Use local CPU only, two compute threads, one inter-operation thread, at most 4 GiB peak process memory during preparation and 16 GiB during the scientific run, and at least 20 GiB free disk. Do not use paid APIs, rented hardware, model training, or new model downloads. Offline mode is set before model loading. The cached model must pass its exact hash checks.

The preparation cap is one hour across the whole roadmap. Prior preparation is 79.79492641699471 seconds. Charge another 120 seconds for all pre-lock syntax, mock, and review checks. Measure the new preparation run from before source checks through saved receipts. Its branch benchmarks use dummy data only. Include their full token costs. Use the fourfold branch projections and another 120 seconds for shared setup and saving. Reserve 120 seconds for post-run checks and archive checks. Launch only when the total projection is less than the two-hour 2A cap.

The stage cap is 7,200 seconds, including failures, replays, saving, and post-run checks. The full preparation and scientific normalization passes share a cap of 1,000,000 forwarded tokens. The planned strict maximum is 528,896 tokens. Count each complete forward, including zero edits and independent replay, before execution. Local tokenization uses no model-forward tokens.

The roadmap cap remains 183,600 seconds (51 hours). Experiment 1 has a cumulative charge of 1,091.6390857510014 seconds. Do not reuse an unspent allocation as a new budget. The new study's final compute record must carry these prior charges forward exactly once.

The supervisor must check live core usage at least once per minute during computation. It may issue an atomic private permit for at most 120 seconds. Stop when any reported core window has 20% or less remaining, the meter is unavailable, or the permit expires. Account data stays in ignored local state. The code checks the permit before each bounded batch and each language-model forward. A one-second signal also checks it between native operations. A native operation can delay signal handling; record any observed overrun. No scientific process may run without active supervision.

Stop on the first failed numerical check, invalid source, token cap, time cap, memory cap, disk limit, or usage rule. Save a stopped receipt and retained outputs. Completion is not implied by a partial result or an upload attempt. Disable future scheduled compute if the usage stop applies.

## Public execution record

First publish this plan, all branch methods, exact inputs, code, source notes, and the separate plan review. Verify the remote bytes. Then run preparation against that immutable commit. Publish its complete receipt and verify it. The scientific run must use a descendant commit containing the same plan and code plus passed preparation. No scientific output may inform these fixed choices.

On completion, save raw results, per-branch checks, summary tables, hashes, a plain-English paper, a separate result review, and cumulative compute charges. Large binary files may be stored in exact, checked archives. Publish the complete evidence package and verify its bytes. Only then update the program count. AI checks are not human peer review. No full independent training reproduction is planned.
