# Normalization test: fixed method

## Purpose

Test one choice in an internal edit. Layer normalization changes a vector by its mean and scale. We compare two rules after the same edit. The normal rule calculates the mean and scale again. The diagnostic rule keeps the old mean and scale.

This test measures numerical differences. It does not test whether a word, fact, or task variable has a known place in the model. It does not test copying ability. A difference between the rules can be a valid result. It is not a measurement failure.

This file and `code/normalization.py` must be in the public source lock before the task tests start. The parent runner controls usage, time, memory, and the source lock. The two module entry points are `prepare(out, guard)` and `run(out, guard)`. Each writes directly to the supplied output folder. It stops if that folder is not empty. There is no automatic run at import.

## Fixed inputs

Use **EleutherAI/Pythia-14M**, revision `cf967c0a9a04383db6f7b1108d86b2962634b4ac`. Use the existing local files. Do not download a model. Check all six file hashes in the code before use. These are the same hashes as the earlier Study 4. The old code is `studies/2026-10-13-patch-calibration-transfer/code/run.py`, read at repository commit `168d952dee2fbb5a6a73c4c9c01bd32cd8bc52c5`.

The local default is the sibling folder `../mi-staging/pythia-14m`. The environment variable `MI_PYTHIA_MODEL_DIR` can point to another existing copy. A different path is permitted only if every fixed hash matches.

Use these four sentence forms, in this order:

1. `The word is {w}. The word is`
2. `The answer is {w}. The answer is`
3. `The label is {w}. The label is`
4. `The code is {w}. The code is`

For each form, use these eight words, in this order: `cat`, `dog`, `book`, `chair`, `red`, `blue`, `rain`, `snow`. This gives 32 prompts. The order is sentence form first, then word. Do not add special tokens. Stop if any prompt has more than 128 tokens. Save all text and token IDs before the task test.

Use eight sites. In each of blocks 0, 1, 2, and 3, use `input_layernorm`, then `post_attention_layernorm`. Edit only the last text position. Other positions stay unchanged. Pythia uses parallel residual paths. The edit changes the input to the selected normalization module only. It does not replace the residual path or the other parallel branch.

## Exact edit

Let `x` be the 128 numbers at the selected normalization input in the intact run. Let `m` be their mean. Let `v` be the mean of `(x - m)^2`. This is population variance; there is no sample correction. Let `r = sqrt(v)`.

Add `alpha × r × sqrt(128)` to coordinate 0. Do not change the other 127 coordinates. Use `alpha = 0.1, 0.5, 1.0`. The root mean square of the added vector is thus `alpha × r`. This is a fixed numerical direction. It is not a donor feature or a selected semantic direction. There are no donor passes in this branch.

For each identical edited input, use both rules:

| Rule | Mean and variance | Output formula |
| --- | --- | --- |
| Recomputed | Calculate from the edited vector | `(edited - new_mean) / sqrt(new_variance + eps) × gamma + beta` |
| Frozen | Keep `m` and `v` from the intact vector | `(edited - m) / sqrt(v + eps) × gamma + beta` |

Use the actual trained gain `gamma`, bias `beta`, and module value `eps`. Save these values for every site. The fixed model config has `eps = 0.00001`. Do not omit the gain or bias. The recomputed main run uses PyTorch's native LayerNorm. The frozen main run uses the stated formula in PyTorch.

The model uses float64 weights and arithmetic. The attention operation keeps its softmax in float64, as in Study 4. Rotary position tables keep the shared float32 calculation used by that earlier study. Thus this is a declared numerical variant of the checkpoint. It is not a test of its usual lower-precision computation. Use Transformers 4.57.6, CPU, one prompt per batch, no gradient calculation, no dropout, and no key/value cache. Use at most two CPU threads.

Check these fixed runtime versions before preparation or the scientific run: Python 3.12.14, PyTorch 2.8.0, NumPy 2.3.5, Transformers 4.57.6, huggingface-hub 0.36.2, safetensors 0.6.2, and tokenizers 0.22.2. Stop if any version differs.

## Controls and separate calculation

The fixed absolute tolerance is **0.00000001** (`1e-8`). Apply it to each saved normalization coordinate and each compared logit. A logit is the model's score before conversion to a probability. Do not change the tolerance after a test failure. All arrays must be finite and have the required shape.

For each prompt and site, also use an edit size of zero. Run both rules and their replays. Compare the result with the intact normalization output and complete next-token logit vector. Every zero-edit control must pass. The zero edit is a control, not a fourth scientific edit size.

A second calculation uses NumPy to calculate the edit, mean, population variance, normalization, gain, and bias. It does not call PyTorch LayerNorm. Check that its edited input matches the main edit. Inject its normalization output at the same site. Replay the complete downstream model. Compare its normalization vector and full next-token logits with the main run. Also compare each rule's change from the intact scores, and the difference between the two rules. All these checks use the same tolerance.

The separate implementation covers the edit and normalization formula. The replay uses the same downstream model code. It is not an independent implementation of the entire transformer. Agreement does not establish that every part of that shared code is correct.

## Measured outputs

Save all 128 normalization output values for every rule, size, site, and prompt. Save the edited input and replay values. Also save the logits for these eight fixed tokens: space plus `cat`, `dog`, `book`, `chair`, `red`, `blue`, `rain`, and `snow`. Stop if any is not one token. Save their token IDs.

For each test, report the Euclidean length and largest absolute coordinate of the difference between the two normalization rules. Report the largest absolute difference among the eight selected scores. Save each rule's changes from the intact output too. Full vocabulary logits are used only for numerical replay checks. Save their maximum replay error, not the full vocabulary array. Do not describe the eight selected scores as a full-distribution result.

Report all prompt, site, and size results. These prompts are a fixed diagnostic panel, not a random sample. Do not use a population confidence interval. Do not select sites, words, edits, or a tolerance from these outputs. A plain-English paper can group the results for display, but must retain every case. There is no required minimum effect and no copying or semantic success claim.

## Preparation and cost

Before task tests, check the actual gains, biases, and epsilons on fixed artificial vectors. The vectors are `sin(0.37 × index) + 0.2`, multiplied by `0`, `0.25`, `1`, or `4`. The constant-vector case also checks the epsilon term. Check native LayerNorm against NumPy, and the unchanged frozen formula against native LayerNorm.

Then run four dummy model passes on the 128-token sequence `[0, 1, 2, 3]` repeated 32 times. The first pass is warmup. Use the slowest of the next three passes for timing. Do not save or inspect these dummy scores as scientific results. Charge all 512 dummy tokens to the shared token cap.

The scientific panel has exactly 4,128 full model passes:

| Pass class | Count |
| --- | ---: |
| Intact prompt; record all eight sites | 32 |
| Zero edit; 32 prompts × 8 sites × 2 rules × main/replay | 1,024 |
| Nonzero edit; 32 prompts × 8 sites × 3 sizes × 2 rules × main/replay | 3,072 |
| Total | 4,128 |

Charge the complete prompt length for every pass. There is no discount for an edit at the last position. The scientific token count is `129 × sum(prompt lengths)`. At the maximum prompt length, the scientific count is 528,384 tokens. With preparation, the bound is **528,896 tokens**, below the shared one-million-token cap. The parent runner must retain the cumulative count across preparation and task tests.

The conservative time projection is `4 × (slowest timed dummy pass × 4,128 + measured setup time + 30 seconds)`. Setup includes the file checks, model load, and artificial vector checks. The fixed 30 seconds allows for saving. It is an estimate, not measured save time. The parent runner must include all branches and all earlier stage time in its two-hour check. The fourfold factor is an allowance, not a measured upper bound. Stop if the forecast does not fit. Keep the 4 GiB preparation memory cap and the 16 GiB scientific-run memory cap. Model weights are about 112 MB in float64; one 128-token output tensor is about 52 MB. These are partial estimates, not a measured peak. The parent memory monitor remains required.

Call the parent guard before every full forward. It checks the usage permit, time, memory, and token allowance. Stop on any failed control, nonfinite value, hash mismatch, or resource limit. Save finished prompt files and any current partial arrays. Label a partial run as stopped. Do not retry the failed panel, enlarge the tolerance, or change the model to obtain a pass. A new design requires a new public source lock.

## Output files

`inputs.json` records the fixed inputs and hashes. `layernorm_parameters.npz` stores the actual gain, bias, and epsilon. Each `prompt_XX.npz` stores the measured arrays. Reload the saved parameters and every finished prompt file. Require exact array names, types, and values before marking that prompt complete. This check needs no extra model pass. The matching checks file stores numerical errors and effects. `rule_gaps.json` contains every between-rule comparison. `status.json` records success or the reason for a stop. Partial files have `partial` in their name. A stopped result is not a completed paper result.
