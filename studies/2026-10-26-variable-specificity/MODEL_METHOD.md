# Fresh selector models and fixed intervention data

This is the base-model part of Experiment 2C. It trains eight fresh models for the fixed selector task. The edit methods use their saved weights later. This part does not train edits or test a graph explanation. The graph branch remains blocked by Experiment 1.

## Fixed ordinary training

Use model seeds **91001 through 91008**, in that order. Use Study 1's original recipe without replacement seeds or extra updates. Its source files are checked against public repository commit `352f1551dde4be89c0e603402a54b7adc534b604`.

The model has 16 inputs, tanh hidden layers of widths 12 and 24, and four output logits. It has 616 parameters. Use float64. For each layer, initialize weights from the seed's NumPy default generator with standard deviation `1/sqrt(input_width)`. Initialize biases to zero, in the original layer order.

The task has all 192 states `(a,b,s,n)`, with four values each for `a` and `b`, two for `s`, and six for `n`. Use the original product order and one-hot input encoding. The correct answer is `a` when `s=0`, otherwise `b`. Every update uses all 192 states.

Train for exactly 4,000 Adam updates, with learning rate 0.02, beta values 0.9 and 0.999, and epsilon `1e-8`. Use the original mean cross-entropy loss, gradients, bias correction, parameter update order, and arithmetic. There is no shuffled minibatch, schedule change, early stopping, tuning, or checkpoint selection. Training uses no random draws after initialization.

The module copies the original numerical recipe without importing its environment-setting statements. The main controller sets two compute threads and one inter-operation thread. No GPU is used.

## Ability and numerical checks are separate

Train all eight planned models before deciding the complete ability gate. A low-accuracy result does not skip a later model. Every completed model must have at least 99% ordinary task accuracy for `ability_gate_pass` to be true.

The returned `passed` flag means the calculation and replay checks passed. It does not mean that task ability passed. A numerical error, source mismatch, or resource stop interrupts the attempt. A completed low-accuracy model remains a scientific result. The controller must not begin the edit comparison unless the complete eight-model ability gate passes.

After each model finishes, save its weights and outputs. Reopen these files. Recompute every first-layer activation, logit, and probability using a scalar implementation with `math.fsum`, `math.tanh`, and `math.exp`. Use a fixed maximum absolute tolerance of `1e-10`. Check all parameter, gradient, and optimizer arrays for their exact shapes, float64 type, and finite values. Do not enlarge a tolerance after failure.

## Three separate intervention sets

Enumerate all 768 receiver/request blocks and all 48 donor contexts in each block. A block fixes a receiver and a requested `a` value. Donors have that `a` value and cover every `b`, `s`, and `n` combination. The required changed input takes only `a` from the donor.

Within each receiver-selector and same-value/different-value stratum, sort blocks by SHA-256 of `selector-specificity-v1:receiver_id:requested_a`. Assign the first half to learning, the next quarter to selection, and the last quarter to test. All donors for a block remain together. This partition is fixed before outputs and shared by every model and edit method.

| Set | All pairs | Different-value pairs | Same-value pairs |
| --- | ---: | ---: | ---: |
| Learning | 18,432 | 13,824 | 4,608 |
| Selection | 9,216 | 6,912 | 2,304 |
| Test | 9,216 | 6,912 | 2,304 |

Each different-value count divides equally between active and inactive receiver cases. These are separate intervention pairs on known ordinary inputs. They are not new task inputs or independent training replications. Ordinary training still uses all 192 states.

## Files and interfaces

`INPUT_FILES` in `code/models.py` lists four original source files: `code/run.py`, `analysis-plan.md`, `data/inputs.csv`, and `requirements.txt`, inside Study 1. The runner checks their public bytes and records SHA-256 values. Generated input states must match the original CSV.

The module exposes:

- `inputs()`: states, one-hot inputs, and task labels.
- `pairs()`: arrays `r`, `d`, `cf`, `split`, `active`, and `same_value` for all 36,864 pairs.
- `split_pairs(name)`: the corresponding arrays for `learning`, `selection`, or `test`.
- `forward(x,p)`: probabilities, first hidden state, second hidden state, and logits.
- `downstream(h,p)`: probabilities from a first hidden state.
- `reference_forward(x,p)` and `reference_downstream(h,p)`: scalar probabilities and logits.
- `prepare(out,guard)` and `run(out,guard)` for the shared controller.

`data.npz` contains direct keys `states`, `x`, `labels`, `r`, `d`, `cf`, `split`, `active`, and `same_value`. Split codes are zero for learning, one for selection, and two for test. Save this file and its schema before training.

For each seed, save `seed-N/initial.npz`, `final.npz`, `training.csv`, `evaluation.npz`, optimizer state, and a model summary. Model weight keys are `W1,b1,W2,b2,W3,b3`. The log keeps every update's pre-update loss. It also keeps post-update accuracy at update 1 and every 100 updates, including update 4,000. Root `summary.json` contains all eight records and the separate ability gate.

## Stops and retained state

Call the controller's guard before every update and bounded evaluation/save phase. The controller enforces the shared two-hour stage cap, usage reserve, memory limit, disk requirement, and single-process rule. Output directories must be absent or empty.

Keep a copy of weights and both Adam moments at the last fully completed update. On interruption, save that complete boundary, its step, the generator state after initialization, optimizer settings, and a stop record. An interrupted later update is not included in that checkpoint. A complete checkpoint is evidence, not permission to resume: **automatic resume is disabled**. Preserve the trace even if its last recorded row precedes the checkpoint. No private account or permit path enters the module's stop record.

## Dummy preparation and cost

Preparation uses random dummy inputs and labels with seed 2026102603 and dummy model initialization seed 2026102604. It does not load trained weights. Check 24 fixed dummy gradient coordinates with central differences of `1e-5`; the maximum scaled gradient error must be at most `1e-6`.

Use 20 warmup updates, then three blocks of 100 updates. Use the slowest block. Include array checks, complete-state copies, training-log writes, and flushes. Separately time weight saving and scalar output replay.

The base-model projection is four times: the measured slowest per-update time for all **32,000 scientific updates**, eight measured save/replay costs, and a fixed 60-second setup and closing allowance. The edit module adds its own cost. The main controller must check their combined projection before scientific execution. Preparation uses at most four GiB; the scientific stage uses at most 16 GiB. These are limits, not measured requirements.

Charge one second for this module's pre-lock syntax check. No dummy benchmark, training, or model-output run was performed while writing these files.
