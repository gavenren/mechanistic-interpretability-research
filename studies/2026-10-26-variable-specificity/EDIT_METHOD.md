# Three-direction edits: fixed method for Experiment 2C

Publish this method and code before the run. Train and verify all eight base models before edit learning. Publish the selected edits and their files before final intervention tests. The parent runner must verify each public record before it starts the next phase.

## Question and scope

Can an edit change the intended variable when it matters, while preserving behavior when it does not matter?

The task has inputs `a`, `b`, `s`, and `n`. Its answer is `a` when `s = 0`; otherwise its answer is `b`. The proposed edit transfers the donor's value of `a` into the receiver. The required input changes only `a`. The receiver keeps its values of `b`, `s`, and `n`.

The study compares three methods that each replace three directions in the first hidden layer. That layer has 12 coordinates. The methods have the same edit size. The learned method has more training and a different search space. Equal edit size does not make their search costs equal.

Experiment 2B's normal-test calibration failed its all-condition rule. This study uses no normal test or statistical-significance claim from that experiment. It uses the fixed practical gates below. This distinction does not mean that all use of separate selection and test data is invalid.

## Models and fixed pair splits

Use fresh base-model seeds 91001 through 91008 from the companion `models.py` module. Use `seed-91001/final.npz` through `seed-91008/final.npz` in the published model folder. Each file has `W1`, `b1`, `W2`, `b2`, `W3`, and `b3`. Shapes are 16-by-12, 12, 12-by-24, 24, 24-by-4, and 4. All values use float64.

All eight models must have at least 99% accuracy on the complete 192-state task before any edit work. Check the published ability gate and independently recalculate each model's outputs before the first edit. Do not replace a failed seed. The companion model method fixes the ordinary training recipe.

The published `data.npz` contains `states`, `x`, `labels`, `r`, `d`, `cf`, `split`, `active`, and `same_value`. The last six arrays identify all 36,864 receiver–donor pairs in receiver-then-donor order. Here `cf` is the required input state; `active` means the receiver uses `a`; and `same_value` means receiver and donor already have the same `a`.

The companion module fixes the split by whole receiver/requested-value blocks. Donor contexts within a block stay together. It uses the fixed hash prefix `selector-specificity-v1`. Each stratum puts half its blocks in learning, one quarter in selection, and one quarter in test. The model module specifies the full ordering. No model output selects a pair.

| Split | All pairs | Active, different value | Inactive, different value | Same value |
|---|---:|---:|---:|---:|
| Learning | 18,432 | 6,912 | 6,912 | 4,608 |
| Selection | 9,216 | 3,456 | 3,456 | 2,304 |
| Test | 9,216 | 3,456 | 3,456 | 2,304 |

The ordinary task states are known during training. The split separates intervention pairs, not unseen task states. Do not describe the final test as generalization to a new task or state population.

## Scores and intended answers

For each pair, the distribution target is the intact base model's probability vector on the coherent required input `cf`. Use zero-to-one total variation distance:

```text
TV(p, q) = 0.5 × sum(abs(p − q))
```

The required answer comes from the exact task rule at `cf`. It is not the donor's answer in general. Same-value answer changes compare the edited answer with the intact receiver's answer. Break answer ties by the lower class index.

Keep three groups separate: active different-value pairs, inactive different-value pairs, and same-value pairs. An active different-value case requires a change in the task answer. An inactive different-value case keeps the required answer. Do not let unchanged inactive answers inflate the claimed ability to make an intended change.

For each group, save its count, required-answer accuracy, target TV, TV from the intact receiver, and answer-change fraction. The inactive target TV remains the original Study 1 coherent-input score. Inactive TV from the intact receiver is an additional preservation measure. These two probability measures can differ even when the task answer stays the same.

## Method 1: three original neurons, chosen by intended change

List all 220 three-element subsets of coordinates 0 through 11 in lexicographic order. Replace exactly those receiver coordinates with the donor coordinates. Evaluate each mask on every selection pair.

Choose the mask with the lowest mean TV on **active different-value** selection pairs. An exact score tie goes to the lower candidate index. Same-value controls do not enter this objective. Do not use final test effects for the choice.

## Method 2: three original neurons, with preservation limits

Use the same 220 candidates and the same selection scores. A mask is eligible only if:

- Mean target TV on inactive different-value pairs is at most 0.05.
- At most 1% of same-value answers change.

Among eligible masks, minimize mean target TV on active different-value pairs. Use the same exact tie rule. If no mask is eligible, record that result. Do not increase a limit or substitute the least-bad mask. An absent eligible mask is a failure of this method's claim in that model, not a numerical failure.

## Method 3: a learned rotation

Freeze all base-model weights. Learn a 12-by-12 real parameter matrix `A`. Define:

```text
Q = exp(A − A.T)
U = first three columns of Q
h_edited = h_receiver + ((h_donor − h_receiver) U) U.T
```

The skew-symmetric matrix exponential makes `Q` orthogonal in exact arithmetic. The edit rotates the state, replaces its first three coordinates, then undoes the rotation. The implementation above uses the equivalent rank-three projection. Check the rotation numerically. A learned direction can mix many original neurons.

For model seeds 91001 through 91008, initialize `A` with Torch generator seeds 92001 through 92008, respectively. Each entry is an independent standard normal draw multiplied by 0.01. Use CPU float64. The initial rotation is not selected from model results.

Run Adam with learning rate 0.01, betas 0.9 and 0.999, epsilon `1e-8`, and zero weight decay. Disable AMSGrad, foreach, fused, maximize, capturable, and differentiable options. Run exactly 4,000 updates per model. There is no learning-rate schedule, early stop, seed replacement, or test-based change.

Use explicit NumPy `PCG64` generators with seeds 93001 through 93008 for learning pairs. At each update, draw with replacement:

- 64 active different-value learning pairs;
- 64 inactive different-value learning pairs;
- 64 same-value learning pairs.

Draw in that group order and save every pair ID. The loss is the sum of the three group mean TVs against the coherent-input probability target. Each group has equal weight in the sum. Base probabilities and hidden states remain fixed; only `A` is trained. The loss uses the probability target, not a later test score.

Save candidate rotations after updates 1,000, 2,000, 3,000, and 4,000. Evaluate all four on the full selection split. Apply Method 2's preservation limits. Among eligible checkpoints, choose the lowest active different-value TV; an exact tie goes to the earlier checkpoint. If none is eligible, record “no eligible checkpoint.” Do not evaluate an unselected replacement on the final test split.

Save the initial parameter, all learning draws, all four loss values at every update, checkpoint parameters and rotations, and Adam's mean, variance, and step at each checkpoint. Check the parameter and all Adam states for finite values after every update. Accept an update only after these checks pass. Keep a separate parameter copy from the last complete accepted update. A numerical or resource failure saves that copy, its exact update count, and the completed loss prefix. An interruption inside an optimizer update cannot be labeled as a completed update. The partial record does not permit a resumed or extended training run under this plan. Refuse to save any nonfinite numeric array.

## Choice publication and final tests

For every model, publish all 220 mask score rows, all four checkpoint score rows, and each method's choice or explicit absence. Save and publish raw selection logits and probabilities for each available chosen method. Bind the choices to the base-model hashes and checkpoint hashes. Publish the complete calibration manifest before final tests.

The final run verifies the calibration files and model hashes. It applies each available chosen method to all 9,216 test pairs. It does not learn a new rotation or choose a different mask.

Also evaluate these fixed controls on all test pairs:

- No edit.
- Replacement of the full first hidden layer with the donor's layer.
- The known coherent input change.
- Twenty three-neuron masks from `PCG64(94001).permutation(220)[:20]`, in that order.

Use the same twenty random-control masks for every model. They can overlap with selected masks; do not redraw them. The coherent input control uses explicit task-variable information that the learned method does not receive as an input. It is a privileged reference, not a competing learned method with equal information.

## Fixed final gates

A method supports the variable-explanation claim only if **every one of the eight models** passes all these rules:

- At least 95% required-answer accuracy on active different-value pairs.
- At least 95% required-answer accuracy on all different-value pairs.
- Mean target TV at most 0.05 on active different-value pairs.
- Mean target TV at most 0.05 on inactive different-value pairs.
- At most 1% of same-value answers change.

The second accuracy rule removes an ambiguity in the roadmap. It is an additional gate; it does not replace the active-case gate. An unavailable selected method cannot pass.

The learned-method improvement claim requires Method 3 to pass every absolute gate above. Its equal-model mean gain over Method 1 in **active different-value accuracy** must be at least five percentage points. At least six of the eight paired model differences must be strictly positive. Report all-different-value accuracy gain separately. These are practical acceptance rules and descriptive counts, not a population significance test. If a checkpoint is unavailable in any model, the improvement claim fails.

## Independent calculation checks

Before scientific edits, compare NumPy and Torch hidden states, logits, and probabilities for every ordinary task state. Check the eight ordinary accuracy gates before the first candidate evaluation.

For every candidate evaluation, compute a separate hidden state and downstream result. The separate affine implementation sums input-coordinate products explicitly. For masks, replace coordinates one at a time. For a learned rotation, use the full rotate, replace, and undo sequence. The primary implementation uses the rank-three formula. Check all logits, probability values, predicted answers, group metrics, and gate decisions. Do not validate an aggregate score without checking its underlying outputs.

The fixed absolute numerical tolerance is `1e-10`. Predicted answers, counts, and gate decisions must agree exactly. Stop for a mismatch. Do not increase the tolerance after results.

Check each saved rotation against a separate matrix-exponential calculation. Scale the skew matrix until its one-norm is at most 0.5, sum 40 Taylor terms, then square back. Permit at most 16 scaling steps; stop beyond that range. Compare against the saved Torch result and check both `Q.T Q` and `Q Q.T` against identity within `1e-10`. This is a numerical cross-check, not a formal error certificate.

Preparation uses only an untrained dummy model. Check identity rotation, a known coordinate rotation, no edit, full-layer replacement, and coherent input change. Check four parameter derivatives with central differences of `1e-6`; the absolute gradient-error limit is `1e-5`. These controls precede scientific outputs.

After each saved array file is closed, reopen it and require every array, shape, value, and data type to match. Recalculate saved logits and probabilities through the separate implementation. Recalculate the saved group metrics. Require an empty output folder and preserve old results. A nonfinite value, altered source hash, failed check, or resource limit stops the run.

## Compute and saved evidence

Use the parent runner's two-thread CPU setting and fixed software record. Use at most 4 GiB of peak preparation memory. Model training, calibration, final tests, and replay all share Stage 2C's two-hour cap. The parent also enforces the program cap and the stop at 20% remaining usage.

Preparation times two full-size dummy mask evaluations, 40 dummy rotation updates, one rotation selection evaluation, and one saved final-size evaluation with replay. The edit-work projection is:

```text
4 × [8 × (220 mask evaluations + 4,000 rotation updates
          + 4 checkpoint evaluations + 29 saved evaluations)
     + 60 seconds]
```

Each term uses its measured dummy time. The 29 saved evaluations allow three chosen selection outputs and all 26 possible final methods and controls. The factor of four covers variation. The 60-second allowance covers file hashes, input checks, draw saving, and summaries. Add the companion base-training estimate. Do not start a scientific phase unless the combined estimate fits the remaining stage and program limits.

Save raw logits and probabilities for every final test pair and every evaluated method or control. Save the common test ledger once. Save all candidate group scores, selected raw selection results, training history, hashes, numerical errors, and elapsed times. Internal raw outputs for every discarded mask need not be stored: all are checked during calibration, and the fixed code and inputs can regenerate them. Do not call an unavailable method complete.

The numerical-completion flag is separate from the scientific gates. A failed scientific gate is a result. These tests concern a small known selector task and a fixed intervention split. They do not establish a semantic mechanism in language models, universal variable isolation, or the absence of a useful explanation when a method fails.
