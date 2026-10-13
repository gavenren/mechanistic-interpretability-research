# Fixed plan: match a spike explanation to the deletion being tested

13 October 2026. AI-authored plan. No study model has been trained or evaluated. The plan, runner, input ledger, source record, requirements, and raw data must be fixed in a public commit before scientific output is collected.

## Question and limits

A spike sends a message to a later layer and resets its own voltage. Removing only its message is different from removing the spike on both paths. Does a gradient that includes the correct local paths select a spike with a larger exact deletion effect?

We compare two local gradient graphs with two exact deletion meanings in a fixed trained network. Both gradient graphs retain every later reset derivative. Only the selected event's immediate reset edge differs. This is not a comparison between globally detached and globally retained reset derivatives. The forward model and its weights are identical across explanation choices.

The sole primary outcome is the benefit of matching the gradient graph to the exact deletion meaning when selecting **one** event. It is a finite selection test. It does not establish a new causal principle, a correct semantic explanation, or exact gradients for a discrete network. The selected spike may have a negative or zero deletion effect. All such cases remain.

## Data and fixed model panel

Use the original UCI optdigits split, copied with attribution from the repository's shift-ablation data: 3,823 training images and 1,797 test images. Each row has 64 counts and one digit label. Divide the counts by 16. Use the 64 values directly, with no image augmentation and no alternate split.

The input ledger records both raw file hashes and every explanation-panel row. For each digit 0 through 9, take its first 12 rows in the original test-file order. Sort the resulting 120 distinct indices in file order. This panel is fixed using labels and row order only. It lies within test-file rows 0 through 127 and is not a random sample of the full test split. No model answer is used. Use all 1,797 test images for the ability check.

Train four seeds: 311, 313, 317, and 331. Keep every seed. Use CPU float64, two compute threads, one interop thread, deterministic algorithms, and disabled MKLDNN. No GPU or paid compute is used.

The model has 64 inputs, two spiking hidden layers of 32 units each, and a linear ten-class readout. Each affine layer has a bias. Its 3,466 trainable parameters use PyTorch Linear's default initialization after setting the model seed. There is no explicit within-layer recurrent matrix, normalization, or dropout. Voltage carries information between steps. Initial voltages are zero. The same input image is supplied at all eight steps.

Let `v1` and `v2` be post-reset voltages. At each step `t = 1,...,8`, in this order:

```
u1[t] = 0.9 * v1[t-1] + W1*x + b1
s1[t] = 1 if u1[t] >= 1, else 0
m1[t] = s1[t]
v1[t] = u1[t] - s1[t]
u2[t] = 0.9 * v2[t-1] + W2*m1[t] + b2
s2[t] = 1 if u2[t] >= 1, else 0
v2[t] = u2[t] - s2[t]
logits = W3 * mean_t(s2[t]) + b3
```

The threshold and reset amount are both one. This is a soft subtractive reset. A voltage can remain above threshold after a spike; at most one spike is emitted per step. These are discrete simulation steps, not measured biological times. The non-spiking readout has no reset.

For backward computation only, replace the threshold derivative with `g(u) = 1 / (1 + 10*abs(u-1))^2`. Its peak is one. This rule is fixed for training and both explanation graphs. All reset paths are included during training. The binary forward rule is unchanged by this surrogate.

Train each model for exactly 40 epochs, 30 batches per epoch, and 1,200 updates. Batch size is 128; the last batch has 111 images. Use Adam with learning rate 0.001, betas `(0.9,0.999)`, epsilon `1e-8`, zero weight decay, and `amsgrad`, `foreach`, `fused`, `maximize`, `capturable`, and `differentiable` all false. Minimize mean cross-entropy of the final ten logits. Use no scheduler, clipping, activity penalty, early stopping, or parameter selection. For each seed, NumPy generator `10000 + seed` produces one fresh permutation per epoch. Save all orders and every training loss.

The exact training budget is 4,800 updates and 611,680 eight-step training examples. Fixed training is not a claim that it will produce strong models.

## Exact deletion queries and matching gradients

The candidates are all 32 first-layer spike nodes at the **fourth update**, which is zero-based index 3. The panel includes inactive candidates. Freeze the weights after training. For each image, fix its intact winner and runner-up by descending logit, breaking ties by smaller class index. All effects use that same winner-minus-runner-up margin. Labels are used for accuracy, not for changing this margin after an edit.

At the fourth update, an exact **message deletion** sets the chosen `m1` entry to zero and leaves its `s1` reset entry unchanged. An exact **node deletion** sets both entries to zero. No future voltage or spike is replaced with a reference value. The later recurrence runs freely in both cases. The first layer has no feedback from layer two, so message deletion should preserve the complete first-layer trajectory; this is also a control.

For the two gradient graphs, replace the fourth-update spike vector by a leaf vector with identical forward values. In the message graph, the leaf supplies `m1`, while the immediate reset uses the original detached spike vector. In the node graph, the leaf supplies both `m1` and the immediate reset. Every future spike uses the fixed surrogate and every future reset remains in the graph. Prefix quantities are fixed. Compute the derivative of the original margin with respect to each leaf entry. The predicted removal effect is that derivative multiplied by the original binary spike. This graph change must leave every forward voltage, spike, message, and output unchanged.

For query `q` and node `i`, save the exact margin drop `E[q,i] = intact_margin - deleted_margin`. For graph `g`, select `i_g` with the largest signed predicted drop; break ties by unit index. Do not take an absolute value and do not discard inactive, negative, or zero effects. Define `R[q,g] = max_i E[q,i] - E[q,i_g]`. This is regret for one selection among the 32 fixed candidates. It is not an unrestricted circuit-search optimum.

The per-image primary score is

```
A = 0.5 * (R[node,message] + R[message,node]
           - R[node,node] - R[message,message]).
```

Positive `A` favours matching. The directional hypothesis is that the mean over fixed seeds is positive. Matching is not guaranteed to win: the surrogate is approximate. The exact maxima cancel, so the same score is

```
A = 0.5 * (E[node,i_node] - E[node,i_message]
           + E[message,i_message] - E[message,i_node]).
```

Thus the primary tests the benefit of matching the selection rule, not absolute fidelity. Report both expressions as an arithmetic check. Report the full 2×2 regret table and both directional gains separately for every seed. A positive mean can hide a loss for one deletion meaning.

Average over all 120 images within each seed, then average the four seed means with equal weight. Use 10,000 whole-seed resamples from NumPy generator 2026101501 and a linear 2.5%/97.5% percentile interval. Report all four seed values. This interval is descriptive and conditional on this small fixed model panel and data design; it does not cover data or design uncertainty. Do not count images or spike candidates as independent trained models.

## Secondary measures and ability gate

Save each exact effect, gradient, selected index, winning class, runner-up, and complete ten-logit output. Report active candidate counts, whether any exact effect is positive under each query, zero-effect counts, selection agreement, selected exact effects, absolute regrets, and wrong-sign fractions among active candidates with nonzero predicted and exact effects. Also report exact-effect differences between queries. Zero-denominator fractions are null with their counts, never silently dropped.

Select the top four candidates under each gradient graph with the same signed rule and index tie break. Delete those four together under each exact query and measure the full joint effect. Also test one four-node random set per image from NumPy generator `20000 + model_seed`, sampling without replacement in panel order. Use the same set under both exact queries. These six joint conditions are secondary. Their effects are measured directly, not inferred by adding singleton effects. They are not joint-optimal regrets: we do not search all four-node combinations.

Report all 1,797-image test accuracies and all 120-image panel accuracies. The fixed ability gate is at least 80% full-test accuracy for every seed. A failed seed remains in every primary table and interval. If any seed fails, task-level claims must state the failure prominently and remain about the tested networks. Also give a descriptive correct-at-baseline panel summary without treating it as the primary or as a matched sample across seeds. No extra training or alternate model will be substituted.

## Controls and output completeness

Use fixed float64 error tolerance `1e-10` for algebraic score and independent tangent checks. Exact no-op and untouched-value checks have zero tolerance. Report measured values before any assertion. Fail on non-finite data, logits, loss, weights, or gradients. Keep all failed records and stop; do not raise a tolerance.

Controls include:

1. Both leaf graphs reproduce the complete intact forward traces and logits exactly.
2. Self replacement under both deletion meanings is a complete no-op.
3. All pre-intervention traces are unchanged, every targeted message is zero, and all other first-layer values at the edit step are exact. Node deletion also zeroes the reset spike. Message deletion preserves the entire first-layer voltage and spike trajectory.
4. Inactive singleton deletions are complete no-ops. Retain their count and outputs.
5. An independent local tangent recurrence on recorded voltages agrees with both autograd leaf derivatives. It uses the same specified surrogate slopes and keeps future reset terms; it is not finite differencing the discontinuous forward program.
6. A second-layer spike at the final eighth update has a linear readout. Its exact margin deletion effect agrees with the direct readout coefficient. This control is not applied to a first-layer event, which still passes through a later threshold.
7. Both primary-score formulas agree, regret is nonnegative within `1e-10`, and the primary aggregation agrees with its saved per-image rows.

Save full binary first-layer spikes, outgoing messages, and second-layer spikes for all baseline, singleton, joint, and self-replacement conditions. Save baseline voltages and both leaf-gradient forward traces. Save all logits, coefficients, event effects, row-level measures, tables, package versions, initial and final weight hashes, schedules, input indices and labels, code and plan hashes, and timing counts. Save a final file-hash manifest. Check public locked bytes both before work and before completion. Refuse an existing output directory.

The planned full model-call count is 5,160: 4,800 training calls, 60 full-test calls, and 300 panel calls. Each model has 75 panel calls: one baseline, two leaf-gradient forwards, 64 singletons, six joint edits, and two self replacements. The panel has 36,000 forwarded examples, and the full test has 7,188. Including training, the total is 654,868 eight-step forwarded examples. Tangent and direct readout calculations are additional arithmetic, not full model calls. The expected records are 1,688 controls, 480 per-image rows, 30,720 singleton-effect rows, 1,920 regret rows, and 2,880 joint-effect rows. All output counts must be checked by the runner. Partial output arrays keep every completed panel condition if a later check fails; a separate last-forward file preserves the failing condition.

## Execution and review

The whole scientific run has a 1,200-second cap. Keep partial logs and the active model on failure. No resume, seed change, epoch extension, or threshold change is allowed under this plan. Before public lock, use only dummy inputs to test batch sizes 128 and 111 in training, 128 and 5 in full-test evaluation, and 120 in the explanation panel. Verify all edit controls and tangent gradients. A separate hand-set dummy witness must contain an active and inactive candidate, an active final readout spike, and a nonzero reset-path effect. Its two deletions must give the same immediate outgoing message while differing in the later first-layer state. Count its calls separately from the timing grid. Use a timing projection with a two-fold safety factor; fix any engineering issue before the public lock. Dummy runs are not task-capability evidence.

A separate reviewer will check the plan and runner before lock. After the fixed run, a separate implementation will replay saved weights, exact discrete queries, gradient/tangent arithmetic, and the full primary summary. It will not be described as a repeat of training unless training is actually repeated. The paper will retain weak results and failed controls. All authorship and review are AI; no human peer review is claimed.
