# Training order and internal shortcut paths

Date: 13 October 2026. This is a fixed plan for a small constructed-model study. No scientific training or test outcome was examined before this plan and its runner were locked on GitHub. A lock is a public record of the proposed analysis, not independent registration or human review.

## Question

If two networks see the same examples the same number of times, can their order leave different internal shortcut effects after a common balanced training tail? Does that difference remain when momentum is reset at the start of the tail?

The directional hypothesis is that the last of two conflicting shortcut regimes leaves a larger signed shortcut path effect. The primary endpoint follows 2,000 common tail updates. Earlier checkpoints cannot replace it. Zero and opposite-direction results will be reported. This study does not test permanent memory, an irreversible critical period, or language-model behavior.

## Data, model, and training

The complete input space consists of all 32 five-bit vectors in lexicographic order, with entries in `{-1,+1}`. Name the coordinates `x1,x2,z,n1,n2`. The signed label is `y=x1*x2`; binary cross-entropy uses `(y+1)/2`. The shortcut `z` and both nuisance bits have no effect on the oracle label.

Regime A is the 16 examples with `z=y`. Regime B is the 16 with `z=-y`. Every update uses a complete regime batch, which has equal label counts and every combination of relevant and nuisance bits. The common tail uses all 32 examples on every update. Thus label recency and unequal example counts cannot explain the order contrast.

The network is `5 → 32 → 32 → 1`, with biases, exact GELU activations, and one output logit. It has 1,281 parameters. Use PyTorch's default Linear initialization after setting each seed, with all parameters created directly in float64. Use CPU float64 for training and evaluation, two CPU threads, deterministic algorithms, and no dropout. No model or learning-rate search is permitted.

Seeds: `1103, 2207, 3301, 4409, 5519, 6607, 7717, 8803`. Within a seed, copy exactly the same initial weights for three schedules:

- `AB`: 500 A updates followed by 500 B updates.
- `BA`: 500 B updates followed by 500 A updates.
- `ALT`: 500 alternating A/B pairs, starting with A.

Use SGD, learning rate 0.05, momentum 0.9, dampening 0, no Nesterov term, and no weight decay. The rate stays constant. Every schedule exposes every example exactly 500 times in its 1,000-update prefix. ALT is a reference; the primary contrast is BA minus AB. ALT's last batch is B, which must be stated when interpreting it.

At prefix update 1,000, make two copies of the model and its optimizer. `keep` retains all momentum buffers. `reset` zeros only those buffers. The copies must have exactly equal weights. Each receives the same 2,000 full-batch tail updates. The complete study has 24 shared prefix runs and 48 final models: `24×1000 + 48×2000 = 120,000` updates. Do not drop a seed, extend a schedule, or change the rate after an outcome is known.

Save initial weights, prefix-update-500 weights and momentum, and all tail conditions at tail updates 0, 200, and 2,000. The tail-0 checkpoints also preserve the complete prefix endpoint. Save aggregate cross-entropy from every training update, plus per-example intact and intervention results at all three tail checkpoints. Training loss is measured before each update; checkpoint loss is measured after the named number of updates.

## Internal intervention

Let the first-layer preactivation be `a(x)=W1*x+b1`. For each receiver input x, its donor differs only by replacing z with -z. Replace selected entries of the receiver preactivation with the donor entries, then apply GELU and the rest of the frozen network. This changes only paths from z into those first-layer units. It does not identify pure semantic neurons.

For each model and checkpoint, select eight paths using calibration contexts only. Calibration inputs have nuisance pairs `(-1,-1)` or `(+1,+1)`; test inputs have `(-1,+1)` or `(+1,-1)`. Each split has 16 directed receivers and includes all relevant-bit values and both z-flip directions. All natural inputs are used in training. This is a held-out intervention-context test, not an unseen-input generalization test.

Rank the 32 individual paths by mean absolute change in correct-class probability over the 16 calibration receivers. Sort descending by the stored float64 score, with exact ties resolved by lower unit index. Select the top eight and apply them jointly to all inputs. Save all 32 calibration scores and the selected indices. No test outcome selects a unit, layer, group size, or checkpoint. Each model gets its own calibration-selected mask; the primary endpoint is the result of this fixed selection rule, not a comparison of identically numbered units across models.

Select the mask again at each checkpoint. A later path effect is not evidence that the same eight units kept the same role during training. The study tests the later result of a fixed selection rule.

For baseline logit l and patched logit l', let `q=sigmoid(y*l)` and `q'=sigmoid(y*l')`. Define the oriented probability effect as:

`e = -(z*y) * (q' - q)`.

Positive e means that the route favors a shortcut that agrees with the correct label. Also save the corresponding oriented logit-margin effect `-(z*y)*y*(l'-l)`. Do not divide by a small effect. Report un-oriented changes as well.

Evaluate each single path, the selected eight jointly, their 24-unit complement, all 32 paths jointly, and these eight fixed random eight-unit masks:

```
[0,6,14,15,17,20,23,25]
[10,13,16,20,21,23,25,28]
[3,6,8,11,17,19,20,26]
[2,7,10,11,14,22,30,31]
[3,8,14,20,22,25,27,30]
[0,2,4,11,15,16,20,31]
[1,2,3,4,5,19,27,28]
[0,6,7,11,14,17,23,30]
```

These masks are fixed descriptive controls. Do not choose among them using results. Report intact accuracy, cross-entropy, margins, weights, momentum norms, and the full z-input contrast. A partial-route effect can remain while omitted routes cancel it. Do not call it whole-model shortcut reliance unless the full input contrast supports that claim. Single-path effects need not add.

## Primary endpoint and report

For each seed, average the selected-eight oriented probability effect over the 16 test receivers at tail update 2,000 in the reset condition. Subtract the AB mean from the BA mean. The sole primary endpoint is the mean of these eight paired seed differences.

Show all eight differences, their median and range. A descriptive 95% percentile interval uses 10,000 paired seed resamples, NumPy PCG64 seed 161803, and linear quantiles at 0.025 and 0.975. The seed is the resampling unit. Rows, contexts, units, and schedules are not independent replicates. The interval describes this small set of initialization outcomes; it is not a confidence claim about other tasks, architectures, or arbitrary seed populations. No significance test is planned.

Fixed secondary results are the same BA-minus-AB contrast with momentum kept, the paired difference of reset and keep contrasts, and the ALT reference. Show all three checkpoints, all schedules, both momentum conditions, all intact-task measures, and all intervention sets. Report probability effects and logit-margin effects together to expose saturation. Equal accuracy does not imply equal loss or confidence. If intact performance differs, do not claim different mechanisms at equal performance.

Treat absolute primary differences below `1e-8` as numerically negligible for the report. This is a fixed description threshold, not a control tolerance or a minimum practically useful effect. Do not select an alternative endpoint if effects are negligible.

## Controls

- Verify exact initial-weight equality across schedules, and exact model and momentum equality at each fork before reset. After reset, weights must still match, all reset buffers must be zero, and retained buffers must match the prefix.
- Hash the full batch sequences. Independently count per-example exposures and per-update label counts. Confirm that all tails have the same batch sequence.
- Self replacement must reproduce baseline logits. Full preactivation replacement must reproduce logits from the true z-flipped input. Restore the saved receiver entries after a selected donor assignment, and verify the restored vector and logits. Assigning the same donor twice is not a restoration test.
- Compute the preactivation change by both direct donor-minus-receiver vectors and `-2*z*W1[:,z]`. The complete vector and all set assignments must agree within tolerance.
- Flip x1 as a label-sensitive positive task control. The oracle label must reverse. Report model accuracy on those contrasts; do not require learning success or exclude failures.
- Evaluation controls use an absolute `1e-10` logit/vector tolerance. Exact copy, counts, and zero-buffer controls require exact equality. Set these bounds before results. Validate only on non-study dummy inputs before the lock.
- Save failures and partial rows. A failed control prevents a completed scientific run. Do not loosen tolerances after results. An independent audit will reconstruct all checkpoint outputs and interventions without importing this runner, and check the complete tables and primary arithmetic.

## Resources, stop rule, and reproducibility

Use CPU only, two threads, and a 20-minute wall-clock cap for the complete scientific run. Check the cap at least every ten updates and each evaluation. If it is reached, save the current weights and optimizer state, preserve all completed rows, and mark the run incomplete. A partial run does not support the planned endpoint. No hidden retry, subset, or training extension is allowed.

The runner must compare the local plan and its own bytes against the full immutable Git commit before and after the run. Root must verify the public remote lock before starting it. Record package versions, platform, seed list, dtypes, code/plan hashes, completed update counts, wall time, checkpoint hashes, and a result manifest. A preflight may test shapes, batch counts, and intervention algebra on a separate dummy network and non-study inputs; it must not train or evaluate a study model.

## Scope and prior work

This is a synthetic test of finite training history, not a new localization method. It does not show that a finite tail reaches an optimum or that any difference is permanent. Momentum reset removes immediate optimizer-buffer carry-over, but weights still encode earlier updates. The remaining causal contrast includes all weight differences induced by the schedule. The study does not identify a unique biological or semantic mechanism.

Close work includes Patterning (Wang and Murfet, 2026), Mechanistic Data Attribution (Chen et al., 2026), Easing Optimization Paths (Odonnat et al., 2025), Critical Learning Periods (Achille et al.), non-stationary grokking (Lyle et al., 2026), Process-Tensor Tomography of SGD (Sevetlidis and Pavlidis, 2026), and optimizer-reset work by Nikishin et al. and Asadi et al. Sources and overlap are recorded in `sources.md`. The proposed contribution is this matched-example, common-tail, internal-path benchmark. A focused source search does not establish priority.
