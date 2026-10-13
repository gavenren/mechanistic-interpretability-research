# Post-failure diagnostic plan

Prepared 13 October 2026 after the original run stopped. This is a diagnostic, not a change to the original analysis plan. No diagnostic model calls have run while this plan is prepared. The original failed result, tolerances, and runner stay unchanged. Do not resume training or complete the missing seeds under the original plan.

## Exact failure

The run fixed by commit `c406a08091cbc74caef1a1c4690aa29482667e0a` stopped in seed 2, direct architecture, shift (0,−4), evaluation batch 6, channel 1. Batch 6 contains original test rows 768 through 895. Channel indices start at zero. The maximum absolute logit difference was `2.1457672119140625e-5`, above the fixed `2e-5` symmetry tolerance. Four earlier models are complete. The fifth model and all its main-test logits are saved, but its structural controls are incomplete.

Use only this saved model and exact batch. Verify the result manifest, original code lock, raw test hash, saved checkpoint, and failure context. The checkpoint is `results/models/2-direct.npz`; the unshifted reference logits are condition 2 in `results/logits/2-direct.partial.npy`. Do not use other images to choose a diagnostic tolerance or method.

## Questions and fixed computation

1. Can a separate first-ReLU output hook reproduce the saved unshifted logits and the exact failed float32 comparison?
2. Does the expected circular-shift relation hold more closely with the same learned weights converted to float64?
3. Does a separate convolution and spatial-reduction implementation agree in float64?

Rebuild the direct network without importing `code/run.py`. Use a first-ReLU output hook that replaces channel 1 with zero. Build the input canvas independently from the raw numeric file, with NumPy padding and division by 16. Make the shifted input with NumPy circular indexing, not the original shift function.

Run the unshifted and shifted batch in four fixed conditions: (a) hook path, float32; (b) hook path, float64; (c) explicit unfolded-window convolutions, float32; (d) explicit unfolded-window convolutions, float64. The hook path uses standard convolution modules and spatial mean. The explicit path uses circular index arrays, `unfold`, a matrix product for each convolution, and separate sums over the two spatial axes followed by division by the spatial area. It zeros the selected channel with a channel mask. Biases and all saved weights are unchanged; float64 conditions only change numerical precision. They are separate computations, not ground truth for the original float32 program.

Save all batch logits in each condition. At each intermediate site, compare the shifted activations with the expected circular permutation of the unshifted activations. The required shifts are −4 input pixels before the first downsampling, −2 positions after it, and −1 position after the second downsampling. Pooled vectors and logits should be invariant in exact arithmetic. Record maximum error and its location at every site. Save removed-channel and untouched-channel errors. These local checks do not claim later states remain unchanged.

## Fixed checks and interpretation

- All arrays must be finite.
- Hook float32 unshifted logits must match saved logits within `1e-10` absolute error. The replayed symmetry maximum must match `2.1457672119140625e-5` within `1e-10`. This strict comparison asks whether the exact recorded failure is reproduced.
- Hook float64 and explicit float64 logit symmetry errors must each be at most `1e-10`.
- Hook and explicit float64 logits must agree within `1e-10` on both inputs.
- Removed-channel and untouched-channel errors must be zero in all four conditions.
- Float32 errors in the alternative convolution/reduction path are descriptive. Do not choose a tolerance for them after inspection or use them to replace the original failed check.

Save every check before any aggregate pass/fail decision. If a check fails, retain its result and all diagnostic arrays. A reproduced float32 failure plus small float64 symmetry errors would be consistent with numerical error in the float32 computation. It would not make the original run pass, validate missing models, or isolate one operation as the sole source of error. If replay differs or float64 checks fail, investigate the implementation before drawing that conclusion. Intermediate-site differences locate where errors appear, but do not by themselves prove a unique cause.

## Budget and authorization

Use the original NumPy 2.3.5 and PyTorch 2.8.0 environment, two CPU threads, one inter-op thread, deterministic algorithms, disabled MKL-DNN. Limit the diagnostic to five minutes. There is no training, no new seed, and no change to weights. The output directory must be new. Save plan and diagnostic-code hashes, input/model hashes, versions, timings, checks, and a manifest. A separate reviewer and root must approve the fixed diagnostic plan and code before execution. The result will be labelled post-failure.
