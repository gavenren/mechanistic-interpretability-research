# Do image filters stabilize the measured effects of channel removal?

**Status:** fixed plan prepared on 13 October 2026, before scientific training or test outputs. A public Git commit must lock this file, the runner, requirements, and raw data before the study runs. Local preparation is not external registration. The study is AI-authored. A separate AI review is required. There has been no human review.

## Question and scope

Low-pass filters can reduce the change in a convolutional model's output when an image moves. We ask whether this known architecture change also reduces the change in the **full output effect of removing an internal channel**. We compare two small training procedures on real digit data. We do not introduce a new filter, attribution method, or principle of causal explanation.

The same channel index is compared across shifts **within one trained model**. Paired initialization does not establish a semantic match between channels in the two architectures. The primary result averages all eight edited channels. The learned weights can differ. The experiment does not isolate the effect of adding a filter to one fixed set of trained weights.

A lower effect-change score can follow from weaker channels, lower confidence, or more stable outputs. It does not by itself show stable semantic units. We therefore report effect strength, ordinary output change, ablated output change, and task accuracy beside the primary score. No outcomes will select seeds, channels, images, or metrics.

## Data and fixed split

Use the original UCI Optical Recognition of Handwritten Digits files. The data creators are E. Alpaydin and C. Kaynak (1998), DOI [10.24432/C50P49](https://doi.org/10.24432/C50P49). The UCI page gives a CC BY 4.0 licence. Copies with attribution are in `data/`. The original split has 3,823 training images from 30 writers and 1,797 test images from 13 other writers. Individual writer IDs are absent. We use the full original split without validation selection, relabelling, or test-based filtering.

| File | SHA-256 |
|---|---|
| optdigits.tra | e1b683cc211604fe8fd8c4417e6a69f31380e0c61d4af22e93cc21e9257ffedd |
| optdigits.tes | 6ebb3d2fee246a4e99363262ddf8a00a3c41bee6014c373ed9d9216ba7f651b8 |

Each input is an 8×8 grid of integer counts in 0..16. Divide by 16, then place it at rows and columns 4 through 11 of a zero 16×16 canvas. The tenth class is digit 9; labels run from 0 to 9. Use no interpolation or further normalization. Test row IDs are the original zero-based row numbers. Training file size, test file size, column count, value ranges, and hashes must pass before training.

## Models and training

Train 16 models: two architectures for each seed 0, 1, 2, 3, 4, 5, 6, and 7. Both use this sequence:

1. A 3×3 convolution from 1 to 8 channels, circular padding of 1, bias, then ReLU.
2. First downsampling stage.
3. A 3×3 convolution from 8 to 16 channels, circular padding of 1, bias, then ReLU.
4. Second downsampling stage.
5. A 3×3 convolution from 16 to 32 channels, circular padding of 1, bias, then ReLU.
6. Mean over both spatial axes, then a linear map from 32 to 10 outputs with bias.

The **direct** architecture takes every second row and column, starting at index zero, at each downsampling stage. The **filtered** architecture first applies a depthwise 3×3 filter with weights `outer([1,2,1],[1,2,1])/16`, using circular padding of 1, then takes the same even rows and columns. The filter is fixed and has no trainable parameter. There is no dropout, batch normalization, other normalization, learned position input, or pooling nonlinearity.

Before each model is created, call `torch.manual_seed(seed)`. Use the standard PyTorch 2.8.0 initialization from `Conv2d` and `Linear`, in the order above. Save and compare one aggregate SHA-256 value that covers the name and bytes of every initial parameter. The two architectures in a pair must have identical initial parameter bytes. Save every final checkpoint. The fixed filter is a buffer and is the same in both stored state dictionaries; the direct model does not use it.

Train each model for exactly 10 epochs. Each epoch uses each of the 3,823 training rows once. Batch size is 128; the final batch has 111 rows. There are 30 updates per epoch, 300 per model, and 4,800 total. The loss is mean cross-entropy. Use Adam with learning rate 0.001, betas (0.9, 0.999), epsilon 1e-8, no weight decay, no AMSGrad, no maximization, and `foreach=False`, `fused=False`, `capturable=False`, `differentiable=False`. Start with a new optimizer for each model. Do not use a scheduler, gradient clipping, early stopping, or checkpoint selection.

For each seed, create a separate NumPy PCG64 generator with seed `10000 + seed`. For each epoch, first draw a permutation of all training row IDs. Then draw an array with shape (3823, 2) of integers from −2 through +2, using `rng.integers(-2,3,dtype=np.int8)`. The two columns are vertical and horizontal shifts. Assign this array to the **ordered training positions**, not the original row IDs. Apply shifts by exact circular index permutations. The two architectures reuse the same saved permutation and shift arrays. Positive vertical shift means down; positive horizontal shift means right. Save all schedules. The tested digits do not cross the canvas boundary under these shifts.

Use CPU float32, two CPU threads, one inter-op thread, deterministic algorithms, and disabled MKL-DNN. Require NumPy 2.3.5 and PyTorch 2.8.0. Do not use a GPU or paid service. Save the Python and package versions. The only reported training outcomes are the full loss log and fixed final checkpoints; none change the design.

## Fixed tests and intervention

After training each model, evaluate all 1,797 test images at these five shifts, in this order:

`(0,0), (-1,0), (1,0), (0,-1), (0,1)`.

Evaluate the intact model and eight interventions for every input. Intervention c sets all spatial entries of first-ReLU channel c to zero, for c from 0 through 7. It leaves all other channels at this site unchanged. It occurs before the first filter or direct downsampling stage. The later network then runs normally. Removal can create an unusual internal state; its effect is a causal effect of this specific edit, not a guarantee of human meaning.

Save all float32 output logits in arrays with shape `(5 shifts, 9 conditions, 1797 images, 10 classes)`, one compressed file per model. Condition zero is intact; condition c+1 removes channel c. Do not save a second probability archive. Instead, define the probability calculation exactly: cast saved logits to NumPy float64, subtract the maximum along the final class axis, take `np.exp`, and divide by the float64 class sum. This definition avoids ambiguity about float32 softmax records. Save test labels once. All raw signed effect vectors can be rebuilt from these logits.

## Primary comparison and uncertainty

Let p(x) be the full ten-class probability vector. Let p_c(x) be the vector after channel c is removed. The signed effect is `e_c(x) = p(x) - p_c(x)`. Its entries can be negative and sum to zero; it is not a probability distribution.

For each model, define

`D = mean over 1797 images, 8 channels, and 4 one-pixel shifts of 0.5 * sum(abs(e_c(shift(x)) - e_c(x)))`.

D is a half-L1 change between signed effect vectors. Its range is 0 through 2. It is not itself total variation between two probability distributions. The sole primary estimate is the seed-paired mean of **D_direct minus D_filtered**. A positive value means lower effect change for the filtered training procedure in this fixed comparison.

Report all eight paired seed differences and their mean. Use 10,000 whole-seed bootstrap draws with NumPy PCG64 seed 2026101417. Each draw samples eight paired seed IDs with replacement. Compute the 2.5th and 97.5th percentiles with NumPy's `method='linear'`. This interval is conditional on this fixed data split and training design. It does not cover uncertainty across writers, data sets, architectures, or tasks. Eight training seeds are the replications. Images, channels, and shifts are not independent model replications. Do not make a stronger general claim from the large raw cell count.

## Required secondary results

Report each architecture, seed, digit class, channel, and shift without choosing a favourable subset. For each model report:

- Original and shifted digit accuracy; top-1 prediction consistency under shifts.
- Baseline output shift TV: `0.5 * ||p(shift(x)) - p(x)||_1`.
- Ablated output shift TV: `0.5 * ||p_c(shift(x)) - p_c(x)||_1`.
- Original-image ablation effect size, and the effect size over all five shifts: `0.5 * ||e_c(x)||_1`.
- Per-channel D, original effect size, and ablated output shift TV; per-shift D, accuracy, and both output shift TV terms; per-digit D and original accuracy.
- The size and D of the fixed subgroup for which the intact model is correct under all five shifts. This is a labelled secondary description. Each model can have a different subgroup. It is not an architecture-matched estimate. The full test set remains primary.

The triangle inequality gives `D_c <= TV(p(shift),p(base)) + TV(p_c(shift),p_c(base))` for each cell. Report both terms and their mean sum; check the inequality numerically. This is known algebra, not a discovery. Lower D alone cannot show that a semantic feature is more stable. In particular, small effect sizes or weak digit accuracy can limit its use. Do not divide by small effects, filter cases by the observed effect, tune a capability threshold, or match outcomes after the run.

## Controls and fixed tolerances

All checks use absolute errors, with no relative term. A failed check stops scientific interpretation. Preserve the failed receipt and partial files. Do not raise tolerances after results.

1. For every model, test image, and five main shifts, a no-op clone at the edit site must reproduce intact logits exactly (tolerance zero).
2. At every channel-removal forward, the edited channel must be exactly zero and all other channels at this site must be exactly unchanged. These local checks do not claim later states are unchanged.
3. For every model, image, and main shift, compare first-ReLU features with the corresponding integer permutation of the original features. Maximum absolute error must be at most 1e-5.
4. For every model, image, and all nine conditions, also shift by `(-4,0), (4,0), (0,-4), (0,4)`. Each full ten-class output must match its unshifted counterpart within 2e-5. The two stride-two stages, circular operations, whole-channel edit, and global pooling imply this symmetry. It is an expected structural control, not a new result.
5. Each float64 probability row must sum to one within 1e-12. The triangle inequality must hold within 1e-12.
6. Before locking, use only random dummy images to test the real training batch sizes 128 and 111 and evaluation batch sizes 128 and 5. Check vectorized augmentation against a separate scalar circular-roll implementation exactly. Use both architectures. Test an untrained stride-one version under every one-pixel and four-pixel shift, with all nine conditions. Its output must be invariant within 2e-5.
7. After results, a separate reviewer will reconstruct shifts and score arithmetic without importing the runner. The reviewer can apply the edit with a first-ReLU output hook and compare saved logits using the fixed 2e-5 output tolerance. Full score reconstruction will use saved logits and the specified float64 calculation. Its scope and any failure will be disclosed.

Control tables record the maximum error over every batch, including all images and classes. This bounds each covered cell; it does not discard failures. The four-pixel outputs do not need a second large archive because the model weights, raw inputs, shift code, and complete error table permit replay.

## Resource limit, preservation, and output

The scientific run has 45,632 control rows and 25,440 full model calls: 4,800 training calls and 20,640 evaluation calls. It has 3,084,352 forwarded examples, including 611,680 training examples. Two extra first-convolution/ReLU calls per main test batch check feature shifts; these 2,400 partial-network calls are not counted as full model calls. The 20,640 evaluation calls include all primary logits, no-op runs, and four-pixel controls. There are 1290 evaluation calls per model. Evaluation batches have 128 rows except the final batch of 5.

A dummy-only run will estimate time from 30 updates and a complete 133-image evaluation per architecture. Scale the measured training time by `611680 / (2*15*239)` and evaluation time by `2472672 / (2*133*86)`, the exact forwarded-example ratios. Apply a two-fold safety factor to their sum. This is a time estimate, not a guarantee. Keep all eight seeds if this estimate is within the 20-minute limit. Any design change for speed must precede the public lock. After the lock, use a hard 1,200-second cap, checked during work and before final completion. Stop on time, numerical, or non-finite failure. Never reduce seeds or cases after seeing results.

Refuse an existing output directory. Write training losses and controls as work proceeds. During each evaluation, keep a partially filled logit array with NaN entries for unmeasured cells. Flush it after each batch. After its complete compressed archive is saved, remove only that temporary duplicate. Preserve partial arrays and the current model checkpoint on failure. Save every completed model's summary and checkpoint. The runner records a failure file and hashes surviving artifacts if a run fails.

Before training and after all summaries, resolve the supplied full lock SHA and compare exact bytes for this plan, the runner, requirements, and both raw data files with that commit. Check expected model, update, call, and example counts. A complete manifest hashes every output file. No publication statement is justified until the remote result commit is verified.

## Prior work and limits

The main prior work is Zhang (ICML 2019) on anti-aliased CNNs, Desai and Ramaswamy (WACV 2020) on channel ablation for visual explanations, and Ahmad and colleagues (WACV 2024) on stable causal graphs and explanations. Sources and inspected sections are recorded in `sources.md`. A focused search found no exact match to this fixed architecture comparison and full signed-effect endpoint. This does not establish priority.

The task has small count images, a small CNN, circular padding, and a fixed original split. It is not evidence for language models or high-resolution vision systems. Low-pass filtering changes learning, feature strength, confidence, and computation. Channel indices need not have the same meaning across architectures. The experiment measures a chosen intervention, not the complete mechanism of digit recognition. Negative and mixed results are equally eligible for the paper.
