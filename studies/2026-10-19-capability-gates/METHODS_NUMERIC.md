# Digit and generated-point methods

This file fixes two task branches for Experiment 1. It contains no scientific results.
The public execution plan must be saved before these jobs run. The old studies remain unchanged.

## Common rules

Each branch has at most eight jobs: two recipes with two development seeds, then one selected recipe with four final seeds.
The digit job trains both required architectures. Thus it has at most 16 trained models. The flow branch has at most eight.
The root plan fixes development seeds 8001 and 8002, and final seeds 8101 to 8104.
The model seeds are the replicate units. Related test rows are not independent models.

The caller fixes two CPU threads and deterministic arithmetic. The caller also checks the public code lock before any scientific run.
Every training batch and evaluation batch checks the time limit and the private account-usage permit through `common.check_stop`.
Non-finite losses, gradients, weights, or outputs stop a job. A failed job retains its trace, error record, and partial weights where possible.
An existing output directory causes an error. There are no automatic retries or extra seeds.

Each complete job saves initial and final weights, every training loss, its input arrays, raw evaluation outputs, controls, times, and metrics.
A separate saved-weight calculation then checks all outputs and all gate decisions. It does not repeat training.
Private account-usage records do not form part of the public result.

## Digit images

The input files are the same UCI Optical Recognition of Handwritten Digits files used in Study 11.
`digits.py` checks both file hashes against that study. It uses 3,823 original training images and 1,797 original test images.
Pixel values are divided by 16. Each 8-by-8 image is placed at the center of a zero-filled 16-by-16 image.

Development uses only the original training set. For each digit class, a fixed random order selects the first floor(20% of the class count) for validation.
The split seed is 2026101901. Both recipes and both development model seeds use this same split.
Final models train on the full original training set. They are evaluated on the original test set.
**The original test images are never used for training.** This test set has already informed the project, so this is not a new-dataset test.

Both architectures have three circular 3-by-3 convolutions, ReLU outputs, two reductions by a factor of two, global average pooling, and a ten-class linear output.
Recipe A has 8, 16, and 32 channels, with 6,218 trainable parameters.
Recipe B has 16, 32, and 64 channels, with 23,946 trainable parameters.
The direct architecture takes every second position after each of the first two convolutions.
The filtered architecture first applies the fixed separable kernel `[1, 2, 1]` in both directions, divided by 16.

Training uses float32, 40 epochs, batches of 128, and Adam with learning rate 0.001.
Adam uses beta values 0.9 and 0.999, epsilon 0.00000001, no weight decay, and no AMSGrad.
Each epoch has a new shuffled order and circular image shifts sampled independently from -2 to +2 pixels on each axis.
The schedule seed is 10000 plus the model seed. The paired direct and filtered models use the same initial seed and schedule.
This retains the original training settings and extends its ten epochs by a factor of four.

Evaluation converts the saved float32 weights to float64. It tests original images and four separate circular one-pixel shifts: up, down, left, and right.
Every model must reach 95% accuracy on original images and 90% on each shift.
The root selects a recipe by its lowest development accuracy across both seeds, both architectures, and all five image groups.
Its shorter measured runtime breaks an exact tie. A final branch passes only if all eight final models pass all five groups.

Unchanged-input controls must reproduce the same logits exactly.
The replay computes convolutions from saved arrays without the training model class. It uses explicit source indices for image shifts.
All saved logits must match within 0.0000000001. All accuracy values and gate decisions must match.
This tests task ability only. It does not test an explanation, channel removal, or semantic stability.

## Generated two-dimensional points

The target is the Study 14 mixture of four isotropic Gaussian distributions.
Their centers are (-2, -2), (-2, 2), (2, -2), and (2, 2). Their respective weights are 0.1, 0.2, 0.3, and 0.4.
Each coordinate has standard deviation 0.55 within a component.

The flow has four alternating affine coupling blocks. Each block uses one hidden tanh layer.
Recipe A has width 32 and 520 parameters. Recipe B has width 64 and 1,032 parameters.
The scale is `1.5 * tanh(raw_scale)`. Each block's final linear weights and biases start at zero.
The transformation therefore starts as the identity. Training and evaluation use float64.

Each recipe uses 16,000 Adam updates, four times the old schedule, with batches of 256 sampled with replacement.
Adam settings match those used for the digit branch. The batch-index seed is 30000 plus the model seed.
The fixed training array has 16,384 points. Density evaluation uses 8,192 other points.
Generation uses 32,768 independent standard-normal latent points.
Development uses input seed base 2026101910. Final work uses base 2026101920.
Offsets 1, 2, and 3 generate the training points, density points, and latent points respectively.
Within each split, all models use the same task arrays. Development and final arrays are separate.

Excess negative log likelihood is the mean true log density minus the mean model log density on the evaluation points.
It is measured in nats. The limit is 0.20.
The four-region error is half the sum of absolute differences between generated quadrant frequencies and the exact target quadrant probabilities.
Its limit is 0.05. This is a coarse four-region measure, not a bound on the full continuous output distribution.

Both development seeds must meet the region-error limit before a recipe is eligible.
Of the eligible recipes, the root selects the lowest mean development excess negative log likelihood.
Shorter runtime breaks an exact tie. If neither recipe is eligible, the final flow branch stops.
Every final model must meet both limits for the branch to pass.

Each evaluation batch checks forward/inverse reconstruction within 0.000000001, cancellation of log determinants within that tolerance, and exact no-op agreement.
The replay computes all affine coupling equations from saved arrays without the training model class.
It checks all encoded points, log determinants, log densities, generated points, and metrics.
It uses a separate log-sum calculation for the true density and a separate Gaussian-tail calculation for quadrant mass.
Output tolerance is 0.000000001. Metric tolerance is 0.000000000001. Gate decisions must agree exactly.

## Preparation and compute estimate

Both preparation functions use random dummy arrays only. They do not evaluate scientific task performance.
Each times 20 training steps with Recipe B. The digit check tests both architectures.
The flow check tests the known identity at initialization, both transform directions, no-op behavior, and the determinant of the gradient matrix.
Both checks compare model outputs with the separate saved-array equations.

Digit cost uses 40 epochs with 30 full-training batches per epoch for each architecture.
It adds five image groups with 15 evaluation batches each, including no-op checks and replay.
Flow cost uses 16,000 training updates plus all density and generation batches, controls, and replay.
Each report gives the projected cost for one job and all eight jobs. The report multiplies the eight-job estimate by four for a safety allowance.
The root must add costs for the other two branches before deciding whether the complete plan fits the four-hour Experiment 1 cap.
These are hardware measurements with a safety allowance, not guaranteed runtimes. File writing and later review can add time.
