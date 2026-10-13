# Fixed plan: likelihood damage and generated region change after unit removal

Prepared 13 October 2026. No scientific model has run. A public commit must fix this plan, runner, requirements, and source record before training. This is AI-authored research. A separate AI review is required. No human review is claimed.

## Question, prior work, and estimand

Does choosing a neuron by the damage its removal causes to calibration likelihood also identify a neuron that strongly changes generated quadrant frequencies? We compare that choice with a choice made directly from calibration quadrant frequencies. Test data and latent samples are separate. This is a bounded selection benchmark, not a new method for editing generators.

Real NVP already provides invertible density models. Theis and colleagues already explain why likelihood and sample criteria need not agree. Bau and colleagues already use internal-unit interventions in generators. Taming Normalizing Flows is particularly close: it already uses a two-dimensional Gaussian-mixture example, compares likelihood and generated attribute counts, and holds latent samples fixed across model edits. It fine-tunes weights and ablates loss terms. We instead freeze trained weights, evaluate every specified single hidden-unit removal, and compare two calibration selectors on independent test samples. The source record states the inspected method sections. A focused search does not prove novelty.

There is a standard local reason to distinguish the objectives. For a differentiable family of normalized densities q_a, if p=q_1 and differentiation can pass under the integral, then the derivative at a=1 of E_p[-log q_a] is minus the integral of the derivative of q_a, hence zero. The derivative of a region probability, integral_R q_a, can be nonzero. This is the usual mean-score identity, not a new theorem. It says nothing certain about full removal (a=0), an imperfectly fitted model, finite samples, or a neuron's semantic meaning. No gate-amplitude sweep will be added.

Quadrant-frequency TV is a **coarse output measure**. It is not TV between the full continuous densities. Movement inside a quadrant and two-way boundary crossings can be missed. Report those changes separately. A gain for the quadrant selector would not make likelihood a worse measure of density fit.

## Fixed inputs

The constructed data distribution is a four-component Gaussian mixture. Component means, in order, are (-2,-2), (-2,2), (2,-2), and (2,2). Their weights are (0.1,0.2,0.3,0.4). Each covariance is 0.3025 I, or standard deviation 0.55 per axis. Use float64 arrays.

Generate each data split with a separate NumPy PCG64 generator. Draw component labels with rng.choice(4, size=n, p=weights), then draw an n-by-2 standard-normal array and scale/add the selected means.

| Array | Count | PCG64 seed |
|---|---:|---:|
| Training data | 16,384 | 26401 |
| Calibration data | 4,096 | 26402 |
| Test data | 8,192 | 26403 |
| Calibration standard-normal latents | 16,384 | 26404 |
| Test standard-normal latents | 32,768 | 26405 |

Save the arrays, component labels, and their hashes. All models use the same arrays. Data labels are for description, not supervised training. Quadrant index is 2*(x0>=0)+(x1>=0), with zero on the positive side. Quadrants and mixture component labels are not identical. Calculate the exact target quadrant probabilities with independent one-dimensional Gaussian CDF factors using math.erf, and save them.

Draw one common array of 4,000 batches of 256 training row IDs using rng.integers(0,16384,size=(4000,256),dtype=np.int32) from PCG64 seed 26406. Sampling is with replacement. Save and use the same batch-ID array in every model. No input selection may use model outcomes.

## Models and training

Use four model seeds: 26501, 26519, 26539, and 26561. Each model has four affine coupling blocks. In encoding order, blocks 0 and 2 keep coordinate 0 and transform coordinate 1; blocks 1 and 3 keep coordinate 1 and transform coordinate 0. Each conditioner is Linear(1,32), tanh, Linear(32,2), with biases. The first output is a raw log-scale and the second is a shift. Define s=1.5*tanh(raw_scale). Encoding maps the changed coordinate v to exp(s)*v+t and adds s to log|det J|. Decoding applies blocks in reverse, maps v to (v-t)*exp(-s), and adds -s to its log determinant.

There are 520 trainable parameters and 128 eligible hidden scalars per model. The base distribution is a two-dimensional standard normal. Log density is the base log density at the encoded point plus the encoder log determinant. There is no dropout, batch normalization, trainable permutation, scale regularizer, or clipping of inputs or gradients.

Before construction, call torch.manual_seed(model_seed). Construct both Linear layers of each block in block order with PyTorch defaults; promote the default float32 parameters to float64, then set the second Linear weight and bias in each block to zero. Thus the initial overall map is identity, but first-layer parameters differ by seed. Save the complete initial state.

Use exactly 4,000 steps of Adam, learning rate 0.001, betas (0.9,0.999), epsilon 1e-8, weight_decay=0, amsgrad=False, maximize=False, foreach=False, fused=False, capturable=False, differentiable=False. The loss is mean negative log density on each fixed batch. No scheduler, early stop, or seed replacement. Save every loss and states at steps 0, 1000, 2000, 3000, and 4000. No training extension after poor fit.

Use Python 3.12, NumPy 2.3.5, PyTorch 2.8.0, CPU float64, two intra-op threads, one inter-op thread, deterministic algorithms, and disabled MKL-DNN. No paid service or GPU.

## Persistent hidden-unit removal

Condition 0 is the intact flow. Conditions 1 through 128 remove one post-tanh scalar: condition 1+32*b+u removes unit u from block b, for b=0..3 and u=0..31. Set that scalar to zero at every call of this conditioner. Keep every other hidden entry unchanged.

Apply the same removal in both encoding and decoding. This changes the conditioner functions but preserves the affine coupling's invertibility. It defines a single edited normalized density and generator. Do not edit the density direction alone. No model weight changes during evaluation.

Evaluate all 129 conditions. Keep every condition, including inactive or apparently unimportant units. Use evaluation batches of 1024. All fixed arrays divide exactly by this batch size.

## Calibration, frozen selection, then test

For each trained model, evaluate every condition on the calibration data and calibration latent arrays first. Do not evaluate test arrays until selectors are saved.

Selector L picks the unit with the largest calibration mean NLL increase, edited minus intact. Do not take an absolute value and do not clip negative damage. Selector M picks the unit with the largest calibration TV between the intact and edited four-quadrant frequency vectors. Both select from the same 128 removals. Exact ties use the smaller condition index. M has the extra cost of generation measurements; it is a behaviour-specific reference, not a free prediction.

Save all calibration scores and both selectors, then evaluate all conditions on independent test data and latents. The sole primary value per model is test quadrant-frequency TV for M minus test quadrant-frequency TV for L. Report all four values and their mean. A positive value favours M for this coarse output endpoint.

Use 10,000 whole-model bootstrap samples with PCG64 seed 26407, drawing four model IDs with replacement per sample. Use NumPy percentile with method='linear'. This four-model interval is descriptive and conditional on the fixed data and latent arrays. It excludes Monte Carlo uncertainty over new latent arrays and does not support broad model-family claims. Do not count neurons or latent samples as model replications.

## Required secondary measures and fit rule

For every unit, on calibration and test sets, save mean NLL damage, its Gaussian-base and log-Jacobian parts, quadrant frequencies and signed changes, quadrant-frequency TV, mean squared same-latent output displacement, and fraction of paired latent outputs that cross a quadrant boundary. Save each 4-by-4 intact-to-edited quadrant transition count table. Report full rankings, selector agreement, and Kendall tau-b between NLL damage and quadrant TV. Tau is descriptive; return null if its denominator is zero. Do not select a subset based on the observed scores.

Report the mean and range over all 128 units, and both chosen units' held-out NLL effects. No test-best unit is used to redefine the selector. No joint-unit edits, new amplitudes, or extra architectures will be added.

For intact models, report test excess NLL relative to the exact known mixture density; generated test quadrant TV relative to the exact target quadrant probabilities; generated mean/covariance; and count, mean, and covariance within each generated quadrant. Give the corresponding test-data moments for context. If a region has fewer than two samples, report the undefined covariance as null rather than filtering it.

The fixed fit gate requires test excess NLL <=0.20 nats and generated quadrant TV <=0.05. It applies only to interpretation. Retain all four models regardless. A panel claim about well-fitted flows requires all four gates to pass. A failed gate means the measured selector comparison concerns a weak density model. Do not call a passed gate proof of every aspect of fit.

## Numerical controls

Write each check before asserting it. All tolerances are absolute. Stop and preserve the record if any fails.

1. On every evaluation batch and condition, do a complete round trip in the opposite direction. Maximum coordinate error and maximum sum of forward/inverse log determinants must each be <=1e-9. This covers every density input and every generated latent.
2. On every intact evaluation batch, clone the conditioner hidden vectors without changing them and repeat the main direction. Outputs and log determinants must match exactly.
3. On every edited call, the removed hidden scalar must be zero and every untouched scalar at that site must match exactly. This does not claim later quantities are unchanged.
4. On the first 64 calibration data points for every condition and model, compare the encoder analytic log determinant with the log absolute determinant of an independently assembled autograd 2-by-2 Jacobian. Absolute error must be <=1e-9. The same check on dummy data precedes the lock. A negative determinant sign or non-finite value fails.
5. All weights, losses, outputs, density terms, and saved scores must be finite. Quadrant and transition counts must cover every sample. Their nonnegative frequencies must sum to one within 1e-12.
6. A separate hand-set dummy witness must have nonzero log-scale and a nonzero removal effect, with both round-trip and Jacobian checks. Dummy-only tests also check the identity initialization; exact mixture density and quadrant formulas against a separate calculation; Kendall ties and sign; selector tie order; and zero-change score arithmetic. No scientific outcome is needed for these checks.

A separate reviewer must reconstruct coupling transforms without importing this runner, replay all saved density terms and all unit quadrant/transition counts, and recalculate selectors, scores, gates, and bootstrap. A saved-weight replay is not a repeated training run.

## Files, counts, time, and preservation

Save raw base-log-density and encoder-log-determinant arrays for every condition and data point. Their sum defines the log density. Split archives by model, phase, and condition groups of at most 32, to keep each file below 10 MB. For all units save generated counts and score sums; raw generated coordinates are saved for intact, L, and M on test latents. Other generated coordinates are exactly replayable from the complete checkpoints and saved latent inputs. Do not imply they were all archived.

Per model and condition, the four main arrays total 61,440 inputs and 60 mapping calls. The reverse round trip doubles this cost. No-op calls add 240 across four models. Jacobian controls add 516 mapping calls of 64 inputs. Evaluation totals are 62,676 mapping calls and 63,684,864 input examples. Training adds 16,000 calls and 4,096,000 examples. Complete totals are **78,676 mapping calls**, **67,780,864 input examples**, **16,000 optimizer updates**, **31,716 control rows**, and four final models. Autograd's internal reverse operations are not counted as extra mapping calls. Check totals before completion.

Before lock, run one dummy model on independent standard-normal inputs only, using the real train/evaluation batch sizes. Train it for 200 steps. Evaluate all 129 conditions on two 2,048-point dummy data arrays and two 2,048-point dummy latent arrays, with all round-trip, no-op, local, and 64-point Jacobian checks. No study mixture points or study model seed may be used. Project full training and evaluation time by their separate example-count ratios, then multiply by two. This estimate must fit the cap before lock; any scope change must precede the lock.

Use a fixed 1,200-second scientific cap, checked after each update/batch/Jacobian check and before completion. Refuse an existing output directory. Write CSV records during work, save each phase before moving on, and preserve partial density arrays and current model weights on failure. A failure manifest must hash surviving files. Do not raise a tolerance, omit a seed, or resume past a failure.

Resolve the supplied full Git lock SHA. Compare exact bytes of this plan, run.py, requirements, and sources.md with that commit before any model call and at completion. Hash all result files. Report local completion separately from remote publication. The public paper must use plain English, state AI authorship and review limits, retain negative outcomes, and state close prior overlap.
