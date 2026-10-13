# Fixed plan: bounds on the sign of an equilibrium neuron-removal effect

13 October 2026. AI-authored numerical verification study. No study model has been called. Plan, source record, input ledger, arithmetic module, runner, requirements, and data need a reviewed public lock before any study training or evaluation.

## Question and target

How often does a small state residual leave the sign of a fixed internal-removal effect unresolved, and how much extra computation establishes its sign or small magnitude?

The model is an implicit ReLU classifier with state equation `z = ReLU(Wz + Bx + b)` and readout `Vz + d`. Every saved float64 parameter and input is treated as an **exact real number with that binary value**. The target is the unique real-valued equilibrium of these equations. It is not the output of a particular finite-iteration float64 program. We use ordinary float64 iterates as candidate states, then enclose their error with outward-rounded calculations.

One intervention is used throughout: replace coordinate equation `z_j = ReLU((Wz+Bx+b)_j)` with `z_j = 0`. All other equations stay fixed. Re-solve the resulting equilibrium. We do not compare clamp duration, one-time zeroing, or alternate equation meanings. The readout is unchanged.

This applies standard contraction bounds to internal effect measurement. It is not a new theorem, model architecture, robustness certificate, or finding of a semantic mechanism. An unresolved sign is not a confirmed wrong sign. All weak models, inactive units, zero effects, unresolved effects, and failures remain in the result.

## Data, model, and fixed training

Use the original licensed UCI optdigits files already used in this repository: 3,823 training and 1,797 test rows. Inputs are the 64 integer counts divided by 16; these values are exactly representable. Reuse the fixed first-12-per-digit explanation panel, sorted by test-file row, from the spike study. It has 120 images in rows 0–127 and is not a random test sample. The ledger records the data hashes and every row/label. No model output selects an input.

Four seeds are 401, 409, 419, 421. State width is 32. Initialize `B,b` with a float64 PyTorch `Linear(64,32)`, `W` with the weight of `Linear(32,32,bias=False)`, then `V,d` with `Linear(32,10)`, in that order after setting the seed. There are 3,434 parameters. Project each row of W after initialization and each optimizer update by dividing it by `max(1, row_absolute_sum / 0.875)`. The binary value 0.875 is exact. No claim depends on the rounded projection giving precisely that norm: the independent outward norm bound must be below one before evaluation.

Train the finite 32-update unrolling from zero, with a shared input injection at every update, ReLU, and final linear readout. This is ordinary backpropagation through a fixed unrolling, not implicit-gradient training. The evaluated object after training is the equilibrium of the saved weights. State this training/evaluation distinction in the paper. Use float64, CPU two compute threads, one interop thread, deterministic algorithms, and disabled MKLDNN. There is no learned initializer, direct input-to-readout branch, dropout, normalization, or activation clipping.

Train each model for 40 epochs, batch size 128, with the final batch of 111. NumPy generator `30000 + seed` supplies one fresh permutation per epoch. Adam has learning rate 0.001, betas (0.9,0.999), epsilon 1e-8, zero weight decay, and all optional flags false, including foreach and fused. Minimize final-logit mean cross-entropy. No early stop, parameter tuning, or replacement seed. Save all 4,800 update losses, all orders, initial and final weight hashes. There are 611,680 training examples and 19,573,760 state-update examples in the 32-step training computation.

For full-test ability, run ordinary fixed-point iteration from zero until each image's float64 infinity-norm residual is at most 1e-12, with at most 512 updates. Save the proposed state, ten logits, residual, iteration count, and outward residual/error bound for every image. The whole study stops if any ordinary solve misses this cap. A small residual is not itself the certificate. Use these ten logits to fix the winner and runner-up, tie by lower class index, for all panel effects. Also report whether the validated logit enclosures establish the same top class. A non-certified top class remains in the fixed panel; the effect still uses its fixed two readout rows.

Report full-test and panel accuracy for every seed. The interpretation gate is at least 80% full-test accuracy for each seed. Failure restricts task claims, not inclusion. Report the zero-state and first-update full-test readouts and their disagreement with the final readout. These are limited recurrence-use diagnostics, following the prior initializer-bypass concern. No generic claim of useful reasoning follows from a nonzero state change.

## Outward arithmetic and mathematical bound

Let `q_upper` be a validated upper bound on the maximum absolute row sum of W. Since ReLU is 1-Lipschitz in the infinity norm, both the intact map and every coordinate-removal map have contraction factor at most q_upper. Require `q_upper < 1` or stop. Use this same conservative bound for all edited systems.

For a candidate state h, evaluate its map with interval arithmetic and form an outward upper bound r on `||F(h)-h||_inf`. Its state error is bounded by `r/(1-q_upper)`. For c equal to the exact difference of the two selected readout rows, an effect between an intact and edited state is enclosed by

```
estimated c^T(h_intact-h_edit)
  +/- ||c||_1 * (r_intact+r_edit)/(1-q_upper).
```

Use an interval for the estimated scalar as well; do not add the radius to an unverified rounded dot product. Readout biases cancel algebraically because the same class pair is used in both outputs. Do not infer exact zero from a small interval.

The arithmetic module will use float64 NumPy elementary operations, with a separate operation and `nextafter` step for each bound. Point inputs begin as zero-width intervals. For addition and subtraction, round each endpoint operation once toward each infinity with nextafter. For multiplication, bound all four endpoint products, then take the lowest lower endpoint and highest upper endpoint. For a positive division, round the numerator upper bound divided by the positive denominator lower bound upward. The denominator `1-q_upper` is rounded downward. Absolute value bounds use the endpoint maximum; nonnegative sums round upward at every addition. ReLU takes the maximum with exact zero at each endpoint. Min/max and exact negation do not need rounding. Dot products use explicit ordered term multiplication and addition; no BLAS reduction, fused operation, or unspecified reduction is used in a certificate. Norms, residual subtraction, the readout coefficient difference, effect dot product, radius, and final endpoints all use this arithmetic.

The numerical assumptions are IEEE-754 binary64 elementary add/subtract/multiply/divide in round-to-nearest mode, functioning nextafter, no fused arithmetic across the explicit ufunc calls, and finite stored parameters/states. Gradual underflow is permitted: the nextafter expansion covers subnormal rounding. Overflow, nonfinite endpoints, invalid operations, nonpositive denominator, or violated interval order stop the study. The checks do not formally verify the CPU or NumPy implementation. State this trusted computing base in the paper. Record software/platform versions.

Before the public lock, compare every primitive, matrix-affine evaluation, residual/norm bound, and final effect interval against exact Fraction arithmetic on fixed small rational cases. Include mixed signs, cancellation, zero, subnormals, and near-contraction cases. Enumerate all ReLU active sets in the small cases and solve each candidate active system with exact rational Gaussian elimination; verify its signs and the removed equation. This gives the unique exact equilibrium without using the study solver. At least one dummy case must have an unresolved coarse effect, at least one a certified positive effect, at least one a negative effect, and at least one exact zero. A deliberately slow contraction must require extra updates. Non-vacuity and exact containment checks are required. No tolerance can replace exact containment for these rational controls.

## Fixed effect protocol and outcomes

For each panel image, solve the intact map and each of 32 removed-coordinate maps from zero. Stop each ordinary solve at the first candidate with float64 residual at most **0.001**, at most 512 updates. Save the candidate before its next update, its actual residual, iteration count, and full state. No outcome screens a candidate. For each of the 3,840 model-specific image–unit pairs, compute the ordinary margin effect and the validated effect interval.

The sole primary quantity per model is the fraction of **nonzero ordinary effects** whose validated interval contains zero. Report the numerator and denominator. If the denominator is zero, the fraction is null, not zero; report this and do not construct a four-seed mean from fewer seeds. Otherwise give the equal-weight mean of all four seed fractions and each seed value. There is no directional or significance test and no bootstrap interval. Also report unresolved fraction among all effects and coarse zero counts, so the primary denominator cannot hide degenerate cases.

Then independently continue the intact and edited iterates for each pair. The baseline is duplicated per pair for this refinement; count that extra work. At each check, stop in this priority order: (1) the interval excludes zero, giving a certified sign; (2) the interval is contained in **[-1e-7,+1e-7]**, giving a certified small magnitude without an exact-zero claim; (3) 128 additional paired updates have been used, leaving an unresolved result. Thus, a small positive interval can be classified as positive before the small-magnitude rule. Freeze a completed pair while others continue. Save every checked interval and its iteration number. The edit is identical at every stage.

Report final positive, negative, small-magnitude, and capped-unresolved counts; extra paired updates; ordinary map calls, certificate evaluations, and wall time. Report naive sign reversals only when a later validated interval excludes zero on the opposite side. A zero naive effect has no sign to reverse. Report effects and interval widths by fixed absolute-naive-effect bins: zero, (0,1e-7], (1e-7,1e-5], (1e-5,1e-3], and above 1e-3. These are descriptive bins, not alternative primary outcomes.

Intervals from successive checks must intersect. A disjoint pair is a numerical failure and stops the run. Check every refinement against the same finite-weight target. A final unresolved interval is a valid scientific result, not an implementation failure. Keep all small or zero cases.

## Controls, cost, and record

Required dummy controls include W=0 (direct ReLU solution), a linear active case with exact rational solution, an inactive removed unit with exact zero effect, a slowly converging contraction, cancellation in the readout, and a no-op map with identical outputs. All arithmetic controls run without UCI model evaluation. Scientific controls check finite weights and losses, q_upper<1, exact clamped coordinate zero, unedited map coefficients unchanged, full output counts, no-op repeated solves, and interval intersection. Save actual errors or boolean evidence before assertions.

The study has four trained models and 15,360 image–unit effects. The coarse panel has 15,840 state solves (120×33×4); the full test has 7,188. Variable solver costs are measured explicitly. Training has the fixed budget above. The whole scientific run has a 1,200-second cap, including evaluation and saved output. Refuse an existing output directory. Keep partial loss tables, current weights, completed states, interval records, and failure manifest. No resume or tolerance change under this plan.

An ordinary map call means one batched recurrent map. Counts include final residual checks, repeated no-op solves, first-update diagnostics, and both refinement branches. Count their row totals as well. A certificate call means a bound request: one full-logit enclosure or one paired effect enclosure. The paired request evaluates two residual maps. Save both request counts and validated-map counts, with their row totals. Count input affine enclosures separately. Training forward counts have 32 state updates each; backward work is additional and included in wall time. Save these definitions with the results.

A dummy-only test must use the real batch shapes (128/111 training, 128/5 test, 120 panel, and 3,840 paired effects), exercise early stops and cap outcomes, and project cost with a two-fold factor below 1,200 seconds. If reliable enclosures cannot meet this budget, narrow the design before the public lock or reject the study. No scientific pilot is permitted.

Save full data hashes, fixed input ledger, schedules, model parameters, all test states/logits, coarse and final paired states, effect and bound components, every refinement record, all controls, summaries, code hashes, package versions, counters, and final file manifest. Check public locked bytes before and after the run. A separate audit must check interval arithmetic against exact rational cases and replay saved weights and effect bounds without importing the scientific runner. It must state whether training was repeated; saved-weight replay alone is not a training replication.
