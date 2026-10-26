# A componentwise bound: fixed method for Stage 5

Publish the method, code, input list, and review before any saved-model calculation. Preparation may use only the dummy cases specified here. The parent runner controls the public lock, software, usage permit, time caps, and memory limits.

## What is different from Study 15?

[Study 15](../2026-10-13-equilibrium-sign-audit/paper.md) already tested a global contraction bound. It refined unresolved effects, proved five initial sign reversals, and replayed the complete result. Repeating those findings, or adding seeds with the same procedure, would not be a distinct result.

This study asks whether retaining the residual at each coordinate reduces the **complete verification cost** on the same known panel. It tests one fixed candidate against the old bound, including bound construction, independent reference work, saving, and validation. A tighter bound that costs more time fails the practical gate.

The mathematical argument uses established contraction and nonnegative-matrix bounds. It is not a new theorem. Output bounds and adjoint methods for equilibrium networks are also established; see [CerDEQ](https://proceedings.mlr.press/v162/li22t.html). This study does not claim priority for those ideas. Its possible contribution is a checked cost comparison for fixed internal-removal effects.

## Fixed inputs and targets

Use Study 15's four saved models, with seeds 401, 409, 419, and 421. The model files are `studies/2026-10-13-equilibrium-sign-audit/results/{seed}/weights.npz`. For each seed, also use its `coarse.npz`, `refined.npz`, and `test-classes.npz`. The root input manifest fixes every exact path and hash. The module exports these paths in `INPUT_FILES`.

Use the existing `inputs.json` panel and `data/optdigits.tes`. The panel has 120 images. Keep every one of its 32 coordinate removals: 3,840 effects per model and 15,360 unique effects in total. Keep the original winner and runner-up for each image. They define the fixed output margin. They are not chosen again after removal.

The exact-real target treats each stored binary64 input and weight as its exact real value. The model equation is:

```text
z = ReLU(Wz + Bx + b)
output = Vz + d
```

Removal replaces one state equation with `z_j = 0` at every update. It does not set the coordinate to zero only once. The effect is the intact margin minus the removed-system margin. A positive sign means removal reduced that margin. The readout bias cancels.

Train no model and change no weight. Do not use the old refined result to initialize the new solve. Load it only for the fixed global-replay control. Recompute all coarse states from zero in each timed arm. Require exact agreement with the saved coarse states, residuals, and update counts. Require the global arm's final states, bounds, counts, and categories to agree with the original saved refinement.

## The two bounds

Both methods use the same outward upper bound `q < 1` on the maximum absolute row sum of `W`. Both methods enclose the effect center with outward arithmetic. The global method uses Study 15's published implementation:

```text
global radius = ||c||_1 × (max(r_intact) + max(r_removed)) / (1−q)
```

Here `c` is the exact difference between the two fixed readout rows. Each residual vector bounds the absolute difference between a candidate state and one exact application of its map.

The candidate uses eight terms. Let `M = abs(W)` and let `g` be a componentwise upper bound on `abs(c)`, obtained from an interval subtraction of the two readout rows. Define:

```text
v0 = g
v(t+1) = M.T v(t)
w8 = v0 + v1 + ... + v7

state readout-error radius
  = w8.T r + sum(v8) × max(r)/(1−q)
```

Calculate this radius separately for the intact and removed states. Add both radii outward. Expand the outward effect-center interval by their sum. Use this pure candidate bound. Do not take a minimum with the old bound, change the term count, or adjust a bound after a containment failure.

The row-vector implementation is `v_next = v @ M`. A transpose error would invalidate the result. Cache the eight-term coefficients once for each distinct fixed class pair within an arm. Map that cache to its effect rows. Do not carry it into another arm or timed repetition.

## Why this is a valid bound

Let `e = abs(z_star − h)` be the unknown state error. ReLU gives the componentwise inequality:

```text
e <= M e + r
```

Since the infinity norm of `M` is at most `q < 1`, repeated substitution gives:

```text
e <= sum(t >= 0) M^t r
```

For `g >= abs(c)`, the margin error is at most the sum of `v(t).T r`. The first eight terms give `w8.T r`. The remaining tail is:

```text
v8.T (I−M)^−1 r
  <= sum(v8) × max(r)/(1−q)
```

The edited map replaces one row of its componentwise Lipschitz matrix with zero. Its matrix is therefore no larger than `M`. The same bound remains valid. No inverse matrix is computed in the implementation.

This is a real-arithmetic argument. The next section states the extra assumptions required for computed intervals.

## Arithmetic and independent code

Use float64 and the fixed runtime from the parent plan. Reuse Study 15's reviewed `code/intervals.py` for basic outward operations and the published global bound. This assumes correct IEEE-754 elementary operations, gradual underflow, and `nextafter`. It is not formal verification of NumPy or the processor.

Form coefficient differences as intervals. Calculate residuals as componentwise interval differences between the exact map enclosure and the candidate state. Do not replace this vector with its maximum before the finite series term.

Round every multiplication and addition in the series upward. Sum nonnegative terms in coordinate order, with an upward step after each addition. Round `1−q` downward and require a positive denominator. Round the division, tail product, radius sum, and final endpoint expansion outward. No BLAS reduction produces a certificate. Ordinary candidate-state updates may use the original NumPy matrix multiplication.

The reference uses Study 15's separate `code/independent_audit.py`. Its solver, affine interval calculations, residual bounds, and global effect bound are independent of the new candidate code. It does not import the candidate or read candidate radii. Compute its input affine enclosure separately and require exact agreement with the original interval implementation.

The source files are part of the locked inputs. Importing their function definitions is common startup work; no scientific state or coefficient cache is shared between timed arms.

## Stops and reference acceptance

Coarse solves start at zero and stop at the first ordinary infinity-norm residual at most 0.001. Their cap is 512 updates. Include the final residual-check map call in the cost. Repeat the intact coarse solve and require an exact no-change replay.

Each effect arm starts from those coarse states. At every step:

1. Stop if the interval has a strict positive or negative sign.
2. Otherwise stop if the whole interval lies within ±`1e-7`.
3. Otherwise advance both states, up to 128 extra paired updates.

Sign has priority. A small-magnitude result does not prove exact zero. Save every interval checked before stopping, including step zero. Check that each effect has one history row per step, no duplicate row, and no update after its first stopping decision. Check final categories with a separate explicit rule.

Construct an independent global reference from separate copies of the coarse states. Continue until the **outward** width of its interval is at most `1e-10`, or until 1,024 extra paired updates. A too-wide reference remains unresolved. Do not stop it early because the candidate has a clear sign. Do not reuse a reference from the other method or a prior repetition.

Compare every recorded effect interval with the final validated reference interval:

- Full containment is required for the conservative acceptance gate.
- Overlap without containment fails that gate. It does not by itself prove that the candidate is invalid. Do not enlarge the candidate afterward.
- Disjoint intervals or opposite certified signs are a fatal numerical contradiction. Preserve the evidence and stop.

Report reference-width failures, noncontainment, and unresolved refinement separately. Passing a finite panel does not prove universal soundness. The derivation, arithmetic checks, and stated machine assumptions remain necessary.

## Timing and practical success

Run five paired repetitions for each model: 40 timed arms in total. For the first model, use global then componentwise in repetition zero, and reverse the order in repetition one. Continue alternating. The next model starts with the opposite order. Use the same hardware and two CPU threads throughout.

Each arm's timer starts before loading its model, inputs, and saved comparison records. It includes coarse solves, input enclosures, coefficient construction, refinement, a fresh independent reference, saving, reloading, validation, logical hashing, comparison with the first repeat, and removal or retention of the repeat archive. The reported outer time also includes writing that arm's result receipt. The final timing table records this outer time. The per-arm receipt also records the inner artifact-work time. Common startup and final summary writing are charged to total stage time and reported separately through the parent receipt.

Take the median of the five global times and the median of the five candidate times for each model. Divide the candidate median by the global median. Do not substitute the median of paired time ratios.

Practical success requires all effects to resolve under the fixed rules, all reference acceptance and repeat checks to pass, and a median-time ratio at most 0.75 in **each** model. A lower update count alone does not pass. No significance test is used.

Report coarse and extra state updates, map calls and state rows, certificate calls and rows, input-enclosure work, coefficient construction, validation calls, and time. The memory record is the process high-water mark at the end of each arm. It is not an isolated per-arm allocation measurement and cannot establish a memory-saving claim.

## Evidence without redundant archives

Every arm writes and reloads its full array archive before comparison. Keep the first full archive for each method/model pair. It contains all coarse states, final effect states and intervals, every refinement-history interval, final independent reference states and intervals, update counts, fixed class pairs, and the candidate's cached series coefficients where applicable.

Later repetitions can delete their archive only after exact array-name, data-type, shape, and value agreement with the first archive. Hash sorted logical array content, including each name, original shape, data type, and raw bytes. ZIP timestamps are not part of this logical hash. Retain each repetition's hash, check counts, receipt, and timing record.

If a repetition differs, retain its full archive and mark the repeat gate as failed. Do not replace the first artifact or discard the difference. Temporary saving, reading, hashing, comparison, and deletion all remain inside the measured arm cost.

After saving, recompute each final reference interval from its saved states with the separate global implementation. Check its width decision and cap. Recheck the saved effect histories and containment. At the end of the study, verify that every original source file still has its starting hash.

During each solve, keep a copied state record at a complete update boundary. For a paired update, take that copy before either state advances and after the current interval has been recorded. Record the phase, attempted step, completed state-update counts, and completed interval-history parts. If a guard or calculation fails, save this record once, with a raw-array limit of 64 MiB. Disable the repeating guard alarm only for this final bounded write, then propagate the failure. Do not run another model calculation during failure saving. A disjoint successive interval adds its offending and previous endpoints before the stop. Preserve previous arm receipts and files. An interrupted half-pair is not counted as a complete paired update. The failure record does not authorize automatic resume.

## Dummy controls and cost projection

No saved scientific model or output can enter preparation. Use these exact small cases:

- A nonsymmetric triangular matrix `M = [[0, 1/4], [0, 0]]`, `g = (1, 0)`, and residual `(1/8, 1/2)`. The series radius must enclose `1/4` and differ from it by at most `1e-12`. This catches a transpose error.
- For that matrix and input injection `(1, 1)`, the intact equilibrium is `(5/4, 1)`. Remove coordinate 1. Readouts `(1, 0)` and `(-1, 0)` give exact effects `1/4` and `-1/4`.
- Remove coordinate 0 instead and use readout `(0, 1)`. Its exact effect is zero.
- Use injection `(1, -1)` and remove the already inactive coordinate 1. The unchanged state `(1, 0)` gives a no-change control.
- Use injection `(0, 2^-40)`, removal of coordinate 1, and readout `(1, 0)`. The exact effect is `2^-42`. Zero candidate states must give a small-magnitude result.
- Use the scalar map with weight `1−2^-20` and injection 1. Its exact removal effect is `2^20`. From zero, the candidate must remain unresolved at 128 updates and the reference must remain too wide at 1,024.
- A strictly positive interval inside the small-magnitude band must take the positive-sign branch.
- With `W = 0`, injection `(1/2, 1/4)`, removal of coordinate 0, and readout `(1, -1)`, the exact effect is `1/2`. Check both bound implementations.

Compare exact targets with the exact rational values of the computed float endpoints. These cases test both the candidate and separate reference enclosures.

For timing only, construct random arrays with seed `2026102650`: 32-dimensional states, 120 coarse inputs, 3,840 effects, and 90 distinct fixed class pairs. Project each recurrent row toward an absolute sum of 0.875 and validate `q < 1`. Use no trained weights. Take the maximum of three measured full-size passes for each operation. Each bound-pass timing includes two full-size failure-record copies; coarse-map timing includes two small record copies. Measure archive saving, reloading, history checks, logical hashing, comparison, and deletion using a full-cap-size dummy history. Random payloads avoid unrealistically cheap all-zero compression.

For each arm, project the complete coarse cap: 34 solves, each with up to 513 map evaluations. Include 129 refinement bounds and 128 paired updates. Include 1,025 reference bounds and 1,024 paired updates. Add both input enclosures, final saved-reference validation, the full-history archive measurement, and two seconds for input loading per arm. Measure the usage/time guard separately and include `34 × 513 + 129 + 1,025 + 6` explicit guard calls per arm. Add series construction for the candidate. Project 20 arms of each type, add 60 seconds for common setup and summaries, then multiply the whole estimate by four.

Use the unchanged two-hour scientific cap and shared preparation limit. The parent enforces the 20% usage reserve and the memory limits. If the worst-cap projection does not fit, retain that preparation failure and do not run the saved models. Do not repeat timing changes to force a pass. No training, paid compute, new dataset, or architecture search is included.
