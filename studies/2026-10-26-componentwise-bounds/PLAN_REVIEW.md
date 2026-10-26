# Plan review

Date: 2026-10-26. Review type: a separate AI review of the plan and code. This is not human peer review.

**Decision: no open blocker for publication of the plan and guarded preparation.** This review does not certify a numerical result. Preparation must pass before the scientific run can start. The public plan, preparation files, input hashes, runtime, live usage permit, and compute limits must pass the controller checks.

## Scope and evidence

The review covered `analysis-plan.md`, `sources.md`, `README.md`, `BOUND_METHOD.md`, `code/run.py`, `code/bounds.py`, `code/test_controls.py`, and the input manifest. It included the final changes to partial records and the zero-weight control. No model run, benchmark, numerical dummy, or test was repeated for this final review.

The controller and input manifest name source commit `8307227b856d6d6cb44155fdb99b1a47a85ef61e`. The main agent reports that all 23 source files match the public source bytes and hashes. The controller checks these inputs again before work. The main agent must verify the published plan bytes before preparation.

The control-test agent reports that 19 stub tests passed in one run of about 0.10 seconds. It then ran the added paired-update interruption test alone; that test passed in about 0.10 seconds. These are 20 tested cases, not a claim that all 20 ran together. The tests use stubs for model and interval calculations. Reported syntax checks and test work fit the shared 120-second preparation charge. This review adds no numerical test work.

## Bound and inference checks

The proposed bound follows from the componentwise inequality `e <= M e + r`, with `M = abs(W)` and a validated maximum row sum `q < 1`. The first eight nonnegative series terms use the output coefficients. The remaining terms are at most `sum(v8) * max(r) / (1 - q)`. The transpose direction is correct. Permanent removal of one coordinate has a Lipschitz matrix no larger than `M`, so the same bound applies to the edited map.

The code encloses coefficient differences, residuals, products, sums, division, and final endpoints. It uses a lower bound for `1 - q`. The claim depends on the stated binary64 arithmetic assumptions. Preparation includes a nonsymmetric matrix control for the transpose direction and exact rational effects. The final zero-weight case checks the candidate and separate reference against the exact effect `1/2`.

The independent reference uses the prior global implementation and separate state copies. It does not import candidate bound arithmetic. Each arm pays for its own reference calculation. The reference is still a computed interval, not an exact oracle. Failure to contain that interval is a conservative acceptance failure. Overlap without containment does not prove the candidate invalid. A disjoint interval or an opposite certified sign is a numerical contradiction and stops the attempt. A reference that misses its width target remains unresolved.

The study tests a fixed method on four saved models and 15,360 fixed effects. It does not establish a new general theorem or priority for equilibrium output bounds. The new test concerns complete measured cost and acceptance of this bound on the fixed removal panel. Study 15's earlier sign reversals are background evidence.

## Fixed choices and complete cost

The plan fixes the models, inputs, removals, class pairs, state order, thresholds, update caps, and first-stop rule. Sign takes priority over the small-magnitude rule. All cases stay in the result. There is no minimum with the old bound and no change of method after a failure.

The five paired repetitions per model give 40 timed arms. Arm cost includes input loading, coefficient construction, coarse solves, refinement, the separate reference, saving, reopening, checks, hashes, and repeat-file retention or deletion. The final timing table includes the arm receipt write. Later full archives can be removed only after exact agreement with the first saved artifact; their checks, hashes, and timing records remain. A difference keeps the artifact and fails the repeat gate.

The preparation projection includes the full update caps, all reference work, two journal copies per bound pass, full-size archive work, guard calls, loading allowance, and the factor of four. It is an admission estimate, not a promise of runtime. The controller retains the two-hour stage cap and the 120-second post-run reserve. It carries forward 453.1494834999976 seconds of preparation and 1568.8212972499932 seconds of scientific work. The program caps remain 3,600 and 183,600 seconds as specified. A larger paper target does not increase these limits.

The shared lock, CPU thread limits, memory and disk checks, zero language-model-token limit, runtime checks, and short live permits remain in force. Source checks, initialization, work, saving, and the main hash scan occur inside phase accounting. Closing writes have a reserved charge; an overrun stops the phase. Signal checks can be delayed by a native operation, so the guard is not an exact operating-system deadline.

## Final partial-record fix

The journal copies arrays before publishing a complete record in one assignment. Refinement and reference updates save both states and the just-checked interval before either side advances. If the second side fails after the first side changes, the retained record still has both old states and the correct completed paired-update count. The added stub test checks this case, including the step-zero interval.

A disjoint successive interval keeps its offending and previous endpoints before raising the error. Completed history parts and prior arm files remain available. Failure saving makes one bounded write of at most 64 MiB of raw arrays. It disables the repeating alarm for that final write, then propagates the failure. This exception permits evidence saving only; it permits no further model calculation or automatic resume. The closing write can still fail because of an operating-system or storage error.

## Required gates after this review

Preparation must pass its exact controls and full-cap cost projection. The scientific result must report numerical completion, reference acceptance, repeat agreement, complete case resolution, and the per-model speed criterion separately. All reliability checks and a candidate/global median-time ratio at most 0.75 on each model are required for the combined success claim. A null result, unresolved reference, resource stop, or failed speed test must remain in the record. The full study counts as one paper.
