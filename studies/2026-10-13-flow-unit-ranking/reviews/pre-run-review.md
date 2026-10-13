# Independent pre-run review

13 October 2026. This is a separate AI review. It is not human peer review.

**Status: ready for the public lock. No scientific result was used in this review.**

## Scope and design

I read the complete plan, runner, source record, and final dummy receipt. The question is a finite test of two unit-selection rules in four small Real NVP models. It does not establish a new principle about likelihood and generation. The source record identifies close prior work, including Taming Normalizing Flows, and limits the novelty claim.

Each of 128 conditions removes one post-tanh hidden scalar on every call to its conditioner. The same removal applies to encoding and decoding. The affine coupling therefore retains its stated inverse and determinant. The intact condition is separate. All four seeds and all units remain in the tables.

The runner selects both units from calibration data before it evaluates test data. The likelihood selector uses signed damage, not absolute damage. The other selector uses a four-quadrant frequency distance on common latent inputs. Its extra generation cost is stated. Lower condition ID resolves an exact tie. The sole primary score is the held-out quadrant distance for the quadrant selector minus that for the likelihood selector. The four-seed interval is descriptive and does not include sample or architecture uncertainty.

Quadrant distance is not full continuous-distribution distance or sample quality. The plan saves displacement, boundary crossings, transitions, likelihood parts, and every unit's scores to make this limit clear. It retains failed fit gates. The fit gates concern the intact density and coarse quadrant frequencies; they do not prove complete distribution fit.

## Code and controls

I checked the coupling equations, alternating coordinates, density signs, inverse order, likelihood decomposition, fixed data generators, phase order, selectors, tie rules, bootstrap, and output counts. The stated initialization matches the code: default float32 Linear initialization, conversion to float64, then zero output layers. The data and latent samples are fixed independently of model results. All models share the same fixed training batches.

Each evaluation batch checks its inverse, determinant cancellation, and local removal. Intact no-op copies must be exact. A separate autograd Jacobian check covers every condition on the first 64 calibration points. The Jacobian is computed from both output coordinates. The fixed 1e-9 limit applies to inverse and determinant errors; local and no-op checks use zero. Failed controls are saved before the run stops. The runner checks locked source bytes before and after the run and preserves partial results on failure.

I independently counted 78,676 mapping calls, 67,780,864 input examples, 16,000 updates, and 31,716 control rows for the full fixed run. These counts match the runner. Every final model has 520 parameters. No new model outcome was collected for this review.

## Dummy evidence

The [fourth dummy receipt](dummy-preflight-v4.json) matches the final plan and code hashes. Earlier receipts remain. The hand-set witness has a nonzero removal effect of 0.05124007 and nonzero total log-scale of 2.42143796. Its inverse and Jacobian checks are nontrivial. The main dummy run covers all 129 conditions. It has 2,401 mappings, 2,181,184 examples, 200 updates, and 1,169 control rows. The separate witness adds five mappings and 15 examples.

All dummy checks pass. Largest coordinate and log-determinant errors are 1.33e-15 and 6.11e-16. All local checks are exact. The estimated full runtime with a factor of two is below the fixed 1,200-second cap. This estimate is an engineering check, not a scientific runtime result.

The fourth revision only removes a local absolute path from failure metadata. It does not alter the model, samples, score, or numerical limits. No design or code issue remains within this pre-run review scope.

## Accepted file hashes

- `studies/2026-10-13-flow-unit-ranking/analysis-plan.md`: `5b0e9d60705c6ba9c9d62c22560b461b87aab946f748a727ab2b55b23d298b8e`
- `studies/2026-10-13-flow-unit-ranking/code/run.py`: `1d3a0af712c6cf671ef0cfd874b4e13f783fc00aa9dac08334f12938f6babf53`
- `studies/2026-10-13-flow-unit-ranking/requirements.txt`: `bf22ae62140ec2188bfb7e3931f9eb0e8f16838592a7bb5527b5d7e546f2d04c`
- `studies/2026-10-13-flow-unit-ranking/sources.md`: `9c8b4c7cc659b89408e5145ef417ae76214f21e64b38f0e0b7c91b4c1c3523d5`
