# Pre-run review of the fixed experiment

Date: 13 October 2026. Status: passed for a public plan and code lock. This is an AI review. No human review is claimed.

## Scope

The lead agent read the full plan, source record, input ledger, numerical module, training code, ordinary solver, refinement protocol, scientific runner, and dummy checks. A separate agent checked the interval proof and exact rational cases in [its review](interval-review.md). No scientific model has been called.

The target is the exact real equilibrium of the saved finite weights. Training uses 32 finite updates. The model is then evaluated at its equilibrium. The study must state this distinction. Removing a unit replaces its named equation with zero. The class pair stays fixed.

## Checks

The four seeds, model size, initialization order, optimizer, 40-epoch schedule, fixed panel, and two residual thresholds match the plan. The ordinary solver returns the first candidate with a passing residual. The validated bound includes input injection error, the coefficient difference, residuals, the downward contraction denominator, and final endpoint error.

The primary outcome uses nonzero ordinary effects only. A zero denominator is null. The four-seed mean is absent if any denominator is null. The full effect denominator and zero count remain visible.

The refinement rule checks strict sign first, small magnitude second, and the 128-update cap last. Completed pairs stop changing. An unresolved interval is not a wrong sign. A reported reversal requires a later interval with the opposite strict sign. The code saves all interval checks, partial states on failure, and output hashes. It refuses an existing output path and checks the fixed time cap.

The original cost counter omitted the first-update diagnostic. This was repaired before scientific calls. Counters now separate ordinary map calls, bound requests, validated residual-map passes, input enclosures, and training state updates. Earlier dummy receipts remain available.

The final protocol dummy used generated inputs only. It passed all 383 recorded controls. Its code and plan hashes match the files below. It covers full batch shapes, all stop outcomes, a known sign reversal, the complete 128-update cap, null and nonzero primary denominators, partial failure records, and time-limit rejection. The timing estimate is 227.4 seconds with a two-fold factor and all pairs at the cap; the fixed limit is 1,200 seconds. This is an estimate, not a measured scientific duration.

The arithmetic review verified 1,697 author checks and passed 1,512 separate exact rational and rejection checks. Its stated hardware and floating-point assumptions remain necessary.

## Publication and result review

This review permits the reviewed files to be published as a plan lock. Scientific execution still requires that remote lock and exact local bytes to be verified. No result is approved by this pre-run review. A separate saved-weight audit and a paper claim review are required after the experiment. Failed ability checks, wrong signs, zero effects, and unresolved effects must stay in the paper.

## Reviewed hashes

| File | SHA-256 |
|---|---|
| `analysis-plan.md` | `01703bd5d11172898c1ab6b6935d019cc3d1c0920a3af46dccf8a7cfbecfe015` |
| `sources.md` | `83ae3e840183fd41e8fca6ec781e81dde76872f5d015c1add6be609e29fc3ac2` |
| `inputs.json` | `cab624b6dccee25262c723457f19eb753c7d6de30695833912b0851f70451dbb` |
| `requirements.txt` | `bf22ae62140ec2188bfb7e3931f9eb0e8f16838592a7bb5527b5d7e546f2d04c` |
| `data/optdigits.tra` | `e1b683cc211604fe8fd8c4417e6a69f31380e0c61d4af22e93cc21e9257ffedd` |
| `data/optdigits.tes` | `6ebb3d2fee246a4e99363262ddf8a00a3c41bee6014c373ed9d9216ba7f651b8` |
| `code/intervals.py` | `0bb5a0275fc645f65dcf6a619b8e0bd6eb217e7fef3b5470bedb1e6d434783ca` |
| `code/check_intervals.py` | `73e0f4294e1f0da0dd93e93167b969e6314528c4cfd385384a8961e4825545a3` |
| `code/model.py` | `028071ecd3a792910e986a3a9e212a32287e660b0678a15c05e7101f82a2644b` |
| `code/protocol.py` | `a5884b49cf462129585bc3670ab8e964e2ab4e52f536be12e2125951e0ed3922` |
| `code/run.py` | `5193f58f11c7e1fcb7c1352e66318d6f7d278550982e0dba547e6eb731264692` |
| `code/check_protocol.py` | `52289be2788b63e2f07940bbed12308faf6cb51b00abc87e37c7a35d69d5b402` |
| `code/dummy_timing.py` | `568ba011b26ea6ce1b568063b5e9603f822bf6e99e8c1e225be9a1c36445fd1f` |
