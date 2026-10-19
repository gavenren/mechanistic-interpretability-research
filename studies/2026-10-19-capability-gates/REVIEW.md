# Preparation and runner review

**Scope:** Code review before scientific training. The reviewer read `code/run.py`, `code/common.py`, and `code/digits.py`. The reviewer wrote the symbolic task modules. This is therefore a separate AI review of the runner and digit module, not an independent review of the symbolic modules. No human peer review or scientific execution is recorded here.

## Findings and corrections

1. **Digit return values did not match the runner.** The digit preparation result lacked `passed` and `projected_job_seconds`. Its replay result lacked `passed`. The runner uses all three fields. Preparation would stop with a missing-key error. Correction checked: the digit module now supplies `passed` after its assertions and sets `projected_job_seconds` to four times the complete train, evaluation, and replay estimate.
2. **The stage deadline was not passed to replay.** The runner called `module.replay(path)`. The replay functions then used an infinite deadline. The usage permit and memory checks remained active, but the stage time cap did not. Correction checked: the runner now sets a private absolute stage deadline. Every common guard uses the smaller of that stage deadline and the supplied deadline. This covers replay and preparation calls even when their local argument is infinite. The symbolic replay functions also accept an explicit deadline.
3. **The public lock omitted an imported source file.** Hall data generation imports the original graph study's `code/run.py`. Correction checked: the runner now verifies that imported source file and both digit data files against the same public lock commit. The digit module also checks its two dataset byte hashes.

## Other checks

The runner has four development jobs and four final jobs per task. Each digit job contains two architectures. This gives the planned maximum of 20 development models and 20 final models. The four-task time estimate correctly multiplies each per-job estimate by eight, provided each estimate already includes its complete job and the safety factor.

Classification selection uses the worst subgroup value across both development seeds. Shorter summed job time breaks a tie; recipe A breaks an exact second tie. Final results do not change that selection. Flow selection and eligibility require a separate review of its final module.

The common guard stops on a missing, denied, or expired private account permit. It also stops at the four-GiB peak process memory bound. Training checks include nonfinite loss, gradients, and updated parameters. Result folders refuse replacement. The modules retain traces and partial weights on a training exception. The runner preserves a stopped status and a file manifest.

The digit development split is inside the original training data. The final run uses the original test data. Its training uses random shifts and its fixed tests use the five specified image positions. Replaying saved weights is distinct from repeating training. The digit schedule currently depends on each model seed; this differs from the symbolic tasks' fixed schedule within each split and must remain explicit in the methods.

## Verification status

Both symbolic modules passed Python compilation. No scientific job was run for this review. All three listed corrections were checked in the updated files. Successful dummy tests remain required; code inspection alone is insufficient. The flow module was checked only for the required return keys, not for its mathematical implementation.
