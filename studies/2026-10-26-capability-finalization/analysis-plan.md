# Finish the fixed task-ability study

**Date:** 26 October 2026. **State:** Execution plan before new scientific results.

This continues Experiment 1 under the [updated roadmap](../../docs/RESEARCH_PLAN.md), version `58251c651ef6397ee26ae0c0159d0f7632063191`. It is the unfinished tenth study from the October 22 batch. It is one study across all attempts, not one paper per task or training seed. It tests task ability, not an internal explanation or a new scientific principle.

## Fixed evidence and method

The [symbolic methods](../2026-10-19-capability-gates/METHODS_SYMBOLIC.md) and [numeric methods](../2026-10-19-capability-gates/METHODS_NUMERIC.md) remain unchanged. Their task code and data are fixed at `c421ca112a0b421862934efb97bdecc0ebf323d3`. The six completed graph jobs and both partial attempts remain fixed by repository commit `58251c651ef6397ee26ae0c0159d0f7632063191`. Verify these files before use.

## Reuse and the one restart

Reuse all six completed graph models and check their saved outputs again. Retain their original paths, scores, and runtimes. The graph recipe remains A; do not select it again.

Final seed 8103 has 22,028 recorded updates. Its weights can contain one additional unrecorded update. No optimizer or random-generator state was saved. These weights cannot support an exact continuation.

This plan permits **one full restart of the same final seed 8103**, from its original initialization, with Recipe A and the same 32,768-update schedule. Save it in this new study directory. Keep the old partial weights and training trace unchanged. Do not initialize from the partial weights, change the seed, extend the schedule, or retry automatically. After this job, run final seed 8104.

Keep the earlier partial development-seed 8002 attempt as well. It was already restarted under the October 22 plan. Do not restart it again. The full record can contain at most 40 complete distinct task models plus two retained partial training invocations if every task runs. Six models are reused, so at most 34 complete model trainings remain. Digit jobs each contain two models.

## Task order and success rules

Run graph matching, register operations, digit images, and generated points in that order. For each unstarted task, run both recipes with development seeds 8001 and 8002. Then select one recipe and run final seeds 8101 through 8104. Use the fixed selection rule and existing code. Do not select from final results.

For classification, select the recipe with the highest minimum development accuracy across both seeds and all required groups. For generated points, both development models must first meet the 0.05 region-error rule. Among eligible recipes, select the lowest mean excess negative log likelihood. Break ties by total development runtime, then Recipe A. If neither generated-point recipe is eligible, stop that branch before final training.

| Task | Required result in every final model |
| --- | --- |
| Graph matching | At least 99% in all five fixed categories |
| Register operations | At least 99% joint-state accuracy in all nine fixed groups: horizon 0 after the history; copy-only and write-only at horizon 1; both operation orders at horizons 2, 8, and 32 |
| Digit images | Both architectures at every final seed: at least 95% on original images and 90% on each of four one-pixel shifts |
| Generated points | Excess negative log likelihood at most 0.20 nats and four-region error at most 0.05 |

All required final models must pass for a task panel to pass. Failure does not permit replacement seeds or a third recipe. Finish the other planned independent tasks if implementation and resource checks still pass. A scientifically failed task blocks its dependent explanation test. A complete negative study is different from an interrupted run.

## Preparation and fixed timing

Publish this plan and its code before preparation. Preparation uses random dummy inputs, not new task results. It repeats the existing arithmetic controls and atomic-permit tests. Use the Hall timing method already fixed on October 22: 20 warmup updates, three blocks of 100 updates, and the slowest block. Keep the original dummy methods for the other tasks. There is no search over timing procedures.

The projection includes two remaining graph jobs and eight jobs for each other task. Each per-job projection retains a fourfold allowance. Include a measured allowance for replaying the six reused models and a fixed setup allowance. Add all prior scientific time. The total must fit the unchanged four-hour cap before training starts. Preserve a failed preparation result and stop; do not rerun the same directory or revise the estimate after task results.

Count preparation, including dummy tests, against its unchanged one-hour cap. Prior preparation used 15.607510499994532 seconds. Charge a further 60 seconds for the pre-lock code and mock checks, including repeated checks. This is a conservative allowance, not a measured duration. Prior scientific work used 317.0650452510017 seconds. Thus the new scientific run has at most 14,082.934954749 seconds before any additional charges. All controls, saving, failed attempts, and replays count. Measure through the full result writes and hash scan. Reserve and charge one additional second for each stage's closing status and manifest writes. Record an overrun if those closing writes exceed that allowance; do not report exact measurement of a write within itself. Do not count reused job runtimes twice: their original execution is already in prior scientific time; their new replay does count.

## Resource and interruption rules

Use one local CPU process, two compute threads, one inter-operation thread, and no GPU. The peak process limit is four GiB. Require at least 20 GiB free disk. Use an exclusive compute lock. Existing result directories cannot be replaced.

The supervisor checks live account usage before substantial work and at least once a minute while computation runs. More than 20% must remain in every reported core window, and ordinary usage must be available. The supervisor issues a private permit for at most 120 seconds, using atomic file replacement. The scientific runner cannot renew it. Missing, invalid, denied, or expired permits stop the process. No account details or private paths are published as research evidence.

Stop on a failed source check, numerical control, replay, time limit, memory limit, disk requirement, or nonfinite value. Preserve partial weights, trace, status, and failure records. Do not retry automatically. No credits, reset, rented compute, or paid model API is used by this plan.

## Evidence, review, and publication

Save code, the public lock, preparation results, all job records, raw outputs, model weights, losses, controls, replay results, elapsed time, and file hashes. Reused files remain at their original paths. Before scientific execution, publish the passed preparation and its checks. The final public lock must include the exact plan and code used for preparation.

A separate AI review checks the code before execution and the claims before publication. This is not human peer review. A saved-weight replay does not repeat training. All graph states were used in training. Digit tests reuse the original UCI test split. Generated-point region error does not bound the complete distribution. Report these limits even if every task passes.

Publish one consolidated report of the fixed study. A stopped attempt remains an execution report and does not count toward the requested 200 additional completed papers.
