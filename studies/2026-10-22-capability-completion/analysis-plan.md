# Complete the fixed task-ability experiment

**Date:** 22 October 2026. **State:** Fixed continuation plan before new scientific runs.

## Question

Can the two fixed training recipes make the four task panels pass their existing ability rules? This continues Experiment 1. It does not test an internal explanation. It does not claim a new scientific principle.

The [earlier execution plan](../2026-10-19-capability-gates/analysis-plan.md), [symbolic methods](../2026-10-19-capability-gates/METHODS_SYMBOLIC.md), and [numeric methods](../2026-10-19-capability-gates/METHODS_NUMERIC.md) remain the full scientific design. Their code and old results are fixed at commit `c421ca112a0b421862934efb97bdecc0ebf323d3`. All old files remain unchanged.

## Treatment of the stopped attempt

The earlier run completed graph Recipe A, development seed 8001. Reuse that complete job with its original path, metrics, runtime, and weights. Run its saved-weight replay again before continuing. Do not train it again.

The earlier attempt at graph Recipe A, development seed 8002, stopped after 13,600 recorded updates. Its partial weights can include one extra unlogged update. Its optimizer and random-generator states were not saved. They are required for an exact training continuation.

For this specified continuation, restart that same model seed from its original initialization and use the full fixed schedule. Do not use the partial weights. Save the new attempt in this separate study directory. Retain the earlier partial weights and trace at their original path. This is an explicit restart of one interrupted seed. It is not a replacement seed or an undisclosed retry. It adds one partial training invocation to the planned model count. No other restart is permitted here.

The panel still has at most 40 distinct base models: 20 development models and 20 final models. If all task branches run, the combined record will have 40 complete model trainings plus one retained partial invocation. One of those complete models is reused from the earlier attempt. The digit jobs each contain two models. Flow final models remain conditional on the development region rule.

## Fixed order, selection, and tests

Run matching graphs, register operations, digit images, then generated points. Each task uses Recipe A with seeds 8001 and 8002, then Recipe B with those seeds. Choose one recipe. Then use fresh final seeds 8101, 8102, 8103, and 8104. No settings change after results are opened.

For classification, choose the highest minimum accuracy across both development seeds and every specified group. For flows, both development seeds must have region error at most 0.05. Among eligible recipes, choose the lowest mean excess negative log likelihood. Use summed development job runtime to break a tie, then Recipe A. The reused graph job retains its measured original runtime in this rule. Timing can vary between runs; report that limit.

Every final model must pass every task gate. Graph and register thresholds remain 99%. Digit thresholds remain 95% for the original images and 90% for each fixed one-pixel shift in both architectures. Flow limits remain 0.20 nats excess negative log likelihood and 0.05 four-region error. Stop the flow branch before final training if neither recipe meets its development region limit. A failed task panel does not justify another recipe or a changed threshold.

Keep all original sources, input generation rules, arithmetic, training schedules, batches, and saved-weight replay tolerances. The continuation imports the original modules without edits. It verifies their bytes, original data, and old result hashes before training. A public lock commit must contain this continuation plan, source record, code, and permit test before scientific execution.

## Resource control and failure rules

Use one local CPU process with two compute threads, one inter-operation thread, and a four-GiB peak memory limit. Require 20 GiB free disk. An exclusive local compute-file lock prevents a second runner from using this research process allocation at the same time.

The preparation cap remains one hour. Count the earlier 5.54-second preparation run and the initial dummy permit test against it. Repeat dummy timing on this run's hardware. Test all four modules, then keep the conservative full eight-job estimate for each task, even though one graph job is reused. Each module includes a fourfold allowance. Add the earlier scientific process time, 52.152487084 seconds. Require the combined projection to fit the unchanged four-hour Experiment 1 cap.

The scientific deadline is four hours minus those prior 52.152487084 seconds. Saving, replay, controls, and the retained partial run are included. No rented compute, paid API, account reset, or purchased credits are used.

The supervisor must read the live account meter at least once per minute and before substantial new work. It writes a permit valid for at most 120 seconds only if more than 20% remains in every core window and ordinary use is allowed. Write a temporary file in the same directory, flush it, and replace the live file with `os.replace`. Never truncate the live file. The scientific loop does not renew its own permit.

The dummy permit test performs 1,000 replacements while another thread reads at least 10,000 valid JSON documents. It also checks that valid permits pass and denied, expired, empty, missing, or late permits stop. This finite check is not a proof for every filesystem failure. Preparation repeats it. If a control, deadline, memory limit, numerical check, or usage check fails, stop and preserve the current run. Do not restart automatically.

## Evidence and report

Save original and final weights, all loss rows, raw evaluations, metrics, input arrays or exact generators, replay records, times, and a file manifest. Save partial weights on a training stop. Store the complete job table and all group scores. Report every development and final model separately. State which task panels passed and which failed or did not run.

A successful calculation replay is not repeated training. A task-ability pass does not identify a neural circuit. The graph states are all seen during training. Digit tests reuse the project's original UCI test set. Register and flow tests use fixed generated inputs. These limits remain in the paper. Separate AI review is required. No human peer review is recorded.

## Timing revision before scientific work

The first fresh dummy preparation failed its four-hour projection gate. Its records remain unchanged. [TIMING_REVISION.md](TIMING_REVISION.md) fixes exactly one longer Hall dummy benchmark, with warmup and the maximum of three timing blocks. The fourfold safety factor, all scientific settings, and the four-hour cap remain unchanged. That method must be published before the benchmark. If its separate revised preparation fails, this continuation stops. If it passes, its records must be in the final scientific lock. No scientific result determined this revision.
