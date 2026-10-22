# Task-ability continuation stopped at the usage reserve

**Date:** 22 October 2026.  
**Status:** Incomplete experiment. This report is not a completed tenth study.  
**Review:** AI-authored. Source review and saved-weight calculation checks are recorded. No human peer review is recorded.

## Main finding

Six complete graph models now give the correct answer in all five fixed test categories. Four are development models. Two are final models. One development model comes from the earlier run; five complete models were trained in this continuation.

The run stopped during the third final model when the account reached its required reserve. The supervisor denied the usage permit. The process saved partial weights and stopped. There was no repeat of the earlier empty-file error.

The graph panel requires four complete final models. Only two have complete test results. The panel therefore has not passed. The register, digit, and generated-point branches did not start. Experiment 1 remains incomplete.

## Plan and continuation rule

The [fixed continuation plan](analysis-plan.md) retained the original recipes, seeds, inputs, thresholds, and numerical checks. The [source record](sources.md) identifies the earlier studies and methods. This is an extension of their task-ability test. It is not a claim of a new scientific principle.

The earlier graph development model A8001 was reused, with its original measured runtime and a new saved-weight replay. The interrupted A8002 could not resume exactly because its optimizer and random-generator states had not been saved. The public continuation plan therefore specified a restart of that same seed from its original initialization. The old partial weights and trace remain unchanged. No replacement seed was used.

The four development jobs all reached 100% in every category. The fixed rule then selected Recipe A because its combined development runtime was shorter. Recipe A has two hidden layers of width 64. Recipe B has width 128. Both used the fixed 256-epoch schedule. Timing is hardware dependent; this choice does not establish that Recipe A is generally faster or better.

## Preparation and the timing revision

The new atomic permit updater passed 1,000 replacements and at least 10,000 concurrent reads. Valid permits passed. Denied, expired, empty, missing, and late permits stopped. This is a finite implementation check, not a guarantee against every possible file-system failure.

The first fresh preparation passed all arithmetic checks but failed its compute projection rule. Its fourfold estimate was 20,995.25 seconds, above the four-hour limit. That failed record remains in [preparation](preparation/status.json).

One revised timing method was then fixed and published before it ran. It used 20 dummy warmup updates and three blocks of 100 updates for the wider graph model. The estimate used the slowest block. All other task estimates stayed unchanged. The fourfold safety allowance and four-hour limit stayed unchanged.

The revised total was 10,707.95 seconds, or 2.97 hours, including the earlier scientific attempt. Its arithmetic checks passed. The method was locked at [commit 1de9626](https://github.com/gavenren/mechanistic-interpretability-research/tree/1de96267c03027226b8ae33b10c23483256d49f6). The revised records and final scientific code were locked at [commit 88e8df7](https://github.com/gavenren/mechanistic-interpretability-research/tree/88e8df7689b65fbeef221423c115cd1b1c198f22) before training.

Total recorded preparation time, including earlier checks and the one revised benchmark, was 15.61 seconds. The shorter warmed estimate does not prove why the first measurement was slower. Both measurements remain available.

## Results by model

Each complete model below reached 100% in all five categories: all 65,536 graphs; 3,680 hard negatives; 432 target negatives; 1,152 positive controls; and 288 other-cause negatives. The categories overlap.

| Split | Recipe | Seed | Job time, seconds | Outcome |
| --- | --- | ---: | ---: | --- |
| Development | A | 8001 | 36.27 | Reused complete model; all gates pass |
| Development | A | 8002 | 39.37 | Explicit restart; all gates pass |
| Development | B | 8001 | 55.67 | All gates pass |
| Development | B | 8002 | 60.70 | All gates pass |
| Final | A | 8101 | 40.55 | All gates pass |
| Final | A | 8102 | 44.94 | All gates pass |
| Final | A | 8103 | Partial | 22,028 recorded updates; no final score |
| Final | A | 8104 | Not started | No score |

All six complete models passed their separate output replay. The replay uses explicit affine and ReLU equations from saved weights. It checks predictions and category accuracy. It does not repeat training.

Seed 8103 stopped during its planned 32,768-update schedule. Its trace has 22,028 recorded updates. A check occurs after an optimizer update but before its trace entry, so the partial weights can contain one extra unrecorded update. No task score was calculated for this partial model.

This continuation used 264.91 seconds. Including the earlier stopped attempt, Experiment 1 used 317.07 seconds, or 5.28 minutes. Peak process memory in this continuation was 320,880,640 bytes. The stop was caused by the usage rule, not the compute deadline or memory limit.

## Limits and retained evidence

All graph states were seen during training. Perfect accuracy establishes task performance on this finite set. It does not establish prediction on unseen graphs or identify an internal matching algorithm. Two passing final models do not pass the required four-model panel. No new claim about the other task branches is supported.

The combined record contains six complete distinct models and two interrupted training attempts. The older partial attempt is A8002. The new partial attempt is A8103. The machine-readable field `retained_partial_training_invocations` counts the one older partial attempt; the new stopped job is recorded separately in `current_job`. The branch's last progress state is `running`, but the overall status is `stopped`. No process remains active.

All 30 files in the new [results manifest](results/manifest.json) matched their recorded hashes during the publication check. The [complete run record](results/status.json) and [group-score table](results/metrics.csv) include every complete job. The old partial attempt remains at its original path. No failed record was erased.

## Next step

Keep both attempts unchanged. Before further research, check the allowance again. A new public continuation plan must specify how to handle partial final seed 8103 and count all earlier compute. It must not silently replace that seed or erase its partial weights. The graph branch remains incomplete, and its dependent explanation test must wait.

See the [reproduction instructions](README.md), [timing revision](TIMING_REVISION.md), and [revised preparation record](revised-preparation/status.json). No more training was started after the reserve stop.
