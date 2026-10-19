# A task-ability test stopped by a usage-check file error

**Date:** 19 October 2026.  
**Status:** Incomplete experiment. One development model completed. A second model stopped during training.  
**Review:** AI-authored. Separate AI source and result checks are linked below. No human peer review is recorded.

## Main result

The preparation checks passed. Experiment 1 then started under the fixed plan. The first graph model gave the correct answer for every graph and every required test category. A separate calculation from its saved weights reproduced the answers.

The run stopped during the second model. The process read the private usage-permit file while the supervisor was updating that file. The read failed because it found an empty JSON document. The resource check stopped the process, as required when the permit cannot be read.

This is an execution report, not a completed study. No final model panel was trained. No recipe was selected. The result does not establish that the planned graph panel passes, and it gives no new result for the other three tasks.

## Question and fixed plan

Experiment 1 asks whether longer training, with or without wider hidden layers, can make four small task settings pass their ability tests. The tasks are matching graphs, register operations, digit images, and generated points. The plan allows two recipes, two development seeds for each recipe, and four final seeds for the selected recipe. The digit task has two architectures.

The maximum is 20 development models and 20 final models. Training uses one local CPU process with two compute threads. Preparation has a one-hour cap. Experiment 1 has a four-hour cap, including checks and saved-weight replay.

The [execution plan](analysis-plan.md), [symbolic methods](METHODS_SYMBOLIC.md), [numeric methods](METHODS_NUMERIC.md), code, and preparation records were published in [commit 9a27126](https://github.com/gavenren/mechanistic-interpretability-research/tree/9a27126071e409ca903ad50dbe978ed42b647f30) before scientific training. All remote file bytes were checked against the local files. The old fifteen studies remain unchanged.

## Preparation results

All four dummy checks passed. These checks used random inputs and labels to test shapes, arithmetic, unchanged-input behavior, known references, and separate output equations. They did not measure scientific task accuracy.

| Task | Projected seconds for eight jobs, with fourfold allowance |
| --- | ---: |
| Matching graphs | 4,466.64 |
| Register operations | 2,704.54 |
| Digit images, both architectures | 3,160.63 |
| Generated points | 1,080.89 |
| **Total** | **11,412.70 (3.17 hours)** |

The total was below the four-hour cap. Preparation took 5.54 seconds inside the runner. Its measured peak process memory was 382,140,416 bytes. The host was an Apple M4 with 24 GiB physical RAM. The process memory limit was four GiB. Free disk space was about 694 GB. The full software and memory record is in [preparation/status.json](preparation/status.json).

The timing estimate uses a short dummy measurement. It is not a guarantee of full-run time. The extra allowance accounts for variation and overhead. It does not permit a longer hard cap.

## Scientific run record

| Item | Observed result |
| --- | --- |
| Complete models | One: graph Recipe A, development seed 8001 |
| Partial models | One: graph Recipe A, development seed 8002 |
| Other planned models | Not started |
| Final models | Zero |
| Selected recipe | None |
| Scientific process time, including replay and the partial run | 52.15 seconds |
| Peak process memory | 361,316,352 bytes |
| Stop reason | Usage-permit JSON read error |

Recipe A has hidden widths of 64 and 5,313 parameters. Seed 8001 completed all 32,768 updates. Its training took 35.30 seconds. The complete job took 36.27 seconds, followed by about 0.034 seconds for the saved-weight replay.

| Required graph category | Cases | Accuracy |
| --- | ---: | ---: |
| All graphs | 65,536 | 100% |
| Negative graphs without an isolated vertex | 3,680 | 100% |
| Target negatives | 432 | 100% |
| Positive controls | 1,152 | 100% |
| Other-cause negatives | 288 | 100% |

These categories overlap. They are not independent samples. The model trained on all 65,536 graph states. Its perfect result therefore establishes correct answers on this finite, known task. It does not establish prediction on unseen graphs or identify an internal matching algorithm.

The separate replay used NumPy affine and ReLU equations. Every answer and category accuracy matched. The largest logit difference was about 0.000000000000171, below the fixed tolerance of 0.000000001. This check used saved weights. It did not repeat training.

Seed 8002 has 13,600 recorded updates from its planned 32,768-update schedule. The guard also runs after an optimizer update but before the trace row. Thus the partial weights can include one additional unlogged update. Its initial weights, partial weights, and completed training trace remain in the results. It has no final evaluation score. It must not be counted as either a task pass or a task failure.

## Cause of the stop and correction for later work

The supervisor used a direct file write to refresh a short-lived private permit. That operation first truncates the file. A training check can read it before the new JSON text is written. The recorded error was `Expecting value: line 1 column 1 (char 0)`.

The stop happened during a permit refresh. The last live allowance check still allowed work. Neither the time cap nor the memory cap caused this stop. Private account values are excluded from the public record.

The local supervisor was corrected to write a temporary file and then replace the permit file in one operation. The [reproduction instructions](README.md) now state this requirement. The published scientific code and saved results were not changed. No scientific job was restarted to test the correction. A later attempt must test that update method before training and publish its continuation rules. This report does not claim that the correction has passed a new scientific run.

## What is new, and what remains unknown

The new evidence is one longer-trained development model that passes the fixed graph checks. The preparation record also gives measured costs for the proposed work. These are additions to this repository, not claims of a new method or principle.

One development model cannot validate four final seeds. It cannot establish that the wider recipe helps. There are no new register, digit, or flow scores. No internal intervention was tested. No claim about large language models follows from this result.

The [source review](REVIEW.md), [symbolic review](REVIEW_NUMERIC.md), and [result review](RESULT_REVIEW.md) describe the AI checks and their scope. External replication and human review remain undone.

## Next step

Keep this attempt unchanged. Before a later attempt, check usage again, test atomic permit replacement, and publish a continuation plan. The plan must state how the partial seed is handled and count all earlier compute. Do not silently replace a failed seed or erase this run. The other task branches and all later experiments remain unstarted. Automatic research remains paused.

The [results manifest](results/manifest.json) gives file hashes. The [run record](results/status.json) gives the complete model metrics, replay result, time, and stop reason. [Reproduction instructions](README.md) identify the environment and commands.
