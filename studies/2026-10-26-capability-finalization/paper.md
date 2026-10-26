# Task ability before explanation: two of four model panels pass the fixed tests

**Date:** 26 October 2026. **Study:** Completion of Experiment 1, the unfinished tenth study from the October 22 batch. **Status:** Complete experiment. Two task panels pass and two fail. **Review:** AI-authored, with separate AI checks. No human peer review is recorded.

## Abstract

An internal change can have a large effect even when a model does not perform its task reliably. Before further explanation tests, we tested four small model panels under fixed training and accuracy rules. The full record contains 40 complete models: 20 development models and 20 final models. Six graph models were reused; 34 model trainings were completed in this run. All four final register models and all four generated-point models passed their rules. Three of four graph models passed. None of the eight final digit models passed. One graph model scored 99.82% overall but failed three required groups. Register models passed their endpoint tests, yet accuracy across all required checkpoints fell to 93.75% in one longer test group. These results support specific later tests. They do not validate an internal explanation, establish a general training method, or estimate failure rates in large language models.

## 1. Question and contribution

The question is whether the models can perform the behavior that later explanations are meant to describe. Earlier repository studies used models with weak results on register operations, digit images, graph matching, and generated points. A measured effect in those models could not establish reliable task use.

This paper completes the fixed task-ability study. It adds a full result panel, saved model weights, training records, separate output calculations, and explicit decisions about later work. It is one study across three execution attempts. Its four tasks are not four papers.

The contribution is this checked experiment and its negative and positive results. The need to check task ability is not claimed as a new scientific principle. The study does not search for the best architecture or identify why one training recipe works better.

## 2. Fixed method

The [execution plan](analysis-plan.md) follows the [research roadmap](../../docs/RESEARCH_PLAN.md). The original [symbolic methods](../2026-10-19-capability-gates/METHODS_SYMBOLIC.md) and [numeric methods](../2026-10-19-capability-gates/METHODS_NUMERIC.md) specify the data, training schedules, arithmetic, and rules in full.

Each task had two training recipes and two development seeds, 8001 and 8002. A seed sets the starting random state. Recipe A kept the original architecture and used four times the original training schedule. Recipe B also increased hidden width, within 100,000 parameters. Recipe choice used development performance. The final seeds were 8101 to 8104. The digit task had two architectures for each seed.

For classification, selection used the lowest required-group accuracy across both development seeds. For generated points, both development seeds first had to pass the region-error limit. Selection then used mean likelihood error. Runtime broke a tie, followed by Recipe A. Final results did not choose the recipe.

Graph Recipe A was already selected. All six complete graph models were reused after output checks. The earlier partial seed 8103 lacked optimizer and random-generator state. A new public plan permitted one full restart of that same seed, from its original initialization. Both old partial attempts remain unchanged. Their compute remains charged. No failed final model was replaced.

The new plan and code were verified on GitHub at commit `95b53af3f603276a082a7bad630f72ffb6508c9a`. Passed preparation was verified at `20caef7307b5e591d4406112555dd33bc2d03cd1` before scientific execution.

| Task | Required result in every final model | Selected recipe |
| --- | --- | --- |
| Graph matching | At least 99% in each of five fixed groups | A |
| Register operations | At least 99% correct joint state at the endpoint of each of nine groups | A |
| Digit images | At least 95% on original images and 90% on each of four one-pixel shifts, in both architectures | B |
| Generated points | Likelihood error at most 0.20 nats and four-region error at most 0.05 | B |

A panel passed only if every required model passed every required group. These are fixed acceptance rules, not general standards. A scientific failure did not stop the other independent tasks. A failed control or resource check would have stopped the entire run.

## 3. Results

All planned training schedules finished. The full panel has 32 jobs and 40 distinct models because each digit job contains two models. All saved-weight replay checks passed. Correct calculation and task success are different results.

### Graph matching: one failure hidden by high overall accuracy

Final seeds 8101, 8102, and 8104 scored 100% in every group. Seed 8103 scored 99.8184% across all 65,536 graphs. It still failed three groups:

| Group in seed 8103 | Accuracy | Required accuracy |
| --- | ---: | ---: |
| Hard negative graphs without isolated nodes | 97.1196% | 99% |
| Target negative graphs | 98.3796% | 99% |
| Negative graphs with the other specified cause | 98.6111% | 99% |

Positive controls scored 99.6528%. The full panel therefore failed, with three passing models out of four. The planned graph-removal explanation test remains blocked. Selecting only the three passing seeds would change the fixed panel.

These graph states were all present during training. This result measures performance on known finite states. It is not prediction on unseen graphs.

### Register operations: endpoints pass, but complete paths are weaker

All four final models passed all nine endpoint groups. Each group contained 1,024 histories. The lowest group accuracy in each model was:

| Final seed | Lowest endpoint accuracy | Lowest complete-checkpoint accuracy at horizon 32 |
| --- | ---: | ---: |
| 8101 | 99.1211% | 94.2383% |
| 8102 | 99.6094% | 96.5820% |
| 8103 | 99.2188% | 94.1406% |
| 8104 | 99.3164% | 93.7500% |

The supporting complete-checkpoint score requires a correct state after the initial history and at every later checkpoint. It was not an Experiment 1 acceptance rule. It exposes errors that a correct final state can hide. Three models have at least one horizon-32 group below 95% on this measure.

This result permits preparation of a later internal-edit test under a new fixed plan. It does not already pass Experiment 4, which has separate complete-sequence requirements. It does not show that a model stores information in the expected form or replaces it correctly after an internal edit.

### Digit images: all eight final models fail

Original-image accuracy ranged from 72.0089% to 83.4725%, below the required 95%. All eight final models failed. Thus the problem was broader than tolerance to image shifts.

| Final seed | Direct architecture: original images | Filtered architecture: original images |
| --- | ---: | ---: |
| 8101 | 78.4085% | 77.3511% |
| 8102 | 74.7913% | 76.7947% |
| 8103 | 79.2432% | 83.4725% |
| 8104 | 72.0089% | 79.2432% |

The models used the original UCI training and test files. Test images were not used for training. However, this test set had already informed the wider project. It is not a new-dataset confirmation. These results do not show that digit classification is inherently difficult or that no better small model exists.

### Generated points: all four final models pass the two rules

The models learned a fixed mixture of four two-dimensional Gaussian distributions. The likelihood score compares the true and model log densities on 8,192 evaluation points. Lower is better. The region score is half the sum of absolute differences between generated frequencies and target probabilities in four regions. It uses 32,768 generated points per model.

| Final seed | Likelihood error, nats | Four-region error |
| --- | ---: | ---: |
| 8101 | 0.170693 | 0.020146 |
| 8102 | 0.144206 | 0.023224 |
| 8103 | 0.179173 | 0.011383 |
| 8104 | 0.177869 | 0.021030 |

Every model was below 0.20 nats and 0.05 region error. This passes the declared panel. Four-region agreement does not bound error in the full continuous distribution. It also does not establish a useful internal feature or better samples under an edit.

## 4. Checks, compute, and complete evidence

Preparation checked shapes, precision, unchanged-input controls, small known answers, explicit equations, permit failure, and timing. All checks passed. The scientific run used an Apple M4 CPU, two compute threads, and one inter-operation thread. It used Python 3.12.14, NumPy 2.3.5, and PyTorch 2.8.0. Peak process memory was 389.88 MiB, below four GiB.

The new run charged 654.5740 seconds. The cumulative scientific runner charge, including prior attempts, was 971.6391 seconds. A separate 60-second allowance covers the saved-result audit. Another 60-second allowance covers result packing and restoration checks. Total Experiment 1 charge is therefore **1,091.6391 seconds, or 18.19 minutes**, under its four-hour limit. These allowances are conservative charges, not measured durations. Preparation has a separate cumulative charge of 79.7949 seconds. See the [compute record](analysis/compute-accounting.json).

The [saved-result audit](analysis/result-audit/audit.json) checked all 40 models and 32 jobs. It checked raw scores, supporting register scores, final decisions, recipe selection, task order, training traces, and old and new file hashes. All checks passed. It did not repeat training or forward calculations. The scientific runner had already performed separate saved-weight calculations.

The [model table](analysis/result-audit/models.csv), [full group scores](results/metrics.csv), and [run record](results/status.json) are directly readable. Four [result bundles](artifacts/manifest.json) contain all 218 new raw task files, including weights and traces. A fresh extraction restored every file with the original hash. Reused models remain in the earlier study folders. The [reproduction guide](REPRODUCTION.md) explains the layout.

## 5. Limits and next steps

The fixed panels are small. Models within a task share test cases, so the rows do not provide independent estimates of a broad population failure rate. The rules were fixed, but the project had prior knowledge of these task families. No result here validates an internal circuit or a semantic interpretation.

The study is not a controlled estimate of the effect of training longer or increasing width. Recipes, seeds, and evaluation arrangements differ from earlier papers. No complete training run was independently repeated as a separate reproduction test. AI code review and output checks are not human peer review.

Next, follow Experiment 2A: test the full internal-edit procedure on its separate diagnostic models. Then test groups, search, and selection in 2B. Keep the failed graph panel as a block on its dependent explanation claim. Use the register endpoint pass only for the scope it supports; retain the weaker full-path result in later planning. A new digit recipe or another graph restart requires a new justified study. It cannot repair this result after the fact.

This completes one previously unfinished study. It adds one completed paper to the requested program after full remote verification. It does not establish 200 discoveries or permit a separate paper for each task.

## References and source records

1. [Fixed continuation plan](analysis-plan.md), [source record](sources.md), and [review before execution](PRE_RUN_REVIEW.md).
2. [Preparation record](preparation/status.json) and [separate preparation review](PREPARATION_REVIEW.md).
3. Original register study: [A small, uncertain gain from training state edits on one future step](../2026-10-13-register-write-order/paper.md).
4. Original digit study: [Image shifts and channel removal in weak digit classifiers](../2026-10-13-shift-ablation/paper.md).
5. Original graph study: [Testing a Hall-rule interpretation in a small matching classifier](../2026-10-13-hall-obstruction/paper.md).
6. Original generated-point study: [Choosing flow neurons by likelihood damage or output change](../2026-10-13-flow-unit-ranking/paper.md).
7. Retained [October 19 attempt](../2026-10-19-capability-gates/) and [October 22 attempt](../2026-10-22-capability-completion/paper.md).
