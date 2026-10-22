# Research plan: from internal effects to tested explanations

**Updated:** 22 October 2026. First version: 13 October 2026.

**Status:** Updated plan only. No experiment was run for this update. Automatic research remains paused.

**Basis:** The [review](REVIEW_PAPER.md) and [one-page summary](REVIEW_SUMMARY.md) cover 24 completed studies at [repository version 802bacf](https://github.com/gavenren/mechanistic-interpretability-research/tree/802bacf849f073c34c838c33876bf2f121ff06d3). The new batch contains nine completed papers. Its planned tenth study is unfinished. Completed Study 10 in the original group is a different study.

**First priority:** Complete the [unfinished task-ability study](../studies/2026-10-22-capability-completion/paper.md), which is Experiment 1 below. It is not counted among the 24 completed studies.

**Review:** AI-authored. The [update check](reviews/2026-10-22-RESEARCH_PLAN_UPDATE_REVIEW.md) applies to this version. The [original plan review](reviews/RESEARCH_PLAN_REVIEW.md) remains part of the history. No human peer review is recorded.

## Aim and decision rules

The aim is to test when an internal explanation predicts the right behavior under a precise change. An output effect alone is insufficient.

Start with small, controlled tasks. Test larger pretrained models only after the test methods work. Keep failed runs and negative results. A useful outcome can be a clear failure of a proposed explanation.

The thresholds below are **acceptance rules**, not established field standards or predictions. Experiment 1 retains its already fixed rules. The new tests have proposed rules that must be fixed in their execution plans. They define a demanding first research program. They must be fixed before the final test results are opened. A failed threshold limits the stated claim; it does not prove that no representation or better method exists.

This document is a research roadmap. Each experiment still needs a public execution plan, code review, fixed input list, and numerical controls. Exact model versions, training recipes, data splits, and example counts must be locked there. These details cannot be chosen from final test results.

## Order of work

Keep the existing experiment numbers so that earlier run records remain clear. Experiment 2 now has three parts. Run the cheaper numerical experiment, 5, before the language experiments, 3 and 4.

| Run order | Experiment | Required earlier work | Decision to continue |
| --- | --- | --- | --- |
| 1 | 0. Refresh preparation and the continuation record | Read current usage and saved records | All controls pass; projected work fits the remaining caps |
| 2 | 1. Finish task-ability panels | 0 | Each task needs all required final models; report failed or unfinished branches |
| 3 | 2A. Check the full edit procedure | 0; resolve the Experiment 1 attempt first | The measurement controls pass; all edit choices are explicit |
| 4 | 2B. Test groups, search, scores, and selection | 2A | Search claims match the exact comparison; the validation rule passes its null test |
| 5 | 2C. Test intended and unwanted effects | 2A and 2B; graph branch also needs 1 | Each model passes separate intended-change and unwanted-change rules |
| 6 | 5. Test numerical reliability and useful savings | 0; saved Study 15 models | Valid bounds and the stated practical saving |
| 7 | 3. Test language transfer | Relevant controls from 2A and 2B; cheaper work attempted first | Both candidate models pass task and explanation rules |
| 8 | 4. Test later use and replacement | Register branch needs 1; language branch needs 3 | Immediate, later, and replacement behavior all pass |
| 9 | 6. Independent repetition and human review | Finished records from 2C and 5 | Report reproduction separately from scientific confirmation |

A stopped resource check pauses all new research. A scientific failure stops the dependent claim or task branch. It does not block unrelated work after that failure is reported. Do not skip the unfinished study to produce more papers. No stage must produce a new paper.

## How the 24 studies inform this plan

| Completed studies | Question carried forward | Planned use |
| --- | --- | --- |
| 1, 2, 3 | Does an edit change the intended variable, and what does the code or coordinate choice mean? | 2A and 2C; a new rare-feature training experiment is deferred |
| 4, 5, 7, 8 | Does a fixed copying explanation transfer across words, positions, and token splits? | 3; do not assume the same head index has the same meaning in another model |
| 6, 10 | Does stored information affect later steps and stop acting after replacement? | 1 for register ability, then 4 |
| 9 | Do training histories affect the result? | Preserve histories in 6; a separate order-effect extension is deferred after the earlier tiny result |
| 11, 12, 14 | Do weak task models meet the required ability or fit rules? | 1; the graph explanation then enters 2C |
| 13, 15 | Does the exact deletion rule and numerical target match the explanation? | Controls in 2A and bounds in 5; a new spike study is deferred |
| 16, 18, 20, 22 | Do donor values, edit order, code ambiguity, and normalization change the test? | 2A; apply the relevant checks again in 3 and 4 |
| 17, 19 | Do individual tests or a greedy search miss smaller working groups? | 2B |
| 21, 23, 24 | Do averaging, score choice, and selection create a misleading result? | 2B; separate final validation in every later stage |

The nine new papers give mathematical tests and reference examples. They do not already establish these effects in trained language models. Deferred questions remain gaps; they are not completed extensions or extra funded stages.

## 0. Prepare the tests

Create a record of every input, model, intervention, score, and stop rule. An *intervention* is the specified internal change. Separate development cases from final test cases.

Use dummy inputs to test shapes, memory, arithmetic, and runtime. Do not use final scientific outputs to adjust the design. Recheck the exact meaning of each intervention in a separate implementation. Include unchanged-model controls and a known correct reference where one exists.

Before each final run, measure work on the selected hardware. Record the processor, available memory, software, thread count, precision, and batch size. Save training and inference times separately where possible.

**Success:** All required controls pass; planned data fit in memory; the projected run and replay fit the remaining stage cap. Otherwise stop. A later revision must be public before scientific testing. Do not reduce the fixed Experiment 1 panel, change its thresholds, or extend its cap to obtain a pass. Numerical controls remain required in every later experiment.

Verify the six complete graph models and their saved files. Check the stop-file writer with atomic replacement: write a new file, then replace the old file in one operation. Test that missing, empty, expired, or denied permits stop the runner. Test saving on a forced stop with dummy data. Do not use a final result for this check.

Preparation already used 15.6075105 seconds. Its unchanged one-hour cap has 3,584.3924895 seconds left. This cap covers dummy controls and timing across the program. Each scientific stage also pays for its own saving, evaluation, and replay. No benchmark is run for this plan update.

## 1. Finish the unfinished task-ability study

**Question:** Can the models reliably perform the specific behavior that the explanation is meant to describe?

**Current state:** Six graph models are complete: four development models and final seeds 8101 and 8102. All passed all five graph categories and saved-weight checks. Recipe A is already selected. Do not select it again. Seed 8103 has 22,028 recorded updates and partial weights, but no final score. Its weights can include one additional unrecorded update. Seed 8104 and the other three task branches have not started. Two passing final models do not pass the required four-model panel.

**Explicit proposed continuation rule:** The partial file for 8103 contains model weights, but no optimizer or random-generator state. It cannot support an exact continuation. The next public execution plan must permit exactly one full restart of **the same seed 8103**, from its original initialization. Keep Recipe A, its data order, and its full 32,768-update schedule. Save the restart in a new directory. Keep both earlier partial attempts and their traces unchanged. Count their compute. Do not replace the seed, extend the schedule, or retry automatically after another stop.

Reuse all six complete graph models after saved-weight checks pass. Complete 8103, then 8104. Next run register operations, digit images, and generated points, in that order. This restart rule is a change from the previous execution plan, which allowed only the earlier development-seed 8002 restart. The new continuation code, tests, and rule must be public before training. This roadmap does not start that training.

Revisit four weak-model settings: matching graphs, register operations, digit images, and generated two-dimensional points. These come from Studies 12, 10, 11, and 14.

Allow at most two training recipes per task. Use development seeds 8001 and 8002, then final seeds 8101, 8102, 8103, and 8104 for the chosen recipe. The completed graph jobs are reused, not repeated. The digit task has both direct and filtered architectures. This gives at most **20 development models and 20 final models** across the four tasks. The two retained partial attempts are extra training invocations, not extra distinct models. Finishing every branch would give 40 complete models plus those two partial attempts. Six complete models already exist, so at most 34 model trainings remain.

Recipe A keeps the original architecture and allows four times the original training schedule. Recipe B also allows up to twice the hidden width, within 100,000 parameters per model. Keep other settings fixed in the execution plan. Define both recipes before development runs. Choose by development task performance, with shorter runtime as the tie rule. Do not use final task or intervention results for selection.

| Task | Required result in every final model |
| --- | --- |
| Matching graphs | At least 99% accuracy in each of the original five categories: all graphs, hard negatives, target negatives, positive controls, and other-cause negatives |
| Register operations | At least 99% joint-state accuracy in all nine fixed groups: horizon 0 after the 12-operation history; copy-only and write-only at horizon 1; and both operation orders at horizons 2, 8, and 32 |
| Digit images | At least 95% accuracy on original images and 90% on each fixed one-pixel shift, in both architectures |
| Generated points | Excess negative log likelihood at most 0.20 nats, and four-region probability error at most 0.05, as defined in Study 14 |

For classification, select the recipe with the highest worst-group development accuracy. For generated points, first require the region-error limit, then select the lowest development likelihood error. If neither recipe meets that limit, stop the generated-points branch. Keep both failed recipes; do not add another. Keep the complete development record.

Retain the original graph task definition: ordinary graphs can be seen during training; new intervention pairs are the later test. Do not call this unseen-graph prediction. Use new generated sequences and points for final tests. For digits, select recipes within the original training split and retain the original test split for comparison. That dataset has already informed this project; a later new-dataset test remains necessary.

**Success:** Every model in a task panel passes every required category. Report a partial pass if only some models pass. A partial pass cannot support a claim about the full panel.

**Compute:** Allow 1.5–3 additional CPU hours for planning. This is not a measured prediction or a fresh four-hour allowance. [Recorded scientific time](../studies/2026-10-22-capability-completion/results/status.json) is 317.065045251 seconds, including earlier attempts. The unchanged four-hour cap leaves **14,082.934954749 seconds**, or about **3.91 hours**.

The [saved conservative job estimates](../studies/2026-10-22-capability-completion/revised-preparation/status.json) give 10.96 minutes for the two graph jobs, 54.76 for register jobs, 60.75 for digit jobs, and 18.23 for generated-point jobs. Their sum is 144.71 minutes, or about 2.41 hours of remaining work. Including prior scientific time gives about 2.50 hours. This is an arithmetic projection from the saved benchmark, not a new measurement. It excludes newly required replay and continuation setup, which must be added before launch.

Refresh dummy timing on the chosen hardware. Keep the fourfold allowance and the previously fixed warmup/timing method. Add the remaining jobs, reused-model checks, controls, saving, and setup. Add all prior scientific time. Launch only if this total fits four hours. Retain both the earlier failed estimate and its published revision. Neither is permission to keep revising timing until it passes.

## 2A. Check the complete edit procedure

**Question:** Does the measured change come from the intended edit, or from an unstated choice in the procedure?

Use the eight saved selector models from Study 1 as a diagnostic panel. They are not fresh confirmation models. Keep the original finite task and its ordinary ability checks. Fix development and test pairs before evaluating new edits. Compare no change, zero replacement, donors with the same variable value, donors with a different value, and a known coherent input change. Use the task rule to state the required answer for each test.

Test two edits in both orders. Check each requested value after the second edit. For decoded features, reuse the six saved reconstruction-only encoders from Study 2 at its primary condition: 1% event frequency and amplitude 0.25. List their paths before testing. Compare only code alternatives that meet declared reconstruction, nonnegativity, and sparsity constraints. If none exists, report that result. A mathematical alternative is not proof that the encoder produces it.

For normalization, reuse the fixed Pythia-14M model version from Studies 4–8. Use 32 fixed prompts of at most 128 tokens, at most eight sites, and three edit sizes. Compare ordinary recomputation of mean and scale with a diagnostic that keeps them fixed. Include every donor pass and replay in a **one-million-forwarded-token cap**. This branch measures numerical spread. It makes no copying or semantic claim unless the relevant task tests also pass.

**Measurement success:** Every no-change control must reproduce the declared computation within its fixed tolerance. The separate implementation must match all recorded test effects within the same tolerance. Publish the tolerance and arithmetic before the test; do not enlarge it after a failure. Save results for every donor class, edit order, code constraint, and normalization rule. This stage succeeds as a diagnostic even if it finds that edits disagree.

**Limit on interpretation:** Do not average away a failed donor class. An explanation that is claimed to work across donors or orders must later pass Experiment 2C's absolute rules in each claimed group. A result for only one procedure must name that procedure.

**Compute:** 0.25–1 CPU hour estimated; 2-hour cap. Reuse saved weights. No new base-model training is included. Stop on any calculation-check failure or either the time or token cap.

## 2B. Test groups, search, scores, and selection

**Question:** Can the search and scoring procedure support the claim made from its largest effect?

Reuse the diagnostic selector panel from 2A. Choose a fixed panel of eight components using development data only. Evaluate all 256 retained subsets on the locked test pairs. Keep all other components fixed. Compare the exact smallest working subset with a one-part-at-a-time deletion search under 32 fixed visit orders. Report individual, paired, and complete-group effects.

**Search check:** A retained subset must keep at least 99% of intact answers and have mean probability-distance error at most 0.01. Use this same rule for every search method. A claim of minimum size passes only if the search matches exhaustive enumeration in every panel. That minimum is only within the chosen eight components and fixed test cases. Larger returned sets or multiple valid groups must be reported. They do not show that the model has no explanation.

From the same saved effects, report signed averages, average magnitudes, score gains before conversion to probabilities, probability gains, and task correctness by context. Fix a separate candidate for each selection rule using development data. Evaluate each on untouched test pairs. A ranking reversal is a result to explain; it is not automatically a failure of either metric.

**Selection check:** Make an explicit zero-effect noise model by centering a fixed bank of development effects. Do not use final test effects to create this bank. Draw selection and validation samples independently from the bank in every trial. Preserve the declared dependence between candidates within each draw. This is a constructed reference, not a claim that real effects are zero. Test 8, 32, and 128 candidates at 64 examples per estimate, with 2,000 trials per condition. Compare reused-data testing, separate validation, and a selection-aware correction. Fix the test law, threshold, and random seeds before trials. Use analytic checks where that law permits them.

The primary separate-validation rule passes this calibration check only if the upper 95% Wilson bound on its false-positive fraction is at most 7% in every condition. The nominal target is 5%; the wider rule allows finite simulation error. These are per-condition intervals, not simultaneous confidence bounds. Report all three methods even if they fail. This check concerns false positives only. It does not measure power or real-circuit error rates. A positive model claim still needs 2C's task criteria.

**Compute:** 0.25–1 CPU hour estimated; 2-hour cap. No new base-model training. The fixed 256-subset search, 32 orders, saved-score comparisons, simulation, and separate calculation check must all fit. If the validation rule fails, stop its use in later positive claims. Do not tune it on the final simulation or test pairs.

## 2C. Test intended and unwanted effects

**Question:** Can an internal edit change one specified variable while preserving unrelated behavior?

After 2A and 2B, start with the selector task from Study 1. Train eight fresh models with its fixed ordinary-task recipe. Require 99% intact accuracy in every model. Use separate cases for learning an edit, choosing it, and testing it. The ordinary task states remain known; only the intervention pairs are separated.

Compare three methods with the same three-direction edit size:

1. Select three original neurons by intended-change error.
2. Select three original neurons under explicit limits on unwanted effects.
3. Learn a rotation of the internal state, then replace three selected coordinates in that rotated state.

Use the same donor cases and the same final test pairs. Limit the learned edit to 4,000 updates per model. Fix its loss and learning rate in the execution plan. If no neuron set meets the selection limits, record that result rather than silently relaxing a limit.

**Success for a variable explanation:** In each of the eight models, require at least 95% correct intended answers. Also require mean probability-distance error at most 0.05 in both active and inactive groups. When the proposed variable has the same value, at most 1% of answers may change. Here, probability distance is the zero-to-one TV measure defined in the review.

**Success for the claimed method improvement:** The learned edit must pass those absolute rules and improve intended-answer accuracy by at least 5 percentage points over Method 1 on average. At least six of eight paired differences must be positive. These are practical acceptance rules, not a population significance claim.

If the graph models from Experiment 1 pass, also repeat Study 12's fixed four-neuron selection. Test the strong claim that removal disables the target rule while preserving other decisions. Require at least 90% target false acceptance after removal, with at least 99% accuracy retained in each control category. Use the original selection and test contexts. Failure leaves only a narrower measured effect; it does not establish absence of the rule.

Include random edits, no edits, full-layer edits, and known correct input changes. A known correct reference can use information unavailable to the learned method. State that difference.

**Compute:** 0.25–1 hour estimated; 2-hour cap. This is the original Experiment 2 allocation. The graph branch is included. Further rare-feature and spike-deletion extensions wait until this test is complete.

## 3. Test transfer to new conditions

**Question:** Does a selected component still support the same explanation when the input conditions change?

Use two public base models as the planned candidates: [Pythia-70M](https://huggingface.co/EleutherAI/pythia-70m) and [SmolLM2-135M](https://huggingface.co/HuggingFaceTB/SmolLM2-135M). Their official model cards were checked again on 22 October 2026 for model identity and size. The cards do not establish ability on this experiment. Record exact weight and tokenizer revisions before use. No language-model training is included.

Within each model, use 256 calibration cases to select from at most 32 prespecified attention-head sites. An attention head is an internal operation that combines information from text positions. Freeze the selected site, donor rule, score, and context labels before final testing. Apply the relevant controls from 2A and the separate-validation rule from 2B. Report score changes by context alongside answer accuracy. Across models, test the selection method and copying account; do not assume that the same head index has the same meaning. A failed search of these sites cannot rule out a useful circuit elsewhere.

Use three final families of 256 cases each: new words, new sentence forms, and valid alternative splits of the same key text into tokens. A token is a text unit used by a model. Keep requested answers to one token within each model. Define compatible words without looking at answer scores. Limit prompts to 224 tokens, leaving up to 32 continuation tokens within a 256-token sequence. Report the families separately.

Test the intact model, the selected edit, random-site edits, and the unchanged-input control. Use the exact donor replacement specified in the plan. Define the required changed answer from the task rule.

**Success:** Both models must give the requested intact answer as their top token in at least 90% of each test family. The selected edit must give the required changed answer in at least 90%, with no more than 2% unwanted answer changes in same-value controls. Require at least a 5-point intended-answer advantage over the fixed random-site average in each model. Report all cases, not only initially correct cases.

If either model fails task ability, report the failure. Do not search for larger replacements within this plan. A follow-up would need a new model choice and budget. A correction for joint patch effects from Study 4 is deferred until this task test passes.

**Compute:** 3–12 hours estimated; 16-hour cap. Allow at most 16 million forwarded tokens across both models, calibration, controls, final tests, and a separate replay. Count a token again each time a new model pass processes it.

## 4. Test whether information is used and then replaced

**Question:** Does an edited state affect later operations exactly when the task requires it?

Use the four capable register models from Experiment 1. Compare the immediate-loss edit from Study 10 with an edit trained on future use. Keep the edit rank at four. Match the total forward and backward recurrent-position budget between methods. Fix update schedules after a dummy cost check. Report optimizer steps as well; equal steps alone do not give equal compute.

Use 1,024 new test sequences at each horizon of 1, 2, 8, and 32 operations. At horizon 1, balance copy-only and write-only cases. At horizons 2, 8, and 32, balance copy-before-write and write-before-copy cases. Use known state rules to determine every required intermediate answer. Train edit methods on separate sequences.

**Register success:** At least 95% of sequences must pass the complete set of immediate, intermediate, and final checks in each order/horizon group, in every model. After replacement, no more than 1% may retain the old value where it should be gone. The future-loss method must improve mean complete-sequence success by at least 5 points. No group may lose more than 1 point.

If both language models from Experiment 3 pass, add a separate language test with 256 new prompts per model. Compare the same selected edit after 8 and 32 continuation tokens. Test an imposed common continuation and free generation separately. Fix the decoding rule. Free generation can change visible words as well as internal state.

**Language success:** Intact task accuracy must reach 90% in each condition. The edited run must give the required later answer in at least 90%, while same-value controls retain at least 98% of their answers. Persistent output change without task success does not pass. This test does not isolate hidden memory from visible text during free generation.

**Compute:** 3.5–14 hours estimated; 18-hour cap. Of this, 0.5–2 hours is for register work and 3–12 for language work. The language work has its own 16-million-forwarded-token limit, including replay. Skip its cost if Experiment 3 fails.

## 5. Test numerical reliability and useful savings

**Question:** Can a tighter valid bound resolve effect directions with less work, without giving an incorrect guarantee?

Use the four saved equilibrium models from Study 15 and its same fixed 15,360 effects. Preserve the exact-real target, permanent coordinate removal, coarse stopping rule, and sign-first decision order. Compare the published global bound with one tighter candidate bound.

Develop that bound only on dummy cases and at most four separately trained development models. Fix their seeds and training schedules before development. Publish the bound's mathematical assumptions and implementation before the fixed panel. Check known positive, negative, zero, very small, and unresolved examples with exact arithmetic where feasible.

**Reliability success:** Every candidate interval must contain an independently validated tighter reference interval. Refine each reference until its width is at most 0.0000000001, or until 1,024 extra paired state updates. A paired update advances both the original and edited state. If the reference is still too wide to test containment, label the case unresolved. Non-containment alone does not prove that the candidate bound is wrong. This is a conservative acceptance check, not a necessary condition for every valid bound.

No certified sign may contradict the validated reference. Report unresolved cases and magnitudes within ±0.0000001 separately. Finite tests alone do not prove universal numerical soundness; the bound also needs a valid derivation and checked arithmetic.

**Practical success:** Resolve every case to a strict sign or the same small-magnitude criterion. Reduce median end-to-end time by at least 25% in each model over five paired timed repetitions. Include bound construction and validation costs. Report state updates, memory, and timing separately. Use identical hardware, alternate run order, and count all repetitions.

This tests efficiency on a known panel. A new dataset or architecture is outside the first budget. Other method gains must also report complete costs. The new digit and flow panels can support later studies, but their old stability scores alone will not establish semantic explanations.

**Compute:** 0.25–1 hour estimated; 2-hour cap. No repeated training of the four scientific models is included here. Development models and the separate calculation audit are included.

## 6. Repeat the work independently

Prepare an external repeat package for Experiment 2C's selector panel and Experiment 5's numerical panel. For the latter, retrain the original four model seeds from the fixed Study 15 training plan. Repeat all Experiment 2C training, with its training history and fixed donor, context, and scoring checks. Also check exactly two fresh seeds per panel as an extension; report them separately.

**Reproduction success:** On the same software and arithmetic, reproduce the recorded gate decisions and effects within locked numerical tolerances. On different hardware, justify tolerances before result comparison. If fresh training differs, report the difference; do not substitute saved weights and call it a training replication.

**Scientific confirmation:** The original acceptance rules must still hold in the repeated original-seed panels. Report fresh-seed results separately. A reproduced negative result counts as successful reproduction, not successful validation of the method.

A human reviewer should assess the questions, controls, novelty claims, failure records, and interpretation. External time and access are not secured. A separate AI review remains useful but does not satisfy this requirement. No external group has been scheduled by this plan, and no invitations are sent for this update.

**Compute:** 0.5–2 hours estimated; 4-hour cap for one external repeat package. This allocation is not a promise that an external group will participate.

## Stop rules for every stage

1. **Usage:** Check the live account meter before preparation, downloads, training, or other substantial work. Keep more than 20% in every reported core usage window. Stop if any window has 20% or less, the meter is unavailable, or ordinary usage is unavailable. Never redeem a reset or buy extra usage for this plan.
2. **Live permission to compute:** While a runner is active, its supervisor checks usage at least once a minute. A local permit lasts at most 120 seconds. Update it by atomic replacement. The scientific runner cannot renew its own permit. A missing, invalid, expired, or denied permit stops the run.
3. **Time and memory:** Include controls, failed attempts, saving, and replay in the applicable elapsed-time cap. Include earlier work in preparation and Experiment 1. Stop at the stage cap, token cap, or memory limit, whichever comes first. Allow time to save before the deadline. Never transfer unused hours between stages silently.
4. **Data and arithmetic:** Stop on an input-hash mismatch, missing required state, nonfinite value, failed no-change control, or failed numerical replay. Retain the error and partial records. A repair needs a new reviewed execution record before a rerun.
5. **Scientific failure:** Keep every planned seed and subgroup. Do not lower thresholds, replace failed seeds, add recipes, extend schedules, or select a metric from final results. Finish only the already planned valid comparisons that fit the budget, then close the failed claim. Do not run its dependent explanation stage.
6. **Interruption:** Save partial state safely. Record the exact step boundary and whether full training state exists. Do not restart automatically. The proposed single restart of final seed 8103 is the only new restart rule in this update.
7. **Scope:** Use local compute only. No rented compute, paid model API, language-model training, or unplanned GPU run is included. Keep automatic research paused. A documentation update does not start an experiment.

## Compute assumptions and budget

Use one local CPU process with two compute threads and one inter-operation thread. Preparation and Experiment 1 keep the stricter four-GiB peak process limit and requires 20 GiB free disk. Other stages use at most 16 GiB of process memory and the same free-disk requirement, subject to actual available memory. These are ceilings, not measured needs. Load one language checkpoint at a time. Save required results in batches.

One CPU hour means one elapsed hour for this process, not one core-hour. Estimates include scientific runs, controls, saving, and one calculation replay unless stated otherwise. They exclude coding, human review, downloads, and assistant usage. They cannot predict this account's usage allowance.

| Run order | Stage | Additional CPU hours, estimate | Total stage cap | Prior seconds charged |
| --- | --- | ---: | ---: | ---: |
| 1 | 0. Refresh preparation | 0.25–0.5 | 1 hour | 15.6075105 |
| 2 | 1. Finish task ability | 1.5–3 | 4 hours | 317.065045251 |
| 3 | 2A. Complete edit procedure | 0.25–1 | 2 hours | 0 |
| 4 | 2B. Groups, scores, and selection | 0.25–1 | 2 hours | 0 |
| 5 | 2C. Intended and unwanted effects | 0.25–1 | 2 hours | 0 |
| 6 | 5. Numerical reliability | 0.25–1 | 2 hours | 0 |
| 7 | 3. Language transfer | 3–12 | 16 hours | 0 |
| 8 | 4. Future use and replacement | 3.5–14 | 18 hours | 0 |
| 9 | 6. External repeat package | 0.5–2 | 4 hours | 0 |
| **All branches** | **Program total** | **9.75–35.5** | **51 hours** | **332.672555751** |

The [budget table](RESEARCH_COMPUTE_BUDGET.csv) gives the same values and the remaining cap per stage. The total cap includes earlier preparation and Experiment 1 work. About **50.91 hours remain under the combined caps**, but each stage must stay within its own remainder. This is a ceiling, not an estimate of necessary work or a promise that all branches will run.

The new diagnostic stages 2A and 2B add four hours to the previous 47-hour roadmap ceiling. The original preparation, task-ability, and later-stage caps are unchanged. The updated estimate for remaining task-ability work uses the saved timing record. Other ranges are planning allowances, not benchmarks. Failed or deferred branches reduce cost. External participation is not secured.

Experiment 2A permits at most one million forwarded language tokens. Experiments 3 and 4 each retain a separate 16-million-token limit. All limits include donor passes, selection, controls, and replay. Count a token again whenever a new pass processes it. The whole program therefore has at most 33 million such tokens if every language branch runs.

For each 16-million-token stage, the earlier assumed rate of 500–2,000 tokens per second gives about 2.2–8.9 hours before other overhead. These rates remain unmeasured planning assumptions. Hooks, sequence length, precision, and cache use can change them. Use dummy timing on the chosen hardware before launch. Include all work and a twofold safety factor. Experiment 1 is the exception: it retains its stricter fourfold allowance. The measured projection must fit the remaining stage cap; a failed projection stops that stage.

At most 40 complete base-model training runs belong to Experiment 1, eight fresh selector runs to 2C, four development runs to 5, and sixteen original- or fresh-seed repeat runs to 6. That is **68 planned complete training runs across the full program**, including repeated original seeds. These are not 68 independent confirmations. Six Experiment 1 models already exist, leaving at most 62 complete training runs. The two retained partial attempts are extra invocations and remain charged. Edit fitting and replay are additional work within the stage caps. Stages 2A and 2B reuse saved models.

### Measured reference points

The repository provides these elapsed times. They show why small tests come first; they do not predict the new program.

| Existing study | Recorded scientific run | Separate replay | Scope |
| --- | ---: | ---: | --- |
| [10. Register operations](../studies/2026-10-13-register-write-order/paper.md) | 32.87 seconds | 13.86 seconds | Four trained models, edit training, tests, and controls |
| [11. Image shifts](../studies/2026-10-13-shift-ablation/paper.md) | 184.84 seconds | 123.15 seconds | Follow-up reused five checkpoints; not full retraining of all sixteen models |
| [14. Flow neurons](../studies/2026-10-13-flow-unit-ranking/paper.md) | 91.71 seconds | 67.29 seconds | Four trained flows and complete intervention evaluation |
| [15. Equilibrium effects](../studies/2026-10-13-equilibrium-sign-audit/paper.md) | 19.80 seconds | 9.03 seconds | Four trained models and numerical effect audit |
| [7. Position and token access](../studies/2026-10-13-position-access/results/run-metadata.json) | 181.20 seconds | 90.58 seconds | One fixed Pythia-14M checkpoint; no model training |

These records use arm64 macOS and generally two compute threads. They do not record the exact chip model or RAM. Run times include more than training. Audit times cover their stated saved-weight replay, not independent retraining. Development checks and earlier failures can add further time.

## Work schedule and deliverables

Allow **four to eight working weeks** for one researcher who can implement and review the tests. This is a planning allowance, not a delivery promise. Machine hours and researcher weeks measure different work.

- **Week 1:** Refresh preparation and address the unfinished task-ability study first. Report every unfinished or failed branch.
- **Weeks 2–3:** Run edit-procedure and selection diagnostics, then qualified specificity tests.
- **Week 4:** Run the numerical comparison before the larger language tests.
- **Weeks 5–6:** Run qualified language-transfer and future-use branches. Skip branches whose required tests failed.
- **Weeks 7–8:** Finish audits and the external repeat package. External replication and human review may take longer. These are scheduling allowances, not deadlines.

For each stage, save the public plan, exact code version, inputs, seeds, model versions or weights, raw measures, costs, controls, and failures. Report every planned seed and subgroup. Use model-level summaries; do not count thousands of related input rows as independent models. Report uncertainty without treating a four-seed interval as proof of broad reliability.

Link each result to its plan and review. Do not promise one paper per stage or label a method new without a focused source check. Keep all 24 completed studies and both partial training attempts unchanged as the starting evidence.

## Sources and current status

The [full review](REVIEW_PAPER.md) and its [source manifest](REVIEW_SOURCES.json) provide the evidence behind the proposed questions. The official [Pythia-70M model card](https://huggingface.co/EleutherAI/pythia-70m) and [SmolLM2-135M model card](https://huggingface.co/HuggingFaceTB/SmolLM2-135M) were checked again on 22 October 2026 for model identity and size. They provide no runtime measurement for this plan.

Experiment 1 retains its fixed success rules. Later-stage rules and future compute amounts remain proposals until their execution records are fixed. The historical status and times come from saved records. No result in this update comes from a new experiment. Automatic research remains paused. This publication adds the plan only.

