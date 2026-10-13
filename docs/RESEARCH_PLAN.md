# Research plan: from internal effects to tested explanations

**Date:** 13 October 2026  
**Status:** Proposed work. No experiment in this plan has started.  
**Basis:** The [review](REVIEW_PAPER.md) and [one-page summary](REVIEW_SUMMARY.md) at [commit 15755f0](https://github.com/gavenren/mechanistic-interpretability-research/tree/15755f0aa61cced6d37aa0a5a765ca1fb0936756).  
**Review:** AI-authored, with a [separate AI plan review](reviews/RESEARCH_PLAN_REVIEW.md). No human peer review is recorded.

## Aim and decision rules

The aim is to test when an internal explanation predicts the right behavior under a precise change. An output effect alone is insufficient.

Start with small, controlled tasks. Test larger pretrained models only after the test methods work. Keep failed runs and negative results. A useful outcome can be a clear failure of a proposed explanation.

The thresholds below are **proposed acceptance rules**, not established standards or predictions. They define a demanding first research program. They must be fixed before the final test results are opened. A failed threshold limits the stated claim; it does not prove that no representation or better method exists.

This document is a research roadmap. Each experiment still needs a public execution plan, code review, fixed input list, and numerical controls. Exact model versions, training recipes, data splits, and example counts must be locked there. These details cannot be chosen from final test results.

## Order of work

| Order | Experiment or preparation | Required earlier work | Decision to continue |
| --- | --- | --- | --- |
| 0 | Check methods, inputs, and resource estimates | None | Controls pass and the planned work fits its cap |
| 1 | Build models that pass the relevant task checks | 0 | Continue each task branch only if its fixed model panel passes |
| 2 | Change the intended variable and preserve other behavior | 0; graph branch also needs 1 | Report each model and each requirement separately |
| 3 | Test a fixed explanation on new language inputs and models | 0; run after the cheaper tests | Both task ability and the explanation must pass |
| 4 | Test stored information over time | Register branch needs 1; language branch needs 3 | Check immediate, later, and replacement behavior |
| 5 | Test numerical reliability and practical value | 0; saved models from Study 15 | Compare correct decisions and total computation |
| 6 | Repeat training and obtain external review | Completed results from 2 and 5 | Separate repeatability from scientific success |

The order sets spending priorities. A failed graph model does not block an unrelated numerical test. If an earlier requirement fails, stop the dependent branch and publish the failure. Do not replace failed seeds or models until a desirable result appears.

## 0. Prepare the tests

Create a record of every input, model, intervention, score, and stop rule. An *intervention* is the specified internal change. Separate development cases from final test cases.

Use dummy inputs to test shapes, memory, arithmetic, and runtime. Do not use final scientific outputs to adjust the design. Recheck the exact meaning of each intervention in a separate implementation. Include unchanged-model controls and a known correct reference where one exists.

Before each final run, measure work on the selected hardware. Record the processor, available memory, software, thread count, precision, and batch size. Save training and inference times separately where possible.

**Success:** All required controls pass; planned data fit in memory; the projected run and replay fit the stage cap. Otherwise reduce scope and publish a revised plan before final testing. Numerical controls remain required in every later experiment.

## 1. Establish task ability

**Question:** Can the models reliably perform the specific behavior that the explanation is meant to describe?

Revisit four weak-model settings: matching graphs, register operations, digit images, and generated two-dimensional points. These come from Studies 12, 10, 11, and 14.

Allow at most two training recipes per task. Use two development seeds, then four fresh fixed seeds for the chosen recipe. The digit task has both direct and filtered architectures. This gives at most **20 development models and 20 final models** across the four tasks.

Recipe A keeps the original architecture and allows four times the original training schedule. Recipe B also allows up to twice the hidden width, within 100,000 parameters per model. Keep other settings fixed in the execution plan. Define both recipes before development runs. Choose by development task performance, with shorter runtime as the tie rule. Do not use final task or intervention results for selection.

| Task | Required result in every final model |
| --- | --- |
| Matching graphs | At least 99% accuracy in each of the original five categories: all graphs, hard negatives, target negatives, positive controls, and other-cause negatives |
| Register operations | At least 99% joint-state accuracy in every operation-order/horizon group, at horizons 0, 1, 2, 8, and 32 operations |
| Digit images | At least 95% accuracy on original images and 90% on each fixed one-pixel shift, in both architectures |
| Generated points | Excess negative log likelihood at most 0.20 nats, and four-region probability error at most 0.05, as defined in Study 14 |

For classification, select the recipe with the highest worst-group development accuracy. For generated points, first require the region-error limit, then select the lowest development likelihood error. If neither recipe meets that limit, stop the generated-points branch. Keep both failed recipes; do not add another. Keep the complete development record.

Retain the original graph task definition: ordinary graphs can be seen during training; new intervention pairs are the later test. Do not call this unseen-graph prediction. Use new generated sequences and points for final tests. For digits, select recipes within the original training split and retain the original test split for comparison. That dataset has already informed this project; a later new-dataset test remains necessary.

**Success:** Every model in a task panel passes every required category. Report a partial pass if only some models pass. A partial pass cannot support a claim about the full panel.

**Compute:** 0.5–2 hours estimated; 4-hour cap. This is a new training budget, not a promise that either recipe will succeed.

## 2. Test intended and unwanted effects

**Question:** Can an internal edit change one specified variable while preserving unrelated behavior?

Start with the selector task from Study 1. Train eight fresh models with its fixed ordinary-task recipe. Require 99% intact accuracy in every model. Use separate cases for learning an edit, choosing it, and testing it. The ordinary task states remain known; only the intervention pairs are separated.

Compare three methods with the same three-direction edit size:

1. Select three original neurons by intended-change error.
2. Select three original neurons under explicit limits on unwanted effects.
3. Learn a rotation of the internal state, then replace three selected coordinates in that rotated state.

Use the same donor cases and the same final test pairs. Limit the learned edit to 4,000 updates per model. Fix its loss and learning rate in the execution plan. If no neuron set meets the selection limits, record that result rather than silently relaxing a limit.

**Success for a variable explanation:** In each of the eight models, require at least 95% correct intended answers. Also require mean probability-distance error at most 0.05 in both active and inactive groups. When the proposed variable has the same value, at most 1% of answers may change. Here, probability distance is the zero-to-one TV measure defined in the review.

**Success for the claimed method improvement:** The learned edit must pass those absolute rules and improve intended-answer accuracy by at least 5 percentage points over Method 1 on average. At least six of eight paired differences must be positive. These are practical acceptance rules, not a population significance claim.

If the graph models from Experiment 1 pass, also repeat Study 12's fixed four-neuron selection. Test the strong claim that removal disables the target rule while preserving other decisions. Require at least 90% target false acceptance after removal, with at least 99% accuracy retained in each control category. Use the original selection and test contexts. Failure leaves only a narrower measured effect; it does not establish absence of the rule.

Include random edits, no edits, full-layer edits, and known correct input changes. A known correct reference can use information unavailable to the learned method. State that difference.

**Compute:** 0.25–1 hour estimated; 2-hour cap. The graph branch is included. Further rare-feature and spike-deletion extensions wait until this test is complete.

## 3. Test transfer to new conditions

**Question:** Does a selected component still support the same explanation when the input conditions change?

Use two public base models as the planned candidates: [Pythia-70M](https://huggingface.co/EleutherAI/pythia-70m) and [SmolLM2-135M](https://huggingface.co/HuggingFaceTB/SmolLM2-135M). Their official model cards confirm these model sizes. The cards do not establish ability on this experiment. Record exact weight and tokenizer revisions before use. No language-model training is included.

Within each model, use 256 calibration cases to select from at most 32 prespecified attention-head sites. An attention head is an internal operation that combines information from text positions. Freeze the selected site before final testing. Across models, test the selection method and copying account; do not assume that the same head index has the same meaning. A failed search of these sites cannot rule out a useful circuit elsewhere.

Use three final families of 256 cases each: new words, new sentence forms, and valid alternative splits of the same key text into tokens. A token is a text unit used by a model. Keep requested answers to one token within each model. Define compatible words without looking at answer scores. Limit prompts to 224 tokens, leaving up to 32 continuation tokens within a 256-token sequence. Report the families separately.

Test the intact model, the selected edit, random-site edits, and the unchanged-input control. Use the exact donor replacement specified in the plan. Define the required changed answer from the task rule.

**Success:** Both models must give the requested intact answer as their top token in at least 90% of each test family. The selected edit must give the required changed answer in at least 90%, with no more than 2% unwanted answer changes in same-value controls. Require at least a 5-point intended-answer advantage over the fixed random-site average in each model. Report all cases, not only initially correct cases.

If either model fails task ability, report the failure. Do not search for larger replacements within this plan. A follow-up would need a new model choice and budget. A correction for joint patch effects from Study 4 is deferred until this task test passes.

**Compute:** 3–12 hours estimated; 16-hour cap. Allow at most 16 million forwarded tokens across both models, calibration, controls, final tests, and a separate replay. Count a token again each time a new model pass processes it.

## 4. Test whether information is used and then replaced

**Question:** Does an edited state affect later operations exactly when the task requires it?

Use the four capable register models from Experiment 1. Compare the immediate-loss edit from Study 10 with an edit trained on future use. Keep the edit rank at four. Match the total forward and backward recurrent-position budget between methods. Fix update schedules after a dummy cost check. Report optimizer steps as well; equal steps alone do not give equal compute.

Use 1,024 new test sequences at each horizon of 1, 2, 8, and 32 operations. Balance copy-before-write and write-before-copy cases. Use known state rules to determine every required intermediate answer. Train edit methods on separate sequences.

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

Ask an external group to repeat Experiment 2's selector panel and Experiment 5's numerical panel. For the latter, retrain the original four model seeds from the fixed Study 15 training plan. Repeat all Experiment 2 training. Also check exactly two fresh seeds per panel as an extension; report them separately.

**Reproduction success:** On the same software and arithmetic, reproduce the recorded gate decisions and effects within locked numerical tolerances. On different hardware, justify tolerances before result comparison. If fresh training differs, report the difference; do not substitute saved weights and call it a training replication.

**Scientific confirmation:** The original acceptance rules must still hold in the repeated original-seed panels. Report fresh-seed results separately. A reproduced negative result counts as successful reproduction, not successful validation of the method.

A human reviewer should assess the questions, controls, novelty claims, failure records, and interpretation. External time and access are not secured. A separate AI review remains useful but does not satisfy this requirement. This plan does not send invitations.

**Compute:** 0.5–2 hours estimated; 4-hour cap for one external repeat package. This allocation is not a promise that an external group will participate.

## Compute assumptions and budget

Use one local CPU process with two compute threads. Allow one BLAS thread for the selector baseline if required for comparison. Reserve 16 GB of usable RAM and 20 GB of disk as planning allowances. These are not measured peaks. Load only one language checkpoint at a time and process activations in batches. Save required scores, predictions, model files, and audit records; do not retain every full-vocabulary activation.

No GPU training, rented hardware, or paid model API is needed for the proposed scope. This is not a cost estimate for training a language model from scratch. If the memory or time allowance fails, stop and revise the plan before final tests. Do not silently add cloud compute.

One CPU hour below means one elapsed hour for the specified process, not one core-hour. The estimates include scientific runs, planned controls, and one separate calculation replay, unless a stage states otherwise. They exclude coding, human review, downloads, and assistant usage. They do not forecast this account's usage allowance.

| Stage | Estimated CPU hours | Hard cap |
| --- | ---: | ---: |
| 0. Preparation runs | 0.25–0.5 | 1 |
| 1. Task ability | 0.5–2 | 4 |
| 2. Intended and unwanted effects | 0.25–1 | 2 |
| 3. Transfer | 3–12 | 16 |
| 4. Future use and replacement | 3.5–14 | 18 |
| 5. Numerical reliability | 0.25–1 | 2 |
| 6. External repeat package | 0.5–2 | 4 |
| **Full program, if all branches run** | **8.25–32.5** | **47** |

The [budget table](RESEARCH_COMPUTE_BUDGET.csv) records these figures in a reusable form. Caps are spending limits, not measured requirements or permission to execute. Failed branches can reduce total use.

For each language stage, the 16-million-token ceiling includes the scientific calculation and one comparable replay. An assumed effective rate of 500–2,000 forwarded tokens per second gives about 2.2–8.9 hours. Add data handling and setup within the 3–12-hour estimate. **These rates are planning assumptions, not benchmarks of the proposed models.** Shape, sequence length, intervention hooks, and cache use can change them substantially.

Before launch, replace that assumed rate with a dummy measurement on the chosen device. Estimate training, inference, saving, and replay separately; add a twofold safety factor to the measured projection. Require the result to fit the unchanged cap. For cached generation, count only tokens actually processed, while timing cache and hook overhead as well.

Small-model estimates are conservative allowances, not linear extrapolations from parameter counts. Stage 1 includes up to 40 base-model training runs. Stage 2 adds eight, Stage 5 adds at most four development models, and Stage 6 adds sixteen original- and fresh-seed runs. The complete program therefore permits up to 68 base-model training runs. Edit training and saved-weight replays are additional work included in the stage time estimates.

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

Allow **four to six working weeks** for one researcher who can implement and review the tests. This is a planning allowance, not a delivery promise. Machine hours and researcher weeks measure different work.

- **Week 1:** Freeze controls, inputs, and recipes. Complete preparation and task-ability work.
- **Week 2:** Complete specificity tests. Publish failures and decide which branches remain valid.
- **Weeks 3–4:** Run qualified transfer and future-use tests. Complete numerical efficiency work.
- **Weeks 5–6:** Complete audits, prepare papers, and seek external replication and human review. External completion may take longer.

For each stage, save the public plan, exact code version, inputs, seeds, model versions or weights, raw measures, costs, controls, and failures. Report every planned seed and subgroup. Use model-level summaries; do not count thousands of related input rows as independent models. Report uncertainty without treating a four-seed interval as proof of broad reliability.

Link each result to its plan and review. Do not promise one paper per stage or label a method new without a focused source check. Keep the existing fifteen studies unchanged as the starting evidence.

## Sources and current status

The [full review](REVIEW_PAPER.md) and its [source manifest](REVIEW_SOURCES.json) provide the evidence behind the proposed questions. The official [Pythia-70M model card](https://huggingface.co/EleutherAI/pythia-70m) and [SmolLM2-135M model card](https://huggingface.co/HuggingFaceTB/SmolLM2-135M) were checked on 13 October 2026 for model identity and size. They provide no runtime measurement for this plan.

All success rules and future compute amounts are proposals. No result in this document comes from a new experiment. Automatic research remains paused. This publication adds the plan only.

