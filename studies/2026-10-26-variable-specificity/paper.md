# A better internal edit can still fail to change one variable cleanly

**Date:** 26 October 2026. **Type:** Completed controlled study, Experiment 2C.

**Research and writing:** AI-assisted work for this repository. A separate AI result audit is complete. No human peer review or full independent training reproduction is recorded.

## Abstract

An internal edit can improve the requested answer while still changing too much else. We tested this problem in eight new small models that all performed their ordinary task perfectly. We compared three-neuron edits selected for the intended change, three-neuron edits with preservation limits, and learned edits in a rotated coordinate system. None passed the fixed explanation rule across the eight models. A learned edit was eligible in six models. Its mean intended-answer accuracy was 83.11% in those six, compared with a required 95% in every model. Two models had no eligible learned edit. The six available comparisons improved over the first neuron method, but the primary eight-model mean was therefore left absent. The result separates relative improvement from evidence that an edit cleanly implements a task variable.

## 1. The question

The task has two values, `a` and `b`, and a selector. When the selector is active, the answer is `a`. Otherwise, the answer is `b`. A proposed explanation says that three internal directions represent `a` well enough to support a specified replacement.

This explanation has two obligations. If `a` matters, replacing its internal representation should give the required new answer. If `a` does not matter, the edit should preserve the answer. An edit should also preserve behavior when the donor and receiver have the same value of `a`, even if their other inputs differ.

Learned distributed alignments are established prior work. [Geiger and colleagues](https://proceedings.mlr.press/v236/geiger24a.html) show why interpretable variables need not occupy single neurons. [Zhang and Nanda](https://arxiv.org/abs/2309.16042v2) examine how patching methods and scores affect results. [Heimersheim and Nanda](https://arxiv.org/abs/2404.15255v1) explain limits on causal interpretations of patching. This study tests one fixed comparison. It does not propose a new general alignment method.

The repository's earlier selector study found that selected neuron groups still produced unwanted changes. Experiment 2A then checked donor context and edit order. Experiment 2B checked search and statistical selection. The present study asks a different, stronger question: can a learned three-direction edit meet explicit intended-change and preservation limits on fresh models?

## 2. Models and separate intervention sets

We trained eight models with fixed seeds 91001–91008. Each had two hidden layers with 12 and 24 units. We kept the original task, initialization, full-batch Adam update, learning rate 0.02, and 4,000-update schedule. Every model reached 100% accuracy on the 192 ordinary task states. No failed seed was replaced and no training schedule was extended.

The ordinary states were known during model training. This is not a test on unseen ordinary inputs. Instead, we separated intervention pairs. All 48 donor contexts for a receiver/request block stayed in one set. A fixed hash rule gave 18,432 learning pairs, 9,216 selection pairs, and 9,216 final test pairs. The models and methods used the same sets.

Each final set contained 3,456 active cases with a changed value of `a`, 3,456 inactive cases with a changed value, and 2,304 same-value cases. The known coherent input change supplied the reference output. It changes the receiver's `a` while preserving its other input fields. This reference has access to the task definition; it is not a learned internal explanation.

The [plan and code](https://github.com/gavenren/mechanistic-interpretability-research/commit/bc5cf3c3d253396ff0aff83af37207bb524fc688), [passed preparation](https://github.com/gavenren/mechanistic-interpretability-research/commit/b11231096cc738723a2c66f415881f4003f9b1d8), and [all ordinary training records](https://github.com/gavenren/mechanistic-interpretability-research/commit/53811c15daa12f99c1b0635270cbb08494c4151d) were public in that order. The [complete calibration evidence and fixed choices](https://github.com/gavenren/mechanistic-interpretability-research/commit/b5f94da233f6b77e13415f1dc071c6cc1feb4faf) were public before final tests.

## 3. Three ways to choose the same edit size

**Method 1: intended change only.** Test all 220 sets of three first-layer neurons. Choose the set with the lowest selection probability-distance error in active, changed-value cases.

**Method 2: add preservation limits.** Test the same 220 sets. Require inactive-case probability distance at most 0.05 and same-value answer changes at most 1%. Among eligible sets, minimize the same active-case error. Record no choice if none qualifies.

**Method 3: learn a rotation.** Learn an orthogonal rotation of the 12-value hidden state, replace three coordinates, then return to the original coordinates. Use exactly 4,000 Adam updates per model. Each update draws 64 learning pairs from each of the active-changed, inactive-changed, and same-value groups. The loss gives the three group mean probability distances equal weight. Select among checkpoints at 1,000, 2,000, 3,000, and 4,000 updates using Method 2's preservation limits and active-error objective.

The probability distance is half the sum of absolute output-probability differences. It ranges from zero to one. A small value means that the two output distributions are close. A correct top answer alone does not guarantee a small distance.

No final test result selected a mask, checkpoint, loss, or threshold. Method 2 had a choice in seven models. Method 3 had a choice in six. Missing choices remained failures of the full-panel method claim.

## 4. What counted as success

Every model had to meet all four requirements:

- At least 95% correct required answers in active changed-value cases, and at least 95% across all changed-value cases.
- Mean probability distance at most 0.05 in active changed-value cases.
- Mean probability distance at most 0.05 in inactive changed-value cases.
- At most 1% answer changes in same-value cases.

The learned method also needed a mean intended-answer gain of at least five percentage points over Method 1, with positive gains in at least six of eight models. These relative conditions were additional requirements. They could not replace the absolute rules.

These are fixed practical criteria, not accepted field standards or population significance tests. We did not use the normal statistical test that failed Experiment 2B's calibration requirement.

## 5. No method met the full explanation rule

The table gives descriptive means among models with an available choice. The denominators differ. These columns do not replace the required eight-model result.

| Measure | Method 1 | Method 2 | Method 3 |
| --- | ---: | ---: | ---: |
| Models with a choice |8 of 8|7 of 8|6 of 8|
| Correct intended answers in active changed-value cases |41.29%|15.68%|83.11%|
| Active-case probability distance |0.5914|0.8425|0.1779|
| Inactive-case probability distance |0.1206|0.0304|0.0437|
| Same-value answers changed |15.70%|0.45%|0.20%|
| Available models passing every final rule |0|0|0|

Adding preservation limits to neuron selection reduced the mean unwanted changes among available choices. It also produced weak intended changes. The learned rotation had much better intended-answer performance than either neuron method among its six available models. It still fell short of the absolute intended-change requirements.

The model-level intended-answer results make that limit clear:

| Model seed | Method 1 | Method 2 | Method 3 |
| --- | ---: | ---: | ---: |
|91001|47.48%|No eligible edit|No eligible edit|
|91002|30.44%|14.00%|No eligible edit|
|91003|50.20%|4.05%|78.82%|
|91004|42.53%|22.16%|82.29%|
|91005|36.81%|36.63%|85.62%|
|91006|49.97%|7.78%|84.20%|
|91007|42.16%|3.01%|82.49%|
|91008|30.70%|22.14%|85.21%|

All six available learned edits improved active-case accuracy over Method 1. The mean paired gain in those six was 41.04 percentage points. That is a conditional descriptive result. The primary eight-model mean remains absent because two models lack an eligible learned edit. The improvement claim fails both this completeness requirement and the absolute performance rules.

Selection eligibility also did not guarantee final preservation. Three of the six learned edits failed at least one final preservation limit. All six failed active-case accuracy and active-case probability distance. Thus, the negative result does not depend only on how missing choices are counted.

## 6. What the controls show

The known coherent input change gave 100% required answers and zero probability error in every model. This confirms that the target behavior exists in these task models.

The no-edit control gave zero correct changed answers in the active group and preserved the inactive group's answers. Across both groups, its changed-value accuracy was 50%. This is why the primary intended-change measure must use active cases separately: unchanged inactive answers can raise a pooled score without implementing the requested change.

Full-layer donor replacement reached 62.5% active-case accuracy but changed 57.29% of same-value answers on average. It copies the donor's other context as well as its value of `a`. It is therefore not the coherent input reference. The 20 fixed random three-neuron masks per model averaged 12.19% active-case accuracy. These controls give context for the method results; they do not relax the success limits.

## 7. What is new, and what the result does not prove

The contribution is a fixed comparison on eight fresh task models, with separate edit-learning, selection, and test pairs. It evaluates a learned rotation against intended-only and preservation-constrained neuron choices at the same edit size. It retains unavailable choices and distinguishes improvement from a successful variable-level explanation.

The result does not show that the task variable is absent from the network. It does not show that every rotation, larger edit, different layer, different loss, or longer schedule must fail. Those alternatives were not tested. It also does not show that the three-direction restriction is the sole cause of failure. The learned objective and optimization can matter.

The task is small and synthetic. Its ordinary states were all known during training. Eight seeds from one recipe do not establish a failure rate for large language models. The paired intervention groups share underlying states, so the large number of pairs is not a large independent model sample. No population significance claim follows.

The next useful extension should test a different explanation or a cause of the failure under a new fixed design. It should not select a more favorable threshold, training duration, or subset of models from these final results. The separate numerical-cost study follows next in the existing roadmap.

## 8. Checks, cost, and reproduction

All fixed numerical controls passed. A separate saved-result audit checked 226 raw output files and made 490 file-hash checks, with 27,549 assertions. Its largest recorded arithmetic difference was `6.00e-15`, below the fixed `1e-10` tolerance. The audit recomputed output metrics, eligibility, choices, and final gates from saved evidence. It did not repeat model training, rotation gradients, matrix exponentials, or unselected candidate outputs. Those limits remain explicit in the [audit record](analysis/audit.json).

The scientific phases completed 32,000 ordinary-model updates and 32,000 rotation updates. Their combined charge was 63.31 seconds. One 120-second reserve covers archive packing, both transport checks, the result audit, and final archive checks. The total stage charge is 183.31 seconds against a 7,200-second cap. Cumulative preparation, including the fixed pre-lock allowance, is 453.15 seconds against its separate 3,600-second cap. The [compute record](analysis/compute-accounting.json) gives the full accounting. The scientific runs used no language-model inference tokens, paid model APIs, or rented compute. AI assistance for code and writing used account usage.

The full calibration ZIP exceeded the browser upload limit. A reviewed amendment split the same bytes into two files before final tests. Review caught an initial folder-layout conflict; the first wrapper and mock check remain in the record. The corrected check called the original calibration verifier. No final scientific test ran under the first layout. Read the [transport plan](TRANSPORT_PLAN.md) and [transport review](TRANSPORT_REVIEW.md) for reconstruction and exact-byte checks.

Six final [result archives](artifacts/manifest.json) preserve all 218 final branch files. Fresh restore, repeat restore, and rejection of a conflicting file passed. The original [result manifest](results/manifest.json) and status are direct files. From the repository root, restore final results with:

```sh
.venv/bin/python studies/2026-10-26-variable-specificity/analysis/result_archive.py unpack
```

The [model method](MODEL_METHOD.md), [edit method](EDIT_METHOD.md), [fixed plan](analysis-plan.md), [source notes](sources.md), [plan review](PLAN_REVIEW.md), and [result review](RESULT_REVIEW.md) give the full record. This is one completed study. Its failed scientific claim is not an unfinished numerical run.
