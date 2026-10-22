# What an internal change can and cannot show
## A plain-English review of 24 completed mechanistic interpretability studies

**First version:** 13 October 2026
**Updated:** 22 October 2026
**Type:** Review of a fixed set of repository papers. No new experiment.
**Research and writing:** AI-authored. A separate AI review accompanies this paper. No human peer review is recorded.

## Abstract

Mechanistic interpretability studies how a model produces its answers. One common method changes part of the model and measures the output. This review covers 24 completed studies: the first fifteen and nine added on 22 October 2026. The first group includes small trained networks and one small pretrained language model. The new group uses mathematical analysis, constructed examples, and a statistical simulation. None of the nine new papers tests a trained language model.

The shared finding is that an output effect alone does not establish a correct explanation. First, the model must perform the relevant task. Then the test must define the internal edit and measure the intended change. Tests of a specific variable must also check unrelated behavior. The new papers show why replacement values, interactions between parts, edit order, score choice, and reuse of selection data matter. Their exact counts describe their declared test sets. They do not measure failure rates in real AI systems.

The specific contributions are reproducible experiments, mathematical conditions, and complete records. They do not establish 24 new general principles. The planned tenth study in the new batch remains unfinished. Only two of its four required final graph models are complete. It is reported separately and is not counted among the 24 completed studies.

## 1. Scope and method

This update covers all 24 completed studies at [source commit 384ea79](https://github.com/gavenren/mechanistic-interpretability-research/tree/384ea795ec685cac1aa883341f7a53ae36bb48c8). The first fifteen were dated 13 October 2026. The nine additions were dated 22 October 2026. We retain the first fifteen study numbers and assign numbers 16–24 to the additions. The unfinished tenth study in the new batch is separate. It must not be confused with completed Study 10 from the first group.

This is a narrative review of a fixed set of repository papers. It is not a systematic review of the whole field. It uses the papers, source records, methods, results, and review records. It adds no model runs, new literature search, or combined statistical estimate. The grouping was developed after the results were available.

The [source manifest](REVIEW_SOURCES.json) records the 24 paper versions and their content hashes. It also lists the unfinished report separately. The [batch index](2026-10-22-RESEARCH_BATCH.md) gives the publication and stop records. Each source paper links its code, saved results, and reproduction instructions.

Studies 4–8 use the same Pythia-14M checkpoint and related copying tasks. Studies 11, 13, and 15 use the same UCI digit dataset. Studies 16–24 use constructed systems or simulated measurements. These are related tests with different types of evidence. They are not 24 independent confirmations of one claim.

Three terms help separate the evidence:

- **Task ability:** Can the model perform the behavior that the explanation is meant to describe?
- **Intervention effect:** What changes when a specified part of the model is changed?
- **Mechanism claim:** Does a proposed account correctly explain those changes, including the required controls?

An *activation patch* replaces internal values with values from another input. An *ablation* removes a specified contribution. A *token* is a unit of text used by a language model. A *logit* is an output score before conversion to a probability. A *margin* compares two output scores. A *total variation distance*, or TV, measures a difference between probability distributions on a scale from zero to one. A larger change is not necessarily a better answer.

## 2. Main findings

### 2.1 Establish task ability before interpreting an effect

A strong effect can occur in a model that does not reliably perform the target task. This problem appeared in several forms.

In Study 8, a copying head still had a positive measured effect when the same text used different token splits. Yet the requested answer ranked first in only one of 1,024 test cases. That effect did not establish successful text retrieval.

Study 12 gave another warning. Its task asks whether each item can receive a different partner. The proposed rule identifies cases with too few available partners. Selected neurons had a much larger effect than random neuron sets in a graph task. But every model failed the fixed task requirements. Removing the selected neurons also caused many errors on a different reason for task failure. The effect score did not validate the proposed graph rule.

Studies 11 and 14 found differences between intervention methods in weak or imperfect models. Those differences remain valid measurements of the tested models. Their meaning is narrower than a result from models that pass the relevant task or fit requirements.

Task ability is also insufficient by itself. Every model in Study 1 solved its ordinary task correctly. But the selected three-neuron patches still had large errors. Patches changed the answer on 15.7% of controls where the proposed variable had the same value. Correct answers from the intact model did not validate the selected neurons.

### 2.2 State exactly what the internal edit changes

An explanation needs a precise intervention. A component name alone can leave the procedure unclear.

A coordinate change gives new numbers to describe the same internal state. Study 3 tested such changes. A correctly transformed intervention preserved the expected answer. A coordinate mask or a different projection rule often did not. Even the same subspace could support different replacement rules after a non-orthogonal coordinate change. The successful reference used a known correct operator. It did not discover that operator from an unknown circuit.

Study 13 examined two meanings of deleting a spike. One removed its outgoing message but kept its immediate reset effect. The other removed both. An explanation matched to the deletion rule reduced mean selection error. However, some comparisons became worse, and sign errors remained.

An equilibrium is a state that no longer changes under the model's update rule. Study 15 removed a coordinate equation and solved the remaining equilibrium system again. Its result concerns that precise edit and the exact-real equilibrium of the saved weights. This target treats the saved numbers as exact mathematical values. It does not automatically apply to a one-step deletion or a finite floating-point program.

Together, these studies support a practical requirement: state the changed equations, the retained paths, and the later computation. Then test the explanation against that exact procedure.

### 2.3 Choose a measure that matches the claim

Several studies show why nearby measures must remain separate.

Study 2 found that a complete learned code carried the required change better than one selected feature. Preserving useful information across a code did not reliably isolate it in a single feature.

Study 5 measured a strong preference between two supplied answers. The same model produced the requested answer as its top token much less often. Study 7 found that a longer input reduced the mean answer margin but increased top-token accuracy. A margin and an accuracy rate answer different questions.

Study 6 selected different patch locations for immediate and later output effects. The later choice produced a larger change after eight fixed continuation tokens. It did not produce successful copying.

Study 14 selected different neurons when it targeted likelihood damage or changes in generated outputs. The output measure counted samples in four regions. It did not measure the full output distribution or sample quality.

The review therefore recommends that each paper state what a score can establish before reporting its value. Relative improvement, absolute error, task success, and unintended changes should have separate results.

### 2.4 Transfer and training benefits need direct tests

A method that fits one set of cases may fail on another set. Study 4 fitted a correction for the joint effect of two patches. Its mean prediction error increased slightly on new words and sentence forms. The result gave no clear evidence of transfer benefit.

A positive average can also hide inconsistent outcomes. In Study 2, rare-event weighting gave a small positive mean benefit for the selected feature. It was worse in five of six main paired comparisons. One large gain drove the positive mean.

Study 10 trained internal state edits with an additional future-step loss. Its mean gain was 0.21 percentage points. The reported interval included both loss and gain. Three of four task models failed the fixed ability rule. The method also required extra training work.

Study 9 tested whether training order left an important internal difference after balanced retraining. The main effect was tiny and opposite to the predicted direction. All final models and recorded final patches kept the correct class.

These results do not show that transfer methods or training changes cannot work. They show that the tested procedures did not establish the broader benefits that a reader might infer.

### 2.5 Check numerical evidence without changing the meaning of success

Float32 and float64 are two levels of numerical precision. The studies retain several failed numerical checks. This matters because a later successful check has its own scope.

Study 4 first failed a float32 check based on an algebraically equal expression. A later check preserved the original addition order and matched the saved outputs. Study 8 exactly replayed its declared float32 outputs but failed an added float64 probability check. Study 11 preserved its original failed run and used a separately planned follow-up evaluation.

Study 15 made a different distinction. At the initial solver stopping point, bounds did not establish 18.66% of nonzero effect signs, averaged over models. This was an uncertainty rate under those bounds. It was not a wrong-sign rate. Further computation proved five initial signs wrong. It established a strict sign or a small-magnitude bound for every tested effect.

A small-magnitude bound does not establish an exact zero. A successful replay of saved weights does not repeat model training. A numerical check also does not establish a semantic explanation. Each claim needs the corresponding evidence.

### 2.6 Replacement values and edit rules are part of the explanation

Study 16 holds ordinary behavior fixed. Six constructed models give the same correct answers on every ordinary input. Yet their answers differ after some internal replacements. A replacement can create a state that does not occur during normal use. A donor with the same answer can also preserve the signal under test. Neither outcome alone shows whether a part is necessary. This extends the earlier concerns about unintended effects and coordinate changes.

Study 18 shows that two edits can interfere. When their directions overlap, changing their order can change the final state. Even values from the same donor do not always prevent this. Solving both requests together works when they are consistent, but it can move the state far from its starting point. A mathematical solution does not establish the intended meaning.

Study 20 holds the decoder fixed. A decoder converts an internal code into a model state. Several different codes can reconstruct the same state. Replacing the same code entry can then give different decoded results. The paper gives an exact condition that prevents this ambiguity. A fixed trained encoder may select only one code. The study therefore shows a limit of reconstruction evidence, not a measured failure of trained encoders.

Study 22 explains another source of wider effects. Layer normalization adjusts values using a shared mean and scale. Changing one value changed every other normalized value in all 2,048 fixed cases. This spread follows from the operation itself. It does not show that every changed number represents a changed concept.

### 2.7 Tests of separate parts do not fully test a group

Study 17 checks all 166 nonconstant monotone Boolean functions of four labeled inputs. These are logical rules where changing an input from off to on cannot turn the output off. Eight tests that changed one input at a time uniquely identified only 14 functions. The other 152 shared their test results with another function. Tests of groups are needed to distinguish these cases. This count applies to the specified class, not to neural networks in general.

Study 19 asks a different question: does a search find the smallest working set of parts? A procedure removes one part at a time while keeping every task answer correct. It can stop with six parts when one part is sufficient. No further single removal is possible, but a smaller working set exists. Thus, correct behavior, resistance to further removal, and proof of minimum size are separate claims.

### 2.8 A score must match the question, and selection needs a separate test

Study 21 shows how an average can hide a useful edit. The correct score change is positive in one context and negative in another. The changes cancel when both contexts have equal weight. In six of ten fully correct constructed models, this score ranked an irrelevant edit above the useful edit. Reporting each context exposes the difference. Aligning effects with the requested answer also works here, but it uses a task rule known in advance.

Study 23 compares output scores before conversion to probabilities with the probabilities themselves. Their rankings reversed in 704 of 7,100 strict score comparisons. All individual changes were nonnegative. The reversal therefore did not require cancellation. Probability gain depends on initial confidence. Neither measure is always better; the research question must determine the measure.

Study 24 adds a selection check. All true effects were zero in its simulation. With 128 independent candidates and 64 cases per estimate, selecting the largest estimate and testing it on the same data gave false evidence in 99.94% of trials. Separate validation data gave 4.38%. A correction for testing many candidates gave 5.08%. These results concern the declared noise model. They do not estimate false discoveries in published circuit research.

## 3. Evidence from every completed study

The following tables give each completed study's main result and its specific contribution. The linked papers provide the full methods and limits. Scores with different units must not be compared as if they were one common measure.

### 3.1 Original studies: 1–15

| Study | Main result | Specific contribution and limit |
| --- | --- | --- |
| **1. [Inactive variables](../studies/2026-10-13-inactive-variable-audit/paper.md)** | Both selection rules chose the same three neurons in all eight models. Mean active-case error remained 0.594 TV. | Complete comparison of small neuron masks and active/inactive controls. Adding inactive cases to this objective did not enforce a separate error limit. All ordinary task states were used in training. |
| **2. [Rare features](../studies/2026-10-13-rare-feature-sae/paper.md)** | Main single-feature error fell from 0.4414 to 0.4173 on average, but five of six pairs became worse. The balanced full code had error 0.1183. | A fixed test of feature frequency, amplitude, and training objective across 72 small autoencoders. The balanced method used true event labels. It was a reference with extra information on constructed data. |
| **3. [Coordinate changes](../studies/2026-10-13-basis-transport/paper.md)** | Rotated coordinate masks gave 43.73% class accuracy. The correctly transformed intervention kept 100%. | Separates the coordinate system, selected subspace, and replacement rule. The correct reference was available by construction. The result is not a new theorem or an unknown-circuit discovery. |
| **4. [Patch correction transfer](../studies/2026-10-13-patch-calibration-transfer/paper.md)** | Mean prediction error rose from 0.030472 to 0.031135 logit units on new prompts. | A fixed transfer test for interaction corrections. It found no clear benefit in one small model. The original failed numerical check remains in the record. |
| **5. [Copying routes and values](../studies/2026-10-13-copy-routing/paper.md)** | Route and value edits each reversed the mean preference between two answers. Their combination largely restored it. | A controlled test of a known copying mechanism in one selected head. Two-answer accuracy was 95.8%; full-vocabulary top-token accuracy was 29.2%. This is not a complete copying circuit. |
| **6. [Later patch effects](../studies/2026-10-13-patch-memory/paper.md)** | The later-effect selector gave test TV 0.002866 after eight tokens, versus 0.000272 for the immediate-effect selector. | A fixed comparison of patch selection times on new words. The continuation was imposed. No requested copy word ranked first. Persistence did not establish useful memory. |
| **7. [Position and token access](../studies/2026-10-13-position-access/paper.md)** | At gap 128, either single edit sharply reduced the margin. Both edits together matched the short-input control. | A joint test of position spacing and access to extra tokens. The combined match was expected by construction. Later internal changes prevent a claim about pure distance or pure attention competition. |
| **8. [Different token splits](../studies/2026-10-13-segmentation-transfer/paper.md)** | Full-vocabulary accuracy fell from 44.43% with matching splits to 0.10% with different splits. A positive head effect remained. | Tests transfer of a fixed head with the same text and equal token counts. The alternative splits were valid but nonstandard. The task-ability rule failed. |
| **9. [Training order](../studies/2026-10-13-training-order-paths/paper.md)** | The main path-probability difference was −0.00274 percentage points. All 48 final models and final patches kept the correct class. | A matched-example test with balanced retraining and saved training histories. It does not show lasting decision errors or irreversible effects of training order. |
| **10. [State edits and future steps](../studies/2026-10-13-register-write-order/paper.md)** | The future-step loss gave a mean gain of 0.21 percentage points. The interval was −1.02 to +1.13 points. | Tests copy-before-write and write-before-copy operations. Three of four models failed the ability rule. Added computation and short operation sequences limit the comparison. |
| **11. [Image shifts](../studies/2026-10-13-shift-ablation/paper.md)** | Filtering reduced the mean change in channel-removal effects from 0.08294 to 0.01177. Original-image accuracy was only 14.91%–35.95%. | Tests intervention stability under exact image shifts. It compares separately trained procedures. Stability does not establish correct image features. A separate follow-up produced these results after the original numerical failure. |
| **12. [A graph-rule explanation](../studies/2026-10-13-hall-obstruction/paper.md)** | Selected edits exceeded random edits by 11.061 margin units. False acceptance of a different failure type rose from 6.34% to 84.72%. | A controlled test of specificity in a matching task. All four models failed the task requirements. A strong selected-versus-random score did not validate the proposed rule. |
| **13. [Spike deletion](../studies/2026-10-13-spike-reset-queries/paper.md)** | Matching the explanation to the deletion reduced mean selection error by 0.01227 margin units. All four models passed the accuracy check. | Compares two explanation graphs with two deletion rules on trained spike traces. Some comparisons lost, and wrong signs remained. It does not establish a general gradient advantage. |
| **14. [Flow-model neurons](../studies/2026-10-13-flow-unit-ranking/paper.md)** | Output-based selection gave test quadrant TV 0.05753, versus 0.03933 for likelihood-based selection. It won in three models and tied in one. | A complete comparison of unit selection objectives on separate test data. All four models failed the density-fit rule. Four-region output change is not sample quality. |
| **15. [Equilibrium effect signs](../studies/2026-10-13-equilibrium-sign-audit/paper.md)** | Initial bounds did not establish 18.66% of nonzero signs. Refinement gave 11,775 strict signs and 3,585 small-magnitude bounds. Five initial signs were proved wrong. | A complete removal-effect audit with established contraction bounds and checked interval arithmetic. All four models passed the accuracy check. The uncertainty rate depends on the bound and stopping rule. |

### 3.2 New completed studies: 16–24

These nine papers contain no trained-model experiment. Their calculations and assumptions remain separate from the trained-model evidence above.

| Study | Main result | Specific contribution and limit |
| --- | --- | --- |
| **16. [Donor choice](../studies/2026-10-22-donor-support/paper.md)** | Six models agree on ordinary answers but differ after some hidden-state replacements. The fixed test contains 312 exact cases. | Holds ordinary behavior fixed while varying responses to states outside normal use. A known coherent edit is a reference with extra information, not a discovered mechanism. |
| **17. [Tests of individual parts](../studies/2026-10-22-coalition-identifiability/paper.md)** | Of 166 logical functions, 152 share their eight individual-part test results with another function. | Complete count for four labeled inputs and monotone rules. It does not measure how often real circuits are ambiguous or identify a unique network structure. |
| **18. [Edit order](../studies/2026-10-22-edit-order/paper.md)** | Across 3,645 exact cases, overlapping edits often depend on order. In each of three direction pairs that are not at right angles, 702 of 729 cases differ by order. | Separates direction overlap from conflicting donor requests. The joint solution needs consistent requests and does not establish a meaningful or safe edit. |
| **19. [Circuit search](../studies/2026-10-22-greedy-circuit-search/paper.md)** | A search that removes one part at a time can retain six parts when one is sufficient. | Checks 17 constructed circuits, 1,016 subsets, and 14,550 deletion orders against exact minimum sizes. It does not test a full practical circuit-search system. |
| **20. [Code-edit ambiguity](../studies/2026-10-22-decoder-edit-ambiguity/paper.md)** | All 9,600 exact edits agree with the derived condition for code-independent results. Exact reconstruction alone is insufficient. | Gives a condition and counterexamples for three linear decoders, including nonnegative codes. No trained encoder or sparsity objective is tested. |
| **21. [Effects that cancel](../studies/2026-10-22-context-effect-cancellation/paper.md)** | With balanced contexts, the absolute signed average ranks a nuisance edit above a useful edit in six of ten fully correct models. | Separates cancellation from task failure using a known task rule and nuisance controls. Correcting the direction uses extra task information. |
| **22. [Normalization effects](../studies/2026-10-22-layernorm-spillover/paper.md)** | Every other normalized value changes in all 2,048 cases. On average, 25.47% of squared output change lies outside the edited coordinate. | Gives an exact formula for a finite edit and checks random constructed states. These are not natural model activations or measurements of concept changes. |
| **23. [Score rankings](../studies/2026-10-22-metric-rankings/paper.md)** | Probability gain reverses 704 of 7,100 strict logit rankings, or 9.92% of that fixed set. | Gives a complete two-context comparison and a condition for ranking agreement. The percentage is not a real-model failure rate. |
| **24. [Selection from noise](../studies/2026-10-22-selection-null/paper.md)** | In the stated 128-candidate example, false positives are 99.94% with reused data and 4.38% with separate validation. | Runs 180,000 trials across 36 fixed conditions with analytic checks. Noise is Gaussian with known variance. The study does not measure detection of true effects or real-network error rates. |

## 4. What is new

The defensible contributions concern the exact experiments, formal analyses, and their records. They include complete candidate comparisons, fixed control sets, held-out intervention pairs, saved model states, and numerical audits.

For example, Study 3 separates changes to coordinates from changes to the intervention rule in one controlled setting. Study 13 crosses two deletion meanings with two explanations. Study 15 records the path from a coarse effect estimate to a checked sign or magnitude bound. These packages make narrower claims testable and reproducible.

The underlying concerns have substantial prior work. The individual [source records](REVIEW_SOURCES.json) identify that work and the claimed difference. Study 5 tests a known copying mechanism. Study 3 uses standard coordinate algebra. Study 15 uses established contraction bounds. Targeted source checks do not establish that no similar experiment exists.

The nine additions make several narrower contributions. Study 17 gives a complete small-class count. Study 18 gives conditions for agreement between edit orders. Study 20 gives a condition for an edit result to be independent of the selected code. Study 19 compares every search order with the true minimum in a fixed panel. Studies 16 and 21–24 provide controlled reference tests for replacement, aggregation, normalization, metric choice, and selection.

The papers identify close prior work on these concerns. They do not claim that donor dependence, overlapping edits, normalization, or selection bias are new principles. A reproducible example can help researchers test a method without establishing priority for its underlying idea.

This review adds an organized account of the evidence and revised research priorities. It adds no experimental result. The reviewed set contains **24 completed studies**. The review and the unfinished execution reports are separate.

## 5. Limits of the combined evidence

**The model and task coverage is narrow.** Many tasks are small and constructed. The language-model studies share one checkpoint. Three digit studies share a dataset. None measures a failure rate across large deployed language models.

**Exact test counts have a limited meaning.** Complete enumeration can establish a result for the stated mathematical class. It does not show how common that class is in real models. Simulation frequencies depend on the declared noise model. The percentages in Studies 17, 22, 23, and 24 must not be read as population rates for language models.

**The studies are related.** They come from the same research project and share methods, task families, and review procedures. A large number of intervention rows does not create a large number of independent trained models. Most of the earlier trained-model experiments use only four to eight fixed training seeds. The new constructed tests add no independent trained models.

**A test split has a specific meaning.** Studies 1 and 3 test new intervention pairs on already trained task states. Study 12 uses all ordinary graph inputs during task training. These tests do not establish generalization to unseen ordinary inputs. Constructed correct operators and true-label comparisons also provide information that a practical explanation method may lack.

**The measures and assumptions differ.** TV, output margins, accuracy, likelihood, and solver bounds answer different questions. We do not pool their effects or rank methods across studies. Reported resampling intervals describe the sampled seeds or fixed cases. They do not cover new architectures, tasks, or research design choices.

**Weak results still require careful interpretation.** A failed ability rule limits a mechanism claim but does not erase a recorded output effect. A failed small-neuron search does not prove that the model lacks the variable. A null result under one training schedule does not prove that another method cannot work.

**Passing a mathematical check is not a complete explanation.** A known correct edit, decoder condition, or group test can rule out a bad inference. It does not by itself discover the computation in a trained model. A large output effect can remain real while its proposed meaning is wrong or untested.

**The review has shared sources of error.** The papers and this review are AI-authored. The source studies include separate AI calculation checks, often with separate implementations. Those checks do not establish external research independence or human peer review. Replaying saved weights does not repeat training. This review did not rerun the experiments.

## 6. Next research steps, in order

### Step 1: Establish adequate task models

The current priority is the unfinished Experiment 1 continuation described below. Before more research, check the usage allowance and publish a rule for the interrupted seed. Count earlier computation and keep the existing stop rules. Complete the required final graph models before the dependent explanation test. Test the other task branches under their fixed plans. This review does not start those runs.

First revisit the task or fit failures in Studies 8, 10, 11, 12, and 14. For Study 8, choose new fixed pretrained checkpoints and retain the original failed transfer result. For the other studies, publish new training and evaluation plans before new results. Use fixed seeds and retain every run. Allow adequate training through those plans. Do not extend training repeatedly until the result looks favorable.

Set separate requirements for the difficult cases that the explanation concerns. Overall accuracy alone was insufficient in Study 12. Continue to a mechanism claim only when the relevant requirements pass. Otherwise report the narrower behavior of the weak model.

### Step 2: Require the intended change and preserve unrelated behavior

Extend Studies 1, 2, and 12 with separate limits for intended-change error and unintended-change error. Choose those limits before testing. Do not combine them into a score that can hide one failure.

Use equal-size random edits, unchanged-input controls, full-layer controls, and known correct references where available. Keep selection cases separate from test cases. Report when a reference has access to information unavailable to the proposed method.

Add the checks from Studies 16–24 to the relevant test plans. Specify the donor values and whether the edited state occurs in ordinary use. Test edit order and effects after normalization. For decoded features, state how the encoder selects a code. Test groups of parts when individual tests cannot distinguish explanations.

Choose the score before examining results. Report effects by context and task outcome. Keep final validation data separate from candidate selection, or use a valid correction for that selection. If the claim is that a circuit has minimum size, compare with an exact solution where feasible. Otherwise report the weaker search result.

Pass this stage only when both intended and unintended effects meet their requirements on the fixed test cases. Better performance than random editing is insufficient. The constructed examples provide test designs; their application to trained models remains future work.

### Step 3: Test transfer of a fixed explanation

Freeze the selected head, neurons, or correction before changing the test conditions. For Studies 4–8, add independent model checkpoints and ordinary language tasks. Test new words, sentence forms, token splits, and donor contexts separately before combining them.

Require both adequate task ability and the same predicted intervention behavior. Use full-vocabulary task accuracy when the claim concerns answer retrieval. A positive two-answer margin alone does not pass.

For coordinate tests, distinguish transformation of a known operator from discovery of an unknown one. Record which claim each test supports.

### Step 4: Test future behavior and competing explanations

Extend Studies 6 and 10 to longer sequences and model-generated continuations. Keep shared forced continuations and free generation as separate tests. In free generation, the visible tokens can change with the internal state. These tests answer different questions. Test intermediate states as well as final answers. A proposed stored variable should affect later operations that use it. Its old value should stop affecting operations after replacement, unless another state correctly retains a copy.

Use cases where different explanations predict different outcomes. The route/value tests in Study 5 and the operation-order tests in Study 10 provide starting points. Compare methods with equal computation budgets.

Pass only when the complete predicted pattern occurs. A persistent output change alone does not establish useful memory.

### Step 5: Measure practical value and check the numerical target

Before testing, define the smallest gain that would justify using a method. Report absolute errors, decision changes, and total computation. Include calibration runs and extra training work.

Extend Study 15 by comparing tighter valid bounds under a fixed computation budget. Keep the model, intervention, and numerical target fixed. Report uncertain signs, proved reversals, and small magnitudes separately. A tighter bound could reduce uncertainty without changing the initial effect estimates.

Use arithmetic controls that match the declared computation. Preserve any original failure when a later diagnostic uses a different procedure.

### Step 6: Obtain independent replication and human review

Ask an external group to reproduce selected results from the public plans, including model training. Prioritize a null result, a failed ability claim, and a positive controlled result. This would test more than repeated calculations from saved weights.

Human reviewers should assess the scientific questions, control choices, source claims, and practical meaning. Publish disagreements and corrections. These steps remain proposed work; this review does not claim that they have occurred.

## 7. Unfinished tenth study in the new batch

**Status: UNFINISHED. Not counted among the 24 completed studies.** This is the [Experiment 1 task-ability continuation](../studies/2026-10-22-capability-completion/paper.md). It is separate from completed Study 10 above.

The continuation asks whether small trained models can perform the tasks needed for later explanation tests. Six complete graph models score 100% in all five fixed test categories. Four are development models and two are final models. One development model was reused from the earlier run; five complete models were trained in the continuation. Every complete model passed a separate check of its outputs from saved weights.

The graph test requires four complete final models. Only two are complete, so this requirement has not passed. The third final model, seed 8103, stopped with 22,028 recorded updates out of 32,768 planned updates. It has partial weights but no final score. The weights can include one additional update that was not recorded in the trace. The fourth final model, seed 8104, did not start. The register-operation, digit-recognition, and generated-point task branches did not start.

The run stopped under the 20% usage-reserve rule. It did not stop because a completed graph model failed, because time ran out, or because memory was exhausted. A first fresh timing estimate failed. One published revision passed with the same fourfold allowance and four-hour limit. Both records remain available. Automatic research remains paused.

All graph states were used during training. Perfect accuracy on this finite set does not establish performance on unseen graphs or identify an internal matching algorithm. Checking saved weights does not repeat training. Results from two final models cannot replace the required four-model test.

Before continuation, a new public plan must specify how to handle the interrupted seed and count all earlier computation. Both partial attempts must be retained. Complete the remaining graph models and the other planned task branches before making their dependent explanation claims. None of this remaining work is reported as complete here.

## 8. Conclusion

The 24 completed studies support a careful way to assess an internal explanation. First establish the model's relevant ability. Then define the edit and test its intended and unintended effects. Use measures, separate validation, numerical checks, and transfer tests that match the claim. Test interacting parts and the full effect of the edit procedure.

The strongest shared result is a limit on inference: a measured internal effect alone does not establish how a model solves a task. The next useful work is to test complete explanations in capable models, across independent conditions, with clear failure rules.

## Research record

The [source manifest](REVIEW_SOURCES.json) fixes the reviewed versions. The [update review](reviews/2026-10-22-REVIEW_UPDATE_REVIEW.md) records the checks for this version. The [original AI review](reviews/REVIEW_PAPER_REVIEW.md) applies to the first fifteen-study version. The [research state](RESEARCH_STATE.md) separates this synthesis from new experiments.

No new dataset, model training, intervention run, or statistical analysis was used for this review. No human peer review is recorded. The source papers remain the primary references for their measurements, code, external citations, and numerical audit records.

The earlier [one-page summary](REVIEW_SUMMARY.md) covers the original fifteen studies. It has not been expanded as part of this review update. The updated review is the full 24-study synthesis.
