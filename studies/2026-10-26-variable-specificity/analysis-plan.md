# Experiment 2C: change one variable without unwanted changes

Date: 26 October 2026. Initial state: code and methods only. No benchmark, training, or new model output has run.

## Question and scope

Can a three-direction internal edit change the task variable `a` while preserving other behavior? This is a controlled comparison of two neuron-selection methods and a learned rotation. It tests absolute task requirements. It differs from 2A's procedure checks and 2B's search and statistical checks.

Use eight fresh models of the original finite selector task. Keep the original training recipe. The task returns `a` when its selector is active and `b` otherwise. There are 192 ordinary input states. The state set is known during ordinary training. Learning, selection, and test intervention pairs are separate; the underlying input states are not new held-out inputs.

The [roadmap](../../docs/RESEARCH_PLAN.md) supplies the stage and success requirements. The [model method](MODEL_METHOD.md) and [edit method](EDIT_METHOD.md) fix all details below. Learned rotations have established prior work. This study makes no claim of a new general method. The graph branch remains blocked by Experiment 1 and must not run.

## Public records and execution order

1. Publish the fixed code, methods, inputs, source notes, and separate plan review. Verify their remote bytes.
2. Run dummy-only preparation. The combined fourfold cost estimate must fit the shared two-hour stage cap. Publish and verify the complete passed preparation record.
3. Train all eight fixed ordinary models. Save every model and its ability checks, including failures. Publish and verify the complete training record. Do not begin edit work unless all eight models pass the 99% ordinary accuracy rule.
4. Learn and select edits using only the learning and selection intervention sets. Save all mask scores and all four rotation checkpoints per model. Before test interventions, publish the complete `calibration/edits` folder in `calibration/frozen.zip`, including all candidate scores, chosen raw selection outputs, learning records, choices, checkpoint tensors, and the inner manifest. Publish calibration status and the complete outer manifest too. The archive file list and bytes must match that completed manifest and local evidence.
5. Run all fixed final tests and controls. Do not change a choice from the test result.
6. Audit saved evidence, publish the plain-English paper and exact results, and verify the remote package. Only a complete study may enter the program count.

A failed ordinary panel stops the edit stages. Do not replace failed seeds or train a new panel within this plan. Numerical or resource failures stop the current attempt. Keep complete records and label the unfinished scope clearly.

## Models and pair sets

Train seeds 91001 through 91008 with the original architecture, initialization rule, learning rate 0.02, and exactly 4,000 full-batch Adam updates. There is no early stopping or replacement seed. Complete all eight ordinary models before the shared ability decision. Saved outputs must match separate arithmetic.

Enumerate all receiver/request blocks and all 48 donor contexts per block. Fix the split by the hash rule in the model method. Keep every donor for a block in the same split. Use 18,432 learning pairs, 9,216 selection pairs, and 9,216 test pairs. All models and methods use those same pairs. No output may select a split.

## Three edit methods

Method 1 tests all 220 three-neuron masks. It chooses the mask with the smallest selection probability-distance error in active, different-value cases. The target is the intact model output on the known coherent input change. This is the intended-change-only comparison.

Method 2 tests the same masks. It first requires mean probability distance at most 0.05 in inactive different-value cases and at most 1% answer changes in same-value cases. Among eligible masks, it minimizes active different-value distance. If none qualifies, record no eligible mask. Do not relax the limits.

Method 3 learns one orthogonal rotation of the 12-value hidden state per model, then replaces three coordinates in that rotated state. Fix the rotation initialization seeds at 92001–92008 and draw seeds at 93001–93008. Use Adam at 0.01 for exactly 4,000 updates. Each update uses 64 active different-value pairs, 64 inactive different-value pairs, and 64 same-value pairs from the learning set. The loss is the sum of the three group mean probability distances. The edit method gives the exact matrix parameterization and arithmetic.

Save checkpoints after 1,000, 2,000, 3,000, and 4,000 updates. Choose among these on the selection set with Method 2's eligibility limits, then smallest active different-value distance. A missing eligible rotation is a result. It is not permission for more updates or a new loss.

Publish every chosen mask or checkpoint, its selection measures, and all missing-choice records before final tests. Publish all retained calibration evidence at the same time, through the complete `calibration/frozen.zip` archive. Keep the calibration source folder unchanged. Create the archive after closing `calibration/manifest.json`, which therefore excludes the archive itself. The runner requires exact archive membership, with no duplicate entries, and checks every file against the public manifest and local evidence. Do not defer candidate scores or chosen raw selection outputs until after final tests.

## Final criteria

Apply these rules separately in every model:

- At least 95% correct required answers on active different-value test pairs.
- At least 95% correct required answers on all different-value test pairs.
- Mean zero-to-one probability distance at most 0.05 in each active and inactive different-value group.
- At most 1% changed answers on same-value test pairs.

The primary intended-change measure uses active different-value pairs. Correct unchanged answers in inactive cases must not inflate this measure. The additional all-different-value accuracy gate makes the broader requirement explicit.

For a practical improvement claim, Method 3 must pass every absolute rule in every model. It must also improve active different-value accuracy over Method 1 by at least five percentage points on average, with positive paired differences in at least six of eight models. Report all-different-value accuracy separately. A missing eligible Method 3 prevents this claim.

These are fixed practical criteria and descriptive paired counts. They do not use the normal significance test that failed Experiment 2B's calibration rule. Do not make a population significance claim or silently adopt another test. Passing these criteria would concern this finite task and model panel only.

## Controls and numerical checks

Use no edit, 20 fixed random three-neuron masks per model, full-layer donor replacement, and the known coherent input change. Full-layer donor replacement also copies donor context. It is not the coherent reference. Keep all controls and all final test pairs.

Use float64. Separate arithmetic must reproduce all candidate logits, probabilities, and selection measures, as well as every saved final output. The fixed absolute tolerance is `1e-10`. Orthogonality must meet the same tolerance. No-change operations must reproduce intact outputs. Nonfinite parameters, optimizer moments, losses, or saved numeric arrays stop the attempt. Do not enlarge a tolerance after outputs.

On interruption, save only a known complete, checked update boundary. Do not label partly updated parameters as a completed step. There is no automatic resume or retry. Any later continuation requires an explicit public rule, retained failure evidence, and cumulative accounting.

## Compute and usage

Use local CPU only, two compute threads, one inter-operation thread, 4 GiB peak memory during preparation, 16 GiB during science, and at least 20 GiB free disk. No paid APIs, rented hardware, new model downloads, or language-model forwards are allowed.

The roadmap preparation cap remains 3,600 seconds. Prior preparation is 328.7724097919963 seconds. Charge 120 seconds for this stage's pre-lock syntax, mock, and review checks, then measure dummy preparation. The projected cost must include all 32,000 base-model updates, all 32,000 rotation updates, mask evaluation, checkpoint selection, all final controls, source checks, saving, and replay. Use the fixed fourfold projections plus 120 seconds for shared setup and 120 seconds reserved for post-run checks.

The stage cap is 7,200 seconds. Training, edit calibration, and final tests share it. Each phase deducts all prior phase charges. Failures, replay, and saving count. The 120-second post-run reserve includes frozen-checkpoint packaging, the result audit, and final archive checks. It is one reserve, not a fresh allowance for each attempt.

The roadmap cap remains 183,600 seconds. Exact prior scientific charges are in the input compute record. Do not create a new budget from unused stage time. Keep all original and later charges once.

During computation, the supervisor must check live usage at least once per minute. An atomic private permit may last no more than 120 seconds. Stop at 20% remaining in any reported core window, on an unavailable meter, or on an invalid permit. Stop on time, memory, disk, token, source, or numerical failures. The signal and batch guards check between native operations; record any observed overrun. Do not leave a runner without active supervision. Pause the schedule at a usage stop.

## Reporting

Report every model, eligibility decision, practical gate, and control. Preserve null results. Distinguish a complete valid negative result from an interrupted run. This is one study, not one paper per method or seed. Publish code, evidence, a plain-English paper, a separate AI check, and exact compute charges. Verify remote bytes before counting completion. AI review is not human peer review.
