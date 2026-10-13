# Post-failure addendum: float64 evaluation of the fixed digit models

Prepared 13 October 2026, after the original float32 run failed. This addendum defines a **new numerical follow-up within the same study**. It does not complete the original primary test or turn its failed control into a pass. The original plan, runner, data, partial outputs, and tolerance remain unchanged.

## Reason for the follow-up

The original run was locked at `c406a08091cbc74caef1a1c4690aa29482667e0a`. It stopped after four complete model evaluations and after training a fifth model. A four-pixel shift control for seed 2 direct had a logit error of `2.1457672119140625e-5`, above the fixed `2e-5` limit. Its failed results and diagnostic plan were published at `5d7a88d9e79d974f7e320359039e11e813d20b15`.

The separate, fixed diagnostic used the exact failed 128-image batch and the same saved weights. Its hook path reproduced both the saved float32 logits and the recorded failure exactly. The same-weight float64 hook path had a symmetry error of `8.881784197001252e-16`. A separate unfolded-convolution path had zero float64 symmetry error; the two float64 paths differed by at most `1.7763568394002505e-15`. These results are consistent with numerical error in the float32 computation. They do not validate the missing models or identify one operation as the sole cause.

The original partial output files already exist. Thus this follow-up is not blind to the existence of those records. The change is made because of the recorded numerical failure. No data, seed, training step, model, channel, shift, estimator, subgroup, bootstrap rule, or interpretation condition is selected from a partial primary score. The author has not read the partial D comparison while preparing this addendum. The full original comparison is still incomplete.

## What stays fixed

Use all original rules in `analysis-plan.md` unless this addendum explicitly changes numerical evaluation, checkpoint reuse, output dtype, or new-run counts. In particular:

- Use the exact original UCI files, 3,823 training rows, and all 1,797 test images.
- Keep seeds 0 through 7 and both direct and filtered architectures: all 16 models.
- Keep the original float32 initialization, Adam, ten full epochs, 300 updates per model, batch order, augmentation, and all hyperparameters.
- Keep all eight first-ReLU channel removals, all five main shifts, all four four-pixel controls, and all no-op and local edit checks.
- Keep the sole D_direct minus D_filtered estimator, all eight paired seed differences, the same 10,000-draw seed bootstrap with seed 2026101417, and every original secondary measure.
- Keep the full test set as primary. The correct-under-all-shifts subgroup remains secondary and model-specific.
- Keep the distinction between within-model channel identity and between-architecture training procedures. No semantic channel alignment is assumed.

## Reuse five completed training runs

The evaluation failure occurred after all 300 training updates for seed 2 direct. Reuse the following five final **float32** checkpoints. Do not retrain or replace them. They are selected by the fixed execution order and completed training status, not by accuracy or effect size.

| Model | SHA-256 of original checkpoint |
|---|---|
| 0 direct | ac8be47aefc85db0c7414157025da5bedeae7a3572479f700c9d188c189e2a06 |
| 0 filtered | e934d69bb24ba3b669408922db68eb34e0e96105628d7597a37d72a7ecd1cef5 |
| 1 direct | 3602fff9ff353f8a6dd0cbe1460cbd2fe2147b7768fe929652b1f56424340b76 |
| 1 filtered | b0c870a6ca82df1110ca18d56e00e23eb05efabf06eb8a8d3212a53961f91d82 |
| 2 direct | b6cc052463f641f3cb9eb1ecccf374de6610f384219de30d96902e1274c9ea8e |

The original failed manifest SHA-256 is `a4c9e610eb86d31fc163586c1de563eee3e6856a08460a5a9d94cd052b9f5e26`. The original training CSV SHA-256 is `e1afd92ec8939fc756ad2f4558546c2f5cc286a13287ebaab5cbf37eebb727cb`.

Before any new training or test forward, verify every original manifest entry. Compare the failed manifest, full training CSV, five checkpoint files, and three shared schedule files byte for byte with commit `5d7a88d9e79d974f7e320359039e11e813d20b15`. Verify each checkpoint hash and that all stored arrays are finite float32 values. Verify exactly 1,500 original training rows: 300 for each listed model, ten ordered epochs, 30 ordered batches per epoch, batch sizes 128 and 111, exact global update IDs 1 through 1500, and finite loss records. Save a hash of each model's canonical training-row records. This check includes seed 2 direct even though its model-summary file was not completed.

Rebuild the original schedules from the fixed generator and compare the complete arrays for seeds 0, 1, and 2. Their file hashes must be:

| Schedule file | SHA-256 |
|---|---|
| seed-0.npz | 9481caf447fdfaae2fe35670d7c283691eac2c0b2b57d92520243f196243e21c |
| seed-1.npz | eccdc3f33e57324920d095a1c2d0a5c4cf31f6688f139e1f2b468abf1cc132ec |
| seed-2.npz | 7f01434ae3de01c6cb793b31e0a9dad515940bdb9eb059e2bb876d39c8ebeccf |

Copy these files and the five checkpoints unchanged into the follow-up output. Copy the full 1,500-row training log as `reused-training.csv`. Save a reuse-verification ledger. The new `training.csv` contains only new training. Reuse does not count as a new training run.

Train only the eleven missing models: seed 2 filtered, then both architectures for seeds 3 through 7, in the original order. Each starts from its original parameter seed and receives its original 300-update training schedule in float32. Save new float32 checkpoints before numerical evaluation. The final panel has the originally planned sixteen trained models, with no omitted or replaced model.

## New evaluation arithmetic

For each reused or newly trained model, convert its complete saved float32 parameter state and fixed filter to float64. Convert the test canvas to float64. Run the full network and all test interventions in float64. Training remains float32. These are different numerical computations from the original float32 evaluation, not ground truth for its exact output program.

Save **float64 logits**, one archive per model, with unchanged shape `(5,9,1797,10)` and the original shift and condition orders. Calculate probabilities using the same NumPy float64 max-subtracted softmax. Do not duplicate the probability archive. Save all model summaries and signed-effect reconstruction data as before. An uncompressed logit array is about 6.47 MB per model, below the per-file repository limit.

Use a new fixed absolute tolerance of **1e-10** for float64 four-pixel logit symmetry and first-feature shift checks. Keep tolerance zero for no-op output equality, removed-channel zero, and unchanged entries at the edit site. Keep 1e-12 for probability sums and the triangle inequality. These are limits for the newly defined float64 condition. The old float32 check still has its unchanged 2e-5 limit and remains failed.

Apply every check to all sixteen models and every original test case. Stop and preserve the new run if any check fails. Do not increase a limit, remove cases, resume past a failure, or add a new condition after looking at the follow-up outcome.

## New count and time limits

There are 3,300 new optimizer updates and 420,530 new training examples. The earlier 1,500 updates and 191,150 training examples are reused through saved checkpoints. Across the study, training remains the originally planned 4,800 updates and 611,680 examples.

The follow-up has 20,640 full evaluation calls and 2,472,672 evaluation examples. Together with new training, it has **23,940 full model calls and 2,893,202 forwarded examples**. Its numerical table has **45,632 control rows**. There are also 2,400 first-convolution/ReLU calls for feature controls. Check the sixteen-model, update, call, example, and control totals before completion. Record reused and new training costs separately. The diagnostic and failed evaluation add to total study cost and must be disclosed.

Use the same packages, deterministic settings, and two CPU threads. Before the new public lock, run a dummy-only preflight with random tensors. Train its dummy models in float32 and evaluate them in float64. Test the actual training and evaluation batch sizes, all original structural controls, the untrained float64 stride-one control, and score arithmetic. Project new training time with `420530/(2*15*239)` and evaluation time with `2472672/(2*133*86)`, then apply a two-fold factor. This is an estimate.

The complete follow-up has a fixed **1,200-second** cap. Do not extend it. Stop if the timer or a numerical check fails. Use a new output directory, `results-float64/`, and refuse any existing directory. Preserve partial arrays, current model weights, training records, and a failure manifest as in the original runner.

## Lock, audit, and paper

Before and after the follow-up, compare the exact addendum, new runner, original plan and runner, requirements, and raw data with the newly supplied full public lock SHA. Recheck all original reuse records at the end. Preserve the original `results/` directory without change. The new runner is `code/run_float64.py`; it does not edit or import the original runner.

The separate AI reviewer must review this addendum, code, and dummy receipt before lock. After the run, the reviewer will rebuild scores from all saved float64 logits and replay the interventions through an independent hook. Report any limits to replay coverage. A saved-weight audit does not establish that training was independently repeated. No human review is claimed.

Write one paper for this study. Its methods and abstract must identify the failed original float32 run and this post-failure numerical follow-up. It may report the full follow-up comparison if its checks and review pass. It must not call the original primary test complete, claim all originally planned controls passed, or split this correction into a second paper. Lower D alone still does not establish faithful semantic units or equal task ability.
