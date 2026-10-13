# Image shifts and channel removal in weak digit classifiers

Read the [paper](paper.md).

The sixteen models had low digit accuracy: **14.91% to 35.95%** on original test images. In the completed float64 follow-up, filtering before downsampling reduced the measured change in channel-removal effects under one-pixel shifts. Mean effect change was **0.08294** for direct sampling and **0.01177** for filtered sampling. The paired difference was **0.07117**, with a fixed seed interval of **[0.06304, 0.07786]**.

This measures edit stability in weak classifiers. It does not show faithful digit mechanisms or better explanations. All 1,797 test images, eight channels, and eight paired training seeds are included.

## Status and failure record

The original float32 run **failed a numerical control**. Its primary comparison remains incomplete. The fixed four-pixel shift limit was 2e-5; the observed error was 2.1457672e-5. The runner stopped after four complete model evaluations and completion of the fifth model's training. The original [failure receipt](results/failure.json), partial data, tolerance, plan, and code are unchanged.

A [fixed diagnostic](diagnostics/failure-batch/receipt.json) reproduced the float32 failure exactly. The same saved weights passed the specified float64 checks. A [separate public addendum](followup-plan.md) then kept all float32 training rules, reused the five completed checkpoints, and trained the eleven missing models. It evaluated all sixteen in float64. This follow-up is complete. It is a new numerical condition within this study, not a completed original float32 test.

All 45,632 follow-up controls passed. The [independent audit](reviews/independent-audit.json) reproduced all 12,938,400 saved logits exactly. It checked every control, score, training record, and schedule. It did not repeat training. The paper and separate AI claim review report the limits. There has been no human peer review.

## Reproduce the follow-up without replacing published outputs

Use Python 3.12 and the full Git history. The runners compare file bytes with the fixed commits. Install [requirements.txt](requirements.txt) from the current checkout. The commands below assume a local virtual environment at `.venv` and that the working directory is the repository root.

```sh
.venv/bin/python -m pip install -r studies/2026-10-13-shift-ablation/requirements.txt
git worktree add --detach ../mi-shift-reproduction 782dc2c8d73a3fa9e67a03c8cbd394a5548d9bf9
.venv/bin/python ../mi-shift-reproduction/studies/2026-10-13-shift-ablation/code/run_float64.py \
  --lock-commit 782dc2c8d73a3fa9e67a03c8cbd394a5548d9bf9 \
  --output ../mi-shift-reproduction/studies/2026-10-13-shift-ablation/results-float64
```

Use a new worktree path if that path already exists. Keep the current working directory unchanged for these commands. The lock checkout contains the raw data and the original failed-run files. The follow-up verifies the five reused checkpoint hashes, all 1,500 original training rows, and the three earlier schedules. It trains only the eleven remaining models and evaluates all sixteen. Thus this command reproduces the defined follow-up; it does not repeat training of the five reused models.

The runner refuses an existing output directory. It uses two CPU threads, no GPU, and a fixed 1,200-second limit. It stops and keeps partial files on a failed check. The published follow-up took 184.84 seconds on the study machine. Other machines can differ. The original float32 failure is environment-dependent and remains part of this study's record.

To repeat the independent saved-weight audit on the published results, use a fresh receipt path:

```sh
.venv/bin/python studies/2026-10-13-shift-ablation/code/independent_audit.py \
  --results studies/2026-10-13-shift-ablation/results-float64 \
  --output .local-state/shift-ablation-audit-repeat.json
```

This step loads saved weights, replays every evaluation condition, and rebuilds every score. It does not train models. To audit the new follow-up output, change `--results` to the new worktree's results path. Use another fresh receipt path for each repeat.

To rebuild the figure and six CSV tables from published summaries:

```sh
.venv/bin/python -m pip install -r studies/2026-10-13-shift-ablation/requirements-plot.txt
.venv/bin/python studies/2026-10-13-shift-ablation/code/plot_followup.py
```

The reporting step makes no model calls and does not change scientific results.

## Files and fixed records

| Record | Purpose |
|---|---|
| [Original plan](analysis-plan.md), [runner](code/run.py) | Original float32 experiment, fixed at `c406a08091cbc74caef1a1c4690aa29482667e0a` |
| [Original results](results/) | Preserved failed run, published at `5d7a88d9e79d974f7e320359039e11e813d20b15` |
| [Diagnostic plan](reviews/failure-diagnostic-plan.md), [code](code/failure_diagnostic.py), [outputs](diagnostics/failure-batch/) | Exact failed batch, fixed numerical checks, and all results |
| [Follow-up plan](followup-plan.md), [runner](code/run_float64.py) | New float64 evaluation condition, fixed at `782dc2c8d73a3fa9e67a03c8cbd394a5548d9bf9` |
| [Data](data/README.md), [fetch code](code/fetch_data.py) | Attributed UCI files, licences, hashes, and retrieval |
| [Sources](sources.md) | Checked prior work and overlap limits |
| [Follow-up metadata](results-float64/metadata.json), [manifest](results-float64/manifest.json) | Versions, fixed hashes, and result integrity |
| [Reuse ledger](results-float64/reuse-verification.json) | Original checkpoint, record, and schedule checks |
| [Summaries](results-float64/model-summaries.json), [primary estimate](results-float64/summary.json) | All fixed measures and paired differences |
| [Pre-run review](reviews/followup-pre-run-review.md), [audit](reviews/independent-audit.json) | Separate AI checks and their scope |

Each follow-up logit archive has shape `(5 shifts, 9 conditions, 1797 images, 10 classes)`. Shift order is unchanged, up, down, left, right. Condition zero is intact; conditions 1 through 8 remove channels 0 through 7. Values are float64 logits. Compute probabilities with the specified float64 max-subtracted softmax. The `models/` archives remain the trained float32 weights. All eight batch and shift schedules are saved. New and reused training records are separate.

The [paper](paper.md) explains why lower edit change, passing numerical checks, and low task accuracy must be assessed separately. No result is presented as human-reviewed or as an established discovery.
