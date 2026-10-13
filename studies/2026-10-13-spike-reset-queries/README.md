# Matching spike explanations to two meanings of deletion

Status: the fixed run, independent saved-weight audit, and [final report review](reviews/independent-review.md) are complete. The [paper](paper.md) is AI-authored; all reviews are AI reviews. No human peer review is claimed.

Four trained spiking networks reached 95.55–96.33% digit accuracy. Matching an explanation's local paths to the exact deletion gave a mean single-event selection gain of **0.01227 margin units**. The descriptive four-seed interval is **[0.01008, 0.01492]**. All seed primary means are positive, but seed 317 loses for message deletion. The four-event secondary test also has per-seed losses. Matching leaves selection error and does not ensure correct effect signs.

The study compares message deletion with deletion of both the message and the immediate voltage reset. Both gradient graphs keep all future reset derivatives. The fixed explanation panel has 120 images from test-file rows 0–127. This result is limited to those images, models, candidates, and output score.

## Evidence

- [Fixed plan](analysis-plan.md), [source record](sources.md), [input ledger](inputs.json), and [licensed data](data/README.md).
- [Public lock](https://github.com/gavenren/mechanistic-interpretability-research/tree/b3d97c132b8162dd2cb21bce031820fd0f078065/studies/2026-10-13-spike-reset-queries) from before all study model calls.
- [Pre-run review](reviews/pre-run-review.md), [dummy history](reviews/dummy-history.md), and [final dummy receipt](reviews/dummy-preflight/dummy-preflight.json).
- [Results](results/), [summary](results/summary.json), [model summaries](results/model-summaries.json), and [file manifest](results/manifest.json).
- [Report tables and figure](report/), made by [report code](code/report.py) without model calls.
- [Independent audit code](code/independent_audit.py) and [audit receipt](reviews/independent-audit.json).

All 1,688 controls passed. The run saved four trained models, all training orders, all test logits, and all baseline and edited panel logits and binary traces. The separate audit reproduced all 431,880 logits and 27,648,000 binary trace values exactly. It checked the gradients with a manual reverse calculation. It did not repeat training.

The fixed run took 16.89 seconds on two CPU threads; the audit took 3.91 seconds. The maximum run time was 20 minutes. No GPU or paid compute was used.

## Reproduce the fixed run

From a clone that contains the public lock, use a detached worktree to keep its bytes fixed. Set a new output path each time. The runner refuses an existing directory and preserves partial output on failure.

```sh
git worktree add --detach ../spike-reset-lock b3d97c132b8162dd2cb21bce031820fd0f078065
cd ../spike-reset-lock
python3.12 -m venv .venv
.venv/bin/pip install -r studies/2026-10-13-spike-reset-queries/requirements.txt
.venv/bin/python studies/2026-10-13-spike-reset-queries/code/run.py \
  --lock-commit b3d97c132b8162dd2cb21bce031820fd0f078065 \
  --output /tmp/spike-reset-reproduction-1
```

The recorded environment used Python 3.12.14, NumPy 2.3.5, and PyTorch 2.8.0. Numerical reproduction on another host is a new check, not a result already established here.

## Reproduce the audit and report

The following commands use the published evidence checkout, which contains the post-run audit and report code. Use fresh output paths. The audit does not train models or change the saved results.

```sh
.venv/bin/python studies/2026-10-13-spike-reset-queries/code/independent_audit.py \
  --results studies/2026-10-13-spike-reset-queries/results \
  --output /tmp/spike-reset-audit-1.json
.venv/bin/pip install -r studies/2026-10-13-spike-reset-queries/requirements-report.txt
.venv/bin/python studies/2026-10-13-spike-reset-queries/code/report.py \
  --results studies/2026-10-13-spike-reset-queries/results \
  --output /tmp/spike-reset-report-1
```

The audit checks the source manifest, data and input rules, saved schedules and loss records, exact forward edits, both gradient graphs, row tables, the primary interval, and the ability gate. The report makes 64 descriptive table rows and one four-panel figure from the saved results.
