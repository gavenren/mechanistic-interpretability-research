# Preparation and task-ability test

Read the [fixed execution plan](analysis-plan.md), [symbolic methods](METHODS_SYMBOLIC.md), and [numeric methods](METHODS_NUMERIC.md).
The [preparation record](preparation/status.json) contains measured checks and the full compute estimate.
The [runner review](REVIEW.md) and [symbolic source review](REVIEW_NUMERIC.md) record separate AI checks. No human review is recorded.

## Repeat the work

Use Python 3.12.14, NumPy 2.3.5, and PyTorch 2.8.0 on a CPU. Run from the repository root.
The runner fixes two compute threads. It refuses to replace an existing preparation or results directory.
For a separate reproduction, use a new checkout and move the recorded output directories outside that checkout. Keep the published original records.

A supervisor must check live account usage and write a private JSON permit with `allowed: true` and an `expires_at` Unix timestamp no more than 120 seconds ahead. Set `MI_USAGE_PERMIT` to that private file's absolute path. Refresh it at least once per minute after a successful live check. Do not refresh it if any core window has 20% or less remaining, or ordinary usage is unavailable. An expired or denied permit stops the run. The permit must not be committed.

```sh
python studies/2026-10-19-capability-gates/code/run.py prepare
python studies/2026-10-19-capability-gates/code/run.py experiment --lock PUBLIC_PLAN_COMMIT
```

Replace `PUBLIC_PLAN_COMMIT` with the public commit that contains these exact code and method files. The runner checks their bytes, the reused graph source, and the digit dataset against that commit. A preparation pass is also required. The experiment performs a separate output replay after every complete job.

An interrupted training job saves partial weights and its completed trace. The runner records the stop reason. There is no automatic resume, replacement seed, or added training. A continuation must retain the failed attempt and fix its next action before more scientific work.

## Recorded attempt and permit update correction

Read the [execution report](paper.md). The first attempt stopped during its second development model. Preserve its results. It has no automatic resume.

Write each refreshed permit to a temporary file in the same directory. Then use `os.replace(temporary_path, permit_path)`. Do not truncate and rewrite the live permit file. A reader can see an empty document during a direct write. This caused the recorded stop. The local supervisor now uses replacement, but no scientific run has tested that correction. A later continuation needs a new public lock and an explicit rule for the partial seed.
