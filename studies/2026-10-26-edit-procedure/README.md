# Experiment 2A: the complete edit procedure

This study tests three parts of an edit procedure: donor choice and order, allowed sparse codes, and normalization. Read the [fixed plan](analysis-plan.md) and its three linked method notes. These branches form one study.

The initial state is prepared code, with no scientific results. A plan is not a completed paper. Later result receipts and the paper must state the actual completion state.

Use the existing environment with Python, NumPy 2.3.5, PyTorch 2.8.0, and Transformers 4.57.6. The normalization branch uses cached Pythia-14M files at the revision fixed in its method. No download is allowed in this run. All model execution is on CPU.

From the repository root, run preparation with:

```sh
MI_USAGE_PERMIT=/absolute/path/to/private-permit.json .venv/bin/python studies/2026-10-26-edit-procedure/code/run.py prepare --lock PUBLIC_PLAN_COMMIT
```

After passed preparation is published and verified, run:

```sh
MI_USAGE_PERMIT=/absolute/path/to/private-permit.json .venv/bin/python studies/2026-10-26-edit-procedure/code/run.py experiment --lock PUBLIC_PREPARATION_COMMIT
```

The supervisor must issue a fresh permit only after a live usage check. The private JSON has `allowed: true` and a finite Unix `expires_at` no more than 120 seconds ahead. The shared runner holds the repository compute lock. It refuses existing output folders. Do not bypass a stopped receipt or rerun it for a preferred result.

Raw results, exact checks, and the plain-English report will be linked here after execution. AI calculation review is not human peer review.
