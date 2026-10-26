# Task-ability finalization

Read the [plan](analysis-plan.md) before running this study. Old studies and result directories remain unchanged.

Use the repository `.venv`: Python 3.12.14, NumPy 2.3.5, and PyTorch 2.8.0. Run from the repository root. Keep the account permit private. Only a supervisor that has just checked live usage may refresh it, at least once per minute while computation runs. The existing atomic writer is `studies/2026-10-22-capability-completion/code/permit.py`. Set `MI_USAGE_PERMIT` to the private file. A permit expires after at most 120 seconds.

Publish the plan and code, then run:

```sh
.venv/bin/python studies/2026-10-26-capability-finalization/code/run.py prepare --lock PUBLIC_PLAN_COMMIT
```

If preparation passes, publish its records and use that later commit:

```sh
.venv/bin/python studies/2026-10-26-capability-finalization/code/run.py experiment --lock PUBLIC_EXECUTION_COMMIT
```

The runner must match the public code and plans used in preparation. It reuses six complete graph models, restarts only final seed 8103 from its original initialization, then runs seed 8104 and the other fixed task branches. It does not overwrite results or retry automatically. It retains all prior time under the same four-hour cap.

A clean reproduction needs a separate checkout and new output directories. Replaying saved weights is not independent retraining. The final report and results will state whether the attempt finished and which task panels passed.
