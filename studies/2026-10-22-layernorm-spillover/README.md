# Reproduction

Read [the plan](analysis-plan.md) and [sources](sources.md). The report will be written after the fixed run.

Use Python 3.12.14 and NumPy 2.3.5. The normalization audit also uses PyTorch 2.8.0. From the repository root:

```sh
.venv/bin/python studies/2026-10-22-layernorm-spillover/code/run.py --lock PUBLIC_PLAN_COMMIT
.venv/bin/python studies/2026-10-22-layernorm-spillover/code/audit.py --lock PUBLIC_PLAN_COMMIT
```

The root supervisor must first verify the public plan/code commit and a live usage allowance above the 20% reserve. Set MI_USAGE_PERMIT to a private JSON file with allowed=true and expires_at no more than120seconds ahead. Refresh through same-directory temporary-file replacement after a live check. Set MI_STAGE_DEADLINE to the monotonic start plus120seconds for the run and audit together. Run only one scientific process at a time. Output directories refuse replacement. Keep a failed attempt. A new attempt needs an explicit revised plan.

No trained network or external dataset is included. Saved arrays and exact generators provide the evidence. A calculation audit does not provide external replication or human review.
