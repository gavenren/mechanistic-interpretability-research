# Experiment 5: complete cost of a tighter numerical bound

Read the [plan](analysis-plan.md), [bound method](BOUND_METHOD.md), and [source notes](sources.md). This study uses four saved equilibrium models. It does not train new models.

Use the existing fixed Python environment. Publish and verify the plan, code, input hashes, and review before preparation. Publish passed preparation before the scientific comparison. The conservative projection must fit the fixed two-hour stage cap.

From the repository root:

```sh
MI_USAGE_PERMIT=/absolute/private/permit.json .venv/bin/python studies/2026-10-26-componentwise-bounds/code/run.py prepare --lock PUBLIC_PLAN_COMMIT
MI_USAGE_PERMIT=/absolute/private/permit.json .venv/bin/python studies/2026-10-26-componentwise-bounds/code/run.py experiment --lock PUBLIC_PREPARATION_COMMIT
```

Each output folder must be new. The private permit follows live usage and cannot override any limit. A stopped run does not restart automatically. The full four-model, five-pair-per-model comparison stays fixed. Keep failed reliability checks and timing results. A mathematical bound, a completed comparison, and a useful speed gain are separate claims.
