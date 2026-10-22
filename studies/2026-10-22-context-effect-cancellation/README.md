# Reproduction

Read [the fixed plan](plan.md) and [the source record](sources.json). This study uses only Python standard-library code. It needs Python 3.12 or later. It trains no model.

The root supervisor must supply a valid private `MI_USAGE_PERMIT` and keep the account reserve. Use a same-directory temporary file followed by atomic replacement when updating that permit. Never publish the permit.

From the repository root, run:

```sh
python studies/2026-10-22-context-effect-cancellation/code/run.py --lock PUBLIC_PLAN_COMMIT
```

Replace `PUBLIC_PLAN_COMMIT` with the verified public commit that contains the exact plan, sources, and code. The runner refuses existing results. To repeat the work, use a separate checkout and preserve the original result directory outside it. The runner writes raw rows, summaries, an independent equation audit, metadata, and hashes. It stops on a failed check, expired permit, or 300-second limit.
