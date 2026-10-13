# Bounds on an equilibrium neuron-removal effect

Status: plan and code prepared for a public lock. No study model has been trained or evaluated. All current model outputs use generated dummy data.

The study asks whether a small solver residual is sufficient to establish the sign of an internal removal effect. It uses one edit: replace one state equation with zero, then solve the other equations again. The target is the real equilibrium of the stored finite weights. It is not a particular floating-point program output.

The method uses established contraction bounds and outward interval arithmetic. It does not introduce a new causal principle or a new verification theorem. See the [fixed plan](analysis-plan.md), [primary sources](sources.md), and [data licence](data/README.md).

The pre-run checks use exact rational answers and generated inputs:

- [Exact rational checks](reviews/interval-dummy-v1/receipt.json): 1,697 checks passed.
- [Independent interval checks](reviews/independent-interval-checks.json): 1,512 comparisons and rejection checks passed.
- [Full protocol dummy check](reviews/protocol-dummy-v3/receipt.json): 383 checks passed. It tests both real batch sizes, every stop rule, the full 128-update cap, a known sign reversal, partial failure records, and the time limit.
- [Cost check](reviews/timing-dummy-v1/receipt.json): the estimate is 227.4 seconds, with a two-fold factor and every effect at its update cap. The study limit is 1,200 seconds.
- [Dummy change record](reviews/dummy-history.md) preserves the earlier receipts and explains the changes.

From the repository root, use a Python environment with [the pinned packages](requirements.txt). These commands must use new output paths:

```sh
python studies/2026-10-13-equilibrium-sign-audit/code/check_intervals.py --output /tmp/equilibrium-rational-check
python studies/2026-10-13-equilibrium-sign-audit/code/check_protocol.py --output /tmp/equilibrium-protocol-check
python studies/2026-10-13-equilibrium-sign-audit/code/dummy_timing.py --output /tmp/equilibrium-cost-check
```

The study command requires the full public commit that contains the reviewed plan and all lock files. Use its exact bytes in a detached worktree if a later version changes any of those files. After the public lock is verified, the command is:

```sh
python studies/2026-10-13-equilibrium-sign-audit/code/run.py --lock FULL_PUBLIC_LOCK_COMMIT --output /tmp/equilibrium-study
```

The code refuses an existing output path. A failure stops the run and saves the available evidence. It does not change the tolerances or resume a partial run. A later independent audit will check the saved outputs; it will state whether training was repeated.
