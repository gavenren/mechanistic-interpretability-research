# Bounds on an equilibrium neuron-removal effect

Status: the fixed experiment and independent result audit are complete. [Final AI paper review](reviews/independent-result-review.md) is complete. This is an AI-authored study; human peer review has not occurred.

Read the [paper](paper.md). At the initial stopping point, the numerical bound did not establish 18.66% of nonzero effect signs. This is an unresolved-bound rate. Only five initial signs were proved wrong. Further updates established a sign or small-magnitude bound for every case. All four models passed the task check.

The public plan lock is [1403654ed629ff41dee40c8d094586c5eb40af73](https://github.com/gavenren/mechanistic-interpretability-research/commit/1403654ed629ff41dee40c8d094586c5eb40af73). The complete [results](results/summary.json), [file manifest](results/manifest.json), [report tables and figure](report/report.json), and [independent audit receipt](reviews/independent-result-audit.json) are included.

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

The study command requires the public lock below. Use its exact bytes in a detached worktree if a later version changes a locked file:

```sh
python studies/2026-10-13-equilibrium-sign-audit/code/run.py --lock 1403654ed629ff41dee40c8d094586c5eb40af73 --output /tmp/equilibrium-study
```

The code refuses an existing output path. A failure stops the run and saves the available evidence. It does not change the tolerances or resume a partial run. The independent audit matched all saved state and effect rows. It did not repeat training.

To repeat the independent audit, use a new receipt path. It does not import the study runner or repeat training:

```sh
python studies/2026-10-13-equilibrium-sign-audit/code/independent_audit.py --results studies/2026-10-13-equilibrium-sign-audit/results --output /tmp/equilibrium-audit-new.json
```

To rebuild the tables and figures, install [the report packages](requirements-report.txt) and use a new output directory:

```sh
python studies/2026-10-13-equilibrium-sign-audit/code/report.py --results studies/2026-10-13-equilibrium-sign-audit/results --output /tmp/equilibrium-report-new
```
