# Choosing flow neurons by likelihood damage or output change

The fixed experiment is complete. The independent AI audit passed. All four models failed the fixed density-fit rule. They all passed the coarse quadrant-frequency rule. The output-based selector gave a larger held-out quadrant change in three models and the same change in one. This is not evidence of better samples. The [final AI claim review](reviews/independent-review.md) is complete.

- [Plain-English paper](paper.md)
- [Fixed plan](analysis-plan.md)
- [Primary sources and close overlap](sources.md)
- [Results manifest](results/manifest.json)
- [Selected-unit table](tables/selector-comparison.csv)
- [All unit rankings](tables/unit-rankings.csv)
- [Per-model unit means and ranges](tables/all-unit-summary.csv)
- [Generated frequencies](tables/generated-quadrants.csv)
- [Intact moments and test-data context](tables/intact-moments.csv)
- [Figure](figures/flow-unit-ranking.png)
- [Independent audit receipt](reviews/independent-audit.json)
- [Pre-run review](reviews/pre-run-review.md)

The pre-run AI review was completed before execution. Its file is included with the result release. It was not part of the earlier public plan and code lock.

The public lock is `4a52bcb4c102851338f7eb99179192ae0153da2d`. The runner checks the exact plan, runner, requirements, and source bytes against that commit before the run and at the end. Do not change those files to repeat the study. The study is AI-authored and AI-reviewed, with no human peer review.

## Repeat the experiment without changing published outputs

Use Python 3.12 and the pinned requirements. From the repository root, create a separate worktree at the public lock. The paths below keep the published results intact. The destination must not exist already.

```sh
python3.12 -m venv .venv
.venv/bin/python -m pip install -r studies/2026-10-13-flow-unit-ranking/requirements.txt
git worktree add --detach ../mi-flow-reproduction 4a52bcb4c102851338f7eb99179192ae0153da2d
.venv/bin/python ../mi-flow-reproduction/studies/2026-10-13-flow-unit-ranking/code/run.py --lock-commit 4a52bcb4c102851338f7eb99179192ae0153da2d --output ../mi-flow-reproduction/studies/2026-10-13-flow-unit-ranking/results
```

If a suitable `.venv` already exists, use it. The runner requires NumPy 2.3.5 and PyTorch 2.8.0. It uses CPU float64, two intra-op threads, one inter-op thread, and deterministic operations. Linear parameters start with PyTorch default float32 initialization, are converted to float64, and then have their output layers set to zero. The four seeds, arrays, and batch schedules are fixed.

The observed scientific runtime was 91.71 seconds. The fixed cap is 1,200 seconds. The runner stops on a failed control and preserves the failure record. It refuses an existing output directory. It does not resume through a failed check, drop a seed, or extend training after a failed fit rule.

## Repeat the independent saved-weight audit

This uses the saved scientific outputs. It does not retrain the models. Use a fresh receipt path on each repeat.

```sh
.venv/bin/python studies/2026-10-13-flow-unit-ranking/code/independent_audit.py --results studies/2026-10-13-flow-unit-ranking/results --output .local-state/flow-unit-ranking-audit-repeat.json
```

The published audit reproduced all saved density terms exactly. It regenerated all conditions to check quadrant and transition counts, displacement sums, crossings, selections, rankings, fit measures, and the fixed bootstrap. Its separate Jacobian check used an analytic construction. It also checked training records and schedules. This saved-weight audit is not a repeated training experiment.

## Rebuild the report

The report reads the published result files. It makes no model calls. It writes five CSV tables and PNG/SVG figures in this study folder.

```sh
.venv/bin/python -m pip install -r studies/2026-10-13-flow-unit-ranking/requirements-report.txt
.venv/bin/python studies/2026-10-13-flow-unit-ranking/code/report.py
```

## What is saved

`inputs.npz` contains all fixed data, component labels, noise inputs, and training row IDs. The result folder also contains initial and scheduled checkpoints, every training loss, every control row, raw density terms for all 129 conditions, calibration selections, full score records, and the bootstrap draws. Density archives group at most 32 conditions; no result file exceeds 4.1 MB in this run.

Generated coordinates are saved for the intact model and selected test removals. The other generated coordinates were not all archived. Their counts, transition tables, displacement sums, and crossing counts are saved. The complete model states and noise arrays permit exact replay. The [dummy check record](reviews/dummy-preflight.md) remains separate from the scientific results.
