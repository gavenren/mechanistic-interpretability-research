# Hall-obstruction test

Read the [paper](paper.md).

Four small neural networks learned to classify all binary 4×4 bipartite graphs. Selected four-unit edits had a larger relative effect on a named impossible-matching case than random edits. The mean difference was **11.06 logit units**. But **all four models failed the fixed ability requirements**. Selected edits also raised false acceptance of a different impossible-matching case from **6.34% to 84.72%**. Removing the whole first layer left a constant output, yet gave positive selectivity of **1.98**. These results do not validate a Hall-rule component.

The experiment, [pre-run AI review](reviews/pre-run-review.md), [result audit](reviews/independent-audit.json), and [result and claim review](reviews/independent-review.md) are complete. The audit reproduced all intact and edited logits exactly. It did not repeat training. No human peer review is claimed.

## Fixed scope

- Four fixed seeds; 64 full training passes per model; no early stop or seed replacement.
- Selection on six-edge graphs; intervention test on seven-edge graphs. Both sets are in the natural training data.
- Exact labelled degree controls, positive controls, and distinct-cause negative controls.
- All 64 single-unit results, eight fixed random four-unit sets, selected sets, empty edits, and full-layer edits.
- Float64 arithmetic on two CPU threads. The completed run took 36.16 seconds.

## Reproduce

Use Python 3.12 with [requirements.txt](requirements.txt). From the repository root, run into a **new empty output path**:

```sh
python studies/2026-10-13-hall-obstruction/code/run.py run \
  --lock 782dc2c8d73a3fa9e67a03c8cbd394a5548d9bf9 \
  --out .local-state/hall-obstruction-repeat
```

The exact locked files must remain present. The runner checks their bytes. It refuses a nonempty output directory and stops at the fixed 20-minute limit. The published run did not reach that limit.

To rebuild the tables and figure from the published results, also install [requirements-report.txt](requirements-report.txt), then run:

```sh
python studies/2026-10-13-hall-obstruction/code/report.py
```

This report step reads saved results and does not call a model. It is a descriptive step added after the run. It does not change the primary endpoint.

To repeat the independent saved-weight audit, use a **fresh receipt path**:

```sh
python studies/2026-10-13-hall-obstruction/code/independent_audit.py \
  --results studies/2026-10-13-hall-obstruction/results \
  --output .local-state/hall-obstruction-audit-repeat.json
```

## Files

- `analysis-plan.md`, `inputs.json`, `sources.md`: fixed design, graph IDs, masks, and prior work.
- `results/run-metadata.json`, `results/manifest.json`: run status, counts, timing, software versions, and result hashes.
- `results/data.npz`: all graph labels and the exact 64 epoch orders.
- `results/model-*.npz`: initial weights and eight scheduled snapshots per model.
- `results/intact-*.npz`: every natural-graph output.
- `results/ablations-*.npz`: every intervention output and removed activation norm.
- `results/masks-*.json`: every unit ranking and selected mask.
- `results/group-scores.csv.gz`: all 64,800 degree-group score rows and category effects.
- `results/training.csv.gz`, `results/controls.csv`: all 32,768 training losses and 1,188 numerical checks.
- `tables/`, `figures/`: compact descriptive reports.

All numerical controls passed. The maximum error was 1.11e-16 against a fixed 1e-10 tolerance. All four models failed the separate scientific ability requirements. These two facts concern different checks.
