# Training order and internal shortcut paths

Status: complete fixed experiment, with a passed independent AI audit. No human peer review has occurred. Read [the paper](paper.md).

The main probability contrast was **−0.00002737**, or **−0.00274 percentage points**. This is very small and opposite to the proposed direction. All 48 final models gave the correct class on all 32 inputs. All 66,048 recorded final patched outputs also kept the correct class. The paper shows both probability and logit-margin effects and keeps the partial-path and full-input measurements separate.

Read [the fixed plan](analysis-plan.md) and [the source record](sources.md). The public plan and runner lock is `c6ff9cb1405e6a65fe161c2f4cb471d51016bf4c`. The last dummy-only receipt is [preflight-v3.json](preflight-v3.json). The two earlier receipts are kept as history. They do not contain scientific outcomes.

The result files include [the summary](results/summary.json), [all checkpoint hashes](results/checkpoints.json), [all model metrics](results/model-metrics.csv), [all set means](results/set-means.csv), [all selections](results/selection.csv), [all intact outputs](results/intact.csv), and [the run receipt](results/run-receipt.json). The complete intervention table and training-loss table are compressed CSV files. The saved model and momentum arrays are in `results/checkpoints/`. The [manifest](results/manifest.json) gives their hashes.

The run completed 120,000 updates and 198,144 intervention evaluations in 51.54 seconds. The [independent audit](reviews/independent-audit.json) replayed all 144 evaluated checkpoints and checked all intervention rows in 10.16 seconds. It did not replay training. See [the independent review](reviews/independent-review.md) for its scope.

From the repository root, install the pinned packages in an isolated Python environment:

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r studies/2026-10-13-training-order-paths/requirements.txt
```

To repeat only the dummy checks, use a fresh output path:

```sh
.venv/bin/python studies/2026-10-13-training-order-paths/code/run.py --preflight --output .local-state/training-order-dummy.json
```

To repeat the fixed experiment, keep the locked plan and runner unchanged and use a fresh output directory:

```sh
.venv/bin/python studies/2026-10-13-training-order-paths/code/run.py --lock-commit c6ff9cb1405e6a65fe161c2f4cb471d51016bf4c --output .local-state/training-order-reproduction
```

The runner refuses to overwrite its output directory. It uses CPU only, two threads, fixed seeds, float64, and a 20-minute cap. It preserves incomplete runs. It does not download a model or use a paid API.

To audit the saved published results and write a fresh receipt:

```sh
.venv/bin/python studies/2026-10-13-training-order-paths/code/independent_audit.py --results studies/2026-10-13-training-order-paths/results --output .local-state/training-order-audit.json
```

To redraw the figure from the saved results:

```sh
.venv/bin/python -m pip install -r studies/2026-10-13-training-order-paths/figure-requirements.txt
.venv/bin/python studies/2026-10-13-training-order-paths/code/plot.py
```

The [PNG](figures/order-effects.png) and [PDF](figures/order-effects.pdf) use percentage points. The report makes no claim about a permanent effect, a unique semantic circuit, or large language models.
