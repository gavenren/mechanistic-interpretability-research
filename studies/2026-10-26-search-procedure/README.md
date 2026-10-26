# Experiment 2B: search, scores, and selection

Read the [fixed plan](analysis-plan.md), [subset method](SUBSETS_METHOD.md), and [selection method](SELECTION_METHOD.md). All branches form one study. The initial state is code and method preparation, with no completed scientific result.

Use the repository's existing Python environment. It fixes NumPy 2.3.5, PyTorch 2.8.0, and Transformers 4.57.6. This stage uses no language model and no training. The shared runtime check keeps preparation and scientific phases in the same environment.

From the repository root, the guarded phases are:

```sh
MI_USAGE_PERMIT=/absolute/private/permit.json .venv/bin/python studies/2026-10-26-search-procedure/code/run.py prepare --lock PUBLIC_PLAN_COMMIT
MI_USAGE_PERMIT=/absolute/private/permit.json .venv/bin/python studies/2026-10-26-search-procedure/code/run.py calibrate --lock PUBLIC_PREPARATION_COMMIT
MI_USAGE_PERMIT=/absolute/private/permit.json .venv/bin/python studies/2026-10-26-search-procedure/code/run.py experiment --lock PUBLIC_CALIBRATION_COMMIT
```

Each phase requires its previous public record and exact source checks. The output folders must be new. Calibration must publish its status, manifest, frozen choices, and `calibration/banks.zip`. This archive contains the eight `subsets/bank-seed-N.npz` files. It stays outside the unchanged `calibration/subsets` folder. Raw development file hashes must also be public before final tests; all raw files must be published with the completed paper.

The private permit has `allowed: true` and a finite Unix `expires_at` no more than 120 seconds ahead. Issue it only after a live usage check. The supervisor must keep the 20% reserve and stop rules. Do not bypass a failed receipt or rerun for a preferred result. The program count changes only after complete evidence is verified on GitHub.
