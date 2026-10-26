# Experiment 2C: variable change and preservation

Read the [fixed plan](analysis-plan.md), [model method](MODEL_METHOD.md), and [edit method](EDIT_METHOD.md). This is one controlled comparison. The initial record contains code and methods only.

Use the existing Python environment with NumPy 2.3.5, PyTorch 2.8.0, and Transformers 4.57.6. This stage does not use a language model. It trains eight small selector models and one rotation per model under fixed local CPU limits.

The phases, from the repository root, are:

```sh
MI_USAGE_PERMIT=/absolute/private/permit.json .venv/bin/python studies/2026-10-26-variable-specificity/code/run.py prepare --lock PUBLIC_PLAN_COMMIT
MI_USAGE_PERMIT=/absolute/private/permit.json .venv/bin/python studies/2026-10-26-variable-specificity/code/run.py train --lock PUBLIC_PREPARATION_COMMIT
MI_USAGE_PERMIT=/absolute/private/permit.json .venv/bin/python studies/2026-10-26-variable-specificity/code/run.py calibrate --lock PUBLIC_TRAINING_COMMIT
MI_USAGE_PERMIT=/absolute/private/permit.json .venv/bin/python studies/2026-10-26-variable-specificity/code/run.py experiment --lock PUBLIC_CALIBRATION_COMMIT
```

Each output folder must be new. All eight ordinary models must pass before edit work. Publish every training file before calibration. Before final tests, publish calibration status, the complete manifest, the choices, the inner calibration manifest, and `calibration/frozen.zip`. The archive must contain every file in `calibration/edits`, with names relative to `calibration`. This includes all mask scores, checkpoint scores, all four checkpoints per model, chosen raw selection outputs, learning records, choices, and the inner manifest. Use the file list in the completed `calibration/manifest.json`; do not use a fixed member count. The archive stays outside the unchanged `calibration/edits` folder and is created after the manifest, so that manifest excludes `frozen.zip`. Before tests, the runner verifies the exact local and archive file lists and every archived file against the public manifest and local evidence.

The shared runner checks source bytes, prior phase records, runtime, and cumulative time. The private permit must follow the live-usage rule. It cannot authorize work past the 20% reserve or any compute cap. Do not bypass a stopped receipt, replace a failed seed, or resume automatically.
