# Copy routing in Pythia-14M

Read the [paper](paper.md) for the result and its limits. The fixed test finds that route and value changes in one attention head each harm use of a supplied binding. Their combination largely restores the mean answer signal. This is a small replication of an established copying mechanism.

## Files

- `analysis-plan.md`: plan written before scientific results.
- `sources.md`: checked primary sources and the contribution limit.
- `code/run.py`: token generator, model checks, experiment, controls, and summaries.
- `code/fetch_model.py`: download and check the fixed model files.
- `code/plot.py`: make the two-panel result figure.
- `code/independent_audit.py`: rebuild tables and rerun the selected-head results without importing the runner.
- `results/`: full input records, all head results, intervals, controls, and hashes.
- `reviews/`: independent AI review and control receipts.

## Reproduce the fixed run

Run these commands from the repository root. Python 3.12 and a CPU are sufficient. The pinned model files use roughly 30 MB of disk space. The experiment does not call a paid inference service.

```bash
python3.12 -m venv .local-state/copy-environment
.local-state/copy-environment/bin/python -m pip install -r studies/2026-10-13-copy-routing/requirements.txt
.local-state/copy-environment/bin/python studies/2026-10-13-copy-routing/code/fetch_model.py --output .local-state/models/pythia-14m
.local-state/copy-environment/bin/python studies/2026-10-13-copy-routing/code/run.py --model-dir .local-state/models/pythia-14m --dummy-preflight --output .local-state/copy-dummy-repeat
.local-state/copy-environment/bin/python studies/2026-10-13-copy-routing/code/run.py --model-dir .local-state/models/pythia-14m --lock-commit 512fdba309df32aa3a98db59c65ec5d5d383635f --output .local-state/copy-repeat
```

Each output directory must be empty. The runner compares its source and the plan with the saved commit. The published scientific run took about 28 seconds in the recorded local environment; time on another computer can differ. Small floating-point differences can also occur across systems.

To remake the published figure from the saved result tables:

```bash
.local-state/copy-environment/bin/python studies/2026-10-13-copy-routing/code/plot.py
```

`results/interventions.csv` has one row for each test prompt and head. It stores all three intervention conditions in that row. Thus 9,216 rows represent 27,648 intervention examples. `baseline-groups.csv` and `head-groups.csv` retain the whole-token-group units used for inference. `bootstrap.json` contains the primary intervals. `manifest.json` records SHA-256 hashes of all scientific result files.

To repeat the independent audit of the published results, use a new receipt path:

```bash
.local-state/copy-environment/bin/python studies/2026-10-13-copy-routing/code/independent_audit.py --model-dir .local-state/models/pythia-14m --output .local-state/copy-audit-repeat.json
```

The published audit passed. It checks all table arithmetic and repeats all baseline attention rows and target logits. It also repeats all 1,152 intervention examples for the selected head. It does not repeat intervention forwards for the other 23 heads.

The model is `EleutherAI/pythia-14m` at revision `cf967c0a9a04383db6f7b1108d86b2962634b4ac`. Do not use a different revision under the same model name. The model and external research sources retain their own terms.
