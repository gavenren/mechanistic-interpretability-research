# Immediate and later effects of a prompt-state edit

This study compares two rules for selecting an internal edit site in Pythia-14M. One rule uses the immediate output change. The other uses the change after eight fixed tokens. The score measures output change, not answer quality.

**Result:** the immediate rule chose block 4. The delayed rule chose block 3. On new words, their later mean output changes were 0.000272 and 0.002866. The fixed difference was 0.002594. All planned controls and the separate AI replay passed. The model did not rank the requested copy word first on any of 64 prompts. This is not evidence of correct copying or useful memory.

Read the [paper](paper.md), [complete score table](results/scores.csv), [layer choices and group results](results/selection.json), and [independent audit receipt](reviews/independent-audit.json). The paper includes the close prior work and limits. No human review has occurred.

The [analysis plan](analysis-plan.md) and [runner](code/run.py) were fixed in Git commit `512fdba309df32aa3a98db59c65ec5d5d383635f` before scientific measurement. The [source check](sources.md), [dummy preflight](preflight.json), and [pre-run AI review](reviews/pre-run-review.md) are available.

## Reproduce the fixed run

Use Python 3.12 from the repository root. The commands below use a new local model folder and a new result folder. They do not overwrite the published results.

```bash
python -m venv .venv
.venv/bin/python -m pip install -r studies/2026-10-13-patch-memory/requirements.txt
.venv/bin/python studies/2026-10-13-patch-memory/code/fetch_model.py --output .local-state/pythia-14m
.venv/bin/python studies/2026-10-13-patch-memory/code/run.py --model-dir .local-state/pythia-14m --preflight
.venv/bin/python studies/2026-10-13-patch-memory/code/run.py --model-dir .local-state/pythia-14m --lock-commit 512fdba309df32aa3a98db59c65ec5d5d383635f --output .local-state/patch-memory-reproduction
```

The runner checks the checkpoint hashes and the exact locked plan and code. The output folder must not exist. It uses two CPU threads and has a fixed 20-minute run limit. No training, sampling, or paid inference API is used.

To audit the reproduced tables with the separate full-prefix implementation, use a new receipt path:

```bash
.venv/bin/python studies/2026-10-13-patch-memory/code/independent_audit.py --model-dir .local-state/pythia-14m --results .local-state/patch-memory-reproduction --output .local-state/patch-memory-audit.json
```

The [final AI review](reviews/result-review.md) records the review scope and its limits.

The arithmetic uses float64 operations with shared stock float32 rotary tables. This is not a fully float64 model reference. The full output vocabulary enters every total variation score. Large vocabulary tensors are not stored; the fixed model and runner support replay.

The fixed sample has 64 directed cases. Four word pairs set the two layer choices. Four other pairs test the choices in the same four frames. All six layers and all nine horizons are measured. Block 5 is excluded from layer selection because its post-block edit cannot change any cache entry. Its later effect is a known zero control.

## Files and validation

- `results/inputs.json`: every prompt, split, and suffix token ID.
- `results/scores.csv`: all 3,456 full-vocabulary TV scores and target-logit measurements.
- `results/group-summary.csv`: all split, layer, horizon, frame, and word-pair summaries.
- `results/baseline.csv`: all 576 baseline measurements, with no task screening.
- `results/controls.csv` and `results/cache-controls.csv`: all 11,072 logit checks and 9,216 exact cache checks.
- `results/hook-calls.csv`: all 4,224 calls that replaced the original prompt position.
- `results/post-block-vectors.npz`: all 64 × 6 × 128 baseline vectors used as donors and receivers.
- `results/model-receipt.json`, `results/run-metadata.json`, and `results/manifest.json`: pinned model hashes, environment, cost, and result hashes.
- `code/independent_audit.py` and `reviews/independent-audit.json`: separate full-prefix replay and its receipt.

The main run made 14,592 model calls in 90.32 seconds. The separate replay made 4,096 calls in 31.82 seconds. These times include different checks and are not inference-speed claims. The main run's maximum cache/full logit difference was `3.7596e-11`, below the fixed `1e-8` bound. The audit's maximum TV difference was `5.6122e-13`.
