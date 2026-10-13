# Inactive-variable audit

This study tests whether two rules select the same small representation of a variable. One rule scores changes when the variable is active. The other also scores the case where the variable is inactive.

All eight trained networks learned the ordinary task perfectly. At the primary budget of three neurons, both rules selected the same mask in every network. Their held-out difference was zero. The selected masks still had substantial counterfactual error. This is a bounded null result, not proof that either rule identifies a complete mechanism.

Read the [paper](paper.md), [fixed analysis plan](analysis-plan.md), and [primary results](results/summary.json).

## Reproduce the run

The original run used Python 3.12.14 and NumPy 2.3.5 on an Apple Silicon computer. It used one process, one BLAS thread, and no external model or data downloads. The full run took about 39 seconds on that computer. Runtime can differ elsewhere. The code uses Python's standard library, NumPy, and the Git command-line program.

The code and plan were fixed before training in commit `ff9d78c15aa6e97b4819e18f4253b60fd165b50c`. The runner checks their exact bytes against this commit both before and after the run. A Git checkout containing that commit is required; a source ZIP without Git history is not sufficient.

To preserve the published results, create a separate checkout at the lock commit. From the repository root:

```sh
git worktree add --detach ../mi-audit-reproduction ff9d78c15aa6e97b4819e18f4253b60fd165b50c
cd ../mi-audit-reproduction
python3 -m venv .venv
.venv/bin/python -m pip install numpy==2.3.5
.venv/bin/python studies/2026-10-13-inactive-variable-audit/code/run.py --check
.venv/bin/python studies/2026-10-13-inactive-variable-audit/code/run.py --run --lock-commit ff9d78c15aa6e97b4819e18f4253b60fd165b50c
```

Use a new checkout path if `../mi-audit-reproduction` already exists. The separate checkout starts before outputs were added. The runner refuses to run when its `results/` folder contains files. It does not silently replace published evidence. If a run fails, preserve its partial results before making a fresh checkout.

The preflight `--check` command does not train models. It checks gradients, pair construction, patch execution, and the exact factorized reference. The `--run` command repeats these checks and runs all eight fixed seeds. It does not replace low-accuracy seeds or extend training.

To check the published weights and results in the current checkout, run:

```sh
python3 studies/2026-10-13-inactive-variable-audit/code/independent_audit.py
```

This separate implementation does not import the runner or retrain the models. It checks the recorded split, mask choices, and summary tables. It recalculates all selected non-random masks from the saved weights. See the [review](reviews/independent-review.md) and [audit receipt](reviews/audit-receipt.json) for the scope and limits.

## Files and units

| File | Content |
| --- | --- |
| `analysis-plan.md` | Locked scientific choices and limits |
| `code/run.py` | Locked training and evaluation implementation |
| `code/independent_audit.py` | Separate calculation and consistency checks |
| `requirements.txt` | NumPy version used in the published run |
| `data/inputs.csv` | All 192 states, their IDs, and task labels |
| `data/pairs.csv` | All 27,648 different-a donor/receiver pairs and fixed split assignments |
| `models/seed-N.npz` | All weights and biases for model seed N |
| `results/candidate-scores-seed-N.csv` | Every mask, donor regime, and split score for seed N |
| `results/selected-masks.csv` | Selected masks and held-out scores, including fixed random controls |
| `results/same-a-controls.csv` | Exact receiver-target tests for masks selected by the two main rules |
| `results/empty-baselines.csv` | No-patch scores on the same pair banks |
| `results/seed-outcomes.csv` | Primary result and training outcome for every seed |
| `results/intact-predictions.csv` | Ordinary-task probabilities for all model seeds and states |
| `results/training-trace.csv` | Training loss and accuracy at recorded steps |
| `results/preflight-controls.json` | Checks before training |
| `results/controls-seed-N.json` | Patch-execution checks for trained seed N |
| `results/summary.json` | Primary summary, environment, runtime, and lock hashes |
| `results/manifest.json` | SHA-256 hashes of data, weights, and run outputs |

`tv` is total variation distance, from 0 to 1. A larger value means more probability-distribution error. `active_tv` and `inactive_tv` compare patched output with the intact model at the specified counterfactual input. `inactive_output_change_tv` instead compares patched output with the original receiver. `intact_inactive_change_tv` shows how much the intact counterfactual itself changes the inactive receiver distribution. A fixed task label does not require an exactly fixed probability vector.

`semantic_accuracy` compares the selected output class with the task's counterfactual label. `donor_copy_rate` is the fraction of outputs that match the intact donor's task label. It does not measure causal fidelity by itself.

Masks contain zero-based neuron indices separated by spaces. `mask_id` follows enumeration by size, then lexicographic neuron order. Stratum ID is `4*receiver_selector + 2*selector_match + b_match`. Missing stratum cells in the matched-donor regime are empty CSV fields, not zero errors. Means use natural pair frequencies.

The matched-donor analysis has 290 discovery pairs and 286 test pairs. The test includes only the matched subset of held-out pair IDs. It never reuses a discovery pair as a test pair. It also has its own discovery mask selection. Thus a comparison between matched and unrestricted results changes both the donor regime and the selected mask.

The bootstrap interval is over the eight model seeds. It is not an interval over architectures or language models. The zero interval follows from identical selected masks and zero differences in this experiment.

Floating-point values may vary slightly across NumPy builds and hardware. Runtime fields and file hashes that include runtimes will also differ. Compare the scientific scores and fixed protocol, not only an exact whole-directory hash.
