# Patch calibration transfer

[Read the paper](paper.md). This study tests whether a correction measured on calibration prompts can improve predictions of joint activation patches on new prompts. It uses one fixed Pythia-14M checkpoint.

The primary comparison changed mean absolute error from 0.030472 to 0.031135 logit units. This small increase did not give clear support for the correction. The fixed-case bootstrap interval included zero. The full results include two numerical conditions and all 256 cases.

**Review status:** Complete with a disclosed audit deviation. The planned float32 whole-block delta check failed in 1,385 of 4,608 patch cells. Its maximum target-logit difference was 0.001754, above the fixed 0.0001 tolerance. A later complete replay used the model's original addition order. It matched every saved target logit and baseline activation exactly in both numerical conditions. The original failure remains recorded. The locked plan, runner, results, and tolerances were not changed. See [reviews](reviews/) for the receipts and method note. This is AI-assisted work without recorded human peer review.

## Reproduce without changing the published results

Use Python 3.12 and a CPU. Run these commands from the current repository root. The detached worktree is made at the plan-and-code lock commit. This keeps the published results in the current checkout intact. The lock checkout does not contain later helper files, so install dependencies and fetch the model from the current checkout first.

```sh
python3.12 -m venv .local-state/patch-environment
.local-state/patch-environment/bin/python -m pip install -r studies/2026-10-13-patch-calibration-transfer/requirements.txt
.local-state/patch-environment/bin/python studies/2026-10-13-patch-calibration-transfer/code/fetch_model.py --output .local-state/models/pythia-14m
git worktree add --detach ../mi-patch-reproduction 4dc5db20c261f9057d08a4ad2c7cc9548dcba1ba
.local-state/patch-environment/bin/python ../mi-patch-reproduction/studies/2026-10-13-patch-calibration-transfer/code/run.py --model-dir .local-state/models/pythia-14m --preflight
.local-state/patch-environment/bin/python ../mi-patch-reproduction/studies/2026-10-13-patch-calibration-transfer/code/run.py --model-dir .local-state/models/pythia-14m --run --lock-commit 4dc5db20c261f9057d08a4ad2c7cc9548dcba1ba
```

The new results will be under `../mi-patch-reproduction/studies/2026-10-13-patch-calibration-transfer/results/`. Keep the working directory at the current repository root for these commands. If the reproduction path already exists, choose another unused path and use it consistently. The runner verifies the lock commit, checks model-file hashes, and refuses to overwrite an existing results directory.

To audit the published result files in the current checkout:

```sh
.local-state/patch-environment/bin/python studies/2026-10-13-patch-calibration-transfer/code/independent_audit.py --model-dir .local-state/models/pythia-14m
```

The audit is read-only with respect to the scientific results. It reports the failed planned float32 check. Run the added diagnostic separately:

```sh
.local-state/patch-environment/bin/python studies/2026-10-13-patch-calibration-transfer/reviews/ordered_reconstruction_audit.py --model-dir .local-state/models/pythia-14m
```

This added diagnostic preserves the original addition order. It does not replace the failed planned check. Final instructions and receipts are in [reviews](reviews/).

## Model and numerical conditions

The external model is not copied into this repository. Fetch it from [EleutherAI/pythia-14m at cf967c0a9a04383db6f7b1108d86b2962634b4ac](https://huggingface.co/EleutherAI/pythia-14m/tree/cf967c0a9a04383db6f7b1108d86b2962634b4ac). The model card states an Apache 2.0 licence. The [source ledger](sources.md) records its version history, including the February 2026 model change at this URL.

The primary run uses float32. The sensitivity run uses double arithmetic, with the same float32 rotary-position tables. The sensitivity run is not ground truth for the exact float32 program. No training, paid API, or GPU was used.

The tested direct dependencies are in `requirements.txt`. Complete installed package versions used by the runner are in `results/provenance.json`.

## Data guide

| File | Contents |
| --- | --- |
| `analysis-plan.md` | Fixed question, prompt set, splits, metrics, controls, and reporting rules |
| `sources.md` | Checked primary sources and limits on novelty |
| `results/prompts.json` | All 256 prompt texts, token IDs, answer tokens, and split labels |
| `results/directions.json` | Receiver and donor IDs for all directed cases |
| `results/provenance.json` | Lock commit, file hashes, model revision, and package versions |
| `results/float32/` | Primary results |
| `results/double_shared_rotary/` | Double-arithmetic sensitivity results with shared rotary tables |
| Each condition's `four-cell-results.csv` | Every raw target logit, margin, patch effect, interaction, correction, and prediction error |
| Each condition's `calibration-corrections.json` | Six calibration medians, frozen before held-out inference |
| Each condition's `intact-behavior.csv` | All intact pairwise preferences and full-vocabulary target decisions |
| Each condition's `summary.csv` | Whole-set, layer, frame, and word-pair summaries |
| Each condition's `activations.npz` | Saved branch outputs, block inputs and outputs, target logits, final residual vectors, LayerNorm values, and target readout rows |
| Each condition's `controls.json` | Experiment-runner numerical checks |
| Each condition's `bootstrap.json` | Fixed-case crossed-bootstrap interval |
| `results/precision-comparison.json` | Differences between the two numerical conditions |
| `results/manifest.json` | Hashes and sizes of all scientific result files |
| `code/plot.py` and `figures/` | Plot code and the saved figure |
| `code/independent_audit.py` and `reviews/` | Separate audit code, findings, and review records |

Layers and prompt IDs are zero-based. In the saved arrays, attention is branch 0 and MLP is branch 1. The four cell indices are no patch, attention only, MLP only, and both. The two target columns are receiver word and donor word. The primary result uses layers 1–5; layer 0 is a known-zero control.
