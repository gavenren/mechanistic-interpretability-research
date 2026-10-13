# Rare-feature SAE study

Read the [paper](paper.md), [fixed analysis plan](analysis-plan.md), and [source ledger](sources.md).

The main result is mixed. Giving rare cases extra output-loss weight was worse for the selected single feature in five of six paired runs. One large improvement made the average difference slightly positive. The method's full code carried more useful output change than its selected single feature. The study uses a specified synthetic representation and output rule; it does not test a language model.

## Reproduce the experiment

The original run used Python 3.12.14 and NumPy 2.3.5, with one BLAS thread. It trained and evaluated all 72 models in about 60 seconds on the recorded Apple Silicon computer. Runtime varies by computer. Matplotlib 3.10.8 produces the figure but is not needed for training.

The runner and analysis plan were fixed in commit `75ab86efd6159a4965b8461a534a0a29592d98ac`, before training. The runner checks their bytes against the commit at the start and end. You need a Git checkout containing that commit. A source ZIP without Git history is not sufficient.

From the repository root, create a separate checkout to preserve the published outputs:

```sh
git worktree add --detach ../mi-sae-reproduction 75ab86efd6159a4965b8461a534a0a29592d98ac
cd ../mi-sae-reproduction
python3 -m venv .venv
.venv/bin/python -m pip install numpy==2.3.5
.venv/bin/python studies/2026-10-13-rare-feature-sae/code/run.py --check
.venv/bin/python studies/2026-10-13-rare-feature-sae/code/run.py --run --lock-commit 75ab86efd6159a4965b8461a534a0a29592d98ac
```

Use another new path if that checkout already exists. `--check` does not train a model. It checks derivatives, the generator, and intervention execution. `--run` runs all fixed cells and seeds. The runner refuses to start if its results directory has files. It does not silently replace published evidence. Preserve incomplete runs before using a fresh checkout.

To redraw the figure from the published results, use the current published checkout, not the earlier lock checkout:

```sh
python3 -m pip install -r studies/2026-10-13-rare-feature-sae/requirements.txt
python3 studies/2026-10-13-rare-feature-sae/code/plot_results.py
```

Prefer a virtual environment for package installation. The plot command writes a PNG and a PDF under `figures/`.

## What is saved

| Path | Content |
| --- | --- |
| `analysis-plan.md` | Fixed question, methods, comparisons, and limits |
| `code/run.py` | Locked generator, training, and evaluation code |
| `code/plot_results.py` | Figure code, added after the fixed experiment |
| `data/generator-records.json` | Exact RNG seeds, array shapes, array hashes, sample ID ranges, rare counts, zero counts, and cross-split value overlaps |
| `data/dictionary-seed-N.npz` | Ground-truth event direction and background dictionary |
| `models/*.npz` | Every model's encoder, decoder, biases, and centering mean |
| `results/all-latent-scores.csv` | All 24 latents in all 72 models, on validation and test data |
| `results/selected-latents.csv` | The fixed validation-selected latent and its test results |
| `results/selected-test-outputs/*.npz` | Per-background selected intervention outputs and full/empty controls |
| `results/population-metrics.csv` | Reconstruction and task measurements, separate for event-absent and event-present cases |
| `results/same-label-controls.csv` | Nuisance replacement tests, separate for the two labels |
| `results/training-trace.csv` | Recorded minibatch loss terms for every model |
| `results/execution-controls.csv` | Execution checks for every trained model |
| `results/preflight-controls.json` | Checks before training |
| `results/paired-contrasts.csv` | Seed-level ordinary-minus-balanced comparisons |
| `results/factorial-summary.csv` | Means for every fixed cell and objective |
| `results/summary.json` | Main estimate, interval, environment, runtime, and lock hashes |
| `results/manifest.json` | SHA-256 hashes of the scientific run outputs |

The label-specific `results/same-label-controls.csv` table was created by the original locked runner. It is covered by the original scientific manifest. The later independent review checked all 144 rows against the saved models. It was not a later data export.

The runner's deterministic `generate(seed)` function recreates the input arrays. Generator records give hashes of their contiguous raw bytes. Duplicate copies of the same training backgrounds are not saved for each objective. Independent draw IDs are separate across splits; the all-zero value is allowed to recur and is counted.

A model archive has `Wenc` with shape (12,24), `benc` (24,), `Wdec` (24,12), `bdec` (12,), and `mean` (12,). Decode a code z as `z @ Wdec + bdec + mean`. Subtract `mean` before encoding. The decoder rows are the learned feature directions.

## How to read the scores

The primary `native_error` applies a learned code difference to the **original** activation. It preserves the receiver's reconstruction residual. Zero means the selected feature produced the specified probability change. A value near 0.964 is the no-intervention baseline in this task. Errors are probability distances, not error rates or percentage accuracies.

The secondary `reconstruction_error` applies the same code change inside the reconstructed vector. It includes reconstruction error. These two scores answer different questions. `full_native_error` changes the whole code but still keeps the receiver residual. `full_reconstruction_error` decodes the donor's whole code.

`same_label_native_change` measures an unwanted output change when the event label stays fixed but the donor background differs. Its aggregate gives the two label groups equal weight. Natural-distribution reconstruction and task metrics instead weight labels by the declared event frequency p. The code keeps these conventions separate.

Unmodified encodings have at most four active latents. A one-latent replacement can have five. Activity after replacement is recorded; the code does not apply TopK again after an intervention.

The primary contrast is ordinary downstream error minus balanced error at p=0.01 and amplitude=0.25. A positive number favours balancing. The bootstrap resamples six paired replication seeds. Each seed changes the dictionary, data, and initialization together. It does not estimate uncertainty over language models or tasks.

Small numerical differences can occur across platforms. Runtime fields, archive timestamps, and hashes can also differ across reruns. Compare the fixed settings and numerical scientific outputs; do not assume whole-directory byte identity.
