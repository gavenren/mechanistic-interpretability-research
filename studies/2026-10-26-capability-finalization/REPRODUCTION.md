# Read and check the complete results

The [paper](paper.md) reports one complete study. Start with the [model table](analysis/result-audit/models.csv), [group scores](results/metrics.csv), and [status](results/status.json).

The original task code remains at `studies/2026-10-19-capability-gates/code/`. The new runner remains in [code/run.py](code/run.py). The public plan lock is `95b53af3f603276a082a7bad630f72ffb6508c9a`. The public execution lock is `20caef7307b5e591d4406112555dd33bc2d03cd1`. Neither the locked code nor the locked plan was changed after training.

## Restore the raw task files

The four ZIP files in [artifacts/](artifacts/) contain every new raw task file. The [archive manifest](artifacts/manifest.json) lists each member and its exact hash. Status, group scores, and the original result manifest remain directly in `results/`.

Run from the repository root with Python 3.12:

```sh
python3 studies/2026-10-26-capability-finalization/analysis/result_archive.py verify
python3 studies/2026-10-26-capability-finalization/analysis/result_archive.py unpack
```

This restores 218 files under the four task folders in `results/`. The tool checks every bundle before extraction. It permits an existing identical file and rejects a conflicting file. It rejects unsafe paths and symbolic links. Extracted task folders are ignored by Git because the ZIP files already contain the complete bytes. File times and permissions are not preserved.

The six reused graph models remain in the October 19 and October 22 study folders. Their exact paths appear in the final status record. The two earlier partial attempts remain there as well. Do not delete or replace them.

## Check the saved result calculations

Use the repository environment: Python 3.12.14, NumPy 2.3.5, and PyTorch 2.8.0. Read [analysis/check_results.py](analysis/check_results.py). This separate check uses saved arrays, not model training or forward passes. It checks scores, gates, traces, recipes, counts, and result manifests.

A fresh checkout does not contain the local Python environment or private state directory. Use an existing environment with the versions above, or create one with an installed Python 3.12.14 interpreter:

```sh
python3.12 -m venv .venv
.venv/bin/python -m pip install numpy==2.3.5 torch==2.8.0
mkdir -p .local-state
```

Check the interpreter version first. Package installation and a repeated audit are new local work; follow the resource rules. This guide does not authorize a new training run.

Its published output is in [analysis/result-audit/](analysis/result-audit/). It refuses to replace that directory. To run it again, use a separate temporary checkout and move that checkout's saved audit output to a backup path first. Keep the reference repository unchanged. Obtain a fresh private usage permit and stay within the compute rules, then run:

```sh
.venv/bin/python studies/2026-10-26-capability-finalization/analysis/check_results.py --max-seconds 60
```

`MI_USAGE_PERMIT` must point to a private permit issued by the supervisor after a live usage check. The existing atomic writer is documented in [README.md](README.md). The checker requires the exclusive compute lock and a completed scientific run. A repeated check is additional compute; add it to the cumulative ledger.

## Repeat training only under a new recorded plan

Saved-weight replay is not independent training. The continuation runner refuses to replace preparation or result folders. Its fixed locks refer to prior published bytes. The existing run is complete and its graph and digit failures stay failed. A new training repetition requires separate output paths, a public repetition plan, all original inputs and schedules, a fresh usage check, and cumulative compute accounting. Do not run the old command over these results or replace a failed seed.
