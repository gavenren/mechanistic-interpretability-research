# Basis transport audit

Changing a network’s internal coordinates can change a coordinate patch while leaving normal answers unchanged. This study also shows why an orthogonal projection onto the same subspace can give a different intervention after a non-orthogonal coordinate change.

Read the [paper](paper.md), [fixed plan](analysis-plan.md), [source record](sources.md), and [independent review](reviews/independent-review.md).

Eight small networks were trained on all 25 states of a modular-arithmetic task. The architecture supplies the causal separation between the two input variables. All eight models reached 100% normal task accuracy. After rotations, selected four-coordinate masks had mean test error 0.5759 in total variation and 43.73% counterfactual class accuracy. The correctly transported operator preserved the counterfactual in all 26 coordinate systems. These are controlled results, not a new claim that coordinate choice matters.

Use the current published checkout to check the saved results and draw the figures. The independent audit and final plot scripts are later additions to the experiment lock. With Python 3.12 and the versions in requirements.txt installed, run from this study directory:

```sh
python code/run.py --check
python code/independent_audit.py
python code/plot.py
```

The first command checks gradients, algebra, pair splits, and all mask mappings with an untrained model. The second checks saved experimental outputs using a separate implementation. The third makes the figures. None repeats training.

For a fresh training run, use a separate checkout at the published analysis lock. Run these commands from the repository root. Choose a new destination if `../basis-transport-reproduction` already exists:

```sh
git worktree add --detach ../basis-transport-reproduction d4ca99ae08121070facd5ff755b6ca4b164dc304
cd ../basis-transport-reproduction
python3 -m venv .venv
.venv/bin/python -m pip install -r studies/2026-10-13-basis-transport/requirements.txt
cd studies/2026-10-13-basis-transport
../../.venv/bin/python code/run.py --check
../../.venv/bin/python code/run.py --run --lock-commit d4ca99ae08121070facd5ff755b6ca4b164dc304
```

This leaves the current checkout and all published outputs in place. The lock checkout contains the fixed plan and runner before the experiment outputs were added. The runner refuses to overwrite scientific outputs. Do not delete published results to make it run. Do not change the locked plan or runner to improve a result. This repository analysis lock is not a third-party registration.

Files:

- `data/`: 25 input states, 500 ordered pairs, 26 coordinate systems, and exact transformation matrices. The 250 discovery pairs include 55 b-matched pairs; the 250 test pairs include 45.
- `models/`: all eight weight files.
- `results/`: all 29,120 candidate-score rows, the 208 chosen masks, all operator comparisons, numerical controls, training traces, per-seed outcomes, a summary, and a SHA-256 manifest.
- `figures/`: the two planned figures in PNG and SVG.
- `reviews/`: the independent AI audit and review record. No human peer review has been recorded.

All network and numerical audit code uses NumPy. Only plotting requires Matplotlib. The original run used NumPy 2.3.5, Python 3.12.14, and one BLAS thread. The eight-model experiment took 18.9 seconds, excluding later review and plotting.
