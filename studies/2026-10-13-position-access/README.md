# Position spacing and access to extra tokens

In a fixed Pythia-14M test, blocking filler access and compressing positions each sharply lower the answer score. Their combination matches the short-input control. The declared contrast ranks two harmful changes; it does not find a successful repair. The normal long input also has a lower mean score but higher full-vocabulary top-1 accuracy than the short input.

The fixed run and a separate AI result audit are complete. All 6,144 replayed full-vocabulary output hashes match exactly. This study is AI-authored and has no human peer review.

- [Plain-English paper](paper.md)
- [Figure](figures/position-access.png)
- [Saved summary](results/summary.json)
- [All 6,144 scored rows](results/scores.csv)
- [Controls](results/control-summary.json)
- [Fixed plan](analysis-plan.md)
- [Sources and overlap](sources.md)
- [Pre-run review](reviews/plan-review.md)
- [Final dummy receipt](reviews/dummy-preflight-v3/manifest.json)
- [Independent audit receipt](reviews/independent-audit.json)
- [Independent result review](reviews/independent-review.md)

## Reproduce without changing published results

From the repository root, install the pinned dependencies and fetch the public checkpoint into local state:

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r studies/2026-10-13-position-access/requirements.txt
.venv/bin/python studies/2026-10-13-position-access/code/fetch_model.py --output .local-state/pythia-14m
git worktree add --detach ../position-access-reproduction c6ff9cb1405e6a65fe161c2f4cb471d51016bf4c
```

Keep the current working directory. Run the locked code in the new worktree with the current environment and cache:

```sh
.venv/bin/python ../position-access-reproduction/studies/2026-10-13-position-access/code/run.py --model-dir .local-state/pythia-14m --lock-commit c6ff9cb1405e6a65fe161c2f4cb471d51016bf4c
```

The result directory in that lock checkout does not exist before the run. The runner refuses an existing output directory and checks plan/code bytes at the start and end. Use a different worktree name if the example name already exists. The published result directory in the current checkout remains intact.

For a fresh dummy check only:

```sh
.venv/bin/python studies/2026-10-13-position-access/code/run.py --model-dir .local-state/pythia-14m --preflight --output .local-state/position-access-dummy-repeat
```

To repeat the independent audit of the published results, use a fresh receipt path:

```sh
.venv/bin/python studies/2026-10-13-position-access/code/independent_audit.py --model-dir .local-state/pythia-14m --results studies/2026-10-13-position-access/results --output .local-state/position-access-audit-repeat.json
```

To rebuild the figure from the published results:

```sh
.venv/bin/python studies/2026-10-13-position-access/code/plot_results.py
```

The run uses two CPU threads, batch size four, and a 20-minute cap. It uses double arithmetic with stock float32 rotary tables. The observed run took 181.2 seconds; another machine can take longer. Weights are fetched from a pinned public revision and checked by SHA-256. They are not copied into this repository.
