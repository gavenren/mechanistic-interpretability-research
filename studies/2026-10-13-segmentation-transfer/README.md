# Same string, different token splits

This study tests whether Pythia-14M can use a supplied binding when the source key and query use different token splits. It also tests one fixed copying head. Both splits preserve the text and token count. The locked run is complete. Matching splits gave 99.12% two-answer accuracy and 44.43% full-vocabulary accuracy. Different splits gave 56.93% and 0.10%. The fixed head still changed the score, but the declared capability gate failed.

- [Plain-English paper](paper.md)
- [Figure](figures/segmentation-transfer.png)
- [Saved results](results/summary.json)
- [All prompt scores](results/scores.csv)
- [Fixed plan](analysis-plan.md)
- [Source record](sources.md)
- [Pre-run review](reviews/pre-run-review.md)
- [Dummy check](reviews/dummy-preflight.json)
- [Independent result review](reviews/independent-review.md)
- [Complete replay receipt](reviews/independent-audit-final.json)
- [Preserved failed probability check](reviews/independent-audit.json)

## Reproduction

Use Python 3.12.14 and the versions in `requirements.txt`. From this directory:

```sh
python -m pip install -r requirements.txt
python code/fetch_model.py --output /absolute/path/to/pythia-14m
python code/run.py --model-dir /absolute/path/to/pythia-14m --dummy-preflight --output /absolute/path/to/new-dummy-results
python code/run.py --model-dir /absolute/path/to/pythia-14m --token-only --output /absolute/path/to/new-token-results
```

A scientific run requires the full public commit that locks the plan and runner:

```sh
python code/run.py --model-dir /absolute/path/to/pythia-14m --lock-commit c6ff9cb1405e6a65fe161c2f4cb471d51016bf4c --output /absolute/path/to/new-results
```

To repeat the independent audit, choose a fresh receipt path:

```sh
python code/independent_audit.py --model-dir /absolute/path/to/pythia-14m --results results --output /absolute/path/to/new-audit.json
```

To rebuild the figure from the saved results:

```sh
python code/plot_results.py
```

Every run output path must be empty. The runner uses two CPU threads and a 20-minute limit. It keeps failures. No paid API is used. This is an AI-authored research project with separate AI review. It has not received human peer review.

The independent replay matched all recorded float32 results. A separate float64 probability calculation differed by up to 0.0000772. Its original failed check is preserved. The main logit and ranking findings are fully verified. See the result review for the numerical limit.
