# Same string, different token splits

This study tests whether Pythia-14M can use a supplied binding when the source key and query use different token splits. It also tests one fixed copying head. Both splits preserve the text and token count. The plan, code, and source record are ready. Scientific results are not yet available.

- [Fixed plan](analysis-plan.md)
- [Source record](sources.md)
- [Pre-run review](reviews/pre-run-review.md)
- [Dummy check](reviews/dummy-preflight.json)

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
python code/run.py --model-dir /absolute/path/to/pythia-14m --lock-commit FULL_LOCK_COMMIT --output /absolute/path/to/new-results
```

Every output path must be empty. The runner uses two CPU threads and a 20-minute limit. It keeps failures. No paid API is used. This is an AI-authored research project with separate AI review. It has not received human peer review.
