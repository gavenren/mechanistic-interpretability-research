# Reproduce this study

This is a constructed formal study. It does not train a neural network.
Read the [fixed plan](analysis-plan.md) and [source record](sources.json).

Use Python 3.12.14. Only the standard library is required. The shared resource guard comes from the prior capability study.
Run from the repository root after a public plan lock and a live usage check:

```sh
python studies/2026-10-22-donor-support/code/run.py --lock PUBLIC_LOCK_COMMIT
```

The supervisor sets `MI_USAGE_PERMIT` to a private JSON permit file with an allowed flag and an expiry timestamp. It must refresh that file with a temporary write and atomic replacement, only while the account reserve rule permits work. The shared guard also accepts a stricter `MI_STAGE_DEADLINE`. Private permit files must stay out of GitHub.

The run cap is 300 seconds. Run the studies in sequence. The three intervention studies have a combined 900-second cap. The script verifies locked inputs and refuses an existing results directory. For an independent repeat, use a separate checkout and retain the original published outputs outside the repeat output path. Never replace a failed record.

The runner saves exact raw data, summary scores, software, elapsed time, plan hash, and file hashes. It uses exact arithmetic and includes separate calculation checks. Those checks are not human peer review. The initial plan had no scientific output. The fixed run is now complete. Read the [paper](paper.md), [metrics](results/metrics.json), and [file hashes](results/manifest.json).
