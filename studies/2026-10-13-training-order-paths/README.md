# Training order and internal shortcut paths

Status: plan and runner prepared before scientific outcomes. No scientific training run has started. A verified public lock is required before the run.

Read [the fixed plan](analysis-plan.md) and [the source record](sources.md). The current dummy-only receipt is [preflight-v3.json](preflight-v3.json). The two earlier receipts are kept as history. They do not contain scientific outcomes.

From the repository root, install the pinned packages in an isolated Python environment:

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r studies/2026-10-13-training-order-paths/requirements.txt
```

To repeat only the dummy checks, use a fresh output path:

```sh
.venv/bin/python studies/2026-10-13-training-order-paths/code/run.py --preflight --output .local-state/training-order-dummy.json
```

The scientific command will be added with the verified public lock. The runner refuses to overwrite its output directory. It uses CPU only, two threads, fixed seeds, float64, and a 20-minute cap. It preserves incomplete runs.
