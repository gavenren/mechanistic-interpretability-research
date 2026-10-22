# Task-ability continuation

Read the [fixed plan](analysis-plan.md) and [source record](sources.md). The methods come from the unchanged [19 October code](../2026-10-19-capability-gates/code/). This directory contains the continuation runner and its records.

Use the repository environment: Python 3.12.14, NumPy 2.3.5, and PyTorch 2.8.0. Run from the repository root. All scientific computation uses one CPU process with two compute threads. Results cannot be overwritten. For reproduction, use a separate checkout and preserve the published result folders outside it before running.

After a successful live account check, a supervisor can issue a private permit:

```sh
python studies/2026-10-22-capability-completion/code/permit.py /absolute/private/permit.json --allowed
```

Set `MI_USAGE_PERMIT` to that path. Refresh at least once per minute, only after another live check. Omitting `--allowed` denies the permit. The runner never refreshes it. The file must not be committed. At or below the 20% reserve, stop research.

```sh
python studies/2026-10-22-capability-completion/code/run.py prepare
python studies/2026-10-22-capability-completion/code/run.py experiment --lock PUBLIC_PLAN_COMMIT
```

Use the verified public commit with these exact plan and code files. Preparation must pass first. The runner also verifies the unchanged original source and completed result. It reuses the completed graph seed 8001, restarts only the explicitly approved interrupted seed 8002, and retains the earlier failure. Do not represent the restart as an exact optimizer-state continuation.

The dummy permit test is in [permit-check.json](permit-check.json). It tested 1,000 atomic replacements and at least 10,000 concurrent reads. No model was trained in that test. The old preparation time and scientific time count against the unchanged stage caps.

## Fixed timing revision

The first fresh preparation failed the compute projection gate. Keep its files. Read [TIMING_REVISION.md](TIMING_REVISION.md) before the one allowed revised dummy benchmark. After that method is publicly locked, run:

```sh
python studies/2026-10-22-capability-completion/code/measure_hall.py --lock TIMING_METHOD_COMMIT
```

The scientific runner now requires a passed `revised-preparation/status.json`. Its public lock must include both failed and revised preparation records. The fourfold allowance and four-hour cap are unchanged. No scientific work has started at this plan stage.
