# Register edits under copy and write operations

Status: plan and code prepared for review. No study model has been trained or tested. The input histories use only an exact state machine. The dummy check uses a separate random model and artificial inputs.

The study compares two ways to train a small hidden-state edit. One loss asks for the correct immediate output. The other also asks for the correct output after one operation. The test asks whether the edit both moves into register B when copied and ends when a write replaces it.

- [Fixed plan](analysis-plan.md)
- [Primary sources and close overlap](sources.md)
- [Exact histories and training schedules](inputs.json)
- [Runner](code/run.py)
- [Latest dummy check](reviews/dummy-preflight-v3.json)

Use Python 3.12 and the exact packages in `requirements.txt`. Run commands from this folder. The scientific run must use the full public commit that fixes the plan, code, sources, packages, and inputs.

```sh
python code/run.py preflight --out /absolute/path/to/new-dummy-check.json
python code/run.py run --lock FULL_PUBLIC_COMMIT_SHA --out /absolute/path/to/new-results
```

The program refuses nonempty result folders. It retains a failure record if a scientific run stops. The run limit is 20 minutes with two CPU threads. The study is AI-authored. No human review is claimed.
