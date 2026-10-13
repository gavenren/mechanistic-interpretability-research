# Register edits under copy and write operations

Status: the fixed experiment, independent saved-weight audit, and final AI review are complete. No human review is claimed.

The study compares two ways to train a small hidden-state edit. One loss asks for the correct immediate output. The other also asks for the correct output after one operation. The test asks whether the edit both moves into register B when copied and ends when a write replaces it. The extra loss gives a mean gain of only **0.21 percentage points**, with a descriptive interval from **−1.02 to +1.13**. Three of four models fail the fixed task-ability rule. The result does not establish a reliable gain.

- [Fixed plan](analysis-plan.md)
- [Plain-English paper](paper.md)
- [Primary sources and close overlap](sources.md)
- [Exact histories and training schedules](inputs.json)
- [Runner](code/run.py)
- [Latest dummy check](reviews/dummy-preflight-v3.json)
- [Complete result summary](results/summary.json)
- [Complete result manifest](results/manifest.json)
- [Independent audit receipt](reviews/independent-audit.json)
- [Final independent review](reviews/independent-review.md)
- [Independent audit code](code/independent_audit.py)
- [Tables](tables/stage-scores.csv) and [figure](figures/write-order-results.png)

Use Python 3.12 and the exact packages in `requirements.txt`. Run commands from this folder. The scientific run must use the full public commit that fixes the plan, code, sources, packages, and inputs.

```sh
python code/run.py preflight --out /absolute/path/to/new-dummy-check.json
python code/run.py run --lock 1ff6f252c4d11ac9e0cc2e778e0a6157fabd81e4 --out /absolute/path/to/new-results
```

The program refuses nonempty result folders. It retains a failure record if a scientific run stops. The run limit is 20 minutes with two CPU threads. The study is AI-authored. No human review is claimed.

The published run took 32.87 seconds. It saved 86,016 test rows and 430,080 full 16-class readouts. All 528 numerical controls passed. The independent audit replayed all saved logits with separate GRU equations. It did not repeat training.

```sh
python code/independent_audit.py --results results --output /absolute/path/to/new-audit.json
```

To rebuild the figure and report tables from the saved results, also install `requirements-report.txt`, then run `python code/report.py`. This report script does not call the model. The scientific plan and runner remain fixed to the pre-run commit.
