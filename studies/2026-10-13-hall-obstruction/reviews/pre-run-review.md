# Separate AI review before the run

Status: ready for a public lock. Reviewed 13 October 2026. No study model was trained or evaluated for this review. This is an AI review, not human review.

Reviewed final bytes:

- Plan: `4b59042f2f97da9920fc0fc2232a1fe61e5490de9e36541df5a6d86c1bb0b253`.
- Runner: `4e1ba4831002f32ab4ac86c35cf69d9de59b37e36888a1e061668715c3e504f3`.
- Input ledger: `ef0cf2aef1e0589feab959547228ea074fddbc19a54de93a47bd543d9842d9b7`.

These hashes match `dummy-preflight-v3.json`. The final dummy check used nonbinary inputs that cannot be natural task graphs. It performed no optimizer update. Its reported cached-forward, batch-split, and bias-only errors are zero.

## Data and task

I independently enumerated all 65,536 binary 4×4 graphs. Complete-matching enumeration and all nonempty Hall subset checks agree. There are 37,823 positive graphs and 3,680 negative graphs with no isolated vertex. I rebuilt the retained degree classes and compared every graph ID for all 24 named obstructions with the input ledger. The category unions also agree exactly. Six-edge graphs enter selection; seven-edge graphs enter intervention testing. Their graph sets are disjoint. Both remain part of the complete natural training set, as the plan discloses.

I also independently enumerated the 317 row/column relabelling orbits. For the first named obstruction, the selection categories have 1, 3, and 1 orbits; test categories have 2, 2, and 1. These agree with the plan. Relabelled graphs are not separate model replications. The full check is recorded in `data-oracle-review.json`.

The eight literal random masks in the ledger match the specified generator and draw order. The complete 64-epoch order array matches the locked hash. No data rule uses a model outcome.

## Code and estimator

The 5,313-parameter model and training loop match the plan. Each of the four fixed seeds receives 8,192 updates over all natural graphs. The loop has no validation choice, early stopping rule, or seed replacement. Final evaluation uses the weights at update 8,192.

Each unit edit zeros the post-ReLU first-layer entry. The cached continuation and local controls test the intended site. Empty-mask and full-layer controls use separate reference calculations. There are 98 mask rows: empty, full, 64 single units, eight fixed random masks, and 24 selected masks. All have saved outputs on the complete union of 3,096 study graphs. This is 1,213,632 ablation-output values over four models, in addition to 262,144 intact natural-graph outputs.

Each four-unit mask is ranked only by the three six-edge degree classes for its name. Exact ties use the lower unit index. The seven-edge score does not select the mask. The joint edit is measured directly. The selected-minus-random endpoint averages degree classes within a name, then all names within a model. The eight random masks do not act as independent training replications. The final interval resamples the four model-level differences.

The five ability gates use the declared intact decisions, with duplicates removed within each test category. Raw selected selectivity and relative selectivity must both be positive for an individual model's stated interpretation. All five gates must also pass. A panel-wide interpretation requires all four models to meet these conditions. Every model still enters the primary summary if a condition fails.

## Limits and record checks

The opening question now concerns relative effect specificity. The score does not guarantee that control decisions are preserved. The saved category acceptance rates, probability changes, removed activation norms, and raw logits allow the paper to report control damage. Positive selectivity does not identify a unique Hall algorithm or a complete circuit.

The training count is 32,768 updates and 16,777,216 examples. The fixed table counts are 1,188 control rows and 64,800 degree-group score rows. The runner verifies these totals, enforces the 20-minute limit, checks locked bytes before and after the run, records numerical errors before stopping, and saves a failure status and file manifest. The latest guard also checks time after each degree-group score.

No pre-run blocker remains. The results still need independent saved-weight replay, score reconstruction, and a claim review. That audit must state whether it repeats training. Do not change the reviewed plan or runner after the public lock.
