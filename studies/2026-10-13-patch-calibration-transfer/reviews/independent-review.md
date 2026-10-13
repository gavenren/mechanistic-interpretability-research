# Independent calculation review

Date: 13 October 2026. Reviewer: a separate AI agent in the same research session. This is not external or human peer review.

**Status:** Review complete with a disclosed audit deviation. The planned float32 block-delta check failed. An added check that preserves the model's addition order reproduced all saved target logits and baseline activations exactly. The failed planned check remains part of the record.

## Scope and sequence

The reviewer read the plan and full runner before the experiment. The [pre-run review](pre-run-review.md) records the design checks and the library's hidden float32 operations. The plan and runner were published at `4dc5db20c261f9057d08a4ad2c7cc9548dcba1ba` before the scientific run. Neither locked file changed during this review.

The reviewer wrote [independent_audit.py](../code/independent_audit.py). It does not import the runner. It checks the saved data, computes the reported statistics again, and uses a different intervention hook. The experiment replaces branch outputs. The planned independent check adds donor-minus-receiver branch changes at the whole-block output.

The first audit stopped at a float32 difference above the fixed tolerance. The exact [initial audit source](initial_audit.py) and [first failure receipt](first-audit-failure.json) are preserved. Their source hashes agree. A later audit records every failed cell instead of stopping at the first one. An intermediate replay was stopped because it repeatedly decompressed the same saved arrays. The current code loads each array once. This speed change did not change a check, tolerance, scientific file, or test case.

## The planned check failed

Each numerical condition has 4,608 nonbaseline patch cells: 256 directions × six layers × three patches. The planned absolute tolerance was 0.0001 for float32 and 0.000000001 for double arithmetic with shared float32 rotary tables.

| Condition | Failed patch cells | Largest absolute target-logit difference |
| --- | ---: | ---: |
| Float32 block delta | 1,385 / 4,608 | 0.0017538071 |
| Double-arithmetic block delta | 0 / 4,608 | 5.94 × 10⁻¹² |

The first failure was case 0, layer 0, attention-only patch. Its target-logit difference was 0.0002031326. The tolerance was not increased. No failed cell was removed. The [complete delta receipt](audit-receipt.json) lists all 1,385 failures and retains a failed status.

## Added diagnosis: preserve the addition order

The original block computes `(MLP + attention) + skip`. The planned check first computes the unmodified block output, then adds a branch difference. These expressions agree over real numbers. They can differ in float32 arithmetic.

After the failure, the reviewer added [ordered_reconstruction_audit.py](ordered_reconstruction_audit.py). This check still uses a whole-block hook. It constructs the patched vector from saved receiver or donor branches and the receiver skip state, in the model's original order and dtype. It does not replace branch module outputs and does not import the runner.

The added check covered **every cell in both conditions**. It used 4,864 forwards per condition: 256 intact forwards and 4,608 patch forwards. Across all **9,728 forwards**, the saved target logits and intact activation vectors matched exactly. The [ordered reconstruction receipt](ordered-reconstruction-receipt.json) reports a maximum model-output or cached-activation difference of zero in each condition.

For the 1,385 failed float32 delta cells, the largest difference between the delta-produced block vector and the ordered block vector was **1.19 × 10⁻⁷**. Reconstructing the ordered vector removed the target-logit discrepancy in every failed cell. The checkpoint, inputs, other positions, downstream computation, and comparison tolerance stayed fixed. This identifies changed floating-point addition order as the source of the failed delta equivalence check in these cases.

The added check supports the recorded branch-patching outputs within this audit scope. It does not make the planned check pass. It does not show that float32 is exact or that other intervention implementations are equivalent. The added diagnostic must be identified as a post-result method in the paper.

## Data and calculation checks

The review verified all 20 files in the scientific result manifest and all six external model-file hashes. The plan and runner match the lock commit byte for byte. The model revision is `cf967c0a9a04383db6f7b1108d86b2962634b4ac`.

The independent calculation checks covered:

- All 256 prompt texts and token sequences, all 256 directed pairs, and the exact four-way split.
- Single-token targets, equal paired lengths, one changed input token, and equal final tokens.
- All 3,072 layer-case result rows across the two numerical conditions. Each row contains four cells.
- Calibration medians, fitted only from the 64 calibration directions in each layer.
- Every prediction, raw logit margin, interaction, absolute error, and sign category.
- All 108 group-summary rows across both conditions.
- Both fixed-case crossed-bootstrap intervals, with the declared seed and retained directions and layers.
- All intact pairwise preferences and full-vocabulary target decisions.
- All saved baseline branch, skip, and block vectors against the pinned model.
- Final LayerNorm parameters, target output rows, and separate NumPy margin calculations.
- Identical rotary-position tables across the two numerical conditions.

The two complete audit receipts each record **2,003,031 compared numbers** and **226,045 checks**. The largest difference in saved-table arithmetic was **2.04 × 10⁻¹⁴**. These counts overlap between audits; they are not counts of independent observations.

The independently computed primary MAEs are **0.0304718424** before the correction and **0.0311346548** after it. The MAE gain is **−0.0006628124**. The fixed-case reweighting interval includes zero. The second numerical condition gives gain **−0.0006710796**. These confirm the reported small negative primary comparison, without a population conclusion.

The separate final-normalization calculation projects onto one centered output direction. This differs from the runner's two-row matrix product. Its maximum margin discrepancy is **8.30 × 10⁻⁶** in float32 and **1.91 × 10⁻¹⁴** in double arithmetic. The frozen-denominator diagnostic also passes its unchanged tolerance. Small differences from the runner's reported residual maxima follow from the different arithmetic order.

## Interpretation and paper review

The correction is a fixed median, matched to mean absolute error. The data do not test all calibration methods. The split holds out the declared words and frames, while keeping the same copy-style task and model. The change in intact accuracy limits any claim that only interaction stability changed.

The double-arithmetic condition retains float32 rotary-position tables. It is a sensitivity test, not a numerical ground truth. The final LayerNorm identity and normalization effects have close prior work. The paper must not present them as a new discovery.

The reviewer read the paper and reproduction guide, checked the numeric claims, and viewed the figure. The plot is readable and agrees with the saved tables. The paper gives the direct-joint cost limit: predicting from two singleton patches is more expensive than one joint patch when only the joint effect is needed. Reuse can save extra joint measurements only when singleton scores already exist and the calibration cost is recovered.

Publication can report the fixed experiment and its limits if it keeps this audit deviation visible. It must not state that all planned controls passed. No further implementation issue was found within the recorded scope. Human peer review has not occurred.

## Reproduce the review

From the repository root, with the declared dependencies and the pinned model files available:

```sh
python studies/2026-10-13-patch-calibration-transfer/code/independent_audit.py --model-dir MODEL_DIRECTORY
python studies/2026-10-13-patch-calibration-transfer/reviews/ordered_reconstruction_audit.py --model-dir MODEL_DIRECTORY
```

The first command lists all planned delta failures and reports the failed status. It continues through both conditions to preserve the full record. The second command runs the added ordered reconstruction on every cell. Both commands leave the scientific result files unchanged. The initial fail-fast behavior can be reproduced with `reviews/initial_audit.py`.
