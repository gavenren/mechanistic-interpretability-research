# Independent calculation review

Date: 13 October 2026. Reviewer: a separate AI agent in the same research session. This is not external or human peer review.

Status: calculation audit and paper review passed.

## Scope

The reviewer read the plan and runner, then wrote [independent_audit.py](../code/independent_audit.py). The audit does not import the runner or repeat training. It reads all eight saved models and computes every candidate intervention again.

The audit replaces coordinate values directly, instead of multiplying by the runner's diagonal mask matrices. It builds a folded dense encoder for the transformed states. It uses an explicit matrix inverse for transported operators and QR factorization for the Euclidean image projector. These calculations give separate implementation paths for the main tests.

## Results

All checked results agree. The maximum absolute numeric difference is **7.11 × 10⁻¹⁵** across **122,560,962** compared values. All **30** files in the scientific manifest match their hashes. The plan and runner match lock commit `d4ca99ae08121070facd5ff755b6ca4b164dc304`.

The audit checked:

- The 25 task states and labels, all 500 intervention pairs, the exact 250/250 hash split, and the counterfactual indices.
- The 26 fixed coordinate transforms and their condition numbers.
- All 29,120 candidate rows, 208 selected masks, and 1,664 operator rows.
- Selection from discovery pairs alone, with the declared numerical tie rule.
- All intact predictions and all ordinary task accuracies. Each model has accuracy 1.0.
- Native branch replacement, correctly transported replacement, empty and full replacement, and the matched-donor image-projector control.
- Every signed-permutation mask correspondence and every coordinate mask under the three diagonal scales.
- Per-seed and across-seed summaries.

The independently computed maximum transported probability difference is **3.43 × 10⁻²⁰**. The complete signed-permutation family agrees within **2.22 × 10⁻¹⁵**. The complete coordinate-mask family agrees across diagonal scales within **6.00 × 10⁻¹⁵**. These are numerical controls, not separate scientific discoveries. Small differences from the runner's residual values arise from the separate calculation paths.

The main rotation result is confirmed: mean selected-mask test TV is **0.57586**, with per-seed rotation means from **0.53351** to **0.61967**. Mean semantic accuracy is **0.43725**. The original coordinate basis and the correctly transported known operator recover the counterfactual with only round-off error.

## Methods and interpretation

The branch architecture supplies the native factorization. The study does not discover a mechanism. The fixed linear interface makes the coordinate transformation explicit. A general dense coordinate transformation would not preserve the separate-branch parameter constraint by itself.

The row-vector transport formula `T^-1 P T` is correct. The transported operator and the Euclidean image projector have the same image space, but their nullspaces can differ. They are different interventions under a non-orthogonal change of coordinates. The results do not show that Euclidean projection is universally wrong.

For `T = Q D`, diagonal scaling commutes with each coordinate mask. Therefore, coordinate-mask outcomes must agree for the three scales associated with a fixed rotation. The experiment verifies this identity. The scaled conditions are not additional independent coordinate-localization results.

All ordinary inputs were used in model training. The test holds out pair identities. Eight trained seeds are the replication units; the eight rotations per model are repeated conditions. The finite task does not establish failure rates in language models. Basis dependence and operator transport have substantial prior work. The valid contribution is this bounded experiment and its complete checkable record.

## Paper and figure check

The reviewer read the complete paper and reproduction guide, checked its numeric claims against the verified results, and viewed both figures. The main table, per-seed ranges, full model-rotation range, unmatched-donor means, and control residuals agree. The figures have readable labels and distinguish within-seed map ranges from independent model seeds.

The matched and unmatched donor analysis is correct. The Euclidean image projector gives the exact intervention when the donor differs only in the supplied a branch. Its unmatched-b mean test errors are 0.19012 and 0.47581 at condition numbers 4 and 16. This is a property of the declared intervention geometry in the constructed task.

The paper states the architecture advantage, pair-only holdout, known prior work, and limits on mechanism claims. The reviewer requested an exact clean-checkout command to improve reproduction instructions. No blocking scientific issue remains within the stated review scope.

## Reproduction

With the declared NumPy version installed, run from the study directory:

```sh
python code/independent_audit.py
```

The command prints a JSON receipt and does not alter the scientific files. The saved receipt is [audit-receipt.json](audit-receipt.json).
