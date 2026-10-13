# Independent calculation review

Date: 13 October 2026. Reviewer: a separate AI agent in the same research session. This is not external or human peer review.

Status: calculation audit and paper review passed.

## Scope

The reviewer read the plan and runner. The reviewer wrote [independent_audit.py](../code/independent_audit.py). The audit does not import the study runner or repeat training. It rebuilds the declared data and reads the saved weights. It constructs the patched 12-dimensional activation vectors directly. This differs from the runner's scalar readout shortcut.

The review checked all 72 models, all 3,456 candidate rows, every saved selected-latent test output, the validation selection rule, reconstruction controls, native full-code and empty controls, nuisance controls, and population metrics. It also checked the primary paired contrast, the bootstrap interval, the factorial summary, and all 144 label-specific nuisance rows.

## Results

All checked results agree. The maximum absolute numeric difference is **5.11 × 10⁻¹⁴** across **2,696,520** compared values. All **162** scientific files in the original manifest match their recorded hashes. The label-specific nuisance table is an original run output: the locked runner writes it, and the original manifest includes its hash. Independent calculation coverage of those 144 rows was added during review. The plan and runner match lock commit `75ab86efd6159a4965b8461a534a0a29592d98ac`.

The primary ordinary-minus-balanced differences are approximately:

| Seed | Difference |
|---|---:|
| 0 | -0.015832 |
| 1 | -0.241624 |
| 2 | 0.796570 |
| 3 | -0.019481 |
| 4 | -0.308711 |
| 5 | -0.066324 |

The mean is **0.02410**, but five of six differences are negative. The 95% paired percentile interval is **[-0.20555, 0.34432]**. These six seeds do not show a reliable improvement in the primary single-latent endpoint. The mean alone would give an incomplete account of the result.

## Plan issues and their resolution

Before training, the reviewer requested an exact definition of mean addition in the decoder, post-patch activity reporting, and a clear distinction between reconstruction failure and single-latent transport failure. The revised primary endpoint retains the receiver residual. The saved results include both native and reconstruction-based controls. A latent patch can give five active code values; the experiment reports this without applying TopK again.

The reviewer also found that the background generator has a point mass at zero. Its probability is `0.9^20`, about 12.2%. Separate random draws therefore do not imply distinct input values across splits. The final plan distinguishes draw identities from values, and the generator records the zero counts and exact overlaps. This limit must remain visible in the paper.

## Interpretation limits

The readout and rare direction are specified by construction. The SAE is trained, but the experiment is not a trained language-model circuit. The balanced loss receives true event labels and must be called a supervised oracle comparator. Correct transfer of this binary output effect does not establish recovery of the complete mechanism or of the background features.

The held-out intervention tests use independent background draws, with repeated zero-valued backgrounds retained. Six random seeds describe this fixed synthetic design. They cannot support a broad claim about sparse autoencoders in language models. No unique mechanism, priority claim, or external replication was verified.

## Paper and figure check

The reviewer read the complete paper and reproduction guide, checked the reported numbers against the independently verified tables, and viewed the four-panel figure. The means, six primary differences, interval, full-code comparisons, zero-background counts, and factorial results agree. The figure shows the seed values on a common scale without clipping.

The paper leads with the mixed primary result. It reports that balanced training has worse natural-weighted downstream KL even though its primary-condition full-code intervention score is better. This prevents a claim that the method improves every output measure. A sentence about unshown geometry scores was removed after review. The final text separates the fixed synthetic finding from prior work and broader language-model claims. No blocking interpretation issue remains within this review's scope.

## Reproduction

With the declared NumPy version installed, run from the study directory:

```sh
python code/independent_audit.py
```

The command reads the saved scientific files and prints a JSON receipt. It does not change them. The saved receipt is [audit-receipt.json](audit-receipt.json).
