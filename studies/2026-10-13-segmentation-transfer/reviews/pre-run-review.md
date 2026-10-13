# Independent AI review before the run

13 October 2026. Reviewer: a separate AI agent. No human review occurred. No scientific model results were available to this review.

Status: ready for the public plan/code lock. All requested repairs are closed.

## Scope and design

I read the fixed plan and full runner. I also read the dummy receipt and download helper. The reviewer did not import the runner or collect scientific model outputs.

The plan separates task ability from the fixed head's causal effect. Its two primary estimates form a declared conjunction. Whole token groups keep paired encodings, bindings, gaps, orders, queries, and ablations together. The code uses the correct supplied-value score and pairs the intact and ablated runs. Its bootstrap follows the plan. The 75% two-answer and 10% full-vocabulary thresholds are acceptable as declared interpretation rules. They are not general standards of language competence. Keep the raw rates and all cases.

The input rules preserve bytes and token counts across the four split cells. Each mismatched query has no token ID in common with either source key. The plan correctly states that the last token ID changes. Thus this is an input-encoding test, not a direct intervention on an isolated semantic variable. Canonical references have different lengths and positions, so they remain capability references only. Subtoken pieces can recur across different groups; group intervals must retain the stated fixed-construction limit.

The local ablation changes only the selected final-query head vector before output projection. The stock attention product and residual ordering remain intact. All-head future-attention checks are present. The full projection check is performed on dummy inputs, as planned.

## Repairs requested and checked

1. The download helper reads `MODEL_ID` from the runner, but the runner has no such constant. Add it and a pinned requirements file. A download from an empty directory must have all required metadata.
2. The plan requires every control error to be saved. The runner currently retains stock/self maxima in memory and writes them only after success. Add a per-batch control table, saved before any tolerance assertion. Preserve partial control records on failure.
3. The `unchanged_attention` error is initialized to zero but never calculated. Measure it against a saved attention tensor around the edit, or remove it from measured statistics and state that the same attention tensor is returned by construction.
4. Preserve a fresh dummy receipt in the study. Include the plan and runner hashes, package versions, and an explicit statement that no scientific outputs were collected.

All four repairs are present in the final files. `MODEL_ID` and the requirements file are present. The runner saves each batch's stock and self errors before the tolerance assertion. It compares attention weights with a saved copy around the edit. The new dummy receipt records versions, model hashes, and an explicit false value for scientific output collection.

I also requested the real batch size. The final dummy check uses 32 distinct inputs at each of the two planned split-input lengths, 9 and 25. All stock, self, projection, attention, ablation, and untouched-output errors are zero. The previous dummy checks remain local records; the current study receipt identifies the final code.

I independently checked the receipt's hashes against the current files. Plan SHA-256: `733b07a06f3d79bbfa749025e02a7252ea5d2c7544b0a1b0b14d9c7cbbcda483`. Runner SHA-256: `c3856ab4632d439fe236483086f8745914564058f8e780a848c3abc0b8bec6bc`.

No pre-run blocker remains within this review scope. No scientific model output was used to choose or change the plan. The root agent must verify the public lock before it permits the fixed run. This review does not replace the independent result audit.
