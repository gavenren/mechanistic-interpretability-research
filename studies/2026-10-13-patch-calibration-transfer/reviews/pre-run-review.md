# Review before data collection

Date: 13 October 2026. Reviewer: a separate AI agent in the same research session. This is not human peer review.

## Design review

The review accepts the design for a bounded experiment. The question is whether one fixed calibration rule improves joint-patch predictions on held-out words and prompt frames. The experiment cannot establish a population rate or show that all calibration methods work or fail.

The median correction matches the primary mean absolute error score. Both patch directions remain in the same split. The four splits separate the declared word and frame sets. Repeated words, frames, layers, and directions are dependent. The fixed cases are not a random sample of language.

The review requested these controls before any outcome collection:

- Use a raw two-token logit margin. Do not divide by a clean-corrupt score gap.
- Keep the receiver skip residual fixed. Patching both parallel branches is not a full donor-state replacement.
- Keep all prompts, including prompts with incorrect intact answers.
- Save all four cell scores and the full activation vectors needed for separate checks.
- Use layer 0 as a known zero-interaction control when the final tokens agree.
- Check the final LayerNorm calculation with a separate NumPy implementation. The frozen-denominator result is a diagnostic on a changed readout, not a corrected effect in the original model.
- Account for joint-patch calibration costs before making any cost claim.

## Source-code inspection

The reviewer inspected the installed Transformers GPT-NeoX implementation before the study run. In parallel mode, each block adds its MLP output, attention output, and incoming residual. Both branches read the same incoming residual. In evaluation mode the dropout operations do not change those outputs.

The stock eager attention function forces its softmax to float32. The stock rotary position code also computes its frequency products and trigonometric values in float32. Converting model weights to float64 does not remove these casts.

The precision sensitivity condition will use double arithmetic and a dtype-preserving attention softmax, with shared stock float32 rotary tables. This is not a fully float64 model evaluation or a numerical ground truth. The paper must keep this limit explicit.

## Planned independent check

The independent audit will not import the study runner. It will recompute the split, score differences, calibration medians, all aggregate metrics, and final LayerNorm margins from saved data. A second model check will add cached donor-minus-receiver branch differences at the whole-block output. This differs from the runner's direct branch replacement method.

This document records the design review only. Runner review, outcome checks, and paper review remain pending.
