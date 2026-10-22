# Larger logit gains need not give larger probability gains

**Date:** 22 October 2026. **Type:** Formal analysis and finite constructed test. No trained model. AI-authored; no human peer review.

## Main finding

Mean logit gain and mean target-probability gain gave opposite rankings in 704 of 7,100 comparisons with a strict logit ranking. This is 9.92% of that fixed comparison set. Every individual margin change was nonnegative. Opposite signs across inputs were not needed for a ranking reversal.

This fraction describes the chosen grid. It is not a failure rate for activation patching or language models. The scores measure different quantities.

## Question and prior work

A logit is a score before conversion to probability. A fixed score increase has a smaller probability effect when the model is already very confident. Can this change the ranking of candidate edits across contexts?

[Zhang and Nanda](https://arxiv.org/abs/2309.16042v2) directly studied score choice in activation patching. Method dependence is therefore not a new finding. This note adds a complete binary-output comparison grid, a fixed counterexample, and a sufficient condition for ranking agreement. It differs from the context-cancellation paper: every margin change here points in the same direction.

## Method and formal result

The two output logits are (m,0). The target probability is `sigma(m)=1/(1+exp(-m))`. Two equally weighted contexts use margins from {-6,-2,0,2,6}. Each candidate adds one value from {0,0.25,0.5,1,2} in each context. We compared all 300 unordered candidate pairs in all 25 ordered context pairs: 7,500 comparisons.

For an increase delta, elementary differentiation gives:

`sigma(m+delta)-sigma(m) = integral from 0 to delta of sigma'(m+t) dt`.

Because `0 < sigma'(u) <= 1/4`, a nonnegative margin increase produces a probability increase between zero and delta/4. Different contexts can have different effective weights under this conversion.

If candidate A adds at least as much margin as B in every context, monotonicity guarantees that A also has at least as much probability gain. A larger average margin alone does not provide that guarantee. The code tests this distinction without fitting a model or selecting favorable examples.

## Results and checks

There were 400 logit ties and 7,100 strict logit comparisons. Of the strict comparisons, 704 reversed under probability gain. No comparison violated the componentwise-order guarantee.

The example fixed in the plan uses margins (6,0):

| Candidate | Margin additions | Mean logit gain | Mean probability gain |
| --- | --- | ---: | ---: |
| A | (2,0) | 1.0 | 0.001069 |
| B | (0,1) | 0.5 | 0.115529 |

A produces the larger logit gain. B produces the larger probability gain because it acts on the less certain context. Both computations are correct. The choice between them depends on the research question.

Zero-edit, monotonicity, upper-bound, and componentwise-order checks passed. Direct two-class softmax matched sigmoid within 2.78e-17. A separate scalar audit reproduced every comparison and reversal within 2.23e-16, below the fixed 1e-12 tolerance. Calculation and audit took about 0.48 seconds inside their scripts, below the shared 120-second cap.

## Limits and next test

The logits and changes are constructed. No actual intervention was made inside a trained network. Equal context weights and the selected margins define this census. Different grids produce different descriptive rates.

Neither metric is universally superior. Logit gain measures a change in relative output score. Probability gain also depends on initial confidence. A report should define its target quantity before selecting an edit. It should show whether the claimed result survives other relevant scores without choosing the score after seeing the outcome.

A later study should use a fixed trained model and report context-level effects. It should keep that test separate from this exact mathematical example.

See [plan](analysis-plan.md), [code and instructions](README.md), [all comparisons](results/comparisons.csv), [audit](results/audit.json), and [separate AI review](PRE_RUN_REVIEW.md). The public code lock is `88e8df7689b65fbeef221423c115cd1b1c198f22`. The [source record](sources.md) states the checked prior work.
