# Separate AI source review

Date: 22 October 2026. The reviewer read the plan, runner, support code, and audit. No experiment was run for this review. The reviewer did not write this study's code. No human peer review is recorded.

The sigmoid integral identity and the delta/4 upper bound are correct for nonnegative margin changes. Componentwise dominance implies weak agreement of the two rankings. Average margin dominance does not imply componentwise dominance. The code keeps these statements separate.

There are 25 candidate vectors and 300 unordered pairs in each of 25 ordered context pairs. The total is 7,500 comparisons. The fixed example is recorded separately. The scalar audit recomputes every comparison and sign, rather than only a reported example. The 1e-12 sign rule is consistent in the runner and audit. The constructed logit ranges avoid overflow in the scalar exponential formula.

No blocking source issue was found. Before execution, the root must verify the public code lock and pass one absolute `MI_STAGE_DEADLINE` to both processes. Otherwise the support module's separate 120-second timers would not enforce a combined run-and-audit cap.

The report must call the reversal rate a result for the fixed grid. The code does not measure language-model edit rankings or estimate how often such reversals occur in practice. The mathematical statements are established calculus facts, not claimed discoveries.

## Correction check before lock

The revised support module now checks the public lock and takes an exclusive compute lease. The root will supply a common deadline across run and audit. No blocking source issue remains.
