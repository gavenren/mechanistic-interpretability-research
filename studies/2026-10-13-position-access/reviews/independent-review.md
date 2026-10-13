# Independent AI result review

Date: 13 October 2026. Status: numerical audit and final paper review passed. No human review occurred.

The audit code does not import the study runner. It reconstructs the tokenizer pool, token groups, bindings, filler sequences, all 768 inputs, physical masks, and position coordinates. The complete input ledger matches. It checks every saved score, all group and cell summaries, the main contrast, the bootstrap interval, and the long-minus-short comparison. This arithmetic matches exactly.

The model replay uses a separate implementation of the causal mask. It blocks filler keys directly in the attention score calculation. The head ablation uses a pre-hook at the attention output projection, where the selected head occupies coordinates 32 through 63. This differs from the runner's edit of the unflattened head tensor inside the attention function.

All 6,144 scored cases were replayed. Every saved full-vocabulary logit hash matched exactly. The target and other logits, top-1 tokens, and selected-head attention masses also matched. No scalar score error was found.

The audit independently replayed all 3,072 blocked-condition cases against the actual short input, including intact and ablated forms with compressed or sparse matching coordinates. It compared the full final vocabulary vector and all six core positions after each block. The maximum logit error was `2.397282372612608e-11`; the maximum core-state error was `8.431033649003439e-13`. These pass the original `1e-8` and `1e-10` limits. All forbidden attention weights and changes outside the local ablated vector were exactly zero. The largest row-sum error was `2.4424906541753444e-15`.

The audit used 2,304 model calls and 9,216 model examples in 90.58 seconds. It checked all original 12,800 comparison records and 4,224 attention-control records for completeness and passed bounds. It did not independently repeat the alternate-filler, default-mask, or no-op control passes. These scope limits do not change the full replay of every scored cell and every short-input equivalence case.

The receipt is `independent-audit.json`. The audit code hash is `f902cbfc36084a44112a065932d537aab6f44810f06313e80a4c5068232ea715`. The plan and runner remained fixed at public lock `c6ff9cb1405e6a65fe161c2f4cb471d51016bf4c`.

The proposed primary contrast can be positive while both single interventions lower the score. A positive contrast must not be described as proof that either intervention repairs performance. The final paper must show all four cells and their interaction, together with the full-vocabulary task accuracy. The result remains a test of artificial operators on one small model and one input family.

I read the final paper and README and viewed the complete figure. The paper shows both harmful single edits, separates the fixed contrast from the secondary interaction, and identifies joint short-input agreement as an expected control. It also states that the natural long input has a lower two-value margin but higher full-vocabulary top-1 accuracy. These claims match the audited tables. The figure labels the two gaps and uses the correct signed score changes.

The final text clarifies that this study's outcomes did not select a head; that choice came from the earlier copy study. The README gives an audit command with a fresh output path. No issue remains within this review scope. The audit code, receipt, and review are now fixed.
