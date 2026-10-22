# Fixed plan: score choice and patch rankings

Date: 22 October 2026. Formal finite-grid study. No trained model. No scientific run has started.

Question: when do mean logit gains and mean probability gains rank two candidate edits differently even though every edit increases the target margin?

Use binary logits (margin,0). Two contexts have margins drawn from -6,-2,0,2,6. A candidate specifies a nonnegative change in each context from 0,0.25,0.5,1,2. There are 25 candidates. Evaluate each unordered pair of candidates in each of 25 ordered context pairs: 7,500 comparisons. All contexts have equal weight. Save each pair's mean margin gain, probability gain, comparison signs, and componentwise-dominance status. Sign ties use absolute tolerance 1e-12. The rate is descriptive for this fixed grid, not a population estimate.

Prove the exact formula sigma(m+delta)-sigma(m)=integral_0^delta sigma'(m+t)dt. For nonnegative delta, the gain lies between zero and delta/4. If one candidate has at least as large a delta in every context, its probability gain cannot be smaller. Under this componentwise order both metrics agree weakly. An average margin advantage alone does not give this guarantee. No claim that these established mathematical facts are new.

Prespecified example: margins(6,0), candidateA(2,0), candidateB(0,1). Report it whether it supports reversal or not. Controls: zero edits, monotonicity, delta/4 bound, componentwise order, direct softmax versus stable sigmoid. A separate scalar audit uses math.exp, recomputes all rows and comparison decisions, and verifies a 1e-12 tolerance. No case is chosen after results.

Two CPU threads, 120-second cap including audit. Root controls usage permit and serial launch after a public code/plan lock. The constructed logits are not evidence about the prevalence of this issue in language models. Distinct from the signed-cancellation study: every per-context margin change here is nonnegative; the question is confidence weighting by the output link function.
