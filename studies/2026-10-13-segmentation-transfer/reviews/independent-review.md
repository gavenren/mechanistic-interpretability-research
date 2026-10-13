# Independent AI result review

13 October 2026. A separate AI agent performed this review. It did not import the study runner. No human review occurred.

Status: the recorded float32 experiment passed the complete replay and arithmetic audit. An added float64 probability check failed. That failure remains part of the record. The final paper must disclose this numerical difference.

## Scope and method

The audit verified the result manifest, package versions, model hashes, and the exact plan/runner bytes at public lock `c6ff9cb1405e6a65fe161c2f4cb471d51016bf4c`. It rebuilt the tokenizer pool, candidate splits, excluded IDs, greedy key pairs, values, fillers, and all input sequences. It checked every four-cell byte equality and the fixed mismatch token exclusions.

It checked all 2,048 paired score records and 512 canonical records. The checks covered score signs, float32 margin arithmetic, head effects, all group means, all bootstrap estimates, and the declared capability gate. All differences were exactly zero.

The independent model replay used a different intervention site in the code. It installed a pre-hook on the attention output projection at block 3. It zeroed dimensions 32 through 63 of the concatenated head input at the final position. This is the same declared head edit. It does not replace the attention function used by the study runner. All other projection-input entries remained exact.

The replay covered every intact and ablated split input and every canonical reference: 4,608 forward examples in 144 calls. Every saved B/D logit, recorded float32 probability, selected-head attention row, and head norm matched exactly. All top-token and rank records matched. The fixed logit tolerance was `1e-5`; the recorded-probability tolerance was `1e-7`. No tolerance was increased.

The full replay and table audit took 6.66 seconds on the local CPU. This time is an observation from one run, not a speed guarantee. The [final receipt](independent-audit-final.json) gives the full scope and maximum errors.

## Preserved failed probability check

The [first audit](independent-audit.json) used a separate NumPy float64 softmax calculation on replayed float32 logits. Its first probability error was `1.0046236819938326e-5`, above the audit's added `1e-7` probability tolerance. The initial code is preserved in [independent_audit_initial.py](../code/independent_audit_initial.py). The scientific runner and results were not changed.

The recorded definition is PyTorch `softmax` on each float32 row. The final audit checked that exact definition and matched every saved probability. It also kept the float64 result as a separate diagnostic. Across the full replay, the largest full-vocabulary difference was `7.718852920790553e-5`, on ablated split prompt 386, token 84. The float32 value was `0.9775402545928955`; the float64 reference was `0.9774630660636876`. Both calculations used the same float32 logits. The sum of that float32 probability row, accumulated in float64, was `1.000078968257326`.

A [separate diagnostic](probability-diagnostic-v2.json) checked the first 32 short split inputs. It compared the full vocabulary logits with attention capture enabled and disabled. Their maximum difference was zero. The saved PyTorch float32 top probabilities also matched exactly. NumPy and PyTorch float64 softmax on these same logits agreed within `4.78e-15`. Thus this observed gap is between numerical softmax calculations on the same logits. It is not evidence of a model-replay mismatch. The diagnostic scope is the first 32 short inputs; the complete final audit records the maximum float32/float64 probability gap across all replayed inputs.

The original float64 check still fails its original threshold. It is not a retroactive pass. The probability records describe the locked float32 computation. They should not be presented as precision-independent probabilities. The main conclusions use logit margins and rankings. Every required target logit and ranking was verified in the complete replay.

## Claim review

I read the paper and README after the full audit. Their main distinction is sound: the primary score conjunction passes, while the capability gate fails. Mismatched prompts have 56.93% two-answer accuracy and one correct full-vocabulary top prediction in 1,024 cases. A positive head effect does not make this successful retrieval. The paper keeps both facts visible.

The paper also states that canonical references have different lengths and positions, that the tested splits are noncanonical, and that the final token ID changes. It does not claim a new detokenization mechanism or equal behavior with different internal causes. The result is bounded to this model, task, and selected head.

The author added the completed audit, the exact replay scope, and the preserved float64 probability-check failure to the paper and README. I checked the final wording. It clearly states that the first check still fails and that no tolerance changed. It also limits the exact untouched-output claim to the edited layer. These changes close the review requests.

Recommendation: ready for publication, with the disclosed numerical qualification. Keep both receipts, the initial audit code, and the separate diagnostic. No change to the locked plan, scientific runner, or result files was needed. The root agent remains responsible for remote publication verification.
