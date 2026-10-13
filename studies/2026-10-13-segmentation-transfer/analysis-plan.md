# Can a copying head use the same string under different token splits?

Fixed analysis plan, 13 October 2026. No study model outputs were inspected when this plan was written. Only tokenizer checks and dummy numerical checks are allowed before the remote plan/code lock.

## Question

A text string can have different token encodings. Does a small model use a supplied string-to-value binding when the source key and query use different encodings? Does a copying head identified in an earlier study contribute to that use?

These are separate questions. A useful head effect requires evidence that the model uses the binding in the first place. We will test one fixed head. We will not search for a different head if it fails.

## Model and budget

Use `EleutherAI/pythia-14m`, revision `cf967c0a9a04383db6f7b1108d86b2962634b4ac`. Verify the six model-file hashes from the earlier copy study. Use float32, eager attention, two CPU threads, deterministic execution, and no training. Keep the stock attention product and residual addition order. Project only the final hidden state to the vocabulary. Set a 20-minute time limit for the scientific run. Preserve partial output on failure. No paid compute is needed.

The selected component is layer 3, head 1, with zero-based indices. It was selected and tested in the earlier copy-routing study. That earlier result does not establish its role in this new task.

## Fixed token pool

Use the pinned tokenizer. Decode with `clean_up_tokenization_spaces=False` throughout.

1. Reconstruct the earlier study's pool: the first 192 IDs at or above 1000 whose single-token decoded text matches ` [a-z]{3,8}` and re-encodes to that same ID. Exclude these IDs as new canonical keys and values.
2. Scan the vocabulary in ID order. A key candidate must have a canonical one-token encoding, match ` [a-z]{5,12}`, have ID at least 1000, and not be in the excluded pool.
3. For each character split after at least the leading space and one letter, require that both pieces are each the exact decoded text of one token. When multiple IDs decode to the same piece, use the lowest ID. Require the two-token concatenation to decode to the original bytes.
4. Split A uses the earliest valid character cut. Split B uses the latest. Require at least two cuts and four distinct token IDs across A and B.
5. Make 32 groups. Take the lowest-ID unused candidate as the first key. Pair it with the lowest-ID remaining candidate whose four subtoken IDs are disjoint from the first key's four IDs. Remove both from the candidate pool. Fail if 32 groups cannot be made.
6. Select 64 value IDs in ascending order from the same single-token rule as step 1. Exclude the old pool and all new canonical key and subtoken IDs. Allocate two consecutive values to each group. Each group therefore has two keys and two distinct values.
7. Select the next 64 eligible single-token IDs as fillers, with the same exclusions and all value IDs excluded. For group `g`, use a permutation from NumPy seed `2026101300 + g`. The first gap uses its first eight tokens; the second uses tokens 8–15.

Save all candidate rules, selected IDs, text, token pieces, rejected pair conflicts, group assignments, and fillers. There is no outcome-based selection. The strings include word parts. We will not call every string a word or infer word meaning.

A tokenizer-only audit before this plan found 10,112 strings with disjoint A/B splits before the old-pool exclusion. Both tested splits are noncanonical. The canonical reference uses one token.

## Prompt grid

Each group has keys K and L and values B and D. Use both assignments, K→B/L→D and K→D/L→B. Use both source-pair orders and both queries. Use gaps 0 and 8.

The token structure is:

`[BOS, source_key_1, value_1, gap_1, source_key_2, value_2, gap_2, query_key]`.

Source keys share a split setting, A or B. The query has an independent split setting, A or B. Thus the four cells are AA, AB, BA, and BB. Each key occupies two tokens. The sequence length is 9 at gap zero and 25 at gap eight. The readout is always at index `length - 1`. For a fixed binding, order, gap, and query, all four cells decode to identical bytes. The final token ID differs with the query split. That is an explicit part of the intervention on input encoding. There is no added delimiter.

The two source keys have no subtoken IDs in common. In AB and BA, the query shares no subtoken ID with either source key. Values and fillers contain none of those subtoken IDs. In AA and BB, only the corresponding source key matches the query tokens.

The main grid has 32 × 2 assignments × 2 orders × 2 queries × 2 gaps × 4 cells = 2,048 prompts. Retain every prompt.

Also run 512 canonical references using the one-token keys. Their lengths are 6 and 22. They are capability references only. Their positions and lengths differ from the main grid. They are not part of the primary causal comparison.

## Intervention

At layer 3, set only head 1's final-query output to zero immediately before the shared output projection. First compute the stock attention output. Clone it and set the selected vector to zero. Keep attention weights, every earlier query output, and all other heads exact. All later layers run normally.

Run this ablation for all 2,048 split prompts. Do not ablate the canonical references. This tests the necessity of one head output for a score. It does not isolate a semantic variable, identify every route, or prove a complete circuit.

## Scores and primary test

For each prompt, let `S` be the correct supplied value's logit minus the other supplied value's logit. Both assignments and queries are balanced. Thus the mean is sensitive to the supplied binding rather than a fixed preference for B or D.

Let `E = S_intact - S_ablated`. Positive E means removal of the fixed head reduces the binding score.

The two primary estimates are the mean S and mean E on the mismatched cells AB and BA, pooled across all other factors. The primary causal pattern is supported only if both 95% lower confidence bounds exceed zero. This is one fixed conjunction. Do not choose a different primary contrast after outputs.

A separate capability gate limits the language of the report. To describe the result as a head contribution to successful binding use, the mismatched intact prompts must have at least 75% two-answer accuracy and at least 10% full-vocabulary top-1 accuracy, as well as a positive lower bound for S. These are modest, declared thresholds for this constructed task. They are not standards for language competence. Report the raw rates even if a threshold fails. Never remove failed cases. If the gate fails, describe the result as a score response on a task where the model has limited ability.

For each prompt and condition, save B/D logits and probabilities, S, target probability, target rank, full-vocabulary top token, top-token probability, two-answer correctness, and full-vocabulary correctness. Save the intact selected head's full attention row and its output norm. Full-vocabulary measures must use all vocabulary entries, including special tokens. Ties count as incorrect for the two-answer metric; target rank is one plus the number of logits strictly greater than the target logit.

## Secondary estimates

Report S and E in each of AA, AB, BA, BB. Also report matched averages (AA/BB), mismatched averages (AB/BA), and their differences. Report results by gap and group. Include the assignment-sensitive binding contrast and the preference for the most recent value at group level. Report canonical reference capability separately. Attention to the two value positions is descriptive evidence only.

Do not calculate a recovery ratio with a small or negative denominator. Do not claim practical equality from a confidence interval that crosses zero. Do not use an all-head sweep.

## Uncertainty and controls

Use 10,000 bootstrap resamples of the 32 complete token groups, seed `2026101320`. Keep all bindings, orders, queries, gaps, encoding cells, and paired ablations together. Use percentile 95% intervals. These intervals concern the fixed construction's group variation. They do not represent all language tasks or models.

Before the lock, use only dummy IDs to compare the wrapped and stock model, self replacement, and final-only versus full projection. Use a maximum all-vocabulary logit error of `1e-5`. A self replacement writes the original selected vector back unchanged. In the ablated output, require exact zero for the selected vector and exact equality for untouched vectors and attention weights.

On the scientific data, repeat no-op checks for all 2,048 split prompts. Verify exact byte equality for every four-cell family, fixed sequence length and readout, all token exclusions, absence of subtoken overlap in mismatch cells, and absence of future attention. Save every control error. Any control failure stops the run and is retained.

Require a full locked Git commit before scientific execution. Check plan and runner bytes against it at start and finish. Refuse a nonempty output directory. Record package versions, model hashes, run times, dtype, thread count, input manifest, and hashes of every saved result. An independent reviewer must check the plan and runner before the lock and audit the result arithmetic and intervention afterward.

## Contribution and limits

Retokenization, word recovery, and copying heads are established topics. This experiment joins them in one small test with byte-identical prompts, equal token counts, disjoint source/query token encodings, fixed binding controls, and a head selected from earlier evidence. It is not a claim to have found a new detokenization mechanism.

Failure can have several meanings. The model may fail to combine pieces, the task may not trigger a copying strategy, or other components may carry the result. The chosen intervention cannot separate all such explanations. A positive result shows a local causal contribution under this construction. It does not show complete retokenization invariance, general reading ability, or a universal circuit.
