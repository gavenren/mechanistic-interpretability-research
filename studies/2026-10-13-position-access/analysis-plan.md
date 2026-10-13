# Position spacing and access to added tokens

Status: plan written before scientific model output. Date: 13 October 2026. A public commit must contain this plan and its runner before the experiment starts. The runner checks their exact bytes against that commit. This is a public record of the plan, not an independent registration. Authors and reviewers are AI systems. No human review has occurred.

## Question and fixed claim

Does blocking access to added tokens raise a binding score more than compressing their rotary position coordinates? The fixed directional hypothesis is that blocking access gives the larger mean increase at gap 128. Negative and mixed results will be reported.

The experiment crosses two model interventions. They change position coordinates and attention access while keeping the full token sequence fixed. The access effect includes all computation caused by reading the added tokens. It is not a pure effect of the softmax denominator. The coordinate effect includes all later changes in contextual features. The two effects can interact.

This is one bounded test in a trained model. Position remapping, retrieval heads, positional bias, and phase interventions are known methods. The proposed contribution is the crossed test with two exact short-input controls. See [sources.md](sources.md) for close prior work and limits on novelty.

## Model and numerical condition

Use `EleutherAI/pythia-14m` at immutable revision `cf967c0a9a04383db6f7b1108d86b2962634b4ac`. The runner pins SHA-256 values for the model, configuration, tokenizer, and model card. The model has six blocks, four heads per block, width 128, and parallel attention and MLP branches. Use evaluation mode, no gradients, eager attention, CPU only, and two CPU threads. Do not train or change weights.

Use float64 parameters and arithmetic, including attention softmax. Keep the stock rotary calculation, which computes positions, frequencies, sine, and cosine in float32 and then casts them to the model dtype. Call this **double arithmetic with stock float32 rotary tables**, not a fully float64 forward pass. Fractional coordinates are rounded by that same rotary calculation. The reference computation has the same double softmax. The difference from stock inference is explicit.

Pin package versions in requirements.txt and check them at startup. Use batches of four equal-length inputs. Read only the final position through the vocabulary output layer. Do not retain full attention matrices. Set a 1,200-second wall-clock limit, including load, token checks, controls, and summaries after the output directory is created. Check the limit between batches and before summary completion. Retain partial files on any failure.

## Fixed inputs

Scan tokenizer IDs in ascending order, starting at 1000. An eligible token decodes to one leading space and 3–8 lowercase ASCII letters, and that string must encode back to that one ID. Save the first 576 eligible IDs. Entries 0–191 were the pool for the earlier copy study; do not use them here. Shuffle entries 192–319 with NumPy PCG64 seed `2026101401`. Reshape them into 32 groups of four distinct IDs, called A, B, C, and D. Entries 320–575 form the filler pool.

For group k, permute the 256 filler IDs with PCG64 seed `2026101402 + k`. Keep the first 128 for the first gap and the last 128 for the second gap. All cases in a group use the same filler order. The alternate-filler control rotates this full permutation by one position before taking the slices. Thus every filler identity changes without changing the core input.

For each group, include both bindings, both pair orders, and both queries:

- Bindings: A→B/C→D, and A→D/C→B.
- Pair orders: A pair first, and C pair first.
- Final query: A, and C.

There are 256 core cases. Each has gaps 0, 32, and 128. The complete input is `[BOS, key1, value1, gap1, key2, value2, gap2, query]`. BOS uses the tokenizer EOS ID. Each gap takes its first g assigned filler IDs. Sequence lengths are 6, 70, and 262. There are 768 complete inputs. Target is the value bound to the query. Other is the remaining value. No case is removed based on model ability. No fitting or head search uses these cases.

Save the exact token IDs, core positions, filler positions, and both position-coordinate arrays before model output. Tokenizer-only checks of this ledger are allowed before the lock. Model preflight uses separate dummy IDs, never these study inputs.

## Four conditions on every input

Let S[p,a] denote the correct-value-minus-other-value logit margin. p=0 means compressed positions; p=1 means native positions. a=0 means blocked filler access; a=1 means readable filler positions.

| Condition | Coordinates | Filler key access |
| --- | --- | --- |
| S[0,0] | Compressed | Blocked |
| S[1,0] | Native | Blocked |
| S[0,1] | Compressed | Readable |
| S[1,1] | Native | Readable |

Native coordinates equal physical token indices. For compressed coordinates, the six core tokens receive 0, 1, 2, 3, 4, 5. In gap 1, filler j receives `2 + j/(g+1)`; in gap 2 it receives `4 + j/(g+1)`, for j=1..g. These artificial coordinates preserve order but compress the span between core positions. They are a diagnostic, not a proposed production method.

Causal masking always uses physical sequence indices, never the rotary coordinates. Blocked access adds negative infinity to every filler key column, for every query, head, and block. BOS always remains available. Filler computation remains in the same tensor. No token can read a filler at any layer. Layer normalization and MLPs act separately at each position in this model.

## Sole primary outcome

At gap 128, define access repair as S[1,0]−S[1,1] and position repair as S[0,1]−S[1,1]. The **sole primary contrast is their difference, S[1,0]−S[0,1]**. It is in logit units. Average the eight balanced cases within each token group, then average the 32 group values. No head ablation enters this primary result.

Use 10,000 bootstrap resamples of the 32 complete token groups with PCG64 seed `2026101403`. Each draw samples 32 group indices with replacement. Use NumPy's default linear 2.5th and 97.5th percentiles. The directional hypothesis is supported on this fixed benchmark only if the lower bound is greater than `1e-8`. An upper bound below `-1e-8` supports the reverse direction. Other outcomes are inconclusive under this rule. Values with absolute magnitude at most `1e-8` are numerically negligible. The interval is conditional on the checkpoint, prompt design, and selected token pool; the groups are not a random sample of language use.

## Secondary outcomes

Report all four conditions at each gap, both repair values, and the interaction `S[1,1]−S[1,0]−S[0,1]+S[0,0]`. Report group values, target-versus-other preference, correct full-vocabulary top-1, and the difference between native-readable gap 128 and gap 0. If this last difference is not negative, do not claim that the standard long input caused mean score degradation. Do not force a percentage decomposition of the long–short difference.

Preselect zero-based block 3, head 1 from the completed copy-routing study. For every condition, zero only this head's final-query output vector immediately before output projection. Use the same zero operator in every cell. It changes neither earlier query outputs nor other heads at this local site. Downstream changes are allowed. Report intact minus ablated binding score, final-query attention to target value, other value, BOS, all filler positions, and the final query itself. No other head is selected. These outputs are secondary and do not become a second primary test. Zero ablation is an artificial intervention. It does not identify all causal routes.

## Required controls and fixed tolerances

Use maximum absolute error `1e-8` for all-vocabulary final logits and `1e-10` for all core-position post-block states. Use `1e-12` for attention row sums. Masked attention weights, forbidden future weights, and local output changes outside the edited head/position must be exactly zero. These tolerances are fixed before dummy model checks. Do not raise them after a failure.

1. On every input, native-readable execution must match a reference pass that uses the model's default causal mask and native positions. Both use double softmax. Check final logits and all six core positions after each block, for intact and ablated versions.
2. In both blocked conditions, compare the long input with its actual six-token core input. For compressed positions, the short input uses coordinates 0..5. For native positions, it uses `[0,1,2,3+g,4+g,5+2g]`. Compare all-vocabulary final logits and all core states after each block, for intact and ablated versions.
3. In both blocked conditions, run the alternate filler identities. Compare the same logits and core states, for intact and ablated versions.
4. At gap zero, compare every cell with the native-readable cell, for intact and ablated versions.
5. In all four cells, a no-op replacement copies the selected final head output back without changing it. It must match the intact logits and core states.
6. On every attention call, verify row sums, causal zeros, and blocked filler zeros. On ablation calls, verify the target vector is zero and all other local head outputs remain unchanged. Record maxima for each model forward. Record the fixed attention-site call count.

Dummy preflight uses four distinct core inputs `[0,10+k,20+k,30+k,40+k,10+k]`, for k=0..3, and filler IDs 101..356. The alternate filler list rotates these IDs by one position. It uses gaps 0, 4, and 128, with four examples in every batch. This tests the real batch size, the same operators and controls, and the maximum sequence length. There are 66 dummy model calls, 264 forward examples, and 96 dummy scored rows. Save all control results, timing, package versions, and plan/runner hashes. No scientific model outputs are allowed in preflight.

The full run retains 6,144 scored condition/intervention examples and 16,896 total model-forward examples, with batch size four. All 768 inputs and all fixed scored cells remain in the record. The controls are collected without adjusting the plan. A failed control prevents a completed scientific status and any causal interpretation. Preserve failures, then investigate them in a separately labelled audit. Do not silently drop failed cases, overwrite a run, or change the tolerance.

## Outputs and checks

Save the input ledger, per-cell target/other logits and top-1 token, signed margins, secondary head effects and attention masses, per-group contrasts, group bootstrap summary, all control maxima, package/model receipts, and exact file hashes. Flush row files during the run. Do not publish home paths or model-cache paths. Do not rehost the model weights.

An independent AI audit must reconstruct the input masks and coordinates, result arithmetic, and primary group bootstrap without importing the runner. It should replay the model checks with a separate implementation. Report its scope and any failure. A separate AI review is not human peer review. Publish the full null or negative result if controls pass. A failed experiment is a research record, not support for the hypothesis.

## Interpretation limits

Readable fillers can change attention normalization, value mixing, and later token representations. Blocking them removes all these paths together. Coordinate changes also alter later representations. The factorial operators do not identify independent natural mechanisms or a unique causal percentage. The compressed-readable geometry is artificial. Reusing a previously selected head does not prove that it has the same role on these new words and lengths. The model can fail the task; all ability results remain beside the causal scores. Results concern one small checkpoint, one constructed binding task, and at most 262 tokens, within its nominal context range.
