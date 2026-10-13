# Copy routing in a small language model: analysis plan

Date: 13 October 2026. Status: plan before scientific results. The remote commit that contains this plan and the runner will be recorded before the experiment starts. A GitHub plan lock is not a formal external registration.

## Question and scope

When a short text contains two key–value pairs, does Pythia-14M use the final key to select the value that followed it? Can changes to one attention head separate the route to that value from the value information that the head reads?

Induction heads and their routing and copying paths are established ideas [R1, R2]. Prior work also tests recency, conflicting episodes, and attention ablation [R3, R4]. This study is a small replication with a fixed checkpoint and a controlled combination of route and value changes. It does not propose a new copying mechanism. It does not test general intelligence or a complete language circuit.

The competing explanations are: (1) a head reads the value bound to the final key; (2) the model prefers the recent value; (3) key–word associations explain the preference without use of the supplied binding; or (4) the task signal is too weak to support a useful mechanism claim. The test retains all four possible outcomes.

## Model and resources

Use `EleutherAI/pythia-14m`, revision `cf967c0a9a04383db6f7b1108d86b2962634b4ac`, in evaluation mode, with float32 CPU calculations and eager attention. Use the model and tokenizer from the same revision. Save their file hashes and package versions. The model has six layers and four heads per layer. The current model address changed in February 2026 [M1]. Do not substitute an older file under the same name.

No model training, paid API, or extra model is planned. Use all 24 heads. Batch inputs by length. A run can stop for a technical error or the account reserve rule. Do not stop because an effect is small, has the wrong sign, or varies across token groups. Preserve failures and partial results.

## Fixed inputs

Build a pool from the fixed tokenizer before any model output is inspected. Eligible token IDs are at least 1000. Their decoded text must match exactly one space followed by 3–8 lower-case ASCII letters. Encoding that text with no added special tokens must return the same single ID. Sort the eligible IDs; take the first 128 as key/value words and the next 64 as filler words. Save the complete pool and decoded words.

Shuffle the 128 key/value IDs with NumPy `default_rng(20261013)`. Divide them into 32 disjoint groups of four, called A, B, C, and D. Groups 0–15 are calibration groups. Groups 16–31 are test groups. No test word is a calibration key or value. The filler pool is shared, so this is not a test on wholly unseen vocabulary.

For each group, use both bindings: A→B and C→D, or A→D and C→B. Use both pair orders, so each key–value pair can occur first. Use gap lengths 0, 4, and 12. The token sequence is:

`[end-of-text, key1, value1, gap, key2, value2, gap, final key]`.

The two gaps use parts of one 24-item filler permutation for each group, from NumPy `default_rng(20261014 + group)`. The first gap uses `filler[:gap]`; the second uses `filler[12:12+gap]`. These parts are disjoint. The same fillers apply to both bindings, orders, and final keys. The final key is A or C. Both query versions have exactly the same prefix. All IDs are supplied directly to the model. This is a constructed token task, not a sample of ordinary prose.

There are 32 × 2 bindings × 2 orders × 3 gaps × 2 final keys = 768 baseline prompts. Every group contains 24 prompts. Test interventions cover all 384 test prompts, all 24 heads, and all three change conditions. No prompt is removed for low model accuracy.

The swapped binding is essential. A fixed association between A and B does not count as use of the supplied binding. Pair order and final key are also balanced. These controls allow binding use and recency to be measured separately.

## Scores and head choice

Let `m = logit(B) − logit(D)`. Let `s = +1` if the prompt binds the final key to B, and `s = −1` if it binds it to D. The answer score is `s × m`, in logit units. The mean over the full balanced set is the binding-sensitive score. Report it for calibration and test data and for every gap. Save each token group's binding-sensitive score and recency score separately. Also report the fraction of prompts with a positive answer score and the mean score of the recent value against the earlier value. This binary comparison does not mean that either answer is the model's top token in the full vocabulary. Report that full-vocabulary top-token rate separately.

At the last query position, let `a_B` and `a_D` be one head's attention weights on the positions that contain B and D. The head's binding-sensitive attention score is the calibration mean of `s × (a_B − a_D)`. Select exactly one head with the largest score. Break exact ties by lower layer, then lower head index. Record all 24 calibration scores and the chosen head before test interventions. Selection uses attention only, not intervention outcomes. A low or negative maximum does not trigger a new choice or a different task.

The selected head is the sole primary head. Report test results for all heads as descriptive results. Do not use the test set to select a second primary head.

## Interventions

Each change affects one head at the final query position. All other head outputs follow the receiver computation. Upstream layers remain unchanged. Downstream layers run normally. The paired donor has the same prefix and the other final key. Thus all past value vectors at a given layer are equal between the two unmodified runs; the current-token vector can differ.

1. **Route change.** Copy the donor attention distribution over past positions. Rescale it to the receiver's original total attention on past positions. Keep the receiver's self-attention weight fixed. If donor past mass is at most `1e-12`, leave the row unchanged and count the case. Use receiver value vectors throughout. This changes where the head reads from the past without changing how much total attention it gives the past. The row is artificial; it is not the full donor row.
2. **Value change.** Exchange the receiver's value vectors at the B and D positions in a temporary value matrix for this head's final-query calculation. Keep its attention row fixed. Replace only that head's final-query output. Do not apply the temporary matrix to earlier queries. This exchanges contextual vectors. They can contain position and context information as well as token information.
3. **Joint change.** Apply both changes to the same head.

The route change can affect many past positions. Thus joint restoration is a testable pattern, not an algebraic identity. Neither path is a pure semantic variable. A successful output change does not prove that these paths form the complete natural mechanism [R5].

For each head and prompt, save baseline, route, value, and joint logits for B and D, plus their answer scores. Save the full baseline and paired-donor output references needed for the stated scores, including target probability, competitor probability, and top-token ID. Do not use a normalized recovery ratio: a small baseline contrast would make it unstable.

## Primary comparisons

For each test prompt, define the following effects:

- Route drop: `s × (m_baseline − m_route)`.
- Value drop: `s × (m_baseline − m_value)`.
- Joint rescue relative to the route change: `s × (m_joint − m_route)`.

The primary pattern for the selected head is that all three means are positive. Report 95% percentile intervals from 10,000 bootstrap resamples of the 16 complete test token groups, using NumPy seed 20261015. Keep all binding, order, gap, and query variants together within each resampled group. Support for the stated pattern requires all three lower interval bounds to exceed zero, a positive lower bound for the test binding-sensitive baseline score, and a positive lower bound for the selected head's test attention score. This is a narrow conjunction for one preselected head. A failure of the pattern does not rule out copying elsewhere in the model.

Also report the joint change relative to baseline, joint rescue relative to the value change, each separate gap result, each order result, and the control effects. These are secondary. Do not describe a rescue relative to the route change as a rescue relative to both separate changes. Do not describe the chosen head as the only copying head, or claim a population rate of copying heads. Bootstrap intervals measure variation in this small constructed pool, not uncertainty over all text or all models.

## Implementation checks

Before accepting any result:

- Compare the unchanged custom attention path with stock eager attention on the first prompt in each group at each gap. Maximum absolute logit error must be at most `1e-5`.
- Confirm that the edited attention row sums to one within `2e-6` and that its self weight is unchanged. Confirm that all other query rows and head rows stay exact.
- Confirm that only the selected value vectors enter the final-query change; other query outputs do not change inside that attention module.
- Preserve the stock batched attention matrix product shape for edited calculations. Use only the selected final-query result. Check dummy no-op inputs before the plan lock; do not inspect study effects in that check.
- Confirm that a same-query route donor is a no-op within `1e-5` logit units, and that exchanging the two value vectors twice is a no-op within `1e-5`.
- Confirm that a full donor attention row combined with all donor value vectors reproduces the donor's pre-output-projection head vector within `2e-6`. This is a local arithmetic control, not a claim that one head reproduces the donor model output.
- Confirm that the paired unmodified runs have equal value vectors at all past positions within `2e-6`.
- Retain zero-past-mass cases and record the maximum errors and counts. Stop on a failed required check, save the failure, and label any repair before rerunning.

Do not add a second model, new token pool, new head rule, or new primary score after seeing results. A useful further study can be proposed in the paper as future work.

## Outputs and review

Save the input generator, runner, dependency pins, plan hash, model-file hashes, token manifest, prompt records, all per-prompt intervention rows, all head summaries, selected-head record, bootstrap intervals, and control receipt. Include a compact figure of the selected head and the full head table. An independent AI reviewer will check the plan before results and audit calculations after results. State clearly that AI review is not human peer review.

The paper will explain the question, baseline ability, all primary effects, controls, close prior work, and limits in plain English. A negative or mixed result is a valid outcome. A local run is not publication; verify the final GitHub commit and files.
