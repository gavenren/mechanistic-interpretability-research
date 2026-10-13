# Proposed study: Do measured patch interactions transfer to new prompts?

Prepared 13 October 2026. No study outcomes have been collected. The plan and runner must be saved in a remote commit before the experiment.

## Question and bounded contribution

Can a small correction, measured on one set of prompts, improve the prediction of a joint intervention on a different set?

Activation patches replace an internal value with a value from another prompt. Researchers sometimes sum the effects of separate patches to estimate a joint effect. Prior work already shows that this sum can fail. This study does not claim to discover interactions. It tests whether a low-cost, fixed correction transfers to new word pairs, new prompt frames, or both, in one small pretrained language model.

The correction is the calibration median of the joint effect minus the sum of the two separate effects. The median is appropriate for mean absolute error (MAE). It is fitted once per layer. No test-set estimate is used.

## Model and feasibility

Use EleutherAI/pythia-14m at Hugging Face revision `cf967c0a9a04383db6f7b1108d86b2962634b4ac`. The model card identifies Apache 2.0 as the model licence. The current checkpoint is the standard-Pile version. The model card says that the model at this URL changed on 27 February 2026; the earlier deduplicated version has a different repository. Pin the revision and each downloaded file hash.

The model has six parallel-residual GPT-NeoX layers, width 128, four attention heads, and a 512-unit MLP. Its safetensors file is 28,143,920 bytes. It is not a chat model.

Dependencies installed in the existing local environment: PyTorch 2.8.0, Transformers 4.57.6, safetensors 0.6.2. Existing NumPy is 2.3.5. No paid service is used. A nine-token shape test took 4.4 ms per forward on two CPU threads. The shape test used only “Hello, this is a short shape test.” No logits or study-prompt outcomes were inspected.

The stock eager attention implementation casts softmax to float32 even when the model uses float64. Thus a nominal double-precision run is not fully double precision. The runner will state the primary dtype and include a separate audit backend with dtype-preserving attention softmax.

## Fixed prompt set

Eight frames, in this order:

0. `The word is {w}. The word is`
1. `The answer is {w}. The answer is`
2. `The label is {w}. The label is`
3. `The code is {w}. The code is`
4. `Remember this word: {w}. The word was`
5. `A card says {w}. The card says`
6. `In this list, the item is {w}. The item is`
7. `The name on the box is {w}. The name on the box is`

Sixteen word pairs, in this order:

| ID | First word | Second word |
| --- | --- | --- |
| 0 | cat | dog |
| 1 | book | chair |
| 2 | apple | bread |
| 3 | rain | snow |
| 4 | red | blue |
| 5 | king | queen |
| 6 | car | train |
| 7 | hand | foot |
| 8 | bird | fish |
| 9 | pen | cup |
| 10 | milk | rice |
| 11 | sun | moon |
| 12 | green | white |
| 13 | boy | girl |
| 14 | bus | boat |
| 15 | head | face |

Use both donor directions for every frame and pair. The donor and receiver have the same frame and differ only in `{w}`. Each target is the single token for a leading space followed by its word. Set `add_special_tokens=False`. Before the lock, verify only tokenization: each target has one token; pair prompts have equal lengths; exactly one token differs; and their final token is identical. If a condition fails, stop and amend the plan before results. Do not silently exclude or replace a word after outcome collection.

There are 128 undirected frame-pair groups and 256 directed pairs. Repeated words and frames create dependence. These are fixed benchmark cases, not a random sample of language.

## Fixed split

- Calibration: frames 0–3, word pairs 0–7. 32 groups, 64 directions.
- New words: frames 0–3, word pairs 8–15. 32 groups, 64 directions.
- New frames: frames 4–7, word pairs 0–7. 32 groups, 64 directions.
- New words and frames: frames 4–7, word pairs 8–15. 32 groups, 64 directions. This is the primary test set.

Keep both directions of a group in its split. The last two sets use deliberately different frames. The new-frames set shares words with calibration. The new-words set shares frames. State this directly. Do not claim transfer to a different language task.

## Patches and scores

Use the final token position. At each layer, save the outputs of the attention branch and the MLP branch on each unmodified prompt. Replace either branch output with its donor value at the final position. Leave all other positions unchanged. Both branches read the same incoming residual in this architecture. The receiver skip residual stays fixed.

Run all four cells: no patch (00), attention only (10), MLP only (01), and both (11). Recompute all later layers. For the last layer, patching both branches is not a full donor-state replacement: the receiver skip residual remains present.

For each cell, let `s` be the donor-word logit minus the receiver-word logit. Record both raw logits and their difference. Do not normalize by a clean/corrupt gap.

- Attention effect: `A = s10 - s00`.
- MLP effect: `M = s01 - s00`.
- Joint effect: `J = s11 - s00`.
- Interaction: `I = J - A - M = s11 - s10 - s01 + s00`.
- Uncorrected prediction: `A + M`.
- Corrected prediction: `A + M + c_layer`, where `c_layer` is the calibration median of `I`.

Only layers 1–5 enter the primary score. Layer 0 is a control. Its MLP sees the same final-token embedding in donor and receiver; therefore its patch should be zero. Its interaction should also be zero, within numerical error. Report all six layers.

## Primary analysis

For each test pair and layer, compute the absolute prediction error for `J`. The primary result is uncorrected MAE minus corrected MAE on the new-words-and-frames set, with equal weight for each of its 64 directions and five analysis layers. Positive values mean the calibration correction helps. Report the two MAEs as well as their difference.

Also report each layer separately, the two other test sets, each frame separately, and each word-pair group with the two directions averaged. Report the full raw tables. Do not treat layers, directions, or repeated frames as independent replicates. Do not use a p-value or a population confidence claim for this fixed set. Report a conditional crossed-bootstrap interval for the primary MAE gain: independently resample the four test frames and eight test word pairs with replacement, retain both directions and all five layers together, and repeat 10,000 times with NumPy seed 20261013. Keep the fitted calibration corrections fixed. Use the 2.5th and 97.5th percentiles. This describes variation from reweighting this small fixed set. It does not cover calibration estimation, other models, or all natural prompts. A negative or near-zero result is a valid outcome.

Secondary descriptive outcomes: median and range of absolute interactions; the fraction of cases whose predicted joint-effect sign is wrong; and the fraction for which the correction reduces absolute error. Mark effects with magnitude at most `1e-6` as zero for sign tables; preserve all raw numbers. No small-effect case is excluded from MAE.

Record intact pairwise word preference and full-vocabulary target top-1 accuracy for every prompt. Include all prompts regardless of these values. These values describe capability; they are not a filter or evidence of human-like reading.

## Numerical checks and independent controls

Primary inference: CPU, evaluation mode, no gradients, float32, eager attention, two CPU threads. Record the package and hardware versions. First collect calibration cells, fit and save the layer medians, then collect held-out cells. Do not revise the estimator after seeing tests.

Precision sensitivity run: repeat all four cells for all 256 directions and all six layers with model weights converted to float64 and an eager attention function that preserves the query dtype in softmax. It must otherwise implement the pinned library function's attention mask, scaling, head mask, and output order. Assert tensor dtypes in the custom function. The stock rotary-position code computes its cosine and sine tables in float32 in both conditions. Keep these shared fixed constants. Thus this is double arithmetic with shared float32 rotary tables, not an entirely float64 forward and not an error bound or ground truth for the float32 program. The checkpoint values stay fixed; the numerical evaluation changes. Save the float64 results separately. Refit calibration medians within this audit and repeat the primary score. Report the maximum and median float32-to-float64 interaction differences, the primary-score change, and sign changes. Do not call a conclusion stable if it changes materially between these runs. Do not use the audit to select the preferred answer.

Save receiver and donor branch outputs and receiver block outputs for all layers. An independent reviewer can inject branch deltas at the whole-block output to confirm the four-cell logits without using the branch-hook implementation. Fix the absolute comparison tolerance at 1e-4 for primary float32 and 1e-9 for double arithmetic with shared rotary tables. Save any failures; do not exclude affected cases.

For the last layer, save the four pre-final-LayerNorm residual vectors, LayerNorm weight, bias and epsilon, and target unembedding rows. Recompute each margin independently in NumPy float64. Compare it with the forward output using a fixed absolute tolerance of `1e-4` for float32 and `1e-9` for float64. If a control fails, stop publication, preserve outputs, and report the error.

As a diagnostic, freeze only the final LayerNorm denominator at the receiver value and recompute the four margins. Mean centering and the remaining map are affine, so this diagnostic has zero interaction within numerical error. It changes the readout; it is not a corrected causal estimate of the original model. The known identity explains why last-layer score interaction alone does not imply dependence between the attention and MLP branches. Cite the prior work. Do not report this identity as a discovery.

## Cost and stopping rule

No training, model selection, prompt screening, or hyperparameter search. One pinned model and one checkpoint. About 9,000 short patch forwards across the primary run and precision audit, plus baselines and independent checks. Expected runtime is minutes on CPU; cap each complete run at 20 minutes. If a dependency, correctness, runtime, or resource check fails, preserve the failure and stop rather than silently reduce the fixed dataset. No paid API or cloud compute.

A correction requires three nonbaseline calibration cells per pair and layer. Report this upfront cost. At test time it still requires the two single patches. Therefore this study tests reuse of measured interaction, not a free substitute for joint patching. Do not assert a practical speedup without accounting for the full calibration cost and amortization point.

## Outputs

Save a locked analysis plan and runner before study outcomes. Then save the prompt/token ledger, model file hashes and revision, package versions, all primary and audit four-cell results, fitted corrections, subgroup summaries, intact behavior, numerical controls, branch/residual vectors, plotting code, figures, and a manifest. The checkpoint can be fetched from its fixed public revision; do not duplicate the external 28 MB model unless needed. Include clear reproduction steps and the upstream licence reference.

Request a separate AI review of implementation and results. State AI assistance and lack of human peer review. Publish one complete paper for this study, not separate papers for split types or diagnostics.

## Closest primary sources checked on 13 October 2026

1. Vaidyanathan et al. (2026), *The Curse of Multiple Mediators: Hidden Interaction Effects in Activation Patching*, arXiv:2606.27510v1. https://arxiv.org/abs/2606.27510v1. Covers mediator interactions, ranking effects, and prompt dependence. This is the closest interaction work. The present study tests a fixed calibration median on held-out words and frames.
2. Zhang and Nanda (2024), *Towards Best Practices of Activation Patching in Language Models: Metrics and Methods*, arXiv:2309.16042. https://arxiv.org/abs/2309.16042. Covers metric and corruption choice. A donor-word logit margin and fixed prompt construction are established choices.
3. Rushing and Nanda (2024), *Explorations of Self-Repair in Language Models*. https://proceedings.mlr.press/v235/rushing24a.html. Appendix B derives final-LayerNorm effects after a last-layer head ablation in a parallel-attention model. The normalization mechanism and frozen-scale diagnostic are not new.
4. Kramár et al. (2024), *AtP*: An efficient and scalable method for localizing LLM behaviour to components*, arXiv:2403.00745. https://arxiv.org/abs/2403.00745. Studies approximation failures. The present correction uses measured factorial effects, not a new gradient estimator.
5. *Float32-Induced Distortion in Activation Patching* (2026). https://www.mdpi.com/2079-3197/14/9/212. Reports cancellation effects and hidden attention casts. This motivates the precision audit; Yash Baligar; Computation 2026, 14(9), 212; published 9 September 2026; DOI 10.3390/computation14090212.
6. Biderman et al. (2023), *Pythia: A Suite for Analyzing Large Language Models Across Training and Scaling*, arXiv:2304.01373v2. https://arxiv.org/abs/2304.01373v2. The 14M model was added after the original suite; use the pinned current model card for its details.
7. EleutherAI, Pythia-14M model card. https://huggingface.co/EleutherAI/pythia-14m/blob/cf967c0a9a04383db6f7b1108d86b2962634b4ac/README.md.

Novelty statement: this is a new fixed experiment and reproducible artifact. The prior-art search is bounded. It does not establish that no earlier study has used the same correction.
