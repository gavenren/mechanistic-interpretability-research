# Opposite output directions can hide the same useful task edit

**22 October 2026 · Formal and computational research note · AI-authored**

## Main finding

A useful task edit can have zero average signed effect when different contexts require opposite output directions. In our fixed panel, ten constructed classifiers answered every ordinary task input correctly. In six of these models, the absolute average effect ranked a nuisance edit above the useful task edit when the two contexts had equal weight.

The useful edit still gave every requested answer. The nuisance edit changed the scalar output but changed no predicted answer in those ten capable models. Thus the average signed score answered a different question from task success.

This is an exact calculation on small specified classifiers. No model was trained. It does not estimate how often this pattern occurs in language models.

## Question and prior work

Cancellation across inputs is established. Kramár and colleagues distinguish the magnitude of an average effect from the average of individual magnitudes. Their AtP* paper also reports cancellation across prompt distributions. We do not claim that this phenomenon is new. [Kramár et al., version 1, Appendix B.2](https://arxiv.org/html/2403.00745v1).

Activation-patching work also shows that methodological choices affect localization results. [Zhang and Nanda, version 2](https://arxiv.org/abs/2309.16042v2).

Our addition is a complete finite construction with a known context rule, a nuisance control, and an explicit task-ability check. It separates cancellation from an incorrect model. It also shows the extra information needed to align effects by the requested task direction. All scores remain on one scalar scale; this is not a comparison of logits with probabilities.

## Model and fixed test

Three variables take values −1 or +1. Context c determines how task variable t maps to the requested answer. Variable n is a nuisance: it does not change that answer.

The correct answer is `c*t`. The model output is

`L = alpha*c*t + beta*n`.

The prediction is +1 when L is nonnegative and −1 otherwise. We use alpha from one to four and beta from zero to four, giving twenty fixed models. We evaluate all eight ordinary inputs for each model. Only a model that answers all eight correctly passes task ability.

The semantic edit changes t from −1 to +1 while preserving c and n. Its required new answer is c. The nuisance edit changes n from −1 to +1 while preserving c and t. Its required answer stays unchanged. Each edit has four input cases.

The [fixed plan](plan.md) gives context +1 probability p, with p equal to 0, 1/4, 1/2, 3/4, or 1. The other unchanged variable is uniform. We save individual effects, context means, aggregate scores, predictions, and task correctness for every case. The panel contains 160 ordinary states, 160 individual edits, and 100 model/mixture combinations.

## Derivation

For the semantic edit, the scalar output change is

`delta = 2*alpha*c`.

Therefore its signed average is `2*alpha*(2*p-1)`. When p is one half, positive and negative changes cancel exactly. Its average absolute change is `2*alpha` for every mixture.

If the task rule is known, multiply each effect by c before averaging. This makes both required directions positive, and the aligned average is also `2*alpha`. This operation uses the known meaning of context. It is not an unsupervised method for discovering that meaning.

For the nuisance edit, the scalar change is always `2*beta`. Its signed average is positive whenever beta is positive. At equal context weights, it therefore outranks the semantic edit under the absolute-average score.

The model answers every ordinary input correctly exactly when `alpha>beta`. In that case, the task term dominates the nuisance term in every state. At equality, the fixed positive tie rule makes some states incorrect. Below equality, the nuisance can determine the prediction.

## Results

Every recorded effect matched the stated equations. Exactly ten models met `alpha>beta` and passed all task inputs. Six of these had positive beta and showed the balanced-context ranking reversal. The other four had beta zero, so both absolute-average scores were zero.

For example, alpha=2 and beta=1 gives:

| Measure, with equal context weights | Semantic edit | Nuisance edit |
| --- | ---: | ---: |
| Signed mean effect | 0 | 2 |
| Absolute signed mean | 0 | 2 |
| Mean absolute effect | 4 | 2 |
| Required answer correct after edit | 100% | 100% |
| Predicted answer changed | 100% | 0% |

Changing the answer is required for the semantic edit and unwanted for the nuisance edit. The two percentages therefore have different meanings. A large output change alone does not establish useful semantic control.

## Checks, limits, and use

A separate scalar audit checked every state, transition, prediction, rational mixture weight, formula, and capability decision. Unchanged-value controls gave zero effect. The complete run and audit took 0.14 seconds. Peak process memory was 24,068,096 bytes. The code and plan were public under lock `88e8df7689b65fbeef221423c115cd1b1c198f22`. [Results](results/summary.json); [reproduction instructions](README.md).

These equations do not invalidate signed averages. Such averages correctly measure net movement in one fixed output coordinate. They answer the wrong question only if interpreted as universal task importance. Average magnitude avoids this cancellation but discards direction. Task alignment retains direction only when the required direction is independently known.

The practical next test is to fix context labels and requested answers before model evaluation, then report conditional effects and task correctness beside aggregate scores. The known context rule in this study is strong prior information. Unknown contexts, multi-token outputs, and learned representations remain outside the evidence. The [separate AI source review](PRE_RUN_REVIEW.md) does not replace human peer review, which is not recorded.
