# Fixed plan: donor choice and hidden-state support

Date: 22 October 2026. Status: plan before results. This is an exact constructed-model study. It has no trained network and no language-model evidence.

## Question and contribution

Can models that agree on every ordinary hidden state give different ablation answers because the patched state is outside that set? How does donor conditioning alter the question being tested?

The contribution is a small exact benchmark with all inputs and all replacement outcomes. The broad issue is known. [Zhang and Nanda](https://arxiv.org/abs/2309.16042v2) studied sensitivity to patching methods. [Heimersheim and Nanda](https://arxiv.org/abs/2404.15255v1) explain why donor selection determines the question. This study does not claim that concern as new.

It differs from the repository's basis-transport study, which changes coordinates while transporting an edit. Here, the coordinates stay fixed. The ordinary model behavior stays fixed while the model's response away from the ordinary state set changes. It also differs from the inactive-variable audit, which trains and selects neuron sets.

## Fixed construction

Let the signal `s` and nuisance `n` be independent, uniform signs. There are exactly four inputs. The hidden state is `h=(2+s,n,s*n)`. The label is `s`. Define the output score as `F_beta(h)=(h1-2)+beta*(h3-(h1-2)*h2)`. The decision is positive when the score is at least zero. Thus a zero score has a fixed positive tie rule.

Use beta values 0, 1/2, 1, 2, 4, and 8. This is a fixed family, not a fitted parameter search. Every beta must give the score `s` on every ordinary state. A complete ordinary state must have `h1` equal to 1 or 3, `h2` equal to -1 or 1, and `h3=(h1-2)*h2`.

Compare seven replacements. Zero sets `h1=0`. Mean sets `h1=2`, the exact marginal mean. Marginal donors draw uniformly from all four inputs and copy only `h1`. Nuisance-matched donors draw from inputs with the same `n`. Label-matched donors draw from inputs with the same `s`. Opposite-label donors draw from inputs with signal `-s`. The coherent opposite-label edit changes both `h1` and `h3` to their values for `(-s,n)`.

For each input and donor, record the exact probability, hidden state, score, answer, and membership in the ordinary state set. Report original-label accuracy, donor-label accuracy where defined, off-support rate, and mean absolute score change. Label matching has access to the target and is a control. It is not a fair method for removing the target information.

## Analysis and checks

Use exact rational arithmetic. Check scores against an independently expanded polynomial. Replay every saved raw row. Check all four intact states for all six settings. The fixed formulas permit a proof of intact equivalence and of exact behavior under the coherent edit. Report all settings and methods, including identical outcomes.

Success means that all implementation controls and saved-row checks pass. It does not require a favorable effect size. Do not use p-values or confidence intervals: the finite constructed domain is fully enumerated. Do not interpret its case counts as independent trained models.

## Resources and stop rule

Use one local Python process and exact arithmetic. No training, GPU, downloads, or paid API is needed. The cap is 300 elapsed seconds, including checks. The three intervention studies together may use at most 900 seconds. The root runs scientific work in sequence. The shared usage guard checks the private, atomically replaced permit and the stage deadline. Stop if the guard fails or any exact check fails. Preserve the failure record. Do not change the grid after seeing results.

The runner verifies its code, this plan, source record, and shared guard against the public lock commit. Save raw CSV, summary CSV, metrics, time, plan hash, and output hashes. Require a separate AI review. No human peer review is recorded. A later learned-model extension needs a new plan.
