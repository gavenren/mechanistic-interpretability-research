# How to read these studies

These studies test how a neural network makes a result. They change part of a model and measure what happens. The papers use small models and fixed tasks. They do not establish how all AI systems work.

This guide explains the terms used in the papers. It is a reading aid, not an additional experiment.

## First check what the model can do

Suppose a prompt gives two pairs: `apple → red` and `pear → green`. It then asks for the value after `apple`. A model can give `red` a higher score than `green` while it gives some other word an even higher score.

The papers therefore report two different rates:

- **Two-answer accuracy:** Does the correct supplied answer score higher than the other supplied answer?
- **Full-vocabulary top-1 accuracy:** Is the correct answer the highest-scoring token among all tokens the model can produce?

The second test is harder. A high score on the first test does not establish a high score on the second. Both measures concern the next token. They do not measure the quality of a complete response.

## Then check what changed inside the model

An **activation** is a number, or a set of numbers, calculated inside a model for a given input. An **activation patch** replaces some of those numbers. A donor input supplies the replacement values. An **ablation** removes a selected component's contribution, often by setting it to zero.

An **attention head** uses earlier token information to help calculate a later internal state. A **token** is one item in the model's input or output code. A token can represent a word, a word part, or punctuation. The same text can have more than one valid token sequence.

These changes can show that a component affects a result under a specified test. A large effect does not, by itself, show that the component has one clear meaning. Other components can share the work. An artificial replacement can also put the model in a state that normal inputs do not produce.

## Read the size and meaning of the score

A **logit** is a model score before conversion to a probability. A **logit margin** is the score of one answer minus the score of another. A positive margin means the first answer has the higher score. It is not the probability that the answer is correct.

Some papers use **total variation distance**, or **TV**. It measures how much two output probability distributions differ. Zero means that the distributions are equal. A larger value means more change. It does not say which output is better.

A small numerical difference can be repeatable without being useful. For example, a change of 0.00003 in probability is 0.003 percentage points. The training-order paper reports both probability changes and logit changes because confidence can make these measures tell different parts of the story.

## Check the comparison before reading the conclusion

Each analysis plan fixes its main comparison before scientific results are collected. Other measurements help explain that comparison. They do not replace it after the results are known.

The position study has four conditions. Its main contrast compares two individual changes. Both individual changes can perform poorly, even when one performs better than the other. The full four-condition table is needed to see their interaction.

A **bootstrap interval** describes variation under repeated resampling of the declared groups or model seeds. The unit matters. Thousands of related prompts are not thousands of independent model tests. These intervals do not include every possible source of error. They do not establish a result for other models, tasks, or populations.

## Check accuracy on the part of the task under study

A model can perform well overall and poorly on a small, important group. In the [matching study](../studies/2026-10-13-hall-obstruction/paper.md), overall accuracy was near 99%. Accuracy on impossible graphs with no isolated vertex was only 84% to 89%. All four models failed the fixed ability requirements. A large internal effect did not repair that failure.

An edit can also affect the intended group more than a control group while changing both groups substantially. The matching study's selected edits raised false acceptance of a different type of impossible graph from 6.34% to 84.72%. Read the separate control outcomes beside the combined score.

## Keep stability separate from correctness

A stable measurement changes little when the input changes slightly. It can still describe a weak model. In the [digit study](../studies/2026-10-13-shift-ablation/paper.md), filtering made channel-removal effects more stable under small image shifts. But accuracy on the original test images was only 15% to 36%. The result does not establish good explanations of reliable digit recognition.

The digit study also keeps three stages separate: an original failed numerical check, a fixed diagnostic, and a later evaluation with higher precision. The later check does not turn the original failure into a pass. The paper includes all three stages in one study.

## State exactly what an edit removes

A spike can send a message and reset its own stored voltage. Removing only the message differs from removing both. In the [spike study](../studies/2026-10-13-spike-reset-queries/paper.md), matching the explanation to the edit reduced average selection error. One model still lost under one deletion meaning. Many predicted effect signs were wrong.

A positive average can contain losses. Read each direction of a comparison and the error that remains. A good task score and a better selection rule do not establish a complete explanation.

## Check what a coarse output measure leaves out

The [flow study](../studies/2026-10-13-flow-unit-ranking/paper.md) counts generated points in four quadrants. A point can move within one quadrant without changing this count. Two points can cross in opposite directions and leave the total counts unchanged. Thus quadrant-frequency TV does not measure every change in the generated distribution.

All four models passed that coarse frequency check but failed the density-fit check. A good score on one measure does not establish a good fit on another. A larger effect after a unit removal also does not mean better generated samples.

## Separate an uncertain effect from a wrong effect

Some models repeat a calculation until the state changes very little. That stopping rule can leave enough error to affect a small neuron-removal measurement. In the [equilibrium study](../studies/2026-10-13-equilibrium-sign-audit/paper.md), the first error bounds did not establish the sign of 18.66% of nonzero effects, averaged across four models.

That percentage is not an error rate. The bounds were conservative. Further calculation proved that five initial signs were wrong. Every tested effect then had either a proven sign or a magnitude bounded by 0.0000001 output-margin units. The small-magnitude result did not prove exact zero.

These bounds concern the exact real equilibrium defined by the saved model weights. They are not confidence intervals from random sampling. Their validity depends on the stated arithmetic assumptions. They do not establish what a neuron means.

## Check the evidence record

Each completed study links its fixed plan, source record, code, results, and calculation review. The review can check the arithmetic and repeat model interventions. Its record states the work that was actually repeated. A review of saved models is different from repeating their training.

The studies are AI-authored. A separate AI review is not human peer review. The papers retain failed checks and negative results. Read the limits with the result, especially when a test uses an artificial task or a small model.

Return to the [paper list](../README.md).
