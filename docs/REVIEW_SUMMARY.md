# What the 24 studies show

**One-page summary | Updated 22 October 2026**

Mechanistic interpretability studies how a model produces its answers. This review covers **24 completed studies**: fifteen earlier studies and nine new papers. The earlier work includes small trained networks and one small pretrained language model. The new work uses mathematical analysis, constructed examples, and simulation. **An output change after an internal edit does not, by itself, explain how the model solves a task.**

## Main findings

- **Check task ability first.** An internal part can affect the output even when the model rarely answers correctly. Passing the task is necessary, but does not prove the proposed explanation.
- **Define the complete edit.** Replacement values can create states absent from normal use. Two edits can interfere. Different codes for the same state can give different edit results. Shared rescaling can spread a local change. Tests must include these effects and check unrelated behavior.
- **Test how parts work together.** Tests of one part at a time can leave different mechanisms unresolved. A search can retain six parts when one is sufficient, even though no further single removal preserves the answer.
- **Match the score to the claim.** A larger effect does not mean a better answer. Useful effects can cancel in an average. Scores and probabilities can rank edits differently. A lasting effect does not establish useful memory.
- **Keep selection and validation separate.** Searching many noisy results and testing the winner on the same data can produce false evidence. Use separate validation data or a valid correction for the selection process.
- **Retain weak results and failed checks.** Some proposed improvements gave no clear or consistent benefit. Numerical uncertainty is not the same as a wrong answer. Checking saved weights does not repeat training.

## What is new, and what the results do not prove

The contributions are specific experiments, exact mathematical conditions, complete small-case counts, and code and results that others can check. They do not establish 24 new general principles. This summary adds no experiment.

Models and tasks are limited. Several studies share models or data, so they are not independent confirmations. None of the nine new papers tests a trained language model. Their exact counts and simulated percentages do not measure failure rates in real AI systems. All work is AI-authored. Separate AI checks are not human peer review; none is recorded.

## The tenth study in the new batch is UNFINISHED

It is **not counted among the 24 completed studies**. Six graph models passed their fixed task tests, but four were development models. Only **two of four required final models** are complete. The third has partial weights; the fourth and the other task branches did not start. Work stopped under the usage-reserve rule. All graph states were used in training, so perfect scores do not establish performance on unseen graphs or explain the internal method.

## Next steps

Check the budget and publish a continuation plan before completing the unfinished task tests. Then test complete explanations in capable models. Check intended and unwanted effects, new conditions, later behavior, numerical error, and cost. Seek independent replication and human review.

**Source:** [Full review and study references](https://github.com/gavenren/mechanistic-interpretability-research/blob/7df068b5e1f14d7c4aef475df3a5707f8ea8fe64/docs/REVIEW_PAPER.md). [Print version](../output/pdf/REVIEW_SUMMARY.pdf).
