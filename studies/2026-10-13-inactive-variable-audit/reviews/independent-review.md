# Independent calculation review

**Date:** 13 October 2026.  
**Reviewer:** A separate AI agent in the same research session. This is not external or human peer review.  
**Status:** Final calculation and paper review passed.

## Scope and method

The reviewer read the frozen analysis plan and study code. The reviewer then wrote a separate NumPy implementation in [independent_audit.py](../code/independent_audit.py). It does not import the study runner or train models. It reads saved weights, reconstructs the input encoding, and calculates the hidden layers, patches, output probabilities, and total variation distances.

The audit checked all eight model seeds. It recalculated every selected non-random mask, across all mask sizes and both donor regimes, on every primary pair and every same-a control pair. This includes every primary selected-mask outcome. It also checked the complete candidate tables for selection, grouping, and summary consistency. It did not repeat forward calculations for every unselected or random-only mask.

## Results

All checks passed:

- The plan and runner match lock commit `ff9d78c15aa6e97b4819e18f4253b60fd165b50c`.
- All 35 scientific files in the saved manifest match their hashes.
- The 192 input states, labels, counterfactual inputs, and 27,648 pair identities are correct.
- The exact hash-based split has 13,824 test pairs and no shared pair identity between discovery and test.
- The matched sensitivity test uses 286 pairs already assigned to the test pool. It does not reuse discovery pairs.
- All 9,536 candidate-table rows and 928 selected-table rows have consistent grouping and summaries.
- The 96 non-random selection configurations follow the declared objective and tie rule. Random-mask choices follow their fixed seeds.
- Saved intact predictions and selected-mask metrics agree with the independent forward calculations.
- Same-a controls, empty baselines, primary seed outcomes, and bootstrap summaries agree.

The maximum absolute difference across 2,692,699 checked numeric values was **2.78 × 10⁻¹⁶**. Each model has intact accuracy 1.0 on the 192 training states. The primary selection rules choose the same three-neuron mask in every seed. Their primary held-out difference is therefore exactly zero in all eight seeds.

## Interpretation checks

The primary prediction is not supported in this run. A zero-width bootstrap interval follows from eight zero differences. It must not be read as proof that the two selection rules are equivalent in other settings.

All ordinary task states were training inputs. The test holds out pair identities, while individual receivers and donors can appear in other discovery pairs. It tests new interventions on known states. It does not test generalization to unseen inputs or tasks. The eight seeds should receive descriptive emphasis.

The primary target is the intact model's counterfactual distribution. Inactive task labels remain fixed, but the intact distribution can change. The saved inactive baseline and same-a controls correctly distinguish these cases. Mask failure need not mean that the variable has no internal representation; the coordinate-mask family may be too restricted.

The review found no blocking calculation error within this scope. Its source review confirms substantial prior work on causal abstraction, context-specific diagnosis, and objective misranking. The contribution must remain the bounded experiment and its complete record.

## Final paper check

The reviewer read the completed [paper](../paper.md) and the reproduction instructions. All numeric claims agree with the saved tables. This includes the eight rounded seed rows, 65.9% unrestricted task-label accuracy, 80.9% matched task-label accuracy, 47.0% random-mask accuracy, and the same-a control mean of 15.6603% changed answers. The stated same-a range, total variation means, inactive baselines, environment versions, and runtime also agree.

The algebra explaining identical selection is correct. When an active-error minimizer already has no larger inactive error, it also minimizes the maximum of those errors. The paper labels this as elementary reasoning rather than a new theorem. Its discussion of numerical ties is appropriate.

The paper reports the negative result directly. It states that training used all ordinary inputs, distinguishes a versioned plan from registry submission, and restricts the contribution to this experiment. The matched-donor comparison changes both the donor pool and selected mask; the paper states this limit. The literature claims match the primary sources checked by the reviewer. No scientific revision was required after this final reading.

## Reproduction

With NumPy installed, run:

```sh
python code/independent_audit.py
```

The script prints a JSON verification receipt and does not alter study data. This review does not independently repeat training or establish unique causal mechanisms.
