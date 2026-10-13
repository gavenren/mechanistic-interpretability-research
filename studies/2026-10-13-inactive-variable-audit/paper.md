# When an input should not change the answer: a fixed-budget test of variable localization

**Date:** 13 October 2026  
**Status:** Completed small-model experiment; working paper. No human peer review.  
**Research and writing:** AI-assisted. No human or external peer review is recorded.

## Abstract

We test whether adding an inactive-variable check changes which neurons a researcher selects as a representation of an input. Eight small networks learn a selector task perfectly. For each network, we evaluate all 298 sets of one, two, or three neurons in the first hidden layer. We select a set either by its error when the input matters, or by the larger error across cases where the input matters and does not matter. Selection and test use different receiver-donor pairs. At the primary budget of three neurons, both rules select the same set in all eight networks. The mean difference in held-out worst-group error is exactly zero. This is a negative result for the predicted benefit of the second rule. It does not establish a correct variable representation: mean active error is 0.594 on a zero-to-one probability-distance scale. Patches also change the answer on 15.7% of a separate same-value control set. We explain why the larger-error objective does not change the choice when active error already dominates. The result concerns this small task, mask family, and donor distribution. It does not establish a result for language models.

## 1. The question

A neural network can give the correct answer without using the internal process that a researcher expects. Mechanistic interpretability seeks to identify that process. One common test replaces part of the network's internal state with values from a second input. This is called activation patching. The first input is the receiver. The second input is the donor.

Suppose a researcher claims that three neurons represent a variable called **a**. If the task uses a, a patch should make the network act as if a had changed. If the task ignores a, the correct task answer should stay the same. A test that measures only the first case leaves the second case untested.

We study this question in a small selector task. Each input has two values, a and b, a selector s, and an irrelevant value n. When s is zero, the correct answer is a. When s is one, the correct answer is b. We compare two ways to select an equal number of neurons. The first uses only cases where a determines the answer. The second also tests cases where a does not determine the answer.

This is a controlled experiment on trained small networks. The task rule is known. The internal circuit learned by each network is not known in advance.

## 2. Relation to previous research

Activation patching is an established method. Zhang and Nanda show that score and corruption choices can change localization results. Our experiment does not claim to discover that sensitivity. [Zhang and Nanda, 2024](https://arxiv.org/abs/2309.16042v2).

Causal abstraction compares a high-level model with a neural model under corresponding changes. Geiger and colleagues also train networks to match such changes. Here, we train on the ordinary task only. We then test possible representations of one variable. [Geiger et al., 2022](https://proceedings.mlr.press/v162/geiger22a.html); [Geiger et al., 2025](https://arxiv.org/html/2301.04709v4).

The MIB benchmark distinguishes the search for task circuits from the search for representations of causal variables. Our experiment concerns the second problem. A neuron set that represents a need not be the full circuit used to solve the task. [Mueller et al., 2025](https://arxiv.org/abs/2504.13151v2).

Two recent papers are especially close. Geng and colleagues compare equally sized circuits and show that a validation score can prefer a circuit with poorer behavior on new test cases. They also examine the effect of signals from the receiver. Li Puyin and colleagues diagnose regions where a proposed causal abstraction succeeds or fails. Thus, neither objective misranking nor context-specific diagnosis is new here. [Geng et al., 2026](https://arxiv.org/html/2610.02098v1); [Li Puyin et al., 2026](https://arxiv.org/html/2605.02234v1).

Our contribution is the recorded experiment: a fixed task, a complete set of small neuron masks, eight fixed training seeds, separate selection and test pairs, and an explicit active/inactive comparison. We make no claim of priority for a general method. The literature check was targeted and does not prove that no similar experiment exists.

## 3. Methods

Each of a and b has four values. The selector has two values, and n has six values. The 192 possible input states receive equal weight. A one-hot encoding gives 16 input coordinates. The model is a fully connected network with hidden layers of 12 and 24 tanh units and four softmax outputs. We train all eight fixed seeds, 0 through 7, for 4,000 full-batch Adam steps at learning rate 0.02. There is no hyperparameter search, early stopping, seed replacement, or intervention training. The [analysis plan](analysis-plan.md) gives all initialization and optimizer settings.

For a receiver x=(a,b,s,n) and a donor x′=(a′,b′,s′,n′), we replace a selected set S of first-layer activations with the donor values. We then recompute the whole downstream network. The target input is x_cf=(a′,b,s,n). Only a changes in the target. The target probability vector is the intact model output p(x_cf).

The primary pool contains all 27,648 ordered pairs with a′ different from a. A fixed hash divides each receiver-selector and donor-match group in half: 13,824 pairs for selection and 13,824 for test. The split is the same for all models. No complete pair is in both pools. Individual inputs occur in both pools. Donors have natural uniform-state weights: b matches in one quarter of pairs, and the selector matches in one half. We do not give each match group equal weight.

For a patched output p_S, error is total variation distance:

`TV(p_S, p(x_cf)) = 0.5 × sum_j |p_S(j) − p(x_cf)(j)|`.

Zero means identical probability vectors. One is the largest possible distance. This score is not a percentage of wrong answers. We also report agreement with the correct task label.

For each set S, let A(S) be mean error in active cases, where s=0. Let I(S) be mean error in inactive cases, where s=1. Both are calculated on selection pairs. The **repair-only rule** minimizes A(S). The **balanced rule** minimizes max(A(S), I(S)). We enumerate all masks at sizes one, two, and three. Size three is the primary comparison. Ties within 10⁻¹² use neuron order. After selection, masks are fixed for the test pool.

The controls include no patch, identity patch, complete first-layer patch, a direct input change, a factorized selector reference, and numerical gradient checks. A separate set of all 9,216 same-a pairs tests whether a donor can change the output even when the proposed variable has the same value. These pairs never enter mask selection. The set includes 192 identity pairs. We also repeat selection and testing with highly matched donors: b, s, and n all match the receiver. This uses only the corresponding subsets of the original selection and test pools.

Every run must have at least 99% intact task accuracy to enter the primary summary. At least six of the eight runs must pass. Failed runs would remain in the release. All eight passed. We report every seed, rather than treating thousands of shared-state pairs as independent training runs.

The code and plan were [published before model training](https://github.com/gavenren/mechanistic-interpretability-research/commit/ff9d78c15aa6e97b4819e18f4253b60fd165b50c). Their hashes were checked at the start and end. This is a versioned analysis plan, not a registry submission. No scientific protocol change was made after results were opened.


## 4. Results

### 4.1 The primary comparison did not improve

Both rules selected exactly the same three-neuron set in each model. Thus, they produced exactly the same outputs on every held-out pair. Table 1 reports test errors. Neuron indices start at zero. The difference is repair-only error minus balanced error; a positive value would favor the balanced rule.

| Seed | Three-neuron set, both rules | Active error | Inactive error | Worst-group difference |
| --- | --- | ---: | ---: | ---: |
| 0 | 7 8 11 | 0.616 | 0.095 | 0.000 |
| 1 | 1 5 6 | 0.627 | 0.112 | 0.000 |
| 2 | 1 2 11 | 0.596 | 0.062 | 0.000 |
| 3 | 0 1 2 | 0.541 | 0.135 | 0.000 |
| 4 | 2 3 4 | 0.572 | 0.161 | 0.000 |
| 5 | 2 3 5 | 0.524 | 0.046 | 0.000 |
| 6 | 0 2 3 | 0.676 | 0.061 | 0.000 |
| 7 | 2 6 10 | 0.599 | 0.096 | 0.000 |

Across the eight seeds, mean active error was 0.593707 and mean inactive error was 0.095859. The worst group was the active group in each seed. Mean, median, minimum, and maximum paired improvement were all zero. The planned seed bootstrap gives [0, 0] because every observed difference is zero. This interval adds no evidence about other tasks or architectures. The same selection agreement occurred at the one- and two-neuron budgets and in the matched-donor analysis.

### 4.2 Identical selection is not evidence of a correct representation

The complete network gave the correct label on all 192 states in every run. However, the selected three-neuron intervention gave the correct counterfactual task label on only 65.9% of unrestricted test pairs, averaged across seeds. High ordinary task accuracy did not make this small neuron set a faithful representation of a.

For comparison, the fixed random three-neuron sets had mean worst-group error 0.864432 and mean task-label accuracy 47.0%. The selected sets were better than this reference, but they still had large absolute error. The no-patch baseline had active error close to one and inactive error below 0.000003. Each candidate's exact results, including donor groups and copying rates, are in the released tables.

The model's own inactive change was small: mean TV between its intact receiver and counterfactual outputs ranged from 0.00000109 to 0.00000203 across seeds. It cannot explain the selected masks' inactive errors of 0.046 to 0.161.

The same-a control gives a second limitation. Although a had the same value in receiver and donor, the selected primary masks changed the predicted label on an average of 15.6603% of the control pairs. The seed range was 9.5920% to 19.0321%. Mean TV from the intact receiver output was 0.156289. These changes show that the tested coordinate sets do not isolate a under arbitrary donor contexts.

### 4.3 Donor restrictions change the result's scope

With b, s, and n matched between donor and receiver, the selected three-neuron sets had mean active error 0.395408 and inactive error 0.0000142. Their mean task-label accuracy was 80.9%. Both selection rules again chose the same masks within that regime.

These scores use masks selected within the matched regime and its smaller test subset. They are not a controlled estimate of the donor restriction's effect on a fixed mask. They show that a conclusion must name both the selection procedure and the donor pool. A good result under a restricted donor pool does not establish validity under unrestricted donors.

### 4.4 Why the larger-error rule can leave the choice unchanged

The explanation is elementary. Let S_R minimize active error. If I(S_R) ≤ A(S_R), then for every candidate S:

`max(A(S), I(S)) ≥ A(S) ≥ A(S_R) = max(A(S_R), I(S_R))`.

Therefore S_R also minimizes the larger-error objective, apart from the stated numerical tie rule. The discovery tables satisfy this condition in all eight primary comparisons. This is a descriptive explanation of the null result, not a new causal theorem or an additional experiment.

The balanced objective sets no independent limit on inactive error. It can leave an inactive error uncorrected while active error is larger. Thus, including inactive cases in an objective is different from requiring small inactive error. A future test could impose an explicit inactive-error bound, but this study does not establish that such a rule works.


## 5. Limits and interpretation

The test concerns one small architecture and one artificial task. It does not measure how often the effect occurs in language models. All ordinary input states are used for task training. Only receiver-donor pair identities are split between selection and test. The test therefore concerns new interventions on known inputs, not generalization to new task inputs.

A neuron mask is a restricted kind of representation. A variable may be distributed across more neurons, a rotated subspace, or several layers. Failure of a three-neuron mask does not show that the network has no representation of a. Success does not establish a unique representation.

The primary target is the intact model's probability distribution for the changed input. This target can preserve a small model error. An inactive a leaves the task label unchanged, but the intact model's probabilities can still change. Exact preservation of the probability distribution is a separate question. We report the model's own inactive change and same-value controls to keep these questions separate.

Eight fixed training seeds provide limited evidence about variation across training runs. They do not sample other tasks, model families, or donor choices. Any interval across seeds has this restricted meaning. No human peer review or replication by an external research group has occurred.

## 6. Reproduction and review

The release contains the fixed plan, input and pair ledgers, all eight weight files, all 298 candidate scores for each model and regime, selected and random masks, training traces, output predictions, controls, and a hash manifest. See the [reproduction instructions](README.md), [source record](sources.md), and [independent AI review](reviews/independent-review.md).

The original run used Python 3.12.14 and NumPy 2.3.5 on an Apple silicon CPU. Training and the complete score calculation took about 39 seconds. No external dataset, downloaded model, GPU, or paid model API was used for the numerical experiment. Code, study design, literature review, analysis, and writing were AI-assisted. AI review does not establish human peer review.

The result does not support the planned prediction that the balanced rule would improve the primary test. It supports a narrower lesson: report absolute active and inactive errors, including same-value controls, even when two selection rules agree. The agreement alone did not validate the proposed neuron sets.
