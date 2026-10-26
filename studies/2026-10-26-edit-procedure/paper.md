# Why the full edit procedure matters in three saved model panels

**Date:** 26 October 2026. **Type:** Completed diagnostic study, Experiment 2A.

**Research and writing:** AI-assisted work for this repository. A separate AI calculation review is complete. No human peer review or full independent model reproduction is recorded.

## Abstract

An internal edit needs more detail than the name of its target. We tested donor choice, repeated writes, alternative sparse codes, and normalization rules in three saved model panels. In eight small selector models, donors with the same requested value still changed test answers in 17.80% of active cases and 13.08% of inactive cases. A constrained search found alternative sparse codes in all 192 test cases. Their mean readout span was 0.98 percentage points. In one small language model, recomputed and fixed normalization statistics gave different scores after the same edit. All fixed numerical checks passed. These findings support precise descriptions of edit procedures. They do not establish a new general principle or a rate of failure across AI models.

## 1. The question

An edit experiment asks what happens when part of a model's internal state changes. But several choices help define that change. A donor can carry information unrelated to the intended target. Two writes can request different coordinates even when they name the same value. A reconstruction limit can allow more than one sparse code. Normalization can change other coordinates after one coordinate is edited.

Prior work already shows that patching methods and scores affect conclusions. [Zhang and Nanda](https://arxiv.org/abs/2309.16042v2) test these choices. [Heimersheim and Nanda](https://arxiv.org/abs/2404.15255v1) explain limits on the resulting circuit evidence. Our question is narrower: how large are these procedure differences in this repository's saved models, under fixed tests and explicit checks?

We use three branches as one study. The [public plan](https://github.com/gavenren/mechanistic-interpretability-research/commit/f4174fe862cd1c660d0bc1db5eddfe873d7e3a56) fixed the inputs, code, measures, and tolerances before new task outputs. [Passed preparation](https://github.com/gavenren/mechanistic-interpretability-research/commit/352f1551dde4be89c0e603402a54b7adc534b604) preceded the scientific run. None of the models was newly trained.

## 2. Donors can change more than the requested value

The selector task has inputs `(a,b,s,n)`. The answer is `a` when `s=0`, and `b` otherwise. We call these active and inactive cases. All eight saved models gave the correct answer on every intact task state.

We kept each model's original three-neuron mask. For each receiver and requested value of `a`, we tested all 48 donor contexts. The receiver and required changed input stayed fixed. We also kept unchanged, zero-replacement, and coherent-input controls. Complete receiver/request groups were split before outputs; no choice was tuned on either split. These reused models make this a diagnostic extension, not fresh confirmation.

The table gives test means across eight models. It includes all declared donor contexts. An answer change is unwanted when `a` stays the same or when the task does not use `a`.

| Request | Task state | Required answer is correct | Answer changes |
| --- | --- | ---: | ---: |
| Change `a` | Active | 42.81% | 61.15% |
| Change `a` | Inactive | 90.17% | 9.83% |
| Keep `a` | Active | 82.20% | 17.80% |
| Keep `a` | Inactive | 86.92% | 13.08% |

A visible change is not the same as a correct change. In active changed-value cases, 61.15% of answers changed, but only 42.81% reached the required answer. The coherent full-input controls reached the required answer. They use information unavailable to a selected three-neuron edit.

We also applied two writes to the same mask in both orders. For equal requested values, the orders gave different answers in 25.07% of active cases and 9.86% of inactive cases. The last write always set its requested coordinates. The two donors could supply different coordinates despite having equal `a` values.

This is expected last-write behavior for one mask. It is not a new law for composing two separate semantic variables. The result shows the output consequences of donor choice in these masks. It does not rule out another mask, a learned rotation, or a more restricted donor rule. Work on [distributed alignment search](https://proceedings.mlr.press/v236/geiger24a.html) also makes clear that individual neuron sets do not exhaust possible representations.

## 3. A reconstruction budget permits different readouts

We used six saved sparse autoencoders from the rare-feature study. Each code has 24 nonnegative weights, with at most four active weights. The decoder maps it to a vector with 12 values. We fixed each input's reconstruction-error budget at its original encoder error, with a declared arithmetic tolerance.

For each input, the search tested fixed candidates on all 12,951 supports of size zero through four. It retained only valid candidates. The known synthetic rare-feature direction defined the readout. This is a constrained witness search, not an unsupervised feature-discovery method.

| Split | Cases | Distinct valid alternatives found | Mean probability span | Largest span |
| --- | ---: | ---: | ---: | ---: |
| Development | 96 | 96 | 0.006629 | 0.055147 |
| Test | 192 | 192 | 0.009762 | 0.152699 |

The mean test span is about 0.98 percentage points. The largest is 15.27 points. Mean test spans across the six models range from 0.28 to 3.41 points. The largest case does not describe every model.

No alternative in this search preserved the original decoded vector within the fixed same-vector tolerance. Thus, these results concern different vectors within the same error allowance. They do not establish that exactly equal decoded vectors have different readouts. Nor do they prove uniqueness when no same-vector witness is found.

The constructed codes need not be outputs of the encoder. The panel has equal numbers of both rare-bit values; it does not estimate behavior at a natural 1% event rate. The broader distinction between reconstruction and downstream function has substantial prior work, including [Braun and colleagues](https://arxiv.org/abs/2405.12241v2).

## 4. The normalization rule changes the result

We used one fixed Pythia-14M checkpoint, 32 short prompts, eight normalization sites, and three fixed edit sizes. At the final text position, the edit changed coordinate zero. The two procedures then either recalculated the mean and scale or kept their intact values. Both used the trained gain, bias, and epsilon.

Each case saved all 128 normalization values and eight fixed token scores. A score here is a logit, before conversion to a probability. For each case, we took the largest absolute gap among those eight scores. The table summarizes that measure over 256 prompt/site cases per size.

| Edit size | Median score gap | Largest score gap |
| --- | ---: | ---: |
| 0.1 | 0.013514 | 0.149417 |
| 0.5 | 0.147835 | 2.416556 |
| 1.0 | 1.367561 | 12.748755 |

Larger edits gave larger median gaps in this fixed panel. This is consistent with the shared statistics in [layer normalization](https://arxiv.org/abs/1607.06450v1). Freezing those statistics changes the procedure. Neither rule is automatically the correct interpretation for every research question.

The experiment does not test copying ability or locate a semantic variable. It uses a declared float64 variant with shared float32 rotary tables. It does not establish the result for usual lower-precision execution, other checkpoints, or the full output distribution.

## 5. Checks, costs, and limits

All 411,648 selector rows, 288 sparse-code cases, and 32 language prompts completed. The runner used separate arithmetic for selector outputs, code reconstruction and readout, and normalization. Every fixed no-change check passed. The largest recorded full-logit replay difference was `1.79e-11`, below the normalization tolerance of `1e-8`.

A separate saved-result audit checked hashes, constraints, cases, and summaries. It did not repeat training, the candidate search, or model forwards. The normalization replay shares the downstream transformer implementation. Full-vocabulary outputs were checked during execution but were not saved; the audit checks their recorded maximum errors. These limits prevent a claim of a fully independent transformer reproduction.

The scientific runner took 29.59 measured seconds through its hash scan, with a two-second closing allowance. Preparation and science used 33,536 forwarded tokens. No paid compute, new model download, or training was used. With conservative audit and archive-check allowances, Experiment 2A is charged 151.59 seconds against its 7,200-second cap. Cumulative roadmap preparation is 205.14 seconds. The [compute record](analysis/compute-accounting.json) gives exact charges.

The rows share inputs and models. They are not independent model replications. The test splits did not select the method, but the source models had already informed this project. No population confidence claim follows from these tables. A passed numerical check supports the calculation, not a correct semantic explanation.

## 6. What is added, and what comes next

This study adds a fixed, reproducible comparison of complete edit procedures on saved assets. It measures donor effects under the same required task change, tests both requests after two writes, keeps sparse-code constraints explicit, and checks normalization with the actual trained parameters. The broad lessons are established concerns; this study adds bounded evidence, not a claim of priority.

The next stage is Experiment 2B. It will compare greedy component removal with exhaustive search, compare score-based choices on separate test cases, and test false positives after selection. A later positive claim about a variable must also pass the stronger intended-change and preservation rules in Experiment 2C. The present results do not lower those rules.

## Evidence and reproduction

Read the [fixed plan](analysis-plan.md), [source check](sources.md), [plan review](PLAN_REVIEW.md), and [result review](RESULT_REVIEW.md). The three branch methods give all case, arithmetic, and tolerance details.

The [result manifest](results/manifest.json) fixes every raw file. Three [result archives](artifacts/manifest.json) contain all 1,169 branch files with exact bytes. The direct status file remains outside the archives. From the repository root, restore and check them with:

```sh
.venv/bin/python studies/2026-10-26-edit-procedure/analysis/result_archive.py unpack
```

The restore procedure verifies every archive before extraction. It permits identical existing files and rejects conflicting files. Its [fresh-restore check](analysis/archive-check.json) passed. The source code, preparation files, raw results, and audit remain part of one study.
