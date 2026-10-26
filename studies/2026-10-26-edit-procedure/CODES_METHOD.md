# Decoded codes: fixed method for Experiment 2A

This is one branch of Experiment 2A. It is not a separate paper. Fix this method and its code in a public commit before the scientific run. The parent runner must check the commit, usage, time limits, and input files.

## Question

Can different sparse codes meet the same reconstruction-error limit and give different outputs through a fixed readout?

The code for an input is a list of 24 nonnegative numbers. At most four numbers can be above zero. A decoder converts this list to a vector with 12 values. We compare the saved encoder's code with codes that we construct from its decoder. These constructed codes are **not evidence that the encoder produces them**.

We keep two results separate:

1. Codes can satisfy the same error limit and produce different decoded vectors.
2. Different codes can produce the same decoded vector, within a fixed numerical tolerance.

The second result would show an ambiguity in the code without a material change in the fixed readout. The first can change that readout. Neither result proves that a model has no useful explanation.

## Saved models

Use exactly these six models from [Study 2](../2026-10-13-rare-feature-sae/paper.md). The event frequency during training was 1%. Its amplitude was 0.25. The objective was reconstruction only. No model is trained or selected in this branch.

| File, relative to the repository | SHA-256 |
|---|---|
| `studies/2026-10-13-rare-feature-sae/models/seed-0_p-0.01_a-0.25_local.npz` | `a5e8543a0e307d6e3c495d3426d8a3f9fa5f3c18bf769dac50d80fe0b29a97e8` |
| `studies/2026-10-13-rare-feature-sae/models/seed-1_p-0.01_a-0.25_local.npz` | `97244a51cc70ee6fd8d5feee023894dd119d6ef5f96f0e70627c3fbb76b55a97` |
| `studies/2026-10-13-rare-feature-sae/models/seed-2_p-0.01_a-0.25_local.npz` | `aecb70005a31abcc7d5ac7447999fb96c6114e054c09c24c77a95729066f8493` |
| `studies/2026-10-13-rare-feature-sae/models/seed-3_p-0.01_a-0.25_local.npz` | `0140de434008a0158172c8f8b7ff28517a77bf47a03bb98743947d68d7aada92` |
| `studies/2026-10-13-rare-feature-sae/models/seed-4_p-0.01_a-0.25_local.npz` | `c312c1b716d51f7ea82875dcb4d3d60138b553260dca949d6430142d17c48d6d` |
| `studies/2026-10-13-rare-feature-sae/models/seed-5_p-0.01_a-0.25_local.npz` | `104d08b8e6a6be9f711dd231cd2ff7811b7dcc6b4b756097f441df747d5af36e` |

Also use the six saved dictionaries at `studies/2026-10-13-rare-feature-sae/data/dictionary-seed-{seed}.npz`, with seed 0 through 5. The input manifest lists all six paths. Code checks their fixed hashes. The other source files are `studies/2026-10-13-rare-feature-sae/data/generator-records.json` and `studies/2026-10-13-rare-feature-sae/code/run.py`.

The six dictionary hashes, in seed order, are:

```text
f65a984477a473417b95affc20b02169701ad467e5ed3cf4685d5bac476fa68e
051274e47977df7de76b9a7cd75ab759e3d653091af16e32d4a046a8cfe0dc35
db770cd092ac5334dbf3ff7d69940209e357f304ed3261e17cbde2e7e829e989
12f172d082ac7b8929340c07888517af55dfe1bb2ae1b0d2f8c158cfc7134487
5b52348530476b078d6523701c68abff42d0451a7c625501c61f529ea118b783
d758c6fad5f0177970267a05ab540d4be1f435db9015d5b6a152dd7421dac052
```

The generator record hash is `71ef2dac1295669ba930e7e7fa30cde65bdcb32947f71d983dad20cfdbf3816b`. The source code hash is `8780933b78c2738df03e41b633765ab687ce5d24acf63a352650f0ec37bc9909`.

## Fixed input cases

Regenerate Study 2's train, validation, and test backgrounds with its saved dictionary and random seed `2000 + model seed`. For each bank, compare the array hash with the original generator record. Stop if a hash differs. Regeneration does not train a model.

For every model, use validation backgrounds 0 through 7 as development inputs. Use test backgrounds 0 through 15 as test inputs. For each background, use both rare-bit values, zero and one. Add `0.25 × bit × rare direction` to the background. This makes 16 development cases and 32 test cases per model: 96 development cases and 192 test cases in total.

The development and test lists are fixed now. There is no parameter selection from either list. Both lists run the same search. We report them separately. This test panel has equal numbers of both bit values. It does not estimate behavior at the natural 1% event rate. Paired bit values share a background. Some background values can also repeat across original splits. The cases are not 288 independent samples. The models and source tasks have already been studied. These are diagnostic cases, not fresh confirmation data.

## Encoder, decoder, and readout

Use float64 arithmetic. The encoder applies its saved linear map to the mean-centered input, adds its bias, and replaces negative values with zero. Keep its four largest values. A tie goes to the lower feature index. All other code values are zero.

Let `D` contain the 24 decoder rows. Let `o` be the saved decoder bias plus the saved training mean. For a code `z`, the decoded vector is `z D + o`. For an input `h`, the original squared reconstruction error is:

```text
E0 = sum((z_encoder D + o - h)^2)
```

Use Study 2's fixed readout:

```text
q(x) = sigmoid(8 × ((x dot u) / 0.25 - 0.5))
```

Here `u` is the saved rare direction. It is unit length. This is a specified one-bit readout. It is not a trained language model. The search uses this known direction. It is not an unsupervised method for finding a new feature.

## Candidate search

List all supports with zero, one, two, three, or four of the 24 features. A support is the list of features that a code may use. There are 12,951 supports. List them by size, then in feature-index order.

For each nonempty support, form the decoder matrix `A` with one column per selected feature. Let `y = h - o`. Find the least-squares center:

```text
G = A.T A
c = inverse(G) A.T y
Emin = sum((A c - y)^2)
v = A.T u
d = inverse(G) v
```

Test the center `c`. Also test these two codes when the error budget permits:

```text
c ± sqrt(max(E0 - Emin, 0) / (v.T d)) × d
```

Before the nonnegative constraint, these are the readout-direction extrema on that support's error ellipsoid. If `v.T d` is at most `1e-24`, use the center for all three candidates. This rule can produce duplicates. Valid-candidate counts include duplicates. For the empty support, test the all-zero code once. Thus, each input has at most 38,851 candidate checks. The original encoder code is also in the reported output range.

Calculate each support's matrix inverse once, then apply it to all 48 cases for that model. Process supports in groups of 256. Check the shared time guard between groups. This avoids a separate inverse for each case.

Use all these acceptance rules:

- At most four coefficients can be positive.
- Each coefficient must be at least `-1e-10` before round-off correction. Replace a value in `[-1e-10, 0)` with zero. Reject a more negative value.
- Recalculate the decoded vector after that correction.
- Its squared error must be at most `E0 + 1e-9 × max(1, E0)`.
- The support's least-squares error must also meet that error limit.
- All computed values must be finite.

The error allowance covers arithmetic error. It is not a new fitted reconstruction budget. It is fixed before the test.

A code is a distinct witness only if its largest absolute difference from the encoder code is greater than `1e-7`. A distinct code preserves the decoded vector only if the largest absolute decoded-coordinate difference is at most `1e-9`.

Keep the lowest and highest readout candidates, the first distinct valid code, and the first valid code that preserves the decoded vector. For equal readout values, keep the first code encountered. The encoder code is first for the two range endpoints. These rules are fixed before results.

This is a finite candidate search. **Do not claim global uniqueness or an exhaustive search of all feasible coefficient values.** If no distinct code is found, report “No valid alternative was found by this search.” Do not relax a constraint. A missing same-vector witness also does not prove uniqueness. The readout range is the range among valid retained candidates and the original code.

## Controls, compute, and stop rules

Before the scientific run, preparation uses only hand-made arrays. One control has two codes for exactly the same vector: `[1, 1, 0]` and `[0, 0, sqrt(2)]`. A second control uses the unit disk around `[1, 1]`. Its first coordinate must range from 0 to 2. These controls check the search without reading trained weights or study cases.

Preparation also times a full-size search on a random 24-by-12 dictionary and 48 fixed synthetic cases. Its seed is 2626. This dictionary is not trained. The branch cost estimate is `4 × (6 × measured dummy-search seconds + 30)`. The factor of four covers timing variation. The 30-second allowance covers data checks, saving, replay, and replay from saved files. The preparation cost is charged to the shared preparation cap. Scientific work is charged to Stage 2A's shared two-hour cap. Use the parent runner's two-thread CPU setting. This branch forwards no language tokens.

Check the Gram matrix for every support. Stop if its smallest eigenvalue is at most `1e-12` times its largest eigenvalue. Stop for a nonfinite value or invalid saved code. Do not drop a bad support and continue.

In the scientific run, compare the encoder with a separate scalar-loop implementation. Decode each saved code with a separate scalar-loop implementation. Recalculate every saved probability with a scalar dot product and a separate sigmoid expression. The maximum absolute error must be at most `1e-9` in each check. The no-change result has the same tolerance. Stop on the first failure. Do not increase the tolerance after results.

The scientific run must have an empty output folder. Refuse to overwrite old results. After each model, close and reopen its array archive. Require an exact match for every intended array, its data type, and the full list of array names. From these reopened arrays, repeat the scalar encoder, decoder, readout, squared-error, nonnegative, sparsity, and witness-mask checks. Recheck the same-vector and readout bounds. Record the largest saved-file replay error. Stop on any failure.

A vector difference of `1e-9` per coordinate can make a larger readout difference. For same-vector witnesses, require the probability difference to be at most `8 × sum(abs(u)) × 1e-9 + 1e-9`. This follows from the sigmoid derivative limit of one quarter and the fixed readout slope of 32. Record that difference separately from replay error.

The parent runner must stop when 20% or less of the applicable usage window remains. It must also stop if the preparation estimate cannot fit the remaining compute cap. It must publish any partial results as partial. No paid service or new base-model training is part of this branch.

## Saved evidence and report

Save all 288 input-case results, including both development and test cases. Each model has a compressed array file with input vectors, input IDs, bits, baseline codes and decoded vectors, both range endpoints, the first distinct witness, and the first same-vector witness. Save the associated probabilities, errors, masks, and candidate counts by support size and candidate type. A false witness mask means its stored placeholder is the baseline code; it is not an alternative. Internal rejected candidates can be regenerated with the fixed code; they are not separate reported edits.

The CSV gives a row for every input case. A JSON file gives each model's control errors. A summary gives separate development and test counts and readout spans. Report variation across the six saved models where it affects the conclusion. Do not turn case counts into confidence claims.

The branch succeeds as a measurement if all controls pass and all saved comparisons satisfy the constraints. A zero span or a missing alternative is a valid scientific result. A large span would show that this reconstruction limit, sparsity rule, and decoder permit different readout results on these cases. It would not show that the encoder emits those other codes, that real models have the same rate, or that a useful semantic feature was found.
