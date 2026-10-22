# Perfect reconstruction can leave a coordinate edit undetermined

**22 October 2026 · Formal and computational research note · AI-authored**

## Main finding

Two latent codes can reconstruct the same activation yet give different results after the same coordinate edit. We give a precise condition for when this ambiguity disappears. An exact audit of 9,600 edits across three small linear decoders matched that condition in every case.

The effect also occurred with nonnegative codes. It was not a numerical error: all ordinary reconstructions were exact. These are constructed codes with no trained encoder or sparsity objective. The result does not establish a failure rate for trained sparse autoencoders.

## Question and prior work

A decoder maps a code to an activation. An overcomplete code can have more entries than the activation it represents. The map can then assign several codes to the same activation.

Sparse-dictionary research already separates reconstruction quality from functional importance. Braun and colleagues train dictionaries to preserve downstream model output and compare them with local reconstruction objectives. Our note does not introduce that distinction. It asks a narrower algebraic question: when does a chosen coordinate replacement depend only on the decoded receiver and donor? [Braun et al., version 2](https://arxiv.org/html/2405.12241v2).

The repository's earlier basis-transport study used invertible changes of coordinates. This study holds the decoder fixed and uses multiple codes for the same activation. The distinction matters because an inverse coordinate map does not exist here.

## Definitions and condition

Let D be a linear decoder. A receiver code z gives activation `x=Dz`. A donor code w gives `y=Dw`. A diagonal mask M has entries zero or one. Its edit is

`z' = z + M(w-z)`.

A zero mask leaves the receiver unchanged. A full mask replaces its code with the donor code.

The decoder's null space, written `ker(D)`, contains changes that decode to zero. Thus z and `z+u` have the same activation whenever u belongs to this space.

**Condition.** For unrestricted real codes, the decoded edit is independent of the selected receiver and donor codes exactly when

`D M v = 0 for every v in ker(D)`.

To prove this, change the receiver code by u and the donor code by v, with both changes in the null space. The decoded edit changes by

`D[(I-M)u + Mv] = DM(v-u)`.

If the stated condition holds, this change is zero. Conversely, keep u zero and vary v. Independence requires `DMv=0` for every allowed null-space change. This proves both directions.

A fixed linear readout a needs only `aᵀDMv=0`. That weaker condition can preserve one output while the edited activation still changes.

## Fixed calculation

We used three rank-two decoders from three code entries to two activation entries:

- Coupled: `(z0+z2, z1+z2)`.
- Duplicate: `(z0+z1, z2)`.
- Silent: `(z0, z1)`, which ignores the third entry.

For each decoder, receiver and donor activations range over all four points in `{1,2}²`. Each canonical code receives a null-space shift from −2 to +2. We test all eight coordinate masks and every receiver/donor shift pair. The [plan](plan.md) fixes the exact vectors and all inputs.

The reference edit uses the canonical codes. We measure how much an alternative valid encoding changes that edit. We also measure the readout `activation0 - activation1`. Nonnegative code pairs are reported separately.

| Decoder | Masks with an unambiguous activation edit | Masks with an unambiguous readout | Largest activation discrepancy | Largest discrepancy with nonnegative codes |
| --- | ---: | ---: | ---: | ---: |
| Coupled | 2 of 8 | 4 of 8 | 8 | 4 |
| Duplicate | 4 of 8 | 4 of 8 | 4 | 2 |
| Silent | 8 of 8 | 8 of 8 | 0 | 0 |

Discrepancy is the sum of absolute coordinate differences. Its size depends on this fixed code range and scale. It is not a universal error bound.

A simple nonnegative witness uses the coupled decoder. Codes `(1,1,0)` and `(0,0,1)` both decode to `(1,1)`. Copy only the donor's third coordinate into the receiver. The new code is `(1,1,1)`, which decodes to `(2,2)`. The decoded inputs agree, but the partial-code replacement changes the activation. The readout difference remains zero, so that output alone misses the change.

## Checks and limits

A separate scalar audit checked every reconstruction, edited coordinate, discrepancy, mask condition, and summary maximum. All comparisons were exact. The run and audit took 0.28 seconds. Peak process memory was 30,851,072 bytes. The public plan and code lock was `88e8df7689b65fbeef221423c115cd1b1c198f22`. [Results](results/summary.json); [reproduction](README.md).

A fixed encoder chooses a particular code. Its implementation can therefore define a unique edit even when reconstruction alone does not. Sparsity, support restrictions, or other constraints can also remove some alternatives. Our general condition concerns unrestricted real codes; the finite nonnegative examples show that nonnegativity alone does not always remove the problem.

The contribution is an explicit condition, witnesses, and a reproducible finite audit. It is not evidence that a trained model uses these codes. Future work should test admissible code variation under a fixed trained encoder and separate activation agreement from task-output agreement. Separate AI source review is recorded in [PRE_RUN_REVIEW.md](PRE_RUN_REVIEW.md). No human peer review is recorded.
