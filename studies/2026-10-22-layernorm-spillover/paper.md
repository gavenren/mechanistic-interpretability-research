# A local edit can spread through layer normalization

**Date:** 22 October 2026. **Type:** Formal analysis and constructed numerical test. No trained model. AI-authored; no human peer review.

## Main finding

Changing one input coordinate changed every other normalized coordinate in all 2,048 fixed cases. On average, 25.47% of the squared output change was outside the edited coordinate. This is a result for the declared random states. It is not a measured rate in language models.

The spread has an exact explanation. Layer normalization recomputes a shared mean and scale after the edit. Thus a local input change can alter outputs whose input coordinates were left unchanged.

## Question, prior work, and contribution

Does a one-coordinate edit remain local after normalization? [Ba, Kiros, and Hinton](https://arxiv.org/abs/1607.06450v1) introduced the normalization used here. [Zhang and Nanda](https://arxiv.org/abs/2309.16042v2) studied how patching choices affect interpretation results. The broad concern about method dependence is established.

This note adds an exact finite-edit decomposition and a complete numerical check on fixed constructed states. It does not claim a new normalization principle. Unlike the earlier basis-transport paper, the coordinates remain fixed. Unlike the earlier language patch study, this test isolates one mathematical operation.

## Method and derivation

Let x contain d coordinates. Define z=x-mean(x) and s=sqrt(mean(z²)+epsilon). The normalized output is z/s. There is no learned gain or bias. Change coordinate zero by delta. Write e0 for that coordinate's unit vector and s' for the new scale.

The new centered vector is z'=z+delta*e0-delta*1/d. Substitution gives:

`change = delta*e0/s' + (1/s'-1/s)*z - delta*1/(d*s')`.

Outside coordinate zero, the first term vanishes. The remaining effect lies in the span of the old centered vector and the constant vector. This identity follows directly from the definitions. It does not require a trained network.

We used widths 4, 8, 16, and 64, with 32 seeded normal vectors at each width. Epsilon was zero or 0.00001. Eight positive and negative edit sizes ranged from 0.01 to four baseline standard deviations. The [public plan](analysis-plan.md) fixed these choices before the run. All cases are saved.

## Results

The following values average both epsilon settings and all 32 inputs. The complete table includes both edit directions.

| Width | Other-coordinate energy share, +0.01 SD | Other-coordinate energy share, +4 SD | Linear approximation error, +4 SD |
| --- | ---: | ---: | ---: |
| 4 | 56.05% | 51.06% | 113.83% |
| 8 | 25.01% | 36.54% | 80.85% |
| 16 | 13.11% | 21.44% | 54.02% |
| 64 | 3.01% | 7.85% | 25.92% |

The energy share divides squared change outside coordinate zero by total squared change. Linear error divides the norm of the Jacobian prediction error by the norm of the actual change. It can exceed 100%. Larger width reduced the spread in these groups. A large edit still made the local linear approximation poor.

The exact decomposition's maximum error was 1.22e-15. Native PyTorch normalization and a scalar calculation matched the saved outputs within 8.89e-16. Recomputed scores matched within 1.94e-13. All passed the fixed 5e-12 tolerance. Zero edits and constant shifts also passed. Calculation and audit took about 2.97 seconds inside their scripts, below the shared 120-second cap.

## Limits and use

These are random constructed vectors. They are not natural model activations. Learned gains, other layers, and output readouts can change the practical effect. A changed normalized coordinate is not proof that a semantic feature changed.

A test with frozen mean and scale answers a different question from a normal forward pass. Researchers should name that choice and measure its effect. The next useful extension is a fixed trained-model test that separates normalization effects from later computation.

See [code and instructions](README.md), [raw results](results/cases.csv), [calculation audit](results/audit.json), and [separate AI review](PRE_RUN_REVIEW.md). The plan and code were locked at `88e8df7689b65fbeef221423c115cd1b1c198f22`. Source versions and limits are in the [source record](sources.md).
