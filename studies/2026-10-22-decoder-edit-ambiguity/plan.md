# Fixed plan: reconstruction does not always define a coordinate edit

Date: 22 October 2026. Status: fixed before calculation.

## Question and contribution

Can two latent codes reconstruct the same receiver and donor, but give different decoded results after the same coordinate replacement? This note studies a non-injective linear decoder. It does not repeat the earlier study of invertible changes of basis. Prior sparse-dictionary work already separates reconstruction from functional importance. This note gives a precise algebraic condition for independence from the chosen code and an exact finite audit. It does not test trained sparse autoencoders or claim a new general principle.

## Fixed definitions

Let D map a three-entry code z to a two-entry activation x. For receiver z and donor w, a diagonal zero/one mask M defines z'=z+M(w-z). The edit is independent of code choices with fixed decoded receiver and donor if and only if D M v=0 for every v in ker(D). State this condition for unrestricted real codes. For a linear readout a, the weaker output condition is a^T D M v=0. Prove both by varying receiver and donor independently in the decoder's null space.

Use three fixed decoders, each of rank two:

- Coupled: D=[[1,0,1],[0,1,1]], canonical code (x0,x1,0), null vector (-1,-1,1).
- Duplicate: D=[[1,1,0],[0,0,1]], canonical code (x0,0,x1), null vector (-1,1,0).
- Silent: D=[[1,0,0],[0,1,0]], canonical code (x0,x1,0), null vector (0,0,1).

Use all four receiver activations and all four donor activations in {1,2}^2. For each, add t times the null vector to its canonical code, independently for receiver and donor, with t in {-2,-1,0,1,2}. Use all eight masks. This gives 9,600 exact cases. Save whether both selected codes are nonnegative. The subset with t in {0,1} always supplies nonnegative controls. No sparsity or encoder optimality is assumed.

Compare each edited activation with the edit from the canonical receiver and donor. Save the activation discrepancy and the readout discrepancy for a=(1,-1). Report the maximum L1 activation discrepancy and maximum absolute readout discrepancy by decoder and mask, for all cases and for the nonnegative subset separately.

## Fixed checks and interpretation

The normal reconstruction error must be exactly zero for every receiver and donor. A separate audit computes coordinates directly without matrix multiplication. It must reproduce every edited activation, every discrepancy, every mask condition, and all summary maxima exactly. Empty-mask edits must preserve the receiver. Full-mask edits must give the donor. Matching receiver and donor codes must produce no change.

Success means the finite cases agree with the stated algebra. A nonzero discrepancy is a witness that reconstruction alone does not determine this coordinate edit. A zero discrepancy on a finite panel is not by itself a proof for all codes; the stated null-space condition provides that proof for these decoders. ReLU, sparsity penalties, a fixed encoder, and code support constraints can restrict admissible lifts. Do not claim their trained outputs have the tested ambiguity.

## Resource and record rules

This is a constructed formal test. It trains no model and uses no paid service. Run one local CPU process. Acquire the shared exclusive research-compute file lock. The test has a 300-second elapsed limit, including its calculation audit. The three measurement notes together have a 900-second limit. The root supervisor must set `MI_USAGE_PERMIT` to a valid private permit, refresh it by atomic replacement after a live account check, and preserve the 20% reserve. The old shared guard checks the permit, time, and four-GiB process limit. A failed or missing check stops the run. Do not alter inputs or acceptance rules after seeing results.

Before calculation, the plan, source list, and code must be in a public commit. The runner checks their bytes against the supplied commit. It refuses an existing result directory. Save raw rows, the complete case list, a summary, a separate equation audit, runtime, environment, and SHA-256 hashes. Keep failures. Successful checking means that the implementation matches the declared construction. It does not validate a claim about trained language models. No human peer review is recorded.
