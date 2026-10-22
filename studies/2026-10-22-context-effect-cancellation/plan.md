# Fixed plan: context alignment and cancellation in patch scores

Date: 22 October 2026. Status: fixed before calculation.

## Question and contribution

Can averaging a fixed signed output coordinate hide a useful semantic edit when the requested answer changes with context? AtP* already studies cancellation across input pairs and distinguishes the magnitude of a mean from the mean magnitude. This note does not claim that observation is new. It supplies an exact known-context construction with a nuisance control and an explicit task-ability gate. It is separate from logit-versus-probability ranking: all scores here stay on one linear output scale.

## Fixed model and inputs

Use three binary variables c,t,n in {-1,+1}. The task answer is c*t. Context c changes how task variable t maps to the answer. The nuisance variable n does not change the specified answer. The model's scalar output is L=alpha*c*t+beta*n. Predict +1 when L>=0, otherwise -1. Alpha is one of {1,2,3,4}; beta is one of {0,1,2,3,4}. All 20 models are constructed, not trained.

Evaluate intact task accuracy on all eight states. A model passes task ability only when all eight are correct. The semantic edit changes t from -1 to +1, while preserving c and n. Evaluate all four (c,n) pairs. The nuisance edit changes n from -1 to +1 while preserving c and t. Evaluate all four (c,t) pairs. Include unchanged-value controls.

For mixture weights p in {0,1/4,1/2,3/4,1}, give c=+1 mass p and c=-1 mass 1-p. The other unchanged variable is uniform. For each model and edit, save all individual output differences, predictions, expected task answers, and correctness. Then save the signed mean, absolute signed mean, mean absolute difference, and each context mean.

For the semantic edit, also save the context-aligned mean E[c*delta]. This uses the known task rule and extra information. It must not be presented as an unsupervised discovery method. For nuisance edits report the rate of unwanted answer changes against the unchanged task rule. Do not treat a large absolute effect as proof of semantic correctness.

## Predictions fixed before calculation

For semantic edits delta=2*alpha*c. Therefore the raw mean is 2*alpha*(2*p-1), while the mean magnitude and context-aligned mean equal 2*alpha. For nuisance edits delta=2*beta. At balanced context p=1/2, a positive-beta nuisance has a larger absolute mean than the semantic edit even when alpha>beta and the intact model is fully correct. This ranking statement concerns the signed-mean score only. It does not apply to all importance measures.

Report every model, including task failures. Separate the fully correct models before discussing useful task edits. The main counts are the number of task-capable models and, among these, the balanced-context ranking reversals. A separate audit enumerates scalar state transitions and exact Fraction weights, rechecks all predictions and aggregate formulas, and demands exact equality. There is no statistical sample or confidence interval: this is an exhaustive panel of fixed constructions.

## Resource and record rules

This is a constructed formal test. It trains no model and uses no paid service. Run one local CPU process. Acquire the shared exclusive research-compute file lock. The test has a 300-second elapsed limit, including its calculation audit. The three measurement notes together have a 900-second limit. The root supervisor must set `MI_USAGE_PERMIT` to a valid private permit, refresh it by atomic replacement after a live account check, and preserve the 20% reserve. The old shared guard checks the permit, time, and four-GiB process limit. A failed or missing check stops the run. Do not alter inputs or acceptance rules after seeing results.

Before calculation, the plan, source list, and code must be in a public commit. The runner checks their bytes against the supplied commit. It refuses an existing result directory. Save raw rows, the complete case list, a summary, a separate equation audit, runtime, environment, and SHA-256 hashes. Keep failures. Successful checking means that the implementation matches the declared construction. It does not validate a claim about trained language models. No human peer review is recorded.
