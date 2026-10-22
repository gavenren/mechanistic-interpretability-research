# Fixed plan: greedy pruning and the smallest faithful circuit

Date: 22 October 2026. Status: fixed before calculation.

## Question and contribution

Can deletion stop at a circuit from which no single component can be removed, while a smaller faithful circuit exists? Earlier ACDC work already explains order effects for OR gates. This note adds an exact size comparison and an audit of every deletion order in a small declared panel. It does not reproduce the full ACDC algorithm or claim a new principle. It is separate from a test of necessity versus sufficiency: the objective here is the search error for one fixed faithfulness definition.

## Constructed model and panel

The input is a bit x. Each of n hidden components copies x. A route outputs the AND of its component values. The output is the OR of all routes. A removed component is fixed to zero. The unedited answer is x. A retained subset is faithful only when it gives the correct output for both x=0 and x=1.

Five analytic cases have one route {0} and another route {1,...,k}, for k=2,3,4,5,6. Add 12 six-component cases. Use Python random seed 22102201. For each case draw six distinct nonempty subsets of size at most four from the lexicographically ordered list. Remove duplicate routes and routes that contain a smaller route. Retain all generated cases, including easy ones. The generated routes are the exact model specification, not learned circuits.

For every case, evaluate all 2^n retained subsets and every n! component order. Start each greedy run with all components. Visit each component once. Delete it if the current subset remains faithful. Save its final subset, size, and objective-query count. Find the minimum faithful size by exhaustive subset search. Use exact integers and Boolean outputs.

## Fixed claims and checks

The main outcome is the fraction of orders for which the greedy size exceeds the exact minimum. Also report the maximum gap, mean size, and counts of each final subset for each case. Do not pool the constructed cases into a population failure rate.

For the analytic family, prove that the minimum is one, the greedy result has size k exactly when component 0 is visited first, and the fraction of such orders is 1/(k+1). The mean size is 1+(k-1)/(k+1). These are deductions fixed before the computation, not data-fitted predictions.

Controls: the full set must match both task inputs; the empty set must fail x=1; every final greedy set must be faithful and inclusion-minimal. An independent audit uses integer bitmasks and ReLU-style Boolean gates to recompute every subset output, every greedy endpoint, the exact optimum, and the closed-form family counts. All comparisons must agree exactly. Any discrepancy fails the audit and blocks interpretation.

## Resource and record rules

This is a constructed formal test. It trains no model and uses no paid service. Run one local CPU process. Acquire the shared exclusive research-compute file lock. The test has a 300-second elapsed limit, including its calculation audit. The three measurement notes together have a 900-second limit. The root supervisor must set `MI_USAGE_PERMIT` to a valid private permit, refresh it by atomic replacement after a live account check, and preserve the 20% reserve. The old shared guard checks the permit, time, and four-GiB process limit. A failed or missing check stops the run. Do not alter inputs or acceptance rules after seeing results.

Before calculation, the plan, source list, and code must be in a public commit. The runner checks their bytes against the supplied commit. It refuses an existing result directory. Save raw rows, the complete case list, a summary, a separate equation audit, runtime, environment, and SHA-256 hashes. Keep failures. Successful checking means that the implementation matches the declared construction. It does not validate a claim about trained language models. No human peer review is recorded.
