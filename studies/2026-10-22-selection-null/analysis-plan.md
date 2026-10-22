# Fixed plan: candidate selection under a zero-effect model

Date: 22 October 2026. Statistical simulation with analytic reference. No neural network is trained. No scientific run has started.

Question: how often can a circuit search appear significant when it selects the largest noisy effect and tests that value on the same data?

The true effect of every candidate is zero. Per-row noise has standard deviation one. For K=1,8,32,128 candidates and n=16,64,256 cases, simulate the exact Gaussian distribution of the sample means. Within one split, candidate errors have common correlation rho=0,0.5,0.9. The mean error is (sqrt(rho)*g_common+sqrt(1-rho)*g_candidate)/sqrt(n). Calibration and validation draws are independent. Use 5,000 trials per cell and independent NumPy seed 452000+cell_index. There are 36 cells. Save every selected calibration value, candidate index, and selected validation value. No outcome-dependent seeds or repeat counts.

Select the largest calibration value. Compare three one-sided 5% rules: naive reuse of its calibration z score; Bonferroni calibration cutoff Phi^-1(1-.05/K); and one independent validation test of the selected candidate at Phi^-1(.95). Known variance is assumed. There are no actual candidate effect sizes to discover. This tests a measurement protocol, not a circuit's truth.

For rho=0, naive familywise error is 1-.95^K. For all rho, the independent validation error is .05. Bonferroni error is at most .05 under the declared marginal Gaussian law, regardless of this dependence. Derive these statements. Report observed counts, mean selected effect, validation mean, and Wilson 95% intervals. Do not claim the interval estimates a real-model rate. A maximum discrepancy of .03 for the analytic probability checks is a fixed simulation check, not a hypothesis test of the field.

Independent audit: read saved arrays; recompute counts, standard errors, and Wilson intervals in scalar Python; check calibration argmax values against saved per-trial selected values where available; confirm selected index range and finite values. Independently implement Gaussian CDF with math.erf for analytic probabilities. Preserve failed checks. Two CPU threads, 120-second cap including audit. Root supplies live private usage permit and serial launch after public lock. This is distinct from earlier patch-transfer papers: it isolates selection noise before any model claim.
