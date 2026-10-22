# Selecting the largest noisy effect can create false evidence

**Date:** 22 October 2026. **Type:** Statistical simulation with analytic checks. No neural network was trained. AI-authored; no human peer review.

## Main finding

In a fixed zero-effect simulation, selecting the largest of 128 independent estimates and testing that same estimate gave a false positive in 99.94% of trials at 64 cases per estimate. A separate validation test gave 4.38%. A multiple-test correction gave 5.08%.

These are results for an explicit Gaussian measurement model. They are not estimates of false-positive rates in published circuit studies. All true candidate effects were zero by construction.

## Question, prior work, and contribution

Circuit searches compare many candidate components. Reusing the selection data for a significance claim can make noise look useful. [Kriegeskorte and colleagues](https://pmc.ncbi.nlm.nih.gov/articles/PMC2841687/) established this broader problem of non-independent selection and testing. [Conmy and colleagues](https://arxiv.org/abs/2304.14997) provide relevant context for automated circuit search. This study does not evaluate their algorithm.

The addition is a reproducible reference test for candidate-selection pipelines. It compares three fixed rules under known zero effects. It does not propose a new statistical principle or infer which real circuits are false.

## Model and derivation

Each candidate's estimated effect has variance 1/n, where n is the number of cases. Candidate errors have common correlation rho. Calibration and validation draws are independent. We used 1, 8, 32, or 128 candidates; 16, 64, or 256 cases; and correlations 0, 0.5, or 0.9. Each of the 36 conditions has 5,000 trials, for 180,000 trials in total.

Select the largest calibration estimate. The naive rule tests its z score at the one-sided 5% threshold. The corrected rule uses the Bonferroni threshold for all K candidates. The validation rule tests only the selected candidate on independent data.

For independent candidates, each calibration estimate stays below the naive threshold with probability 0.95. All K stay below with probability `0.95^K`. Thus the selected estimate crosses it with probability `1-0.95^K`.

For the independent validation split, selection does not change the selected candidate's validation distribution. Its false-positive probability remains 0.05. The Bonferroni rule bounds the familywise probability by 0.05 under the declared marginal distributions, even with dependent candidates. These probability bounds do not force every finite simulated fraction to be at most 0.05.

## Results

This table shows the middle sample-size setting, n=64, with independent candidates. [All 36 conditions](results/cells.csv) are retained.

| Candidates | Reused selection data | Bonferroni rule | Independent validation |
| --- | ---: | ---: | ---: |
| 1 | 5.34% | 5.34% | 5.36% |
| 8 | 34.06% | 5.00% | 5.54% |
| 32 | 81.64% | 4.58% | 4.96% |
| 128 | 99.94% | 5.08% | 4.38% |

Across all conditions, validation differed from its 5% reference by at most 0.62 percentage points. Independent-candidate naive rates differed from the analytic formula by at most 1.02 points. All fixed simulation checks passed. Correlation reduced the number of effectively distinct candidates, but did not make reuse of selection data valid.

The audit regenerated the exact seeded candidate banks. It checked every selected index, saved value, count, and Wilson interval with separate scalar calculations. Maximum metric error was 6.94e-18. Calculation and audit took about 50.27 seconds inside their scripts, below the shared 120-second limit. The audit dominated the time.

## Limits and use

Known Gaussian variance and a zero-effect model make the reference transparent. Real patch estimates can be heavy-tailed, dependent across examples, and affected by adaptive design. The simulation does not establish their calibration. It also does not measure statistical power for true effects.

Wilson intervals in the saved table describe simulation uncertainty for each condition. They are not simultaneous intervals and do not estimate a real-model rate. More data reduce the raw noise scale, but under this known-variance null model they do not remove selection bias in the standardized test.

Use a separate validation set or an appropriate selection-aware test before making a claim from the largest effect. A later experiment should test that procedure with declared real-model dependencies and nonzero effects.

See [plan](analysis-plan.md), [code and instructions](README.md), [saved draws](results/draws.npz), [audit](results/audit.json), and [separate AI review](PRE_RUN_REVIEW.md). The code lock is `88e8df7689b65fbeef221423c115cd1b1c198f22`. Sources were checked on 22 October 2026.
