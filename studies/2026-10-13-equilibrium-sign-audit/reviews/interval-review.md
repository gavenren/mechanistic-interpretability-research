# Independent pre-run interval review

**Status: passed within the stated scope. No arithmetic blocker found.**  
**Date:** 13 October 2026.  
**Reviewer:** a separate AI agent. No human review or formal software verification is claimed. No scientific model was called or trained for this review.

## Scope and target

I read the fixed plan, interval module, exact rational checker, and its receipt. I also inspected the protocol's effect-bound calls. Root owns the complete training and runner review. This note approves the interval method under its stated arithmetic assumptions. It does not approve scientific results, which do not yet exist.

The target is the exact-real equilibrium defined by the stored binary64 coefficients and inputs. The target is not a finite floating-point iterate. Coordinate removal replaces one state equation by exact zero and keeps the other equations fixed.

## Mathematical and code checks

- `contraction_upper` sums absolute stored weights with an upward step after each addition. The final maximum does not round. Thus it bounds the exact infinity operator norm. The code checks this bound below one. It does not rely on the training projection reaching exactly 0.875.
- ReLU is nonexpansive in the infinity norm. Replacing one coordinate equation by zero preserves the same contraction bound. Both maps therefore have one exact-real equilibrium.
- `affine_point` encloses each exact product and each ordered addition. The injection interval encloses the exact `Bx+b`, rather than treating a rounded matrix product as the target injection. ReLU and the zero-coordinate assignment preserve containment.
- The residual subtraction encloses the exact `F(h)-h` at the stored candidate. The maximum endpoint magnitude bounds its infinity norm. `down(1-q_upper)` is a lower bound on the positive contraction denominator. Upward division then bounds the state error by the contraction residual theorem.
- The readout-row difference is itself an interval. Its absolute endpoint bounds and upward sum bound the exact coefficient L1 norm. The effect centre is also an interval, including state subtraction and all dot-product operations.
- The summed residual, division, radius multiplication, and final endpoint expansion all round outward. The resulting interval contains the exact equilibrium margin drop. The readout biases cancel because the same fixed class pair is used for the intact and edited states.
- The intact output-logit bounds use the same state-error argument with each readout-row norm and an enclosed affine centre.

These checks support the declared real-valued bound. They do not prove that every returned interval is narrow or that an unresolved ordinary sign is wrong.

## Exact tests performed

I verified that the author's receipt matches the reviewed module and checker hashes. Its 1,697 recorded checks all pass. Its exact active-set enumeration includes positive, negative, and zero effects, as well as slow and inactive cases.

I wrote and ran a separate dummy rational checker. It made **1,512 exact comparisons and failure-rejection checks**. It tested interval addition, subtraction and multiplication; signed and zero values; subnormal values; ReLU; ordered sums; and absolute-value bounds. Forty fixed two-dimensional systems tested the norm, denominator, input injection, residual, coefficient norm, centre, radius, and complete effect interval. Their exact equilibria came from independent active-set enumeration with closed-form two-by-two linear solves. The largest checked contraction upper bound was **0.9999999990686776**. A cancellation case used adjacent large binary64 readout values. All full output-logit intervals also contained their exact targets.

The checker confirmed rejection of a noncontractive norm, a zero denominator, infinite input, reversed endpoints, and upper-endpoint overflow. It used no UCI examples and made no scientific model calls. The [separate receipt](independent-interval-checks.json) records the counts and hashes. The [checker](../code/review_interval_checks.py) is reproducible with a fresh output path.

## Assumptions and limits

Containment depends on IEEE-754 binary64 elementary operations in round-to-nearest mode, gradual underflow, correct `nextafter`, and separate operations without fused arithmetic across NumPy calls. Finite inputs, valid endpoint order, and positive denominators are required. The code rejects nonfinite bounds, including expansion beyond the largest finite float. The hardware, floating-point mode, Python, and NumPy form a trusted computing base. These tests do not formally verify that base or all possible inputs.

This review does not test the scientific training, input panel, stop-rule implementation, output counts, time cap, or later result interpretation in full. Those remain subject to root's runner review and the result audit. No change to the interval module was needed.

## Reviewed file hashes

These hashes identify the exact files inspected. Later changes require review of the changed scope.

| File | SHA-256 |
|---|---|
| `analysis-plan.md` | `01703bd5d11172898c1ab6b6935d019cc3d1c0920a3af46dccf8a7cfbecfe015` |
| `code/intervals.py` | `0bb5a0275fc645f65dcf6a619b8e0bd6eb217e7fef3b5470bedb1e6d434783ca` |
| `code/check_intervals.py` | `73e0f4294e1f0da0dd93e93167b969e6314528c4cfd385384a8961e4825545a3` |
| `code/protocol.py` | `a5884b49cf462129585bc3670ab8e964e2ab4e52f536be12e2125951e0ed3922` |
| `code/review_interval_checks.py` | `883e9453bc0089b5cb6fa0d5eb7691f8747dfcdf59a70da2c3bcd05520c56dab` |
| `reviews/interval-dummy-v1/receipt.json` | `1d08abe41df59432b75c85c57d822652b5a0f655ddde8a45f54875ec3dca9260` |
| `reviews/interval-dummy-v1/checks.json` | `4a513cc8d54dd9fe258988a5cfd52a1b1948e080500782caccb1a3718760f764` |
| `reviews/independent-interval-checks.json` | `e368f930234b910ab0533d7fe34bbd9d88443a577c7089f003d904c1491046ee` |
