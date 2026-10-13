# Dummy checks before the study

The fourth receipt matches the current plan and runner. The first three receipts remain unchanged. Version 2 added control maxima to the record. Version 3 added the exact initialization sequence to the metadata. Version 4 removed absolute local paths from failure tracebacks. Those changes did not change the model or score calculation.

No scientific model was trained or evaluated. The run used standard-normal dummy inputs and seed 29002. The separate hand-set witness used seed 29003. It made a nonzero removal effect of 0.0512400707515139 and a nonzero total log-scale of 2.42143796250682. Its largest round-trip or Jacobian error was 4.44089e-16.

All 129 conditions passed the checks. The main dummy run used 2,401 mapping calls, 2,181,184 input examples, 200 updates, and 1,169 control rows. The hand-set witness added five calls and 15 examples. The largest coordinate error was 1.33227e-15; the largest log-determinant error was 6.10623e-16. All local hidden-unit checks were exact. The fixed limit is 1e-9 for inverse and determinant errors, and zero for no-op and local checks.

The example-scaled full-run estimate, with the fixed factor of two, is 197.41 seconds. The scientific cap is 1,200 seconds. This estimate is a budget check, not a measured scientific runtime.

The exact file hashes, versions, counts, and result hashes are in [the fourth receipt](dummy-preflight-v4.json). These are AI checks. They are not human peer review.
