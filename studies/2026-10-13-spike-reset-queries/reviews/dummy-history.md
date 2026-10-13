# Dummy test history

All four tests used only random inputs and a hand-set demonstration. No UCI input was passed through a model. No study model was trained or evaluated.

- v1 passed 422 numerical controls. It tested all real batch sizes and an active/inactive reset example. It produced a PyTorch scalar-conversion warning. The receipt remains at [dummy-preflight-v1.json](dummy-preflight-v1.json).
- v2 passed the same 422 controls after a scalar-output cleanup and record clarifications. It added the separate witness call count. Its receipt remains at [dummy-preflight-v2.json](dummy-preflight-v2.json).
- v3 added the reviewer's active final-readout equality check and both witness gradient/tangent checks. These passed, with a nonzero difference between the two local gradients. It revealed one more harmless scalar-output warning in the trace comparison. Its receipt remains at [dummy-preflight-v3.json](dummy-preflight-v3.json).
- v4 contains only the final scalar-output cleanup after v3. It passed all 422 controls and the complete witness tests. The complete outputs and their manifest are in [dummy-preflight/](dummy-preflight/). This receipt matches the final plan and runner bytes.

The final conservative estimate is 813.59 seconds with a two-fold factor. It scales the measured training, test, and panel costs separately. It is an estimate, not a measured scientific run time. The final witness has an exact readout error of zero, tangent errors of zero and 3.47e-18, identical forward leaf traces, and a local-gradient difference of 0.0050752. The original dummy folders remain in local staging.
