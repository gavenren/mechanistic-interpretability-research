# Follow-up dummy checks

Both runs used random dummy inputs only. Training was float32; evaluation was float64. No UCI image was passed through a model.

- v1 passed all 1,197 controls.
- v2 passed the same controls after a final time check was added after the end-of-run reuse verification. The full v2 files are in `followup-dummy-preflight/`. Its plan and runner hashes match the proposed lock files.

The final projected time, including the two-fold factor, is 406.5 seconds. This is an estimate. The hard scientific cap remains 1,200 seconds. No tolerance, input, seed, estimator, or training rule changed between these dummy runs.
