# Exact archive transport before final tests

Date: 26 October 2026. Calibration is complete. Final tests have not run.

The complete calibration archive contains 90 files and 30,963,232 bytes. It exceeds the 25 MB browser upload limit. This amendment changes only how the same bytes reach GitHub. It does not change training, edit selection, final cases, scores, gates, tolerances, or compute limits.

Keep the original code and plan unchanged. Divide the frozen archive at 20,000,000 bytes into `frozen.zip.part01` and `frozen.zip.part02`. Store the parts and manifest in the separate `transport` folder. Keep the calibration folder unchanged. Publish both parts, their ordered byte and SHA-256 manifest, this plan, the separate review, and `analysis/final_transport.py` before final tests. The original archive SHA-256 is `64c6fe719663885b2db61f8a9c91e6a59abd5b16eef93c79bc8624e73b57aec7`.

The wrapper changes only the public-byte check for `calibration/frozen.zip`. It checks its own public source, this plan and review, the transport manifest, and each part's public bytes. It checks that concatenation equals the local frozen archive exactly, including length and digest. It rejects changed parts, missing parts, order changes, and extra bytes. The original runner then checks the full 90-file archive and all original phase locks. All other byte checks use the original function.

The wrapper calls the unchanged final phase under the same exclusive compute lock, live-usage permit, memory and disk limits, fixed runtime, and cumulative stage clock. Archive verification is inside that clock. Earlier calibration results cannot alter final choices. Two absent eligible learned edits remain absent.

The original frozen-archive pack and replay is charged 20 seconds. Charge another 10 seconds from the same 120-second post-run reserve for splitting and transport mock checks. This leaves 50 seconds for the result audit and 40 seconds for final archive checks. Do not add a new reserve.

To reconstruct the exact ZIP from the repository root:

```sh
cat studies/2026-10-26-variable-specificity/transport/frozen.zip.part01 studies/2026-10-26-variable-specificity/transport/frozen.zip.part02 > studies/2026-10-26-variable-specificity/calibration/frozen.zip
```

Verify the declared digest before use. Extract the archive into `calibration` to restore the `edits` folder. The ordinary training files and preparation remain published directly.

Run the final phase only after public verification:

```sh
MI_USAGE_PERMIT=/absolute/private/permit.json .venv/bin/python studies/2026-10-26-variable-specificity/analysis/final_transport.py --lock PUBLIC_TRANSPORT_AND_CALIBRATION_COMMIT
```

This is a transport amendment after calibration and before final testing. It is not a new experiment or a completed paper.

The first transport mock checked only the wrapper. Review then found that parts in the calibration folder would fail the original membership check. The first wrapper, test, and receipt are retained in `analysis`. The corrected test also calls the original calibration verifier. Both check attempts share the same 10-second transport allowance. No final test ran under the first layout.
