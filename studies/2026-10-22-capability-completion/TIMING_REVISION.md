# One fixed revision to the Hall timing method

**Date:** 22 October 2026. **State:** Method fixed before the revised benchmark.

The first fresh preparation attempt passed all arithmetic and permit checks. Its combined fourfold compute estimate was 20,995.247604 seconds. This exceeded the unchanged four-hour cap. Its files remain unchanged in `preparation/`. No scientific model ran in this attempt.

The dominant estimate came from the Hall branch. Its 20-update dummy training time was 0.240139 seconds, compared with 0.078935 seconds in the earlier preparation. A short measurement can include start-up cost. This observation does not establish the reason for the difference. One fixed measurement revision will separate start-up work from steady update work. It will not reduce the safety factor, extend the deadline, or change scientific models or data.

## Fixed measurement

Use the same wider Hall model, width 128, batch 512, binary64 arithmetic, and Adam learning rate 0.002. Use random continuous input arrays and independent random binary labels with seed 99301. Fix model seed 99302. These inputs do not measure graph-task accuracy.

Run exactly 20 warmup updates. Then run three blocks of 100 updates. Keep the maximum block time per update. Do not choose the minimum or repeat the benchmark to obtain a lower value. Each update includes the same loss, gradient, finite-value, usage, and deadline checks as scientific training.

Use the larger of ten seconds and the measured time for model/optimizer setup, all warmup updates, and all three timing blocks as the per-job setup allowance. The estimate is:

`4 * (maximum seconds per update * 32768 + old evaluation/replay allowance + setup allowance)`.

Recover the old evaluation/replay allowance from the saved fresh Hall estimate by subtracting its measured training term and ten-second setup term before multiplication by four. Use it without change. Preserve the register, digit, and flow estimates from the failed fresh preparation without change. Keep the conservative eight-job count for every task, even though one graph job is reused. Add the prior 52.152487084 seconds of scientific work.

The revised total must be at most 14,400 seconds. The initial failed projection remains a failure. A pass under this revised measurement is a separate acceptance decision. Save both estimates and all measurements.

## Caps and decision

This benchmark has one attempt. If any control, usage rule, memory limit, or revised budget check fails, stop this continuation. Do not repeat this benchmark or add another measurement revision.

Count the original preparation, initial permit test, failed fresh preparation, and this revised measurement against the same one-hour preparation cap. The fourfold safety factor and four-hour scientific deadline remain unchanged. No scientific training starts until the revised preparation passes and its records are included in a verified public code lock.

The public timing-method commit must precede this dummy benchmark. The public scientific lock must also include the resulting `revised-preparation/status.json` and file manifest. The runner verifies the original failed preparation records as well as the revised records.
