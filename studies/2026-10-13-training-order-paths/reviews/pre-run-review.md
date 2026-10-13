# Independent pre-run review

13 October 2026. Reviewer: a separate AI agent. This is not human review. Status: ready for the public plan/code lock. No scientific model was trained or evaluated during this review.

I read the complete analysis plan, runner, source record, and dummy preflight receipt. The current receipt is `preflight-v3.json`. Its plan and runner hashes match the final files. The first receipt is retained as an earlier record.

The primary endpoint is fixed: the paired BA-minus-AB difference in the selected-eight path effect, after 2,000 balanced tail updates with momentum reset. The runner uses that endpoint. It does not replace it with an earlier or stronger result. The effect sign is consistent with the stated shortcut direction.

The schedule code exposes each of the 32 examples exactly 500 times in every prefix. Each batch has balanced labels. The three schedules start from exact copies of the same seeded model. Both tail conditions start from the same prefix weights and optimizer state. Only the reset branch clears the momentum buffers. The full tails have the same batches and update count.

The intervention changes only the first-layer input from the shortcut coordinate into selected units. Full replacement has a natural donor reference. The direct and analytic preactivation changes are checked. Self replacement and true restoration have separate checks. The selected mask uses calibration contexts only. Test contexts determine no rank, unit count, checkpoint, or schedule. The report must retain the distinction between held-out intervention contexts and unseen training inputs.

The seed is the correct resampling unit for the stated endpoint. The runner pairs schedules before resampling. It records all eight seed differences. Probability and logit-margin effects, full-input effects, and intact loss are saved. These records are needed to distinguish a partial path effect from whole-model shortcut use, and to detect confidence differences at equal accuracy.

The code preserves partial tables and the current model and optimizer on failure. It checks the time cap during training and evaluation. It verifies locked bytes at the start and end. The result receipt and manifest include the relevant counts and hashes.

The supplied dummy receipt reports zero self, full-donor, and restoration errors. The analytic edge error is approximately 1.11e-16, below the fixed 1e-10 limit. This check does not show that scientific training will pass. No blocking issue was found in the code or plan. The receipt issue was resolved by checking `preflight-v3.json`. The final code also checks package versions, expected checkpoint and table counts, and the time cap after summary generation.

Limits for the final paper: this is a small constructed parity task. A finite tail cannot show permanent memory. Momentum reset does not remove history from the weights. Each model and checkpoint has its own selected mask. The endpoint compares a selection procedure. It does not track the same units across models or time. The source record states the close overlap. Do not claim a new training-order principle or a general result about language models.
