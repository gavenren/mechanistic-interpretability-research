# Independent result review

13 October 2026. Reviewer: a separate AI agent. This is not human peer review.

## Audit result

The independent audit passed. It did not import the scientific runner. It used saved weights to reconstruct all 144 evaluated checkpoints and all 198,144 intervention rows. It also reconstructed the calibration ranks, selected masks, set means, primary paired-seed differences, and bootstrap interval.

The audit checked all 176 checkpoint hashes, the exact initial parameters, equal fork weights, zero reset buffers, input and donor tables, batch sequences, example counts, and 120,000 training-record rows. It verified the result manifest. It did not replay training trajectories. Thus the saved-state calculations have a full independent replay; the optimizer's full update history has record checks only.

Maximum numerical error was 1.33e-15, below the fixed 1e-10 audit limit. The receipt is `independent-audit.json`. The code is `../code/independent_audit.py`.

## Findings checked from the saved rows

The primary reset BA-minus-AB mean is -2.7367253279468075e-5 in probability units, or -0.0027367253 percentage points. The descriptive interval is [-4.813700031744332e-5, -7.691894210918699e-6]. Its direction is opposite to the planned hypothesis. Its small size must be central to the report.

All 48 models have perfect intact classification accuracy at both tail update 200 and tail update 2,000. Their losses and margins still differ. Equal accuracy must not be described as equal predictive distributions or equal confidence.

At tail update 2,000, all 66,048 recorded patched outputs retain the correct class. The counts are 49,152 single-unit outputs, 1,536 selected-set outputs, 1,536 complement outputs, 1,536 full-set outputs, and 12,288 random-mask outputs. These are repeated interventions on 48 models, not independent successes or general robustness evidence.

The mean reset-minus-keep change in the primary contrast is approximately -9.714659e-9 probability units. This comparison does not show that all momentum effects vanish. It concerns the declared endpoint and schedules.

## Claim limits

The main result is a small conditional probability effect in a fully learned, constructed parity task. It is not a discovered failure of classification. Probability saturation and full-input contrasts must stay visible beside partial-path effects. A partial route can have an effect while other routes cancel it.

The masks are selected again for each model and checkpoint. The report cannot infer that the same units retained their roles across training. The finite tail also cannot establish permanent history or an irreversible phase transition.

The final paper, README, and figure were reviewed. Their numerical claims agree with the saved records. The report leads with the small absolute effect and opposite direction. It keeps probability and margin effects separate, states the complete finite-table accuracy scope, and discloses that training was not independently replayed. The figure is legible and uses percentage points. No scientific or presentation blocker remains. The review is closed.
