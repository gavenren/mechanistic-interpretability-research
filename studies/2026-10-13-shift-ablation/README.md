# Digit shifts and channel ablation

Status: the original fixed experiment stopped at a numerical control. The planned primary comparison is incomplete. This study is AI-authored and has no human review.

The study asks whether filtering before downsampling makes the measured effect of removing an internal channel more stable when a digit moves by one pixel. The data are real handwritten digits from UCI. The two trained architectures use the same initial parameters and data schedules within each seed pair.

The [original plan](analysis-plan.md), [runner](code/run.py), and raw data were fixed at public commit `c406a08091cbc74caef1a1c4690aa29482667e0a`. Four models completed. A control in the fifth model gave a maximum logit error of 0.0000214577, above the fixed 0.00002 limit. The runner stopped after 1,500 updates and 57.50 seconds. All completed and partial records remain in [results](results/). The original tolerance remains unchanged.

A [separate post-failure plan](reviews/failure-diagnostic-plan.md) fixes an exact-batch numerical diagnostic. Its [code](code/failure_diagnostic.py) and [review](reviews/failure-diagnostic-review.md) are included. This diagnostic does not train a model or complete the missing comparison. Its results must not turn the original failed check into a pass.

See [sources](sources.md), [data attribution](data/README.md), and [pre-run review](reviews/pre-run-review.md). A later follow-up, if justified, must have a separate public plan and clear status.
