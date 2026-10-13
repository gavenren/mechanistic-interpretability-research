# What the exact-batch diagnostic shows

The diagnostic reproduced the original failure. It does not change its status. The float32 experiment remains stopped and incomplete.

The check used the saved seed-2 direct model, test rows 768 through 895, channel 1, and a four-pixel left shift. No model was trained. The plan and code were public before these diagnostic outputs, in [commit 5d7a88d](https://github.com/gavenren/mechanistic-interpretability-research/commit/5d7a88d9e79d974f7e320359039e11e813d20b15). The [receipt](../diagnostics/failure-batch/receipt.json), [arrays](../diagnostics/failure-batch/batch-logits.npz), and [manifest](../diagnostics/failure-batch/manifest.json) preserve the evidence.

All 22 diagnostic checks passed. The separate hook path exactly reproduced the saved unshifted float32 logits and the recorded maximum shift error, `2.1457672119140625e-5`. That value still exceeds the original `2e-5` limit.

| Computation with the same saved weights | Maximum logit change under the four-pixel shift |
|---|---:|
| Standard convolution modules, float32, separate edit hook | 2.14577e-5 |
| Unfolded-window convolution and separate spatial sums, float32 | 4.76837e-7 |
| Standard convolution modules, float64, separate edit hook | 8.88178e-16 |
| Unfolded-window convolution and separate spatial sums, float64 | 0 |

The two float64 paths agreed with each other within `1.78e-15` on both inputs. In the float32 hook path, the input, edited first-ReLU output, and first downsampling output obeyed the expected spatial permutation exactly. The first nonzero error appeared after the second convolution and ReLU (`1.584e-5`). The largest intermediate error was after the third convolution and ReLU (`4.053e-5`). The final logit maximum occurred at test row 816, output class 0.

These observations are consistent with numerical error in the float32 computation. The result is not evidence that the channel edit or input shift changed the intended mathematical architecture. The alternative float32 implementation had a smaller error, which also shows that evaluation order can matter. This small diagnostic does not assign all error to one operation, establish an error rate across models, or make float64 ground truth for the exact float32 program.

The test covered one model, one edited channel, one batch, and one shift. It did not check missing models or complete the original primary comparison. A full float64 evaluation must be a separately fixed, explicitly post-failure analysis. Its results must not replace the failed original record.

The diagnostic took 0.31 seconds on two CPU threads. It used eight batch conditions and no training. The diagnostic plan, code, receipt, and arrays are now held fixed.
