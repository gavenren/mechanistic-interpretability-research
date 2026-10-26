# Separate AI review before execution

**Date:** 26 October 2026. **Decision:** No blocking issue remains in the reviewed plan and code. Publish this review with the fixed plan before preparation. Scientific execution still requires passed preparation and its published record.

This was a separate AI review. It was not human peer review. The reviewer read the plan, three method files, source notes, input manifest, shared runner, branch code, and resource tests. The reviewer did not run preparation, a full benchmark, trained-model calculations, or training. Only this review file was edited by the reviewer.

## Method and evidence checks

- The selector branch keeps the eight saved models and their existing three-neuron masks. It fixes complete receiver/request blocks before model outputs. Each model has 36,864 donor rows, 2,304 control rows, and 12,288 order rows. The eight-model total is 411,648 rows. Both edit orders stay in the same split. The direct last-donor calculation checks the composition code. The method states that last-write behavior is a known property of this operator.
- The code branch fixes six saved encoders and 48 cases per model. The 288 cases include 96 development cases and 192 test cases. Its finite search can produce valid witnesses. It cannot prove uniqueness or establish which codes the encoder produces. Candidate counts include duplicates. The saved arrays are reopened and checked with separate scalar calculations, constraints, error limits, and witness flags.
- The normalization branch fixes the model revision, cached file hashes, runtime versions, prompts, sites, and edit sizes. The 4,128 scientific forwards and four dummy forwards have a strict bound of 528,896 tokens. The runner carries preparation tokens into the scientific count. Full logits are used for calculation checks; only eight fixed token scores form the saved score panel.
- The normalization reference uses a separate formula for the edit and normalization. It shares the downstream transformer code. The method states this limit. It makes no copying or semantic claim.
- All branches fix their tolerances before results. A numerical failure stops the attempt. A valid zero effect or disagreement can be a completed diagnostic result. The three branches form one study.

The main agent reported that all 31 repository input files match the fixed source commit. The branch code also checks the relevant saved assets. The cached language checkpoint has separate fixed file hashes. No model download is permitted.

## Resource and failure checks

The runner uses one CPU process, two compute threads, one inter-operation thread, and an exclusive compute lock. Preparation has a four-GiB memory limit. Scientific execution has a 16-GiB limit. Both require 20 GiB free disk. Permits must be valid, finite, and no more than 120 seconds ahead. The stage and token limits include controls and replays.

The runner now keeps a failure receipt if the first permit or source check fails. It requires the declared runtime versions and the same runtime record in preparation and science. Main saving and the full hash scan count toward elapsed time. The two-second closing allowance is charged. An overrun changes the result to stopped and updates the cumulative charge fields.

The review raised and checked fixes for the preparation memory limit, an unset preparation record on early failure, saved-file replay, runtime matching, and closing-write accounting. No unresolved blocker remains from these checks.

## Small tests and limits of this review

The reviewer ran the selector's index and arithmetic checks, a hand-made duplicate-code case, and an analytic disk case. All passed in **0.1499945 seconds**. No saved model was loaded. No preparation timing benchmark ran. This time is within the shared 120-second pre-lock check allowance.

The main agent separately reported that all four resource tests passed. The reviewer read those tests. They cover token charging, invalid permits, time, memory, disk, and a failure receipt after an initial denied permit.

These tests do not establish scientific results or prove that all defects are absent. Dummy timing is still required. The fourfold forecast is an allowance, not a guaranteed runtime. A long native operation can delay the signal check. Keep the recorded stop rules and report any observed overrun. Do not change the design to obtain a pass.
