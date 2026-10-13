# Independent review before the run

Date: 13 October 2026. Reviewer: a separate AI agent. This is not human peer review.

## Status

Ready for the plan and code lock. The plan and runner have no unresolved design blocker. I read the passing dummy cache receipt. No scientific outcome was used in this review. The experiment must still wait for a verified remote lock.

## Design checks

- The layer choice uses only four calibration word pairs. Both directions and all frames for one pair stay in one split. The four test pairs use new words in known frames.
- The endpoint compares the two selected layers on the same test cases. All other layers and horizons remain public. Ties, negligible values, and null results have fixed report rules.
- The common suffix is fixed as token IDs. It does not come from model output. The profile therefore tests a controlled continuation, not free generation or a pure effect of elapsed time.
- The edit occurs once, after a block at the original final prefix position. No suffix position is edited. Full recomputation must use that same original position.
- Cache copies, a fresh-cache reset, and cache-versus-full-recompute checks can separate a persistent cache effect from a hook or mutable-cache error.
- The final block has no later cache writer. Its zero-future-effect rule is a known architecture control, not a new scientific result.
- Total variation uses the full vocabulary. It measures generic output change. It does not show correct steering, useful memory, or semantic fidelity.
- The residual-change norms help describe edit size. Larger later effects can result from larger initial changes. The study does not divide by a small immediate effect or claim a memory decay rate.
- The plan reports each test pair and frame. It makes no broad statistical claim from four held-out pairs.
- Double arithmetic retains shared float32 rotary tables. The plan names this limit and fixes strict numerical control bounds.

## Code checks before the lock

I read the complete runner. Each continuation starts from its own cache copy. The hook is removed after prefill. Full recomputation uses the original final prefix position. Cache checks cover all required layers and positions. The selector reads calibration rows only. Total variation uses all 50,304 output tokens. The fixed loops give 14,592 model forwards and 3,456 score rows, as stated. Control failures are saved and prevent a complete status. Plan, runner, and model hashes are checked before and after the run. The output directory cannot be reused.

The [preflight receipt](../preflight.json) records 46 model calls on separate dummy strings. The maximum cache-versus-full logit error was `4.424904886946024e-12`, below the fixed `1e-8` bound. All 64 scientific prompts passed token checks only. The receipt records the plan and runner hashes. This result shows that the fixed tolerance is attainable. It does not replace the complete study controls.

## Review limits

This review checks the plan and code. It does not show model task ability or a causal result. A later result audit must check saved scores and replay the model with a separate implementation. The report must retain weak task performance, small effects, null results, and failed controls.
