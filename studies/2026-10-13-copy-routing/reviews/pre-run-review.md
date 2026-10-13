# Independent review before the run

Date: 13 October 2026. Reviewer: a separate AI agent. This is not human peer review.

## Status

Ready for the plan and code lock. The study design is suitable for a fixed, small replication. The code review found no error in the input split, head choice, intervention signs, or group bootstrap. The record checks and dummy numerical checks below are closed. No scientific outcome was used in this review. The experiment must still wait for a verified remote lock.

## Design checks

- The two binding assignments are needed. They prevent a fixed association between A and B from counting as use of the given key–value binding. Both pair orders and both final keys also stay in each group.
- The calibration and test keys and values are separate. Shared filler words limit the claim about new vocabulary. The plan states this limit.
- The head is selected with calibration attention scores only. The code saves the choice before test interventions. Test results for other heads remain descriptive.
- The route edit keeps self-attention and total attention to the past fixed. It can change attention to many past positions. It is an artificial route edit, not a pure semantic change.
- The value edit uses a temporary value matrix for the final query of one head. The code replaces only that query's head output. It does not change earlier query outputs.
- The three primary effects form a fixed conjunction. The baseline and test attention checks further limit a positive mechanism claim. A failed conjunction must remain a valid result.
- The bootstrap samples 16 whole test groups. It keeps both bindings, both orders, all gaps, and both queries together. The code does not count prompt variants or heads as separate independent samples.
- The primary rescue is relative to the route edit. Rescue relative to the value edit is separate. Neither result proves a complete circuit.

## Code checks and changes before the lock

1. The runner checks the plan and runner bytes against the lock commit before the run and again before completion.
2. The runner saves a result manifest with file hashes and refuses a nonempty output directory.
3. The plan states the exact filler slices. The code uses the first `gap` elements and elements `12` through `12 + gap` of each group's fixed filler permutation.
4. The first runner used a separate final-row matrix product. Float32 rounding can change when the matrix shape changes. The final runner uses the stock full batched matrix product shape. It then writes only the selected final-query head output. Earlier query results from the temporary product are discarded.
5. I read the [dummy preflight receipt](dummy-preflight.json). Stock versus custom attention, same-query routing, and the double value swap each had zero logit error. Local donor-vector reconstruction and past-value equality also had zero error. The maximum row-sum error was `1.1920928955078125e-7`, below the fixed `2e-6` bound. Self weight and all other head outputs were exact. These tests used non-study tokens at all three lengths.

The dummy result shows that the fixed checks are attainable. It does not replace the same checks on the study inputs. A later failure must remain public and must not cause a silent tolerance increase.

## Review limits

This review checks the design and the code by inspection. It does not establish that the model can do the task. It does not replace the required result audit. The final report must retain weak baseline performance, zero effects, opposite effects, control failures, and all 24 head results.
