# Independent AI review before the run

Date: 13 October 2026. Status: ready for the public lock.

This is a separate AI review. It is not human peer review. The reviewer read the full plan, source notes, runner, and dummy receipts. No study model outputs were available for this review.

## Scope and design

The question has a clear limit. A correct current output need not mean that a model will use the changed state in a later operation. This point is known from the cited work. The added test compares two fixed edit losses on two orders of the same copy and write operations. It does not establish a new method of state editing.

The exact state machine gives the required current and future targets. The two branches need different future uses of the edit. The copy-then-write branch must retain the changed A value in B. The write-then-copy branch must remove that value. A single tendency to preserve an edit does not meet both tests.

The plan keeps all four model seeds and all test contracts. The ability rule is a limit on interpretation, not a rule to remove seeds. The primary score includes the current output and both final outputs. It does not include the two first-step outputs. Those outputs are retained as separate checks. The two same-state routes use factual targets, as the plan requires. Their scores must not be read as scores on the requested changed state.

The transition arm has more computation per update. The plan states this cost. Four model seeds give a small panel. The bootstrap interval is descriptive. The test does not establish results for language models, a unique internal circuit, or a complete four-dimensional state representation.

## Code checks

The reviewer checked the operation order, state transition rules, history quotas and splits, data schedules, paired initial edit weights, frozen model parameters, and gradient path through the future step. The code applies the edit once. It does not use future target labels as model inputs or apply another edit during a continuation.

The complete test grid has 3,072 contracts per model and route. The code retains seven routes, all five sets of 16 logits, all target labels, edit norms, and per-contract scores. The primary score and the seed-level bootstrap match the plan. The run refuses a nonempty output directory, checks public-lock bytes, checks its time limit, saves partial records on failure, and makes a result hash manifest.

The author made these changes before the lock in response to review:

- Check orthonormality and projection idempotence for the initial subspace as well as both learned subspaces. The expected numerical-control row count is now 528.
- Save factual states after every operation in the locked input file.
- Check that dummy histories do not occur in any study history bank.
- Test the future-only loss gradient. This loss must give the edit matrix a finite, nonzero gradient while frozen model parameter gradients remain absent.

The earlier dummy receipts remain in this directory. The latest receipt is [dummy-preflight-v3.json](dummy-preflight-v3.json). It uses artificial histories at batch sizes 64 and 256. Its largest listed numerical error is 4.44e-16, below the fixed 1e-10 limit. The future-only edit gradient norm is 0.0001609237244. It records that no study model was trained or evaluated.

## Reviewed file identities

The reviewer computed these hashes from the files and checked that they match the latest dummy receipt:

| File | SHA-256 |
|---|---|
| `analysis-plan.md` | `b2e1c9e76926f04b517dde0e9af8d1498f70b15224896b4ff1299516c60b5a59` |
| `code/run.py` | `cc282ca30b1b0a23173fc486aa37e15f63fe234a48b61378197c9d875d06e39e` |
| `inputs.json` | `637b990e99f28ad36ea2205bedf4afaa09b0cf1c1bfa91d1b2cbc7e5beb9c2ff` |

No design or code issue remains open within this review. The study must wait for the public lock before model training or evaluation. A later result audit must check saved model outputs and primary calculations. Such an audit must state whether it also repeats training.
