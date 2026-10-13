# Separate review before the run

Date: 13 October 2026. This is an AI review, not human peer review.

**Status: ready for a public lock. No blocking issue remains. Study models must wait for the verified public lock.**

The final dummy receipt is [dummy-preflight/dummy-preflight.json](dummy-preflight/dummy-preflight.json), version 4. Its manifest was checked. The accepted file hashes are:

| File | SHA-256 |
|---|---|
| `analysis-plan.md` | `cc43676cc86456164267b0c9dc480212a02d6192495476827ed75dddc4441dcf` |
| `code/run.py` | `a7cd55a48ea3b219d1063e748bd5cec9eeb010e493b2218fb63645d2e0428872` |
| `inputs.json` | `057e197a859c797db1c9f180a6ecd1870f2b6b6c07f3335c8e8180c9867da8de` |
| `sources.md` | `e936296dfb0d6369fcac994ec11d0c16f80ed5e8d7def5e27e99d8c516c9eda4` |
| `requirements.txt` | `bf22ae62140ec2188bfb7e3931f9eb0e8f16838592a7bb5527b5d7e546f2d04c` |
| `data/optdigits.tra` | `e1b683cc211604fe8fd8c4417e6a69f31380e0c61d4af22e93cc21e9257ffedd` |
| `data/optdigits.tes` | `6ebb3d2fee246a4e99363262ddf8a00a3c41bee6014c373ed9d9216ba7f651b8` |

## Design and data

The question is a finite comparison of two internal deletion meanings. Message deletion changes only the outgoing message. Node deletion also changes the immediate reset at the selected event. Both gradient queries keep future spike and reset derivatives. The plan does not call this a new causal principle or compare global reset-gradient detachment rules.

The reviewer checked the raw input files without calling a model. Both hashes, 3,823 training rows, 1,797 test rows, 65 columns per row, input value ranges, and labels agree with the ledger. An independent file-order selection gives the same 120 rows, with 12 per digit. These rows lie between indices 0 and 127. They are a narrow file-order panel, not a random sample. The receipt is [independent-input-check.json](independent-input-check.json).

The proposed model has 3,466 parameters. Forty epochs give 1,200 updates per model. The last training batch has 111 rows; the last full-test batch has five. The planned count of 5,160 full model calls and 654,868 forwarded examples is consistent with four models and the fixed panel conditions.

## Estimator and limits

The original winning class and runner-up stay fixed for every edit of that image. Class and event ties have explicit lower-index rules. The single-event candidate set contains all 32 first-layer nodes at the fourth update. Inactive nodes and zero, negative, or incorrect prediction effects are retained. This avoids an outcome-based change to the candidate set.

The full two-by-two table compares each gradient selection with each exact deletion meaning. The oracle maxima cancel from the primary off-diagonal-minus-diagonal regret contrast. Thus, this endpoint measures relative selection benefit, not absolute fidelity. The separate absolute regrets and directional gains are needed. One directional loss can be hidden by a larger gain in the other direction.

The plan fixes the surrogate, training schedule, random masks, data panel, seeds, primary endpoint, and bootstrap. It does not divide by a small exact effect. The four-event edits are direct joint replays and remain secondary. They are not called optimal joint regret. Four seed values and a descriptive seed interval do not justify broad model-family claims.

The 80% full-test gate is a stated task-ability limit. No failed model is removed or retrained. The correct-at-baseline subset is secondary and can differ across seeds. Neither the ability gate nor a positive matching benefit establishes a complete semantic explanation.

## Implementation and dummy checks

I read the full runner. Local leaf placement, fixed prefix values, both immediate reset meanings, and all later surrogate/reset derivatives match the plan. The controls check complete forward identity, self replacement, inactive edits, preserved first-layer trajectories for message deletion, and an independent tangent calculation. The exact final-readout check uses a second-layer spike at the final step. Full joint edits are replayed directly.

All 422 checks in the dummy panel pass. These consist of 18 leaf checks, 160 message singleton checks, 192 node singleton checks, 33 joint checks, 16 self checks, one readout check, and two score checks. The maximum tangent error is 5.20e-18. All exact no-op checks have zero error.

The added hand-set witness has one active and one inactive selected event. The message edit preserves the first-layer state; the node edit changes it. Their maximum state difference is one, and the output difference is 0.125. Both leaf forward traces remain exact. The leaf gradients differ by 0.0050752, and their tangent errors are zero and 3.47e-18. Thus, the gradient check exercises a nonzero local reset contribution. Two final readout spikes are active. Their deletion has effect 0.125 and zero error against the direct coefficient. These checks use no task input or task-trained model.

The latest dummy receipt matches the final plan, code, and ledger hashes. Earlier dummy receipts remain. The two-fold timing projection is 813.59 seconds, below the fixed 1,200-second run cap. This is a projection, not the observed scientific runtime. The runner preserves partial files and the active model on failure, verifies the public bytes before and after execution, and checks the full output counts. Four models should give 1,688 controls, 480 case rows, 30,720 event rows, 1,920 regret rows, and 2,880 joint rows.

## Prior-work boundary

The source record correctly treats surrogate spike attribution, spike deletion, temporal relevance, and reset-gradient choices as prior work. In particular, Gygax and Zenke analyze reset derivatives; Bitar and colleagues evaluate spike deletion; EventRPG uses layer-time relevance for event selection. The NeurIPS 2020 paper *Unifying Activation- and Timing-based Learning* also compares backward reset paths. The current proposal is a narrow application of these established ideas. The source search does not prove novelty.

No task model was trained or evaluated for this review. The accepted version is ready for a public lock. After the fixed run, a separate saved-weight replay and claim review are still required. Those checks must not be described as a repeat of training unless training is actually repeated.
