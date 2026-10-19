# Symbolic task methods

These methods implement the matching-graph and register parts of Experiment 1. They test the unmodified models. They do not test an internal explanation. No scientific training or evaluation occurred during code development.

## Matching graphs

Recipe A has two ReLU layers of width 64. Recipe B has width 128. Both use binary64 arithmetic, Adam with learning rate 0.002, batch size 512, and 256 epochs. This is four times the original 64-epoch schedule. Each epoch contains all 65,536 graphs. All model seeds in a split use the same epoch orders. The order generator seed is 94210 for development and 94211 for final models. The model seed controls weight initialization only.

The input has one bit for each edge of a 4-by-4 bipartite graph. A positive label means that a perfect matching exists. The original study's data function supplies the exact labels and category ledger. Its labels are checked against both permutation and Hall-condition oracles. The five gates are the original `all_graphs`, `no_isolated_negatives`, `test_target`, `test_positive`, and `test_hall3` categories. The last three use the original seven-edge test unions. Each final model must reach 99% accuracy in every category. These are known graph states used during training. This does not measure prediction on unseen graphs.

Each job saves initial and final weights, all 32,768 training losses, all graph predictions and logits, category masks, counts, and gate results. The replay uses NumPy affine and ReLU equations. It must reproduce every predicted label and category accuracy. The largest allowed logit difference is 0.000000001.

## Register operations

Recipe A has one GRU layer of width 64 and a 16-class readout. Recipe B has width 128. Both use binary64 arithmetic, Adam with learning rate 0.003, batch size 64, and 6,000 updates. This is four times the original 1,500-update schedule. Loss includes every operation in each length-12 training history. The 14 operations and state encoding match the original study. Both registers start at zero. Each register has four values. The joint-state label is `4*A+B`.

The training set contains 1,024 distinct histories, drawn with seed 71201. Each update samples 64 training indices with replacement. The schedule seed is 71202 for development and 71208 for final models. All model seeds in a split use the same histories and schedule.

There are nine evaluation groups, each with 1,024 different histories:

- Horizon 0: the state after the 12-operation history.
- Horizon 1: one copy from A to B, or one write to A. These are two separate groups. A one-operation horizon cannot contain both operations.
- Horizons 2, 8, and 32: copy A to B then write A, or write A then copy A to B. These are two groups at each horizon. The first two operations fix the order. Any remaining operations are independent fixed draws from all 14 operations.

The write changes the current value of A. Its nonzero offset cycles through 1, 2, and 3 across rows. Later random operations can replace earlier information. These groups test task execution at a longer horizon. They do not require retention of an old value after the task has replaced it.

Development histories use seed 89001. Final histories use seed 89002. The code draws both pools in that order and rejects every history already in the training or earlier evaluation pool. No history belongs to more than one group or split. Suffix seeds are 89003 and 89004. All models use the same fixed rows within a split. Histories are not balanced by their resulting state.

Each gate is joint-state accuracy at the final checkpoint of its group. Every final model must reach 99% in all nine groups. The complete-checkpoint result also checks the initial state and every suffix checkpoint. It is saved as a supporting result, not an additional gate. This choice is fixed before the final runs.

Each job saves the exact token arrays, labels, group IDs, schedules' generating rules, all training losses, initial and final weights, every intermediate prediction, final logits, and group results. Replay uses explicit GRU gate equations and the affine readout. It does not call the GRU module. All predicted states, labels, and gate values must match. The largest allowed final-logit difference is 0.000000001. Saved-weight replay does not repeat training.

## Limits, checks, and computation

Both recipes have fewer than 100,000 parameters. The job refuses to replace an existing result directory. It checks the deadline and the private usage permit before each training batch and each evaluation chunk. It checks loss, gradients, and weights for nonfinite values. A stop during training preserves the current weights and completed trace.

Preparation uses random dummy data and independent random labels. It tests model shapes, arithmetic, unchanged-input output consistency, exact small label examples, and explicit-equation replay. It times 20 dummy training updates for the larger recipe. It does not use scientific labels for this timing run. The register oracle check covers all 224 state-operation pairs.

The compute estimate multiplies measured larger-recipe training and replay costs by four and includes a ten-second setup allowance before that multiplier. The register estimate assumes the longest 44-token sequence for every evaluation row. The graph estimate uses all 32,768 updates and all 65,536 replay rows. Multiply each per-job estimate by eight for its maximum four development plus four final models. The root runner adds the other tasks and checks the combined stage cap before any scientific training starts. A failed budget check requires a revised public plan; it does not permit an automatic scope reduction.
