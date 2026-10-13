# A state edit must survive copying and end after replacement

Fixed analysis plan, 13 October 2026. This is an AI-authored controlled study. No human review is claimed. No study model will be trained or tested before a public commit fixes this plan, the runner, and the input histories.

## Question and limits

Can a learned edit to a recurrent model's hidden state pass two different tests of future use? One test must carry the edited value into a second register. The other must remove that value after a new write. We compare two training losses for the same small edit method. We do not propose a new theory of causal state, a new alignment method, or a result about language models.

Close work already separates an immediate output change from a change that the model uses in later computation. Shih, Winnicki, and Darve test state use, continuation controls, and persistence. Liu and Chen test one-time edits followed by autonomous rollouts and specificity controls. Csordás and colleagues test learned interventions in recurrent sequence models. The added experiment is a paired, non-invertible copy/write test and a fixed comparison of immediate and one-step training losses. The source search does not establish priority for this test.

## State machine

Registers A and B each hold one value in {0, 1, 2, 3}. Both start at zero. The joint label is 4A+B. There are 14 input operations in this exact order:

1. WRITE_A_0, WRITE_A_1, WRITE_A_2, WRITE_A_3.
2. WRITE_B_0, WRITE_B_1, WRITE_B_2, WRITE_B_3.
3. COPY_A_TO_B, COPY_B_TO_A.
4. INC_A, INC_B, each modulo 4.
5. SWAP, NOP.

The model gets only operations. It predicts the joint state after each operation. Predicted outputs are never fed back. An exact state machine supplies the target labels.

## Model and factual training

The model is a single PyTorch GRU with one-hot input size 14 and hidden size 64, followed by a linear 64-to-16 readout. It has 16,400 parameters. All model and edit calculations use float64 on CPU with two threads. Model seeds are 17011, 17021, 17027, and 17033. The four seeds are the model-level units of replication.

Use 1,024 unique operation histories of length 12. Draw operations uniformly with NumPy generator seed 71201. Keep the first unique histories. Start each history at state (0,0). Train each model for exactly 1,500 updates, batch size 64, Adam learning rate 0.003, default betas (0.9,0.999), epsilon 1e-8, no weight decay. The loss is mean cross-entropy over every state at all 12 positions. Sample batch indices with replacement from generator seed 71202; the same index schedule is used for all four models. Do not stop early, select checkpoints, retry a seed, or change training after the ability check.

Save every update's loss and final weights. Save a recovery checkpoint after each 250 updates. No checkpoint selection is allowed. Only the final model is used for the edit study.

## Separate history banks

Build four additional banks by exact state-machine labels, not by model outputs. They must be mutually disjoint and must exclude the 1,024 factual training histories. For each bank, draw unique length-12 histories with a separate NumPy generator. Retain the first histories that fill each final-state quota. Stop with a recorded failure if one million draws do not fill a bank.

| Bank | Seed | Histories per joint state | Total |
|---|---:|---:|---:|
| Edit-training receiver | 71203 | 16 | 256 |
| Edit-training donor | 71204 | 16 | 256 |
| Test receiver | 71205 | 8 | 128 |
| Test donor | 71206 | 8 | 128 |

Sort each bank by joint state, then acceptance order within that state. Save all 1,792 histories, exact states, the operation list, and data rules in `inputs.json` before the public lock. This step uses no model outputs. The study holds out complete histories, not operation types or current register values.

## The edit method and its two losses

Freeze the trained GRU and its readout. Learn a 64-by-4 matrix W. Obtain U from reduced QR decomposition of W. Its four columns are orthonormal. For receiver hidden state h and donor hidden state d, apply one edit:

`h_edit = h + ((d-h) U) U^T`.

This replaces the donor's coordinates in a four-dimensional subspace. It does not replace the full hidden state. The receiver has state (a,b), the donor has (a_new,b_donor), and the edit target is (a_new,b). Thus donor B can differ from receiver B. Donor histories can carry other information. The method has to avoid transferring information that changes the required B value.

For each model, initialise W with standard normal values using PyTorch seed 81000 plus the model seed. Both learned arms start from these exact same W bytes. Also evaluate this untrained initial subspace as a descriptive baseline.

Train each arm for exactly 1,000 Adam updates, learning rate 0.01, batch size 64, default betas and epsilon, no weight decay. Use the same fixed request schedule for both arms and all model seeds. NumPy seed 71207 supplies independent choices of: a receiver index; a nonzero modular offset in {1,2,3} for a_new; donor B in {0,1,2,3}; donor history index in {0,...,15}; and a uniform operation in {0,...,13}. Save the schedule in `inputs.json`.

- **Immediate arm:** cross-entropy of the edited current joint-state readout.
- **Transition arm:** one half of that immediate loss plus one half of the joint-state loss after the one sampled operation. The operation is applied to the edited hidden state. Its label comes from applying that operation to (a_new,b).

Only the first edit is applied. There is no later hidden-state correction. The transition loss uses one future operation. It never trains on a two-operation continuation as one loss example. Save every loss, both components where present, final W and U, and recovery checkpoints each 250 updates. Never use test outputs to select an edit, step count, rank, loss weight, or checkpoint.

## Fixed paired test

For each of 128 held-out receiver histories with state (a,b), use every a_new different from a, every donor B, and a fixed donor history index equal to the receiver's within-state index. This gives 1,536 requests. Use both write values w outside {a,a_new}, for 3,072 test contracts per model and route.

Apply the edit to A once, then read the immediate state and two branches:

| Branch | First operation and target | Second operation and target |
|---|---|---|
| Copy then write | COPY_A_TO_B: (a_new,a_new) | WRITE_A_w: (w,a_new) |
| Write then copy | WRITE_A_w: (w,b) | COPY_A_TO_B: (w,w) |

The branches have the same operation types in different orders. The first must preserve the edited value in B after A is overwritten. The second must release the edit. A generic tendency to retain or erase an edit cannot pass both requirements. Correct outputs alone still do not identify a unique internal circuit.

Save the complete 16 output logits for all five readouts, their exact targets, and per-contract scores. Score all cases. Do not select cases that pass the immediate edit.

## Reference routes and controls

Evaluate these seven routes on the full fixed test grid:

1. Original receiver, with no edit. Score this route against the requested edited targets; also score it separately against its factual targets for ability.
2. Natural full-state donor with state (a_new,b), chosen from the test donor bank at within-state index (receiver index + 1) modulo 8. This donor has the required whole state and a separate history. It is a capability reference, not evidence for a successful low-rank edit.
3. Untrained initial subspace using the requested donor.
4. Immediate-trained subspace using the requested donor.
5. Transition-trained subspace using the requested donor.
6. Immediate-trained subspace with a separate donor history having the same state (a,b), at index (receiver index + 1) modulo 8.
7. Transition-trained subspace with that same-state donor.

The last two routes test irrelevant-history sensitivity. Score them against factual targets: the register values should not change. Do not expect identical hidden states or logits from two histories that end in the same state.

For both learned subspaces, also apply a self patch with donor equal to receiver. Compare every current and future logit with the original receiver route. Check full-state donor injection against running the complete donor history followed by the two-operation suffix. Check each U's orthonormality and projection idempotence. Fixed numerical tolerance is 1e-10 maximum absolute error. Save per-batch errors before assertions. A failed numerical check stops the run and preserves outputs. Ability failures do not stop the run.

## Ability gate and primary comparison

For each model, require at least 99% full joint-state accuracy separately for (a) current factual test receivers; (b) factual first-step outputs over both orders; (c) factual second-step outputs over both orders; (d) all current and future natural full-state donor outputs. These are fixed interpretation rules, not a general standard of competence. Retain every model even if it fails. If any seed fails, state prominently that the full comparison includes a model that did not meet the declared task-ability rule. Do not claim successful state editing for that model from its edit score alone.

The **primary score** is the fraction of all 3,072 contracts with a correct immediate joint state and correct final joint states in both branches. The first-step outputs are separate diagnostic scores; they are not part of this primary conjunction. The primary comparison is transition-arm score minus immediate-arm score, paired within each of four seeds. The directional hypothesis is that this difference is positive. Support requires a positive mean and positive differences in at least three seeds. This small-panel rule is not a significance test.

Report all four paired differences, their mean, and a descriptive 95% percentile interval from 10,000 bootstrap samples of the four whole seeds (NumPy seed 71208). Four seeds give limited uncertainty resolution. Do not treat the thousands of related test cases as independent model replications.

Report, for every route and model: immediate joint-state, A-only, and B-only accuracy; first-step and final accuracy for each branch; primary conjunction; target logit margin and target probability at every readout; and mean hidden edit norm. Report the same-state routes' factual score to show semantic preservation under irrelevant history. Report the requested donor B equal/different from receiver B as fixed descriptive strata. Report factual and natural ability measures even if their gate fails.

## Cost, verification, and stopping rules

Factual training has 6,000 optimizer updates and 4,608,000 GRU position examples. Each edit arm has 4,000 optimizer updates and 256,000 requests. Both arms have the same number of edits and optimizer steps. The transition arm additionally has 256,000 recurrent position examples and 256,000 readouts with gradients. It therefore has a larger computation cost. Do not call this a compute-matched comparison. Frozen history encodings, test evaluation, QR work, controls, and independent audits add further cost; report wall time and exact example counts separately.

Set a 20-minute run limit with two CPU threads. Check the limit at training updates and test batches. Preserve partial work and a failure record if the limit or a numerical check fails. Do not extend the budget or silently resume. Refuse a nonempty output directory. Check public-lock bytes before and after the run. Save package versions, seeds, times, row counts, all model/edit weights, and a SHA-256 manifest for every result file. Python is 3.12; NumPy 2.3.5 and PyTorch 2.8.0 are required.

Before lock, run only data/oracle checks and a separate random dummy model on distinct artificial inputs. Check the state-machine truth table, paired suffix targets, shapes, batched/stepwise continuation, self edits, full donor interface, orthonormality, gradient routing, and real training/evaluation batch sizes. Mark the receipt `scientific_model_outputs_collected=false`. A separate AI reviewer checks plan and code before lock and saved results after the run. State the scope of any independent replay; do not imply that saved-weight checks retrain the models.
