# Fixed selector edits: donors and order

This is a diagnostic extension of Study 1. It uses the same eight trained models. It does not train models or select new masks. The models have already informed the research. This is not fresh confirmation.

## Fixed sources

All paths below are inside `studies/2026-10-13-inactive-variable-audit/`. The exact SHA-256 values are in `code/selector.py`, in `SOURCE_HASHES`. The source files are checked before use and after the run. The fixed source repository version is `168d952`.

| Files | Use |
| --- | --- |
| `models/seed-0.npz` through `models/seed-7.npz` | All saved float64 model weights |
| `data/inputs.csv` | The complete 192-state task; regenerated states must match |
| `data/pairs.csv` | Historical source record; this stage fixes its own pair blocks |
| `results/selected-masks.csv` | Original unrestricted, repair-only, three-neuron selections |
| `results/manifest.json`, `code/run.py` | Source identity and original arithmetic |

The masks for seeds 0 through 7 are: `(7,8,11)`, `(1,5,6)`, `(1,2,11)`, `(0,1,2)`, `(2,3,4)`, `(2,3,5)`, `(0,2,3)`, and `(2,6,10)`. Neuron numbers start at zero. The saved selection table must match these masks. There is no reselection for a donor group.

## Donor comparison

The input is `(a,b,s,n)`. The answer is `a` when `s=0`; otherwise it is `b`. There are four values each for `a` and `b`, two for `s`, and six for `n`.

Use every receiver and every requested value of `a`. This gives 768 receiver/request blocks. For each block, use all 48 donors with the requested `a` value. Donor `b`, `s`, and `n` cover their full Cartesian product. The receiver and required changed input stay fixed across these 48 donors.

Replace only the frozen mask with the donor's activations. Keep all other receiver activations. Recompute the downstream network. The required input is `(requested_a, receiver_b, receiver_s, receiver_n)`. Report same-value and different-value requests separately. Also report active and inactive cases and every combination of donor matches for `b`, `s`, and `n`. Do not infer that a same-value donor removes the proposed signal.

For every receiver/request block, also evaluate three controls:

1. Keep the receiver unchanged.
2. Set the three selected activations to zero.
3. Use the complete hidden state of the required changed input.

The last control is a known coherent input change. It has access to the full task input. It is not a learned three-neuron edit. Zero replacement has no automatic interpretation as a valid variable value. All controls retain the same required input as their donor block for comparison. The identity donor and unchanged-model controls must reproduce the intact receiver output.

## Fixed split

Split complete receiver/request blocks, with every donor context in the same split. Within each receiver-selector and same-value/different-value stratum, sort blocks by SHA-256 of `selector-procedure-v1:receiver_id:requested_a`. The first half is development and the second half is test. Each split has 384 blocks and 18,432 donor rows per model.

This split is fixed before model outputs. No method selection or tuning uses either split in this stage. Inputs and models are shared. The split does not make this a fresh independent confirmation. Save all pair ledgers before loading trained weights.

## Two edits and both requests

Apply two replacements to the same frozen mask. The first requested value covers all four values. The second is either the same value or `(first_value+1) mod 4`. For each donor, its other inputs either match the receiver or all change by the fixed rule: `b` increases by one modulo four; `s` flips; `n` increases by one modulo six. Cross both donor-context choices and both edit orders.

This gives 6,144 base cases and 12,288 ordered results per model. Each case uses the split of its receiver and first request. The two orders remain in the same split.

After the second edit, check both requested activation vectors, both required task answers, and both probability distances. Also report the probability distance and answer difference between the two orders. Equal requested `a` values are logically compatible, but their donors can request different neuron values. Record this coordinate conflict separately. Different requested `a` values conflict as simultaneous variable requests. In an inactive case, they can still require the same visible answer.

The last edit must set its requested coordinates. This follows from replacement on the same mask and is a calculation control. The experiment tests the output consequences and donor-context dependence of this operation. It does not discover a new composition law or test edits of two independently identified variables.

## Calculation checks and outputs

Use float64 throughout. Fix the maximum absolute calculation tolerance at `1e-10`. Do not enlarge it after a failure. Every model must retain at least 99% ordinary task accuracy.

The main implementation uses matrix multiplication. The separate calculation uses explicit columnwise reductions. Both start from the saved weights and task inputs. For the order check, the separate calculation constructs the final state directly from the last donor. It does not repeat the composition code. Each saved result batch is reopened and checked. Both the edited coordinates and all four output probabilities must match. A mismatch, nonfinite value, incorrect shape, or failed no-change control stops the branch.

Save raw input identifiers, probabilities, edited coordinates, per-batch replay checks, intact outputs, masks, source hashes, and summaries for every group. There are 36,864 donor rows, 2,304 control rows, and 12,288 ordered rows per model. The total is **411,648 probability rows across eight models**. These rows are dependent observations; they are not independent model replications.

Primary summaries are task-answer accuracy, zero-to-one total variation distance from the intact required input, and unwanted answer changes. Order summaries also check both coordinate requests. A passed numerical check means the declared computation was reproduced. It does not mean that the mask correctly represents `a`.

## Preparation and resource estimate

The module exposes `prepare(out, guard)` and `run(out, guard)`. It requires a new output directory. The shared runner supplies `guard()`, which must raise on resource or usage stops. The module calls it before each batch and before result saves. Completed batches remain after interruption.

Preparation uses only random dummy inputs and random weights, with fixed seed 2026102601. It times one 768-row batch for each of the three operation types, including saving and separate replay. The projection is four times the measured per-row cost for 411,648 rows plus a fixed 60-second setup and summary allowance. The shared Stage 2A runner must include this cost within its two-hour cap. No trained model outputs are calculated in preparation.

Before measurement, allow roughly one to five CPU minutes and less than 100 MiB for saved data. This is a planning estimate. The fixed dummy measurement determines whether launch fits the remaining stage budget. No language tokens are used by this branch. The main stage has separate time, memory, disk, usage, and token rules.

The local code syntax and small mock checks have a conservative pre-lock time charge of ten seconds. They are not scientific results. The shared preparation record must account for this charge.
