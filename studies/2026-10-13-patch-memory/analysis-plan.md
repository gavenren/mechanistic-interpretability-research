# Does an immediate patch score select a layer with a later effect?

Date: 13 October 2026. Status: fixed before scientific results.

This is an AI-authored study plan. A separate AI agent will review the plan and code. No human review has occurred. The plan and runner must be published in one immutable Git commit before scientific runs. That commit is a public time-order record, not a formal external registration.

## Question and scope

We will change one internal state when a small language model reads a prompt. We will then give the changed model and the unchanged model the same eight tokens. Does the layer with the largest immediate output change also have a large output change after those tokens?

The main comparison is between two fixed rules for selecting a layer. One rule uses immediate output change. The other uses output change after eight tokens. Each rule sees only the calibration word pairs. We compare their later effects on held-out word pairs.

The score measures any output change. A larger score need not mean a better answer, a correct state change, useful steering, or longer semantic memory. We do not test free generation. The suffix content changes with the horizon, so the profile does not isolate elapsed time.

Cache persistence, one-time cache edits, and the difference between output bias and a state used later are known ideas. Close prior work includes Belitsky et al. (2025), Kang et al. (2026), Liu et al. (2026), Li (2026), and Shih et al. (2026). Our proposed contribution is a small, fixed, held-out comparison of immediate versus delayed layer selection. No scientific priority claim is planned.

## Model and arithmetic

- Model: `EleutherAI/pythia-14m`, revision `cf967c0a9a04383db6f7b1108d86b2962634b4ac`.
- Six parallel transformer blocks, indexed 0 through 5. The checkpoint and tokenizer hashes are in the runner.
- CPU, two PyTorch threads, evaluation mode, no gradients, no training, no sampling, and deterministic algorithms.
- PyTorch 2.8.0, Transformers 4.57.6, NumPy 2.3.5, Python 3.12.
- One scientific arithmetic condition: float64 parameters and operations, including a dtype-preserving eager attention softmax. Stock GPT-NeoX rotary tables still use float32 arithmetic. They are shared unchanged by all conditions. This is double arithmetic with shared float32 rotary tables, not a fully float64 reference.
- All cache and output controls use a fixed maximum absolute logit error of `1e-8`. Cache positions that must remain unchanged use exact equality. Failure does not permit a wider tolerance or a smaller sample after results.

## Fixed inputs and split

The four frames are:

1. `The entry is {w}. Copy the entry:`
2. `The code is {w}. Copy the code:`
3. `The password is {w}. Copy the password:`
4. `The label is {w}. Copy the label:`

The eight word pairs, in order, are `fox/wolf`, `tree/leaf`, `glass/plate`, `rock/sand`, `gold/silver`, `door/wall`, `road/path`, and `fire/wind`. Each pair is used in both directions. This gives 64 directed cases and 64 distinct prompts. We keep every case regardless of model task ability.

Pairs 0–3 form the calibration set. Pairs 4–7 form the test set. Each set has 32 directed cases. Both directions and all four frames for one word pair stay together. The test set has new words in known frames. It is not a new-frame test. These are fixed, related cases, not random independent language samples.

The pinned tokenizer must produce one token for each target with its leading space. The target IDs are: fox 30013, wolf 25872, tree 5202, leaf 10617, glass 5253, plate 5340, rock 5561, sand 7537, gold 5328, silver 9711, door 3369, wall 3402, road 3971, path 1854, fire 3289, wind 5448. Paired prompts must have equal length, differ at one token, and end in the same token. These checks can run before the model experiment.

The exact forced suffix is ` The record will be checked again after this`. Its token IDs are `[380, 1924, 588, 320, 10141, 969, 846, 436]`. The eight decoded pieces are ` The`, ` record`, ` will`, ` be`, ` checked`, ` again`, ` after`, and ` this`. We supply these IDs directly. We do not select tokens from model outputs. The suffix contains neither target word. It is a common controlled continuation, not a natural answer trial.

## Intervention

For each layer, save the donor prompt's residual vector after that block at the final prefix position. Replace the receiver's vector at that same block and position with the donor vector. This is a full vector replacement, not an added difference.

The patched prefill pass produces an output distribution and a key-value cache. Remove the hook after prefill. Feed the eight fixed suffix tokens one at a time using that cache. No suffix position is patched. This is one prompt-state edit. It is not continuous activation steering or direct optimization of the cache.

At the edited block, attention has already written its cache. Thus the edit cannot change that block's cache or any earlier block's cache. It can affect later blocks' cache at the final prefix position. A post-block edit at block 5 cannot change any cache entry. With a common forced suffix, its future outputs must therefore equal the unchanged outputs. This last fact is a known architectural control, not an empirical discovery.

## Scores and fixed layer selectors

Let `E(c,l,h)` be total variation distance between the patched and unchanged next-token distributions for case `c`, layer `l`, and horizon `h`. Total variation is half the sum of absolute probability differences across all 50,304 output tokens. Its range is 0 to 1. Horizon 0 is the output just after the prefix. Horizon 8 is the output after the full fixed suffix.

Only layers 0–4 are candidates for selection. Layer 5 remains in all descriptive tables as the known zero-future control.

1. Immediate rule: select the layer with the largest calibration mean `E(c,l,0)`.
2. Delayed rule: select the layer with the largest calibration mean `E(c,l,8)`.
3. If more than one layer is within `1e-10` of the maximum, select the smallest index among them.

**Sole primary endpoint:** test mean `E(c, delayed_layer, 8) - E(c, immediate_layer, 8)` over the 32 test cases. A positive number means that the delayed rule selects a layer with more later output change in this fixed test set. The two choices may be the same. We will retain a zero or negative result.

No layer is selected using test scores. No hyperparameter is fitted. We make no population significance claim and do not treat 32 related directions as independent draws. We will report the primary difference separately for each of four test word pairs and four frames. We will also report the uniform mean across candidate layers as a descriptive reference.

TV values and absolute primary differences at or below `1e-8` are labeled numerically negligible. If every calibration candidate mean at horizon 8 is at or below this level, report no resolved best delayed layer. Still save the formal tie-rule choice. This conservative reporting label does not change any control tolerance or score.

Save the L2 norm of the donor-minus-receiver residual change and that norm divided by the receiver residual norm for every case and layer. A larger later effect can reflect a larger initial disturbance. We do not divide by immediate TV, and we do not interpret the result as a rate of memory decay.

Complete descriptive summaries, fixed now: mean, median, minimum, and maximum TV for each split × layer × horizon; mean TV for each split × frame × layer × horizon; and mean TV for each word pair × layer × horizon. All 3,456 case × layer × horizon rows remain public. All horizons other than 8 are descriptive. Also report the two target logits, their signed donor-minus-receiver margin change, top-1 token IDs, and whether the top-1 token changes for each row. The target margin at later horizons is not task accuracy. At horizon 0, report baseline target top-1 frequency and receiver-versus-donor target margin, with no screening.

## Controls and alternative computation

Run the following checks for every directed case and every layer. Save each error, not only the overall maximum.

1. **Self replacement.** Replace the receiver vector with its saved unmodified vector. Check every output from horizons 0–8 against the baseline.
2. **Cache structure.** Compare patched and baseline prefill keys and values for all six cache layers. All positions in cache layers up to and including the edited layer must be exactly equal. Earlier prefix positions in higher cache layers must also be exactly equal. Save maximum changes at the final prefix position for all cache layers.
3. **Final block.** All cache entries after a block-5 edit must equal baseline entries. All eight later outputs must agree within `1e-8`.
4. **Cache reset.** Discard the patched cache and use a fresh clone of the unchanged prefill cache. Feed the same suffix. Check all eight later outputs against the baseline. Each trajectory starts from its own cache copy; the cache is mutable.
5. **Cache versus full recomputation.** For all 64 cases, all six layers, and all nine horizons, run the receiver prefix plus the first `h` suffix tokens in one pass without a cache. Replace only the original final prefix position with the same donor vector. Compare its full-vocabulary output logits to the cached path within `1e-8`. Also compare the unchanged cached and full paths at every horizon.
6. **Hook position.** Record each replacement call's sequence length and absolute index. The hook must run once per prefill or full recompute pass, always at the original final prefix position. No hook is installed for incremental suffix steps.

The full recomputation path is a separate execution path in the runner. The independent reviewer will replay the model with their own implementation and recalculate the scores. This is AI review, not human replication.

## Cost, stopping rules, and records

The scientific run is capped at 20 minutes. It must retain partial CSV files and a failure record if an error, timeout, or control failure occurs. Complete the fixed tables when possible; do not search for a more favorable subset. A failed control blocks a normal scientific conclusion pending a separately labeled audit. A code correction after results needs a disclosed new version and a new locked run; the original failure remains available.

Both rules cost model calls to measure their calibration effects. This study also pays for the delayed test measurements and all controls. It does not show a reduction in inference cost. Report actual forward counts and elapsed CPU wall time.

Before the lock, permissible checks are syntax checks, tokenization of the fixed inputs, and cache/hook smoke tests on separate dummy strings. No study output, layer ranking, task capability screen, or effect size may be viewed before the lock.

Save the input ledger, every scalar result, every control error, calibration layer means, selected layers, complete group tables, pinned model/config/tokenizer hashes, package versions, machine class without private paths, wall time, and forward counts. Save all baseline donor/receiver post-block vectors. Full-vocabulary tensors need not be stored; the pinned model and deterministic runner permit replay. The runner must verify its own and this plan's bytes against the supplied full Git lock commit before and after the run. It must verify model hashes and refuse to overwrite a result directory. A manifest will hash all result files.

## Limits to retain in the paper

One very small pretrained model; one checkpoint; artificial copy frames; one chosen suffix; four held-out word pairs; eight suffix tokens; complete post-block replacements; no task screening; generic output-change metric; one arithmetic condition with float32 rotary constants; no generated trajectories; no evidence of useful control, semantic fidelity, model safety, or population frequency. Larger models, other prompts, cache sites, suffixes, and horizons require new studies.
