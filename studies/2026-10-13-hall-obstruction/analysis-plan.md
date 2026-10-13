# A degree-matched test for shortage-specific components in a graph classifier

Date: 13 October 2026. Fixed pre-run analysis plan. This is an AI-authored study with a separate AI review. No human review is claimed. No study model has been trained or evaluated. Only exact graph labels and control sets have been enumerated. The plan, runner, sources, requirements, and input ledger must be reviewed and publicly locked before any model result.

## Question

Does removing a fixed set of components move a small network toward accepting graphs with a specific matching obstruction more than it moves the network toward accepting matched control graphs? The controls have exactly the same vertex degrees but different causes of success or failure.

A bipartite graph has four left and four right vertices. The task is to decide whether all four left vertices can be matched to distinct right vertices. One obstruction occurs when a named pair of left vertices both connect only to one named right vertex. The two vertices compete for one possible partner. Call this a named Hall-2 obstruction. This term refers to the exact input condition, not to an inferred feature in the network.

We will not claim that a selective component implements Hall's complete algorithm. The network could mix many local conditions and matching templates. The proposed contribution is a complete, degree-matched causal test of one input-defined obstruction family.

## Primary sources and scope

Georgiev and Liò, [Neural Bipartite Matching](https://arxiv.org/abs/2005.11304v4), already train neural components for matching through a Ford-Fulkerson procedure. Palumbo and colleagues, [Validating Mechanistic Interpretations](https://arxiv.org/abs/2407.13594v2), ICML 2025, study exact algorithmic explanations for a 2-SAT solver. Narad, Boussioux, and Wagner, [Mechanistic Interpretability for Neural TSP Solvers](https://arxiv.org/abs/2510.21693), study internal features in a combinatorial optimizer. The model task and the principle of algorithmic interpretation are therefore established.

[MINAR](https://arxiv.org/abs/2602.21442) already studies neuron-level circuits in trained graph models. [Slaying the Hydra](https://arxiv.org/abs/2610.04017) studies joint component effects and task specificity. We do not propose a new circuit-search method, a new causal estimator, or a new warning about individual versus joint ablation. A focused source search found no exact duplicate of this finite Hall-2/Hall-3 comparison. That search does not establish priority.

## Exact data and oracle checks

All 65,536 binary 4×4 adjacency matrices form the natural task set. Bit `4*left+right` in an integer graph ID specifies an edge. Label a graph positive if any of the 24 complete matchings exists. Verify every label by a separate calculation of Hall's condition over all 15 nonempty left-vertex subsets.

Oracle-only enumeration gives 37,823 positive graphs and 3,680 negative graphs with no isolated vertex. The full set has 317 row/column-permutation orbits. These are exact facts about this finite input set, not model results.

For a named obstruction, define three categories, with no isolated vertices in any category:

- **Target negative:** the named pair both have the named right vertex as their only neighbour.
- **Positive control:** a complete matching exists.
- **Other-cause negative:** no complete matching exists, but no pair of left vertices has fewer than two neighbours. With no isolated vertex, these graphs fail first at a three-left-vertex Hall condition.

Match categories by the exact four labelled left degrees and four labelled right degrees. Edge count follows from these degrees. Retain a degree class only when it contains every category. Include every graph in each retained category. All retention rules use the exact oracle only.

For named left pair (0,1) and right vertex 0, the fixed control sets have:

| Context | Target negatives | Unique positive controls | Unique other-cause controls | Degree classes |
|---|---:|---:|---:|---:|
| Six edges: selection | 6 | 78 | 12 | 3 |
| Seven edges: test | 18 | 48 | 12 | 6 |

The six-edge row has one target orbit, three positive orbits, and one other-cause orbit. The seven-edge row has two, two, and one. No orbit crosses the edge-count split. This is a small structural test, despite the number of labelled matrices.

Apply these same fixed rules to all 24 choices of left pair and single right neighbour. The counts per named obstruction stay the same by vertex relabelling. Record every graph ID and all repeated uses. Related graphs are not independent replications. The complete labelled test lets us avoid choosing an especially favourable vertex name.

The network will train on the whole natural graph set. Only intervention-selection and intervention-test contexts are separate. This is **not** a test of prediction on unseen natural graphs. It tests whether a component set selected with six-edge graphs has a specific effect on seven-edge graphs.

## Fixed model and training

Use a feed-forward network `16 → 64 → 64 → 1`, with ReLU after each hidden layer. Inputs are 0/1 adjacency entries. Its scalar output is the logit for a complete matching. It has 5,313 parameters. This is a graph-classification task in a feed-forward model, not a general claim about message-passing GNNs.

Use four fixed parameter seeds: 19121, 19139, 19157, and 19181. All calculations use float64 on CPU with two threads. Train for exactly 64 complete epochs, with batch size 512. Each epoch uses a permutation from NumPy generator seed 94210; use the same full order for all four models. Thus every graph is seen exactly 64 times by each model. There are 8,192 updates and 4,194,304 training examples per model.

Use binary cross-entropy with logits and Adam, learning rate 0.002, default betas and epsilon, no weight decay. Use no validation-selected schedule, early stopping, checkpoint choice, seed replacement, or extension. Save all losses, exact epoch orders, initial and final weights, and weights after each 1,024 updates. The final weights alone define the scientific evaluation. All four fixed models are retained.

## Intervention and selection

For each model and each named obstruction, measure every one of the 64 first-hidden-layer unit ablations on every six-edge selection context. An ablation sets the unit's post-ReLU output to zero. It changes no weight or other hidden entry. This is a fixed component intervention. It may create a hidden state that no natural graph produces.

For a mask S and graph G, let `e_S(G)=logit_ablated(G)-logit_intact(G)`. Positive values move the output toward accepting a complete matching. For degree class g, let `e_T`, `e_P`, and `e_C` be the mean changes for target negatives, positive controls, and other-cause negatives. Define

`selectivity(S,g) = e_T - max(0, e_P, e_C)`.

This score is positive only when ablation moves target negatives toward acceptance more than zero and more than it moves either matched control category. A uniform shift toward acceptance has score zero. A uniform shift toward rejection has a negative score. Report `e_T`, `e_P`, `e_C`, `e_T-e_P`, and `e_T-e_C` separately so the combined score cannot hide its source. This is relative effect specificity. Even a positive score can accompany changed control decisions. It does not require the ablated model to preserve control accuracy. Retain all category-level classification changes and state this limit when interpreting the score.

Rank single units by the mean selectivity over the three six-edge degree classes. Break exact ties by lower unit index. Select the first four units, even if all scores are negative. Do not change the mask size or take only positive units. Then ablate those four units together on every seven-edge test graph. Measure the joint effect directly; do not add the single-unit effects.

Use eight fixed random masks of four distinct units. Make a single NumPy generator with seed 94301. Make eight sequential calls to `choice(64, size=4, replace=False)` and sort each returned list. Store the eight literal masks in the locked input ledger. The masks may overlap each other. Reuse them across models and named obstructions. They provide the fixed primary comparison; they are not independent model replications. Retain all 64 single-unit effects in both selection and test contexts. Also retain empty-mask, selected-mask, and full-first-layer ablations. Do not select units from test scores.

## Primary endpoint and ability rules

For each model, first average joint selected-mask selectivity over all six seven-edge degree classes within each named obstruction, then equally over all 24 names. Compute the same average for each of the eight fixed random joint masks. The sole primary comparison is the selected-mask average minus the mean of the eight random-mask averages, paired within each model. The primary result is the mean of these four model-level differences. Report all four paired differences and raw selected and random-mask scores prominently. Also report every named-obstruction score and its full range. A selective-component interpretation requires positive raw selected selectivity, a positive advantage over random masks, and all declared ability gates to pass. If any required condition fails, state that this interpretation is not supported. A positive result is specific to this finite contrast. It does not identify a unique Hall circuit.

Apply the interpretation conditions separately to each model: its raw selected score and selected-minus-random difference must both be positive, and its five ability gates must pass. A claim that the component interpretation holds across the complete four-model panel requires every model to pass those conditions. If any model fails, the pooled mean remains a reported result, but it does not establish this panel-wide interpretation. Report the failing models and conditions explicitly.

Use a descriptive 95% percentile interval from 10,000 resamples of the four whole model seeds, generator seed 94302. Do not bootstrap graph relabellings as if they were independent models. No formal significance claim is planned.

Report intact accuracy and cross-entropy on all 65,536 graphs. Separately report intact accuracy for all 3,680 negative graphs with no isolated vertices. On the union of the seven-edge test contexts, report accuracy separately for target negatives, positive controls, and other-cause negatives, with duplicates removed within each category. The ability rule requires at least 99% accuracy in each of these five categories. Keep all seeds if a gate fails and state the failures prominently. Do not use an ablation score to claim a correct obstruction mechanism in a model that fails the required decisions.

Also report the same effects on probability, per-category classification changes, all intact margins, and the first-layer activation norm removed by each mask. Do not divide by a small intact score or effect. Differences in activation scale and confidence remain possible explanations for differences in magnitude; the degree controls alone do not remove every such difference.

## Numerical checks, storage, and cost

Use a fixed maximum absolute error of 1e-10 for replay and component controls. Before lock, test only artificial nonbinary inputs and a random dummy model. The dummy inputs must not occur in the binary task set. Verify full forward versus cached-hidden continuation, empty-mask identity, changing only selected entries, restoring the saved receiver entries, and full-first-layer ablation versus a direct calculation from the two later biases and weights. Verify both independent label oracles and the exact context split without a model.

The scientific runner must verify the locked plan, runner, source list, requirements, and input specification before and after the run. It must refuse a nonempty result directory. Save complete intact outputs; all 64 unit-ablation outputs on the finite study contexts; selected, random, and full-layer outputs; group scores; masks; losses; weights; controls; package versions; and result hashes. Save the actual data and batch orders or enough exact arrays to replay them byte for byte. Save per-mask graph rows without silently counting repeated uses as new graphs.

The training budget is 32,768 optimizer updates and 16,777,216 examples in total. The intervention set is small relative to the full input set. Set a 20-minute total CPU limit, with checks after every training batch and evaluation group. Preserve partial results and a failure receipt if a bound is exceeded. Do not extend the budget or tune a failed run. The runtime is unknown until the fixed experiment is run. Use Python 3.12, NumPy 2.3.5, and PyTorch 2.8.0. Use deterministic algorithms.

A separate reviewer must check the input construction, plan, and runner before lock. A saved-weight audit should independently recompute all intact and intervened outputs, context scores, mask selection, ability gates, and the seed summary. It must state that it does not repeat training if it only replays saved weights.

## Locked input ledger and reporting details

Generate `inputs.json` using only graph oracles. It records all 24 names, all nine degree groups for each name, all target and control graph IDs, exact duplicate-free category unions, eight random masks, and SHA-256 identities of the complete graph labels and the 64 epoch permutations. Store each array hash over its C-order bytes, with dtype and shape in the ledger. Save the exact graph IDs, labels, and epoch permutations in the results. The ledger is a public-lock file.

For each seed, save final intact logits for every graph and every first-layer ablation output on the union of all selection and test contexts. Retain the 64 singleton masks, empty mask, full-layer mask, eight random masks, and all 24 selected masks. Save the full mask matrix, output matrix, removed-activation norms, graph IDs, and group-level score table. Repeated masks remain distinct named rows. Each group score uses means over all unique graphs in that category and degree group. No test outcome determines a mask.

Save the all-natural-input ability tables, the five gates, and each named obstruction separately. Classify a graph as positive when its logit is at least zero. Never condition intervention scores on intact correctness. The primary interval resamples the four selected-minus-random model differences. Save numerical errors before assertions. Save all failures, including a time-limit failure; do not resume or extend a run.
