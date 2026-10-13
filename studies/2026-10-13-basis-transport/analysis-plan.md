# Analysis plan: when a coordinate change changes an intervention

Date: 2026-10-13. Status: fixed before training or trained-model result inspection. The runner requires a Git commit with these exact plan and runner contents. The remote commit must be published before the run. This is a project analysis lock, not a third-party registration. This is a controlled methods audit. The broad basis-dependence result and the change-of-coordinates algebra are established ideas.

## Decision and contribution boundary

Proceed with a new modular-arithmetic task and a deliberately factorized interface. Measure three distinct facts: a signed permutation preserves the complete coordinate-mask family; a general rotation changes that family but preserves the correctly transported interventions; a non-orthogonal coordinate change requires an oblique intervention operator, so an orthogonal projection onto the transformed subspace is usually a different intervention. The new contribution is the complete reproducible experiment, test data, and operator checks on trained networks. Do not claim a new theorem, a new feature-discovery method, or that basis sensitivity itself is new. The architecture supplies the true factorization. The experiment does not discover it.

## Closest primary work and overlap

- David Bau, Bolei Zhou, Aditya Khosla, Aude Oliva, and Antonio Torralba (2017), *Network Dissection: Quantifying Interpretability of Deep Visual Representations*. Section 3.2 already rotates representations and compensates the downstream map. Unit interpretability changes while predictions do not. This rules out claiming the basic rotation experiment as new. https://openaccess.thecvf.com/content_cvpr_2017/papers/Bau_Network_Dissection_Quantifying_CVPR_2017_paper.pdf
- Atticus Geiger, Zhengxuan Wu, Christopher Potts, Thomas Icard, and Noah Goodman (2024), *Finding Alignments Between Interpretable Causal Variables and Distributed Neural Representations*. DAS already searches non-standard bases and performs subspace interchange interventions. A transported known operator here is an oracle control, not an improvement over DAS. https://proceedings.mlr.press/v236/geiger24a.html
- Charles Godfrey, Davis Brown, Tegan Emerson, and Henry Kvinge (2022; revised 2023), *On the Symmetries of Deep Learning Models and their Internal Representations*. Model symmetries depend on the activation function and architecture. Place the change of coordinates at an explicitly linear interface. Do not claim arbitrary rotations are symmetries of tanh neurons. https://arxiv.org/abs/2205.14258
- Nelson Elhage et al. (2022), *Toy Models of Superposition*. Features need not correspond to single neurons. This experiment uses an invertible coordinate change without compression; it does not test the overcomplete feature packing studied in superposition. https://arxiv.org/abs/2209.10652
- Aleksandar Makelov, Georg Lange, and Neel Nanda (2023), *Is This the Subspace You Are Looking for? An Interpretability Illusion for Subspace Activation Patching*. Successful subspace interventions need not identify the mechanism used in intact execution. Exact transport proves that two interventions are the same under a coordinate change. It does not prove semantic truth, unique localization, or mechanism recovery. https://arxiv.org/abs/2311.17030
- Chuqin Geng, Li Zhang, Haolin Ye, Mark Zhang, Luke Zhang, and Xujie Si (2026), *Are We Recovering Mechanisms? Objective-Level Recovery Gaps in Mechanistic Interpretability*. This October paper warns that equal-size candidate circuits and good objective scores do not suffice for mechanism recovery. Our study isolates a coordinate-induced change in the candidate family with a supplied factorization; it does not resolve their general recovery problem. https://arxiv.org/html/2610.02098v1

These sources establish direct overlap. A targeted check found no basis for a broad originality claim. A web framework page called “Mechanistic Views” discusses gauge orbits and transport, but its primary paper and authors were not verified in this check. Do not use that page to support a formal novelty claim.

## Task and model

Inputs are a,b in {0,1,2,3,4}. All 25 ordered states occur in ordinary model training. The class is the ordered pair ((a+b) mod 5, b), encoded as 5*((a+b) mod 5)+b, with 25 classes. The second output component makes donor b leakage visible. There is no nuisance input and no selector. The labels use modular arithmetic, but the model can memorize this finite table. The study does not establish that it learned a modular-arithmetic algorithm.

Use two separate trainable branches: one-hot a → 4 linear features; one-hot b → 4 linear features. Concatenate the branches into an 8-dimensional linear interface z. All conditions include an explicit fixed 8→8 linear interface map, equal to the identity in the native model. A shared decoder maps this interface → 32 tanh units → 25 softmax classes. Branch biases are allowed, but the interface must remain linear after the branch outputs. Train only the branch and decoder parameters with full-batch Adam and seeds 0–7. The interface map is fixed at identity during training. Use 4,000 steps, learning rate 0.02, beta1=0.9, beta2=0.999, epsilon=1e-8, float64. Initialize weights with independent normal draws of standard deviation sqrt(2/(fan_in+fan_out)) and biases at zero. Draw matrices in order Ea (5×4), Eb (5×4), W1 (8×32), W2 (32×25). Use NumPy default_rng(seed). There is no regularization, early stopping, validation-based model choice, or learning-rate schedule. Record every seed and do not replace failed seeds. Require at least 99% intact accuracy for a seed to support the semantic label interpretation; since there are 25 states, this requires 25/25. Keep ineligible seeds in all descriptive tables and numerical covariance checks. Aggregate all eight seeds with equal weight; if any seed is ineligible, report its failure and do not make a semantic success claim for that seed.

The native projector P=diag(1,1,1,1,0,0,0,0) changes the a branch only. For any receiver r and donor d, the patched interface equals the intact interface for state (a_d,b_r). Thus this is a known causal intervention by architecture, independent of whether training succeeds.

## Intervention pairs and selection

Enumerate all 500 ordered pairs with a_d != a_r: 25 receivers × 20 donors. The target is the intact model distribution at (a_d,b_r), with semantic target class as above. Sort all 500 ordered pairs by SHA-256 of the literal UTF-8 string "basis-audit-v1|{receiver_id}|{donor_id}". Define state ID as 5*a+b and pair ID as 25*receiver_id+donor_id. Break any hash tie by pair ID. Assign the first 250 pairs to discovery and the remaining 250 to test. Save every pair, hash, split, and whether b_d=b_r. Report the b-match counts by split. State clearly that the split holds out pairings, not ordinary task states. No training state is held out.

Enumerate all C(8,4)=70 rank-four coordinate masks in each basis. Select the mask with smallest discovery mean total variation distance to the intact counterfactual. Find the smallest discovery TV value. Treat every value no more than 1e-12 above that minimum as tied. Choose the lexicographically first mask from that set. This numerical tie rule is fixed before the run. Report signed-permutation score invariance separately from tuple identity. Use the selected mask on the test pairs. Record all candidate scores and all selections. Do not use test scores to choose a basis, transform condition, seed, or mask.

## Predeclared coordinate changes

Use row vectors throughout: z'=zT. Include identity, one fixed signed permutation generated by NumPy default_rng(99) using permutation(8) followed by choice([-1.0,1.0], size=8), and eight fixed orthogonal matrices Q_j shared across model seeds. Save the signed-permutation matrix to remove any library-convention ambiguity. Generate Q_j by QR of standard-normal matrices with fixed seeds 100–107 and the standard positive-diagonal sign correction. Store the matrices. Haar sampling over O(8) is sufficient; do not force determinant +1.

For each Q_j also use T_j,c=Q_j D_c, where D_c is diagonal with geometrically spaced positive entries from c^(-1/2) to c^(1/2), for c=4 and c=16. These matrices have known condition number c. Thus the study has 26 bases: identity, signed permutation, eight rotations, and sixteen non-orthogonal transforms. The non-orthogonal conditions are predeclared secondary tests of the intervention operator, not additional evidence that rotations alone change coordinate localization. A diagonal coordinate mask commutes with D_c. Therefore every coordinate-patch output must remain the same between Q_j and Q_j D_c. The scaling conditions keep the coordinate candidate family fixed while changing the Euclidean image projector. Check this identity for every candidate and pair before interpreting the operator comparison.

Set the explicit interface map to T and fold T^-1 into the decoder’s first weight matrix. Biases must be transformed consistently. There is no nonlinear activation between these two maps. Save transformed parameters or enough information to reproduce them. Verify the reparameterized forward pass independently; do not only multiply and immediately undo T in one expression. A second check can fold the two source branches and T into one dense 10→8 encoder. That dense encoder is not block-sparse: a generic T does not preserve the separate-branch parameter constraint. The architectural claim is a coordinate change at the explicit linear interface, not a symmetry within the source branch class.

## Operators and exact control

For row vectors, native patching is z_r+(z_d-z_r)P. The same intervention in the new coordinates uses P_T=T^-1 P T:

(z_r T)+(z_d T-z_r T)P_T = [z_r+(z_d-z_r)P]T.

P_T is idempotent. It is symmetric under orthogonal T, but can be oblique under a general invertible T. With the compensated decoder, all patched output distributions must match to numerical tolerance. This is elementary change-of-coordinates algebra, presented as a test specification rather than a new theorem.

For the non-orthogonal test, the transformed a-subspace is the row span of A=T[0:4,:]. Its Euclidean orthogonal projector is E=A.T @ solve(A @ A.T, A). Patch with E as a separate comparator. E and P_T have the same image subspace, but can have different nullspaces. Do not call their different output behavior a failure of transport: E is not the transported operator.

For every basis compare: (1) the known transported P_T; (2) the discovery-selected coordinate mask of rank four; (3) for non-orthogonal conditions, E. Do not add a naive-index comparator. Report empty and full patches, and the Euclidean image projector for all bases as controls; the non-orthogonal image-projector comparison is the predeclared secondary result. Rank matching alone does not make the oracle transported operator a fair learned-method baseline.

## Fixed outcomes

Primary descriptive outcome: held-out mean TV error for the selected rank-four coordinate mask in each of the eight rotation bases, relative to the native basis. Report all eight model seeds separately, the per-seed rotation mean and range, and the across-seed mean with range. Do not assume every rotation makes every model worse. Do not count 64 model/rotation combinations as 64 independent models.

Primary implementation control: maximum absolute difference in all class probabilities between native P patches and correctly transported P_T patches, across every pair, seed, and basis. Also report intact-output residuals. Expected residual is numerical round-off, not an empirical claim of greater faithfulness.

Secondary operator outcome: held-out TV for E under condition numbers 4 and 16, against P_T, plus ||E-P_T||_F and the projector idempotence/symmetry residuals. Report semantic class accuracy as a secondary descriptive score. The reference model may have nonzero softmax uncertainty; TV is to its counterfactual distribution, not to a one-hot label.

## Required checks and report limits

1. All 25 intact distributions remain equal after each coordinate change, tolerance 1e-10 for float64.
2. Native branch patch equals the intact counterfactual interface and distribution for all 500 pairs.
3. Correctly transported patches match native branch patches for all pairs and bases, tolerance 1e-10.
4. Every signed-permutation coordinate candidate has a corresponding native coordinate mask with identical predictions. This checks candidate-family closure, not just the winning score. Ties can alter the chosen tuple but not the score.
5. Every coordinate-mask output agrees across condition numbers 1, 4, and 16 for a fixed Q. Each Q is orthogonal, T has the declared condition number, P_T is idempotent, and E is symmetric/idempotent. Include empty/full patch sanity checks.
6. Discovery/test pairs are disjoint and exhaustive. Verify mask selection from raw candidate CSVs. An independent verifier must recompute the primary table from weights, matrices, and pair IDs without importing the experiment runner.
7. Figures: coordinate-mask TV by basis class with all seed points; non-orthogonal E versus transported P_T. Separate near-zero residuals from substantive errors. No significance test is required.

The designed architecture makes the native factorization favorable by construction. This is useful ground truth, but not evidence that ordinary trained networks usually have a similarly privileged representation. The finite fully observed task is a methods audit, not a language-model result. It does not identify a natural or unique coordinate basis. It does not show that orthogonal projectors are universally wrong: they are correct under an explicit Euclidean geometric definition and orthogonal coordinate changes. A general coordinate transform changes that geometry unless it too is transported.

## Budget and output

Eight tiny CPU models, 26 coordinate systems, 70 candidates per system, and 500 pairs fit a small local NumPy run. Batch masks to limit memory. Keep training and selection code NumPy-only. Plotting may use the repository .venv with matplotlib 3.10.8. Publish the frozen plan, code, weights, transforms, pair ledger, full scores, selected-mask table, verification, and a short plain-English methods paper. Include an AI-use statement and state whether human peer review occurred. If results are null or invariant where variation was expected, publish that result without new settings. Do not start a second run to seek a better headline.
