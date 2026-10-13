# Versioned analysis plan: rare causal features under a fixed SAE budget

Date: 13 October 2026. Status: final plan before training. The plan and runner must be committed before the first experiment. This document is not an external preregistration.

## Main question

Does rare-event weighting improve a learned SAE feature's held-out causal replacement test when the true task effect is fixed, but feature frequency and activation amplitude vary? The main comparison is between ordinary downstream-loss training and a version that balances rare and common task cases. A local-reconstruction SAE is a third baseline.

The contribution is a new controlled experiment. It is not the first report that reconstruction can miss useful features, that downstream loss can help, or that decoder alignment does not prove causal use. The balanced objective uses true labels. It is a supervised oracle comparator, not a proposed label-free method.

## Close primary work checked

- Braun et al., *Identifying Functionally Important Features with End-to-End Sparse Dictionary Learning* (2024), https://arxiv.org/abs/2405.12241. Downstream training is already established. We isolate rare-event weighting at a fixed code budget.
- Gao et al., *Scaling and Evaluating Sparse Autoencoders* (2024), https://arxiv.org/abs/2406.04093. TopK training and downstream feature evaluation are established.
- Chanin and Garriga-Alonso, *SynthSAEBench* (2026), https://arxiv.org/abs/2602.14687. Synthetic ground-truth benchmarks and reconstruction/recovery disagreement are established. This small study makes no realism or benchmark superiority claim.
- Chanin and Garriga-Alonso, *Sparse but Wrong: Incorrect L0 Leads to Incorrect Features in Sparse Autoencoders* (ICML 2026), https://proceedings.mlr.press/v306/chanin26a.html. A fixed sparsity budget is a substantive restriction, not a guarantee of correct features. We hold it constant and report realized activity.
- Bal, *From Geometric Recovery to Causal Validation* (July 2026), https://arxiv.org/abs/2607.12166. This directly separates geometric alignment, firing, and causal use. Our primary endpoint therefore uses an actual learned-latent intervention, not a cosine match.
- Hoang et al., *Sparse Autoencoders Encode Both Concepts and Functions: The Downstream Geometry of Feature Effects* (July 2026), https://arxiv.org/abs/2607.24645. Context-dependent feature effects are established. We test paired backgrounds and a same-task-label donor control, without claiming a new feature-effect framework.

This was a targeted primary-source search. It cannot prove that the exact design has no prior instance.

## Factorial design

Use six independent replication seeds, 0–5. Each seed sets one background dictionary, all data, and one initial SAE. Across objectives, hold dictionary, data, initial weights, and minibatch index sequence exactly fixed. Also couple random background samples across the four frequency/amplitude conditions within a seed.

- Rare-event probability p: 0.01 or 0.16.
- Rare-feature amplitude a: 0.25 or 1.0.
- Objectives: local reconstruction; reconstruction plus ordinary downstream KL; reconstruction plus label-balanced downstream KL.
- Total: 4 data conditions × 3 objectives × 6 seeds = 72 small SAEs.

The conditions (p=0.01,a=1) and (p=0.16,a=0.25) have the same expected rare-feature squared energy, p*a^2=0.01. This makes a useful descriptive comparison. Equality of this moment is simple arithmetic, not a research discovery, and it does not make the two training distributions identical.

## Known representation and readout

Input dimension is 12. For each seed, construct a random orthogonal basis. Reserve one unit vector u for the task-relevant feature. Draw 20 unit-length background feature directions in the other 11 dimensions. The rare direction is therefore exactly orthogonal to the background span. This deliberate simplification separates rarity from non-identifiability.

Each background coefficient is zero with probability 0.9; otherwise it is uniform on [0.5,1.5]. Let r be the rare-event bit. The activation is h = background + a*r*u. There is no observation noise.

The frozen downstream binary readout is q(h)=sigmoid(8*(u dot h/a - 0.5)). Thus the correct probabilities are sigmoid(-4) and sigmoid(4), and the true output effect is identical across all four conditions. This readout is specified, not trained. The SAE is the learned system. Do not call this a trained language-model circuit.

## Data split and rare-case support

Training: 10,000 independent backgrounds per seed. Fix exactly 100 or 1,600 rare cases using a seeded permutation. The rare cases at p=0.01 are a subset of those at p=0.16. All other randomness is held constant across the frequency conditions. Report the exact positive counts; no zero-positive training runs are possible.

Validation: 1,024 new backgrounds, each evaluated once with r=0 and once with r=1. Test: 2,048 further new backgrounds, each evaluated with both values of r. Sample IDs are disjoint and draws are independent across splits. Background values can recur. In particular the all-zero background has probability 0.9^20, about 12.2%. Report each split's all-zero count and the number of exact shared unique background values for each split pair. Do not claim all test values are unseen. These paired stratified banks provide 2,048 held-out rare cases regardless of p. Use p-weighting to report natural-distribution reconstruction and task metrics. Do not describe the test bank itself as an iid draw from the rare-event distribution.

Selection of a learned latent uses validation pairs only. Final scores use test pairs only. Save exact generation seeds, dictionary, array shapes, split identifiers, and SHA-256 hashes of the generated arrays. The deterministic generator reproduces the arrays. Do not duplicate identical background arrays for each objective.

## SAE and training

Use a 12→24→12 SAE. Center inputs with the training mean. The encoder is linear plus a learned bias, followed by ReLU and per-example TopK with k=4. Set all other activations to zero. The decoder is linear with learned bias. Add the fixed training mean back to the decoded centered vector before calculating reconstruction loss or the downstream readout. Thus hhat = TopK(ReLU((h-mu)W_enc+b_enc))W_dec+b_dec+mu. Each decoder direction has unit norm after each optimizer update. Ties use the lower latent index. Report actual nonzero counts, since ReLU can give fewer than four nonzero latents.

Initialize decoder directions from normalized independent Gaussian vectors; initialize encoder weights as the transpose of the initial decoder and both biases to zero. Each objective starts from exactly these same arrays. Use float64 NumPy, Adam, 4,000 steps, minibatches of 128 drawn with replacement, learning rate 0.005, beta1=0.9, beta2=0.999, epsilon=1e-8. No auxiliary dead-feature loss, resampling, or learning-rate changes. The same index sequence is used for all objectives within a seed and condition. Save every training run, including poor runs; no seed replacement or after-the-fact training extension.

Let hhat be the reconstruction. The reconstruction term is the sample mean of the squared Euclidean norm ||hhat-h||^2; it is not divided by dimension. Local loss uses only this term. Ordinary downstream training adds 0.1 times the sample mean of KL(Bernoulli(q(h)) || Bernoulli(q(hhat))). Balanced training uses the same coefficient and weights the KL term by 1/(2p) on positive cases and 1/(2(1-p)) on negative cases. The reconstruction term remains naturally weighted in all three objectives. Use stable log-sigmoid and probability computations.

TopK gives the same maximum code budget, not equal realized sparsity or equal learned quality. The selected KL weight and optimizer schedule are fixed design choices, not tuned optima.

## Primary causal endpoint

For a paired background, encode h0 and h1 into z0 and z1. For each latent j, perform both replacements:

1. Start from the original activation h1 and add (z0[j]-z1[j])*decoder[j]. Compare its readout with q(h0).
2. Start from the original activation h0 and add (z1[j]-z0[j])*decoder[j]. Compare its readout with q(h1).

These residual-preserving patches leave the SAE reconstruction error at its original receiver value. They isolate the output effect of the selected decoded contribution. Do not replace the original activation by its SAE reconstruction in the primary endpoint.

The residual-preserving single-latent interchange error is the mean of the two absolute probability errors. Select the latent with the lowest validation mean error, breaking ties within 1e-12 by lower index. Freeze that latent. The primary endpoint is its held-out test interchange error. This measures whether one learned latent can transport the specified output change under a residual-preserving intervention. It does not identify recovery of the full internal mechanism. Also report the reconstruction-based version, which patches the latent inside the SAE reconstruction; that secondary score combines reconstruction error with transport error. Keep residual-preserving and reconstruction-based full-code and empty replacement errors beside the single-latent results. Native full-code replacement is h_receiver + (z_donor-z_receiver)W_dec; reconstructed full-code replacement decodes the donor code. The native version retains the receiver residual in both the single-latent and full-code cases. A patched code can have five active latents even though the original encoder permits at most four. Report active counts after each patch and do not apply TopK again to a patched code.

Primary contrast: ordinary downstream error minus balanced downstream error at p=0.01 and a=0.25. Positive values favour the balanced objective. Report all six paired seed differences, their mean, median, range, and a 95% percentile paired bootstrap interval with 10,000 draws and seed 20261013. Report the other three factorial cells without moving the primary hypothesis. Local reconstruction is a fixed baseline. Do not infer generality across tasks from an interval over six replication seeds.

Directional prediction: balanced downstream loss will reduce primary single-latent error. Null or negative outcomes remain publishable. Strong full-reconstruction performance without single-latent improvement is a meaningful failure to localize, not a successful mechanism-recovery claim.

## Required secondary measurements

- Natural-weighted reconstruction error and downstream KL.
- Rare-case and common-case classification performance and probability error separately.
- Full-code replacement, empty replacement, and single-latent replacement errors.
- Chosen latent's signed decoder cosine with u, activation rate given r=1 and r=0, and its mean paired activation difference. Do not infer causal use from these values alone.
- Actual mean code activity, activity after each latent patch, and count of latents never active in training data.
- Equal-energy cell comparison, clearly descriptive.
- Latent selection and held-out results for all 24 candidate latents, not only the winner.

For a same-label nuisance control, pair each test background with the next background in a fixed cyclic order, keeping r unchanged. Replace the chosen latent between these cases using the primary residual-preserving intervention. Report the change from each receiver's original readout. Also run the reconstruction-based version and report both its change from the unmodified reconstruction and its error against the true readout. The correct task output should not change. This checks whether a seemingly useful intervention also transfers irrelevant background information. It is not used to select the latent.

## Correctness controls and run gate

Before training, check analytic gradients by finite differences on points with a strict gap at the TopK boundary and no ReLU kink. Check all three objectives, including balanced weighting, decoder gradients, and bias gradients. Check unit decoder norms after an update.

Check an identity reconstruction and the known rare-direction oracle on the complete validation/test construction. The oracle's single-coordinate replacement must recover the exact counterfactual readout. Verify empty replacement returns the receiver reconstruction and full-code replacement returns the paired donor reconstruction. Report disjoint draw IDs and exact repeated background values rather than requiring distinct values. Check that q(h) is independent of every background and has the same rare-event effect in all four cells.

Lock code and plan in a commit before training or held-out evaluation. Record hashes and exact package versions. No training until root approves the locked implementation. Save model weights, exact generation records with array hashes, all candidate latent scores, selected latents, per-test-background outputs for the selected latent, loss traces, controls, runtime, and summary tables. Use one BLAS thread. The primary bootstrap uses seed 20261013. Initialization uses RNG seed 3000+replication_seed; dictionary uses 1000+seed; data use 2000+seed; minibatch indices use 4000+seed. Each condition starts a fresh generator at the same applicable seed. An independent implementation should verify the readout, selected-latent interventions, and primary contrast from saved weights.

## Interpretive limits

The rare feature is made easy to identify geometrically. This isolates frequency and amplitude but does not represent realistic LLM superposition. The downstream readout uses only one bit. The balanced loss receives labels unavailable to a fully unsupervised SAE. A good interchange score on this generator does not prove interpretable semantics elsewhere. Unit decoder normalization, fixed TopK, and the chosen loss coefficient restrict the result. Full-code recovery can coexist with failure of a one-latent description. Better task preservation does not establish that all background features were recovered.

This should be one paper of about 2,000 words with a compact factorial result table, a primary comparison, and a clear negative-outcome policy. Do not split its cells into separate papers.
