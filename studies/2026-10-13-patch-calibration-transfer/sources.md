# Sources and contribution limits

Sources checked on 13 October 2026. This was a focused search of primary sources. It was not a systematic review. The search does not prove that the experiment or correction is new.

## Model and checkpoint

**M1. EleutherAI. Pythia-14M model card and files.** Fixed revision: `cf967c0a9a04383db6f7b1108d86b2962634b4ac`. [Model card at this revision](https://huggingface.co/EleutherAI/pythia-14m/blob/cf967c0a9a04383db6f7b1108d86b2962634b4ac/README.md). [File list at this revision](https://huggingface.co/EleutherAI/pythia-14m/tree/cf967c0a9a04383db6f7b1108d86b2962634b4ac).

The card states that the model at this address changed on 27 February 2026. The earlier model used the deduplicated Pile. It moved to `EleutherAI/pythia-14m-deduped`. This study uses the current standard-Pile model at the fixed revision above. The name alone is not sufficient to identify the weights. The study records file hashes as well as the revision.

The card identifies six layers, width 128, four attention heads, and the Apache 2.0 licence. It states that 14M was added after the original Pythia suite. The model is a pretrained next-token model. It is not a chat model. Model rights remain with the upstream rights holders.

**M2. Stella Biderman et al. (2023). Pythia: A Suite for Analyzing Large Language Models Across Training and Scaling.** Checked [arXiv:2304.01373v2](https://arxiv.org/abs/2304.01373v2), revised 31 May 2023. This source describes the original suite and its research purpose. M1 supplies the details specific to the later 14M release. We do not imply that the original paper evaluated 14M.

## Research closest to this study

| ID | Checked primary source | Use and limit |
| --- | --- | --- |
| R1 | Sankaran Vaidyanathan, David Arbour, Aaron Mueller, Scott Niekum, and David Jensen (2026). [The Curse of Multiple Mediators: Hidden Interaction Effects in Activation Patching](https://arxiv.org/abs/2606.27510v1). Version 1, 25 June 2026. Abstract and full HTML checked. | Establishes close prior work on interactions in patching. The present study tests a fixed correction on new prompts. It does not introduce interaction measurement. |
| R2 | Fred Zhang and Neel Nanda (2024). [Towards Best Practices of Activation Patching in Language Models: Metrics and Methods](https://arxiv.org/abs/2309.16042v2). Version 2, 17 January 2024; ICLR 2024. Abstract and full HTML checked. | Supports fixing the output metric and donor rule. Sensitivity to these choices is established prior work. |
| R3 | Cody Rushing and Neel Nanda (2024). [Explorations of Self-Repair in Language Models](https://proceedings.mlr.press/v235/rushing24a.html). ICML, PMLR 235:42836–42855. Also checked [arXiv:2402.15390v2](https://arxiv.org/abs/2402.15390v2), 26 May 2024, including Section 3 and Appendix B. | Establishes final-LayerNorm effects after last-layer interventions. Our final-layer diagnostic checks a known mechanism. |
| R4 | János Kramár, Tom Lieberum, Rohin Shah, and Neel Nanda (2024). [AtP*: An Efficient and Scalable Method for Localizing LLM Behaviour to Components](https://arxiv.org/abs/2403.00745v1). Version 1, 1 March 2024. Abstract and full HTML checked. | Studies gradient approximation and exact verification. We use measured patch effects. We do not test or improve AtP*. |
| R5 | Luyang Zhang and Jialu Wang (2026). [When Attribution Patching Lies: Diagnosis and a Second-Order Correction](https://arxiv.org/abs/2606.09899v1). Version 1, 5 June 2026. Abstract and full HTML checked. | Studies nonlinear approximation error and curvature corrections. Our correction is a calibration median of observed residuals. It is not a gradient or curvature method. |
| R6 | Maxime Méloux, François Portet, and Maxime Peyrard (2026 revision). [Mechanistic Interpretability as Statistical Estimation: A Variance Analysis](https://arxiv.org/abs/2510.00845v4). Version 4, 28 May 2026. Current abstract and version record checked. | Establishes variation in causal scores as a research concern. Our fixed-set transfer test does not introduce statistical analysis of interpretability. |
| R7 | Yash Baligar (2026). [Float32-Induced Distortion in Activation Patching: Precision Floors and Displacement-Dependent Endpoint-Curvature Error](https://www.mdpi.com/2079-3197/14/9/212). *Computation* 14(9), 212. Published 9 September 2026. DOI: [10.3390/computation14090212](https://doi.org/10.3390/computation14090212). Publisher metadata and indexed publisher text checked. | Motivates the numerical audit of four-term interactions. The two numerical conditions are distinct. The double-arithmetic condition keeps the same float32 rotary tables. It is not a fully float64 forward or ground truth for the exact float32 program. |
| R8 | Emmanuel Ameisen et al. (2025). [Circuit Tracing: Revealing Computational Graphs in Language Models](https://transformer-circuits.pub/2025/attribution-graphs/methods.html). Published 27 March 2025. Author research page checked, including the treatment of normalization denominators. | Fixing normalization denominators for conditional linearity is established practice. Our diagnostic does not introduce that procedure. |
| R9 | Neel Nanda. [Attribution Patching: Activation Patching At Industrial Scale](https://www.neelnanda.io/mechanistic-interpretability/attribution-patching). Author technical article checked as served on the check date. It has no numbered paper revision. | Gives early treatment of attribution patching and limits from final normalization. This is an original technical article, not a peer-reviewed paper. |

## Why the correction is a median

**S1. Tilmann Gneiting (2011). [Quantiles as Optimal Point Forecasts](https://www.sciencedirect.com/science/article/pii/S0169207010000063).** *International Journal of Forecasting* 27(2):197–207. DOI: [10.1016/j.ijforecast.2009.12.015](https://doi.org/10.1016/j.ijforecast.2009.12.015). Publisher abstract and article record checked.

For absolute error, a median is an optimal constant prediction. Here, the error in adding the two single-patch effects is the interaction `I`. Thus a median of calibration interactions minimizes calibration absolute error among constant offsets. This is standard statistical reasoning. It gives no guarantee on new prompts. The held-out test measures whether the correction transfers.

## Contribution boundary

The study supplies one fixed test of a simple correction on one small pretrained checkpoint. Its primary question is prediction error on new word pairs and new prompt frames. It does not claim a new causal theorem, a new normalization mechanism, or a complete circuit explanation.

The intervention definition, metric, split, estimator, and numerical checks are fixed in [the analysis plan](analysis-plan.md). The results apply to that design. The literature search cannot establish that no prior study used the same correction. Scientific value depends on the measured result, full records, and reproducibility, not a guaranteed claim of novelty.
