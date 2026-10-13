# Source record and contribution check

Checked: 13 October 2026. Sources are primary research papers or their official publication pages. This was a targeted review. It was not a systematic review and does not establish priority. No full external paper is copied into this repository.

| Source and version | Use in this study |
| --- | --- |
| Fred Zhang and Neel Nanda. [Towards Best Practices of Activation Patching in Language Models: Metrics and Methods](https://arxiv.org/abs/2309.16042v2), January 2024 revision | Score and corruption choices can alter localization. This general effect is prior work. |
| Stefan Heimersheim and Neel Nanda. [How to Use and Interpret Activation Patching](https://arxiv.org/abs/2404.15255v1), April 2024 | Patching direction and intervention design need explicit interpretation. |
| Atticus Geiger et al. [Inducing Causal Structure for Interpretable Neural Networks](https://proceedings.mlr.press/v162/geiger22a.html), ICML 2022 | Prior use of aligned changes in causal and neural models. Our networks receive ordinary task training. |
| Atticus Geiger et al. [Causal Abstraction: A Theoretical Foundation for Mechanistic Interpretability](https://arxiv.org/abs/2301.04709v4), May 2025 revision | Defines the causal abstraction framework. Context-specific interchange tests are established. |
| Rohan Gupta et al. [InterpBench: Semi-Synthetic Transformers for Evaluating Mechanistic Interpretability Techniques](https://arxiv.org/abs/2407.14494v3), October 2025 revision; first posted in 2024 | Prior benchmark with known circuit structure. Known labels in our task do not imply a known learned circuit. |
| Aaron Mueller et al. [MIB: A Mechanistic Interpretability Benchmark](https://arxiv.org/abs/2504.13151v2), June 2025 revision | Distinguishes circuit localization from causal-variable localization. |
| Li Puyin et al. [Bucketing the Good Apples: A Method for Diagnosing and Improving Causal Abstraction](https://arxiv.org/abs/2605.02234v1), May 2026 | Close prior work on regions of successful and unsuccessful interchange interventions. |
| [The Curse of Multiple Mediators: Hidden Interaction Effects in Activation Patching](https://arxiv.org/abs/2606.27510), June 2026 | Prior work on interactions and unstable patch scores. We do not claim to discover these effects. |
| Chuqin Geng et al. [Are We Recovering Mechanisms? Objective-Level Recovery Gaps in Mechanistic Interpretability](https://arxiv.org/abs/2610.02098v1), October 2026 | Closest comparison: equal-size circuits, selection objectives, held-out behavior, and recipient context. |

The first candidate topic concerned aggregation of normalized patch scores. It was set aside as the main contribution because direct prior work already compares aggregation choices. The selected study instead records a complete finite comparison of two variable-localization selection rules. Its contribution is an experiment and its artifacts, not a new general causal principle.

The task states and intervention pairs are generated locally. There are no external training data, downloaded model weights, human subjects, or paid model calls. The code and data record the actual executed experiment. Citations provide context; they are not evidence for the numerical results of this study.
