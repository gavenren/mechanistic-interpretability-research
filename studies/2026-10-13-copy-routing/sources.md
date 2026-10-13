# Source record

Checked on 13 October 2026. These are primary research sources or the official model card. The search does not prove that the experiment is unique.

## Model

**M1. EleutherAI. Pythia-14M model card**, fixed revision `cf967c0a9a04383db6f7b1108d86b2962634b4ac`. [Fixed card](https://huggingface.co/EleutherAI/pythia-14m/blob/cf967c0a9a04383db6f7b1108d86b2962634b4ac/README.md).

This card specifies the model used here. It records a 27 February 2026 change: the former deduplicated-data model moved to `pythia-14m-deduped`. A model name alone is not a sufficient version record. The 14M model was added after the original Pythia paper.

## Closest work

**R1. Elhage et al. (2021). A Mathematical Framework for Transformer Circuits.** Published 22 December 2021. [Author publication](https://transformer-circuits.pub/2021/framework/index.html).

The paper separates the attention route, called the QK circuit, from the value and output path, called the OV circuit. It also describes copying and induction in small attention-only models. The route/value distinction in this study comes from that work. The author page was checked through indexed primary-page text; direct access exceeded the browsing size limit.

**R2. Olsson et al. (2022). In-context Learning and Induction Heads.** arXiv:2209.11895v1, submitted 24 September 2022. [Fixed version](https://arxiv.org/abs/2209.11895v1).

The paper studies heads that complete repeated token patterns. It gives causal evidence in small attention-only models and more limited evidence in larger models with MLPs. Our experiment does not discover this mechanism. It tests specific route and value changes in one pretrained model that has MLPs.

**R3. Crosbie and Shutova (2025). Induction Heads as an Essential Mechanism for Pattern Matching in In-context Learning.** arXiv:2407.07011v3, revised 2 April 2025; first submitted in 2024. [Fixed version](https://arxiv.org/abs/2407.07011v3), [full text](https://arxiv.org/html/2407.07011v3).

This work tests head ablation and targeted attention knockout in Llama-3-8B and InternLM2-20B. It is close prior work for the use of attention interventions to test pattern matching. Our controlled source-only route swap and contextual-value exchange form a different small experiment; they are not a new general causal method.

**R4. Bajaj, Mistry, Maini, Aggarwal, and Tiganj (2026). Beyond Semantics: How Temporal Biases Shapes Retrieval in Transformer and State-Space Models.** EACL 2026, pages 7562–7581. DOI: 10.18653/v1/2026.eacl-long.355. [Published record](https://aclanthology.org/2026.eacl-long.355/), [paper](https://aclanthology.org/2026.eacl-long.355.pdf).

This work tests repeated anchors, serial position, overlapping episodes, and ablation of heads selected by induction score. It already links temporal retrieval patterns to induction-style copying. Our order control and calibration attention score overlap with this approach. The smaller contribution here is the fixed, balanced binding test and the combined route/value intervention record on Pythia-14M.

**R5. Che and Li (2026). Does the Model Use the Feature? Separating Steering from Mechanism in LLMs.** arXiv:2610.07270v1, submitted 5 October 2026. [Fixed version](https://arxiv.org/abs/2610.07270v1), [full text](https://arxiv.org/html/2610.07270v1).

The paper distinguishes feature control from natural use. It tests installation, removal, and downstream rescue and shows that conclusions depend on the update rule. This supports the scope limit here: an artificial route or value intervention gives evidence about that intervention, not a full identification of the natural algorithm.

## Proposed contribution boundary

This study adds one fully specified experiment with one public checkpoint, disjoint calibration and test token groups, balanced key–value bindings, pair order controls, all 24 head results, and separate route/value changes. It does not claim a new induction principle, a new attention decomposition, a new recency effect, or broad evidence about modern language models. No scientific outcomes were used to write this source record or the initial plan.
