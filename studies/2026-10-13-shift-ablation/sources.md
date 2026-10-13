# Checked primary sources

Checked 13 October 2026. These sources support the design and its limits. They do not prove that this experiment is the first of its kind. No source result is presented as a result from this study.

## Anti-aliasing

Richard Zhang. **Making Convolutional Networks Shift-Invariant Again.** ICML 2019, PMLR 97, pages 7324–7334.

- [Official proceedings](https://proceedings.mlr.press/v97/zhang19a.html)
- [Primary paper](https://proceedings.mlr.press/v97/zhang19a/zhang19a.pdf)
- Scope checked: abstract, introduction, and method description.
- Overlap: low-pass filtering before downsampling, shift stability, and CNN accuracy are established. The proposed binomial filter is a known method. This study measures the effect of channel removal under shifts; it does not claim a new anti-aliasing architecture.

## Channel ablation

Saurabh Desai and Harish Guruprasad Ramaswamy. **Ablation-CAM: Visual Explanations for Deep Convolutional Network via Gradient-free Localization.** WACV 2020.

- [Official proceedings and abstract](https://openaccess.thecvf.com/content_WACV_2020/html/Desai_Ablation-CAM_Visual_Explanations_for_Deep_Convolutional_Network_via_Gradient-free_Localization_WACV_2020_paper.html)
- Scope checked: primary proceedings record and abstract; indexed primary appendix.
- Overlap: feature-map ablation is an established way to measure class-related effects. This study does not claim to invent channel ablation or a new image attribution map. It compares full signed output-effect vectors before and after exact pixel shifts.

## Stability of causal explanations

Ola Ahmad, Nicolas Bereux, Loïc Baret, Vahid Hashemi, and Freddy Lecue. **Causal Analysis for Robust Interpretability of Neural Networks.** WACV 2024; arXiv:2305.08950, inspected version 2 dated 20 June 2023.

- [Primary arXiv record](https://arxiv.org/abs/2305.08950)
- [Full text](https://arxiv.org/html/2305.08950)
- Scope checked: full main text, particularly methods in Section 3 and stability tests in Section 5.
- Overlap: the paper uses path/weight interventions to find causal graphs. It tests graph stability under different intervention strengths and explanation stability after white noise is added to images. It covers several image models and data sets. Stable causal explanation is therefore an established topic.
- Difference: the inspected text does not compare a fixed anti-aliasing architecture change using all channel-removal output vectors under integer image shifts. Our smaller experiment is an empirical extension, with strong structural controls and limited scope.

## Data

E. Alpaydin and C. Kaynak. **Optical Recognition of Handwritten Digits.** UCI Machine Learning Repository, 1998. DOI: 10.24432/C50P49.

- [Official data page](https://archive.ics.uci.edu/dataset/80/optical+recognition+of+handwritten+digits)
- [DOI](https://doi.org/10.24432/C50P49)
- [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)
- Scope checked: input construction, original split, file list, attribution, and licence. Raw data validation checks the two original files. Exact hashes and attribution are in `data/README.md`.
- Limit: the 64-count images are not raw high-resolution photographs. Writer identities are not supplied in the numeric files. The seed bootstrap is not a writer bootstrap.

The source search also used combinations of “anti-aliasing”, “attribution stability”, “channel ablation”, and “shift”. It found related work on output robustness and feature maps. No exact match to this bounded comparison was found in that search. This absence is not a proof of novelty or a comprehensive literature review.
