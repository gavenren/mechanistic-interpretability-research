# Data source and licence

These are unchanged copies of the UCI `optdigits.tra` and `optdigits.tes` files, already retrieved for the repository's shift-ablation study on 13 October 2026.

Attribution: E. Alpaydin and C. Kaynak (1998). *Optical Recognition of Handwritten Digits*. UCI Machine Learning Repository. [DOI 10.24432/C50P49](https://doi.org/10.24432/C50P49). The [UCI record](https://archive.ics.uci.edu/dataset/80/optical+recognition+of+handwritten+digits) states the [CC BY 4.0 licence](https://creativecommons.org/licenses/by/4.0/).

No raw values, labels, or row orders were changed. This study divides the 64 counts by 16 in memory and uses them directly as input currents.

| File | Rows | SHA-256 |
|---|---:|---|
| optdigits.tra | 3823 | e1b683cc211604fe8fd8c4417e6a69f31380e0c61d4af22e93cc21e9257ffedd |
| optdigits.tes | 1797 | 6ebb3d2fee246a4e99363262ddf8a00a3c41bee6014c373ed9d9216ba7f651b8 |

The original split separates 30 training writers and 13 test writers. These numeric files do not include individual writer IDs. The 120 explanation-panel row indices and labels are fixed in [inputs.json](../inputs.json). All test rows remain in the model-ability check.
