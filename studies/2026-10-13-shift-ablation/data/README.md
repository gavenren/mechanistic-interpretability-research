# Data source and licence

These are unchanged copies of `optdigits.tra` and `optdigits.tes` from the UCI Optical Recognition of Handwritten Digits data set.

**Attribution:** E. Alpaydin and C. Kaynak (1998). *Optical Recognition of Handwritten Digits*. UCI Machine Learning Repository. DOI: [10.24432/C50P49](https://doi.org/10.24432/C50P49).

**Licence:** [Creative Commons Attribution 4.0 International](https://creativecommons.org/licenses/by/4.0/), as stated on the [UCI data page](https://archive.ics.uci.edu/dataset/80/optical+recognition+of+handwritten+digits). Retrieved 13 October 2026. No raw values, row orders, or labels were changed in these copies. The experiment places the 8×8 values in a larger canvas and divides them by 16 in memory.

| File | Bytes | SHA-256 |
|---|---:|---|
| optdigits.tra | 563639 | e1b683cc211604fe8fd8c4417e6a69f31380e0c61d4af22e93cc21e9257ffedd |
| optdigits.tes | 264712 | 6ebb3d2fee246a4e99363262ddf8a00a3c41bee6014c373ed9d9216ba7f651b8 |

The original split has 3,823 training images from 30 writers and 1,797 test images from 13 different writers. Each row has 64 integer counts in 0..16, then a digit label in 0..9. Individual writer IDs are not provided in these files. Do not treat image-level resampling as writer-level uncertainty.

The source archive URL is fixed in `../code/fetch_data.py`. The downloaded ZIP had SHA-256 `0d7b054fea010270e9b3f06411c654c5e59547732ad626381980baffe0a23fb0`. Reproduction checks the two raw file hashes; a ZIP metadata change alone need not change the data.
