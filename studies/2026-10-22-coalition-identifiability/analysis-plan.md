# Fixed plan: what singleton patch tests leave unknown

Date: 22 October 2026. Status: plan before results. This is a census of constructed Boolean mechanisms. It includes no trained model.

## Question and contribution

How often do all single-component noising and denoising results fail to identify the full coalition structure within a finite mechanism class?

[Heimersheim and Nanda](https://arxiv.org/abs/2404.15255v1) already explain the difference with AND and OR examples. [Wang and colleagues](https://arxiv.org/abs/2211.00593v1) use faithfulness, completeness, and minimality to assess circuits. The [Hydra study](https://arxiv.org/abs/2307.15771v1) reports compensation in language models. This work does not claim those observations as new. It gives a complete small-domain census of how much information the eight singleton queries retain. It does not test adaptive self-repair.

No earlier repository study enumerates the full class of monotone four-input Boolean functions or groups them by the same singleton evidence. This study is separate from the donor study: replacement values are fixed at zero and one, and the question is which groups must be tested.

## Domain and scores

Enumerate all 65,536 four-input Boolean truth tables. Retain functions with `f(0000)=0`, `f(1111)=1`, and monotonicity under each one-bit increase. This removes both constant functions. Use the standard finite labeled-input class; do not weight it as a distribution of real neural circuits.

The clean state is `1111`. The corrupt state is `0000`. For each of four inputs, denoising restores that input alone to one. Noising removes that input alone from the clean state. The eight-bit signature records the four restored outputs and the four output losses. Group functions by this signature.

A sufficient coalition is a set S with `f(1_S)=1`. A disabling coalition is a set D with `f(1_(N\D))=0`. Minimal means no proper subset has the same property. It does not mean the smallest cardinality among all such sets. Enumerate all 16 coalitions for every retained function. Save sufficient sets, disabling sets, and both minimal families.

Report the number of retained functions, distinct signatures, functions in ambiguous groups, uniquely identified functions, largest signature group, functions with no sufficient singleton, functions with no disabling singleton, and functions where neither singleton test detects a component. Report minimum coalition sizes separately from minimal-set counts. Do not claim that a minimal sufficient set is a unique explanation.

## Independent checks and formal statement

The main implementation filters full truth tables by local monotonicity. A second generator builds functions from every antichain of nonempty subsets. An antichain contains no pair where one set contains the other. Require both generators to give exactly the same function set.

Check the classical finite duality: D disables a monotone function if and only if D intersects every minimal sufficient set. This is a mathematical identity for this construction, not a proposed discovery. Reconstruct every saved truth table from its saved minimal sufficient sets. Require exact agreement. Record all coalitions and signature groups so the census can be checked without rerunning enumeration.

The controls define success. The study does not require a particular ambiguity rate. Use exact integers and Boolean logic. There is no sampling error in the retained finite class and no population inference about language models.

## Resources and outputs

Use one local Python process. Cap this run, including checks, at 300 elapsed seconds. The three intervention studies together have a 900-second cap. No training, paid API, or GPU is used. The root runs scientific work in sequence. Check the shared private usage permit and deadline during enumeration. Stop on any control or resource failure and save the failure record. Do not change the four-input scope after the result is seen.

Before execution, verify the code, plan, source record, and shared guard against the public lock. Save functions JSON, coalition CSV, signature groups, metrics, time, plan hash, and file hashes. Obtain a separate AI review. No human peer review is recorded.
