# Preparation and Experiment 1: test task ability

Date: 19 October 2026. This is a follow-up to the [research plan](../../docs/RESEARCH_PLAN.md), version `f482f479866de4d3dba5c80d92e9f5914f8c2389`.

## Question and scope

Can a longer training schedule, with or without wider hidden layers, make the four weak task settings pass the fixed ability tests? This work tests intact models. It does not identify an internal explanation. It makes no claim of a new scientific principle.

The starting methods are Studies [10](../2026-10-13-register-write-order/paper.md), [11](../2026-10-13-shift-ablation/paper.md), [12](../2026-10-13-hall-obstruction/paper.md), and [14](../2026-10-13-flow-unit-ranking/paper.md). Their code and data at repository version `1f289aee2b2218437789c7217c5ad1c9f30abb19` are the source record. See the [review](../../docs/REVIEW_PAPER.md) for prior work. This preparation does not add a literature novelty claim.

## Fixed design

Run tasks in this order: matching graphs, register operations, digit images, then generated points. Run Recipe A with development seeds 8001 and 8002. Then run Recipe B with the same seeds. Choose one recipe from development results only. Run the chosen recipe with final seeds 8101, 8102, 8103, and 8104.

Recipe A keeps the original width. Recipe B doubles that width. Both use four times the original training schedule. No model may exceed 100,000 parameters. There are at most 20 development models and 20 final models. Each digit job contains two architectures. No failed seed is replaced. No schedule is extended.

The [symbolic methods](METHODS_SYMBOLIC.md) and [numeric methods](METHODS_NUMERIC.md) fix the exact data, seeds, batches, arithmetic, controls, schedules, gates, and saved files. These documents and all six code files form the execution plan. They must be in a verified public commit before scientific training starts. Dummy preparation may run before that commit.

For classification, choose the highest minimum accuracy across all development groups and both seeds. For flows, require region error at most 0.05 in both development seeds. Then choose the lowest mean excess negative log likelihood. Use total development job runtime to break ties, followed by Recipe A. Stop the flow branch if neither recipe is eligible.

All four final seeds must pass every task gate. The digit branch requires both architectures. A partial pass does not pass a branch. A failed ability test blocks later explanation tests in that branch. Do not start Experiments 2 to 6 in this run.

## Resources and stop rules

Use one local CPU process, two compute threads, and one inter-operation thread. No GPU, paid API, rented compute, account reset, or purchased credits are included. Automatic research stays paused.

Preparation has a one-hour process elapsed-time cap. Experiment 1 has a four-hour cap, including evaluation, controls, saving, and saved-weight replay. Coding and review time are outside these compute caps. Each task times 20 dummy training updates of the wider recipe. Project all eight jobs per task, include evaluation and replay, and apply a fourfold safety factor. The combined estimate must fit four hours. A failed estimate stops the run; it does not permit scope changes after results are seen.

The roadmap's 16 GB RAM allowance is a planning allowance. This run uses a stricter four-GiB process peak limit. The host has 24 GiB physical RAM. Preparation records its current memory state. Require 20 GiB free disk. Check the process peak, deadline, and private usage permit within training and evaluation loops. The permit expires after 120 seconds. It is refreshed only after a live allowance check. Stop new research if any core window has 20% or less remaining, the meter is unavailable, or ordinary use is unavailable. Save partial files after a stop.

## Required evidence

Preparation must pass shape, arithmetic, known-reference, unchanged-input, and explicit-equation checks. A separate AI reviewer checks the code and its rules. No human peer review is recorded.

Save each model's initial and final weights, input arrays or exact generators, training trace, raw evaluation, all group results, and costs. Save partial weights on a training stop. Recalculate outputs and gate decisions from saved weights with separate equations. This is a saved-weight replay, not a repeat of training. Hash result files. Keep every completed and failed attempt. Report model counts and incomplete groups. Do not treat thousands of input rows as independent models.

Publish an execution report even if preparation fails. A preparation failure is a resource or method result; it gives no new evidence about model ability.
