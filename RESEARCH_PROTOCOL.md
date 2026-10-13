# Research protocol

Established: 13 October 2026.

## Purpose

Produce useful, testable research on the internal operation of neural networks. Aim for a specific contribution. Do not promise a new discovery or a completed paper on each run.

## Study process

1. Read the current research state and existing studies. Resume incomplete work before starting another study.
2. Check primary research sources. Record the search date, source versions, close prior work, and the proposed difference. Reject a claim of novelty when prior work already supports it.
3. Write the question, method, controls, fixed comparisons, stopping rules, and report plan before examining test results. Save a content hash. A local plan is not an external registration.
4. Run an experiment or a formal analysis. Preserve failed runs and negative results. Clearly distinguish trained models, constructed examples, external observations, and hypotheses.
5. Save code, dependencies, seeds, data generation rules, model weights where practical, and complete result tables. Use explicit units and define all scores.
6. Check the implementation and the interpretation. Use a separate AI reviewer for substantial claims. Record review issues and their resolution. State whether any human review occurred.
7. Write a clear paper. Explain the question and result in plain English before technical detail. Include assumptions, related work, methods, results, limits, and reproduction instructions. Do not invent results or citations.
8. Publish complete studies only. Verify the remote commit and file contents. A local commit, upload attempt, or page view is not proof of publication. Preserve history and use a pull request if the branch requires it.

## Scope of evidence

A small synthetic model can test a precise claim under controlled conditions. It cannot establish how often the result occurs in language models. Task labels do not reveal a trained network's internal circuit. An accurate task model need not use the expected mechanism.

Separate a new artifact or experiment from a new scientific principle. Cite close work even when it reduces the claimed contribution. Do not divide one experiment into several papers merely to increase output.

## Resource budget

Check the available account usage at the start of each run and before substantial new work. Keep a 20% reserve in every reported core usage window. Stop new work if a window has 20% or less remaining, or ordinary usage is unavailable. Save the current state and pause the repeat task. If the meter cannot be read, preserve work and defer new research until it is available.

Use small local experiments by default. Do not buy credits, redeem usage resets, invoke paid model APIs, or rent compute without a new user instruction. Limit parallel work to specific tasks that improve the study. Account usage is shared; changes in the meter are not an exact cost for this repository.

## Continuation

Maintain `docs/RESEARCH_STATE.md`. Record the study state, unresolved issues, next steps, and published paper links. Keep private account data and credentials out of the repository. Keep exact local run state in the ignored `.local-state/` directory when needed.

Repeat runs are research attempts. A run can continue an experiment, reject a weak idea, fix an error, or publish a supported negative result. It need not produce a new paper.
