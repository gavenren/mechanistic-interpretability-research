# Separate source review of graph and register branches

**Date:** 19 October 2026.  
**Reviewer:** Separate AI agent that wrote the digit and flow branches. No human review is recorded.  
**Scope:** Source review only. No scientific training or evaluation was run for this review.

The review checked `code/hall.py`, `code/register.py`, and the root runner against the research roadmap and the earlier study code.
No blocking source error was found in the graph or register branches.

- The graph model retains the original finite graph definition, label source, five categories, learning rate, and architecture. Both hidden widths stay below the parameter cap. The new schedule has four times the original 64 epochs.
- Graph training and evaluation cover the same 65,536 graph states. The code states this limit. The result cannot establish prediction on unseen graph states.
- The register model retains the original 14 operations, GRU, 16 joint states, learning rate, and 12-operation training histories. Its 6,000 updates are four times the original 1,500.
- The register test includes separate groups at horizons 0, 1, 2, 8, and 32. Longer groups start with the fixed copy/write orders. Their later operations are random. The gate tests the final joint state in each group. Complete checkpoint accuracy is a separate supporting measure.
- Register histories are distinct across training, development, and final banks. Final inputs do not determine recipe selection.
- Both branches use explicit saved-weight equations for output replay. They preserve a partial model when training stops. They check the private usage permit and stage deadline during training and replay.
- The root runner selects from two recipes using development scores, then uses four new final seeds. A failed final model is retained. A partial panel does not pass the full-panel rule.

The preparation run must still confirm runtime, shape, no-op, arithmetic, and reference-rule checks. This source review does not replace those checks.
The saved-weight replays do not repeat training. They also share software and task definitions with the project. External replication remains separate work.

The digit and flow modules received a syntax check with Python's compile step. That check passed. Their author did not perform a separate review of their own scientific claims.
