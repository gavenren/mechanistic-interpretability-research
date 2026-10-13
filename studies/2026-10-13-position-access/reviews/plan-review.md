# Independent pre-run AI review

Date: 13 October 2026. Reviewer: a separate AI agent. Status: ready for the public plan and runner lock. No human review occurred. This review does not certify scientific outcomes.

I read the complete plan and runner. I checked the four-cell contrast, physical causal mask, fractional position coordinates, fixed token groups, all-layer filler-key block, selected-head operation, complete row counts, group bootstrap, failure handling, and lock-byte check.

The proposed primary contrast matches the code: native positions with filler access blocked, minus compressed positions with filler access allowed, at gap 128. The control input uses the correct sparse coordinates when native positions remain. No query can read a filler key in the blocked condition. Thus the short-input comparison can check removal of all filler-mediated paths, including later effects of filler queries. It does not isolate only attention normalization.

The pre-run review requested one addition: test four distinct dummy inputs in each batch. The earlier dummy checks used batches of one. The final dummy receipt now tests the actual scientific batch size of four at gaps 0, 4, and 128. The earlier receipts remain unchanged.

I checked every file hash in `dummy-preflight-v3/manifest.json` and compared its plan and runner hashes with the current files. The receipt records 66 model calls, 264 dummy examples, and 96 scored dummy rows. All 200 comparison checks and 66 attention checks passed. The largest logit error was `4.50461890011411e-12`, the largest core-state error was `2.0930479571745764e-13`, and the largest attention row-sum error was `1.887379141862766e-15`. Each is below its fixed bound. The receipt states that no scientific model output was collected.

Reviewed plan SHA-256: `671f15e628977538a111f5eff1220db46d13684b16c68e183f9424c3faefbb70`.

Reviewed runner SHA-256: `18a7cac510f8e67147da5746b252c959e2c760be3c5e6b085b1c23ad52a56cc7`.

The 16,896 scientific model-forward examples and 6,144 scored rows agree with the loop structure. All fixed cells remain in the analysis, and failed controls prevent a complete status. A later independent audit must still check the actual outputs, complete input construction, arithmetic, and model replay.

No pre-run blocker remains within this review scope. The result will concern one small checkpoint and one artificial binding task. The coordinate and access operators can interact. The compressed-readable geometry is artificial. A selected head need not keep its earlier role on this task. These limits must remain in the paper.
