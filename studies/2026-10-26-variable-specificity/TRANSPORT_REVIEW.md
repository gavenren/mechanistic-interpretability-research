# Separate AI review of the transport amendment

Date: 26 October 2026. Result: the corrected transport method has no open review blocker. Public verification is still required before final tests.

This is a review after edit calibration and before final testing. It changes file transport only. It does not change model training, edit selection, missing eligibility, final cases, gates, or tolerances. The reviewer ran no model, benchmark, or saved-result calculation. This is an AI review, not human peer review.

The reviewer read `analysis/final_transport.py`, `TRANSPORT_PLAN.md`, the original calibration verifier, the corrected transport test, and its receipt. The wrapper delegates every ordinary file check to the original function. Its only special case is the exact path `calibration/frozen.zip`.

For that path, the wrapper first checks its own source, the transport plan, this review, and the transport manifest against the specified public commit. It checks both ordered parts with the original public-byte check. It checks each part's length and SHA-256 value, compares its bytes with the corresponding segment of the local ZIP, rejects trailing data, and checks the total length and fixed original digest. The expected archive remains 30,963,232 bytes with SHA-256 `64c6fe719663885b2db61f8a9c91e6a59abd5b16eef93c79bc8624e73b57aec7`.

The original calibration verifier then checks its complete local file list, raw-file hashes, archive members, and every archived payload. The split files do not replace those checks. The original training, preparation, code, runtime, and phase checks remain in place.

The first proposed layout placed the transport files inside `calibration`. That layout would fail the original exact-membership check. The corrected layout places both parts and their manifest in the separate `transport` folder. The original calibration folder remains unchanged. The earlier wrapper, test, and receipt are retained as amendment history. No final model test ran under the first layout.

The corrected test checks changed parts, reversed order, missing parts, and extra archive bytes. It also checks delegation for other paths and calls the original calibration verifier. The root agent's receipt reports a pass in 0.309988 seconds. Together with the first 0.159768-second check, measured work was 0.469755 seconds. Both attempts share the one 10-second transport charge. The reviewer inspected these tests and receipts but did not repeat them. Their public-byte calls use test stubs, so they do not establish GitHub publication.

Transport verification occurs inside the unchanged final runner's timed and guarded execution. It uses the same exclusive compute lock and resource limits. The reserve remains 120 seconds: 20 for the original frozen archive, 10 for transport preparation and tests, 50 for the result audit, and 40 for final archive checks. The amendment creates no new compute allowance.

Before final tests, the root agent must verify the final public commit's exact wrapper, plan, review, manifest, and two part files. The final tree must place both parts outside `calibration`; an earlier uploaded part at the old path must not remain in that tree. Preserve the earlier commit history. Reconstruct the exact ZIP, then run the original calibration checks through the wrapper. A failed public-byte check or changed scientific source must stop the run. This review does not authorize a scientific rerun or a change to the selected edits.
