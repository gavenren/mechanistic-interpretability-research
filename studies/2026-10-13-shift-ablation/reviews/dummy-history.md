# Dummy check history

All runs used random dummy tensors. No UCI image was passed through a model. No scientific model was trained.

- v1 checked the first complete runner. All controls passed. The plan still described the parameter hash as per-parameter hashes. The implementation used one aggregate hash.
- v2 followed the plan wording correction. All controls passed.
- v3 corrected the time projection to use exact forwarded-example ratios before the two-fold factor. All controls passed.
- v4 also exercised the float64 probability and full score-summary code on dummy outputs. All controls passed.
- v5 changed the labels of aggregate probability checks to show their full-model scope. It is the final preflight for the present runner and plan. Its full files are in `dummy-preflight/`. Earlier summary receipts remain next to this note.

No scientific tolerance changed. The final projected time is an estimate from dummy data. The scientific run still has a fixed 1,200-second cap.
