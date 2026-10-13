# Match a spike explanation to the deletion being tested

Status: plan and code under separate AI review. No study model has been trained or evaluated. Only input checks and dummy tests have run.

The study compares two meanings of spike deletion: removing its outgoing message, or removing both its message and its local voltage reset. It asks whether a gradient with the matching local paths selects a more consequential single event. It uses four fixed trained spiking networks, a fixed 120-image explanation panel, and the full 1,797-image test split for accuracy.

Read the [fixed plan](analysis-plan.md), [source record](sources.md), and [input ledger](inputs.json). The [runner](code/run.py) requires a verified public lock before scientific work. It refuses an existing output directory and preserves partial data if a check fails. The maximum run time is 20 minutes on two CPU threads. Passing a dummy test does not show task ability.

The exact scientific command and final result record will be added after the public lock is verified. The paper will retain all weak models and failed checks. No human peer review is claimed.
