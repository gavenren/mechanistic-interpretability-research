# Separate AI source review

Date: 22 October 2026. The root agent reviewed the plan and code before results. This record was written by the implementation agent from that review. No human peer review is recorded.

The review accepted the constructed-model scope and the polynomial identity on ordinary states. It requested two corrections. The zero-versus-mean comparison must be calculated from results, not fixed as a Boolean constant. The saved-row audit must also recompute weighted summary values, rather than checking raw scores alone.

The code now derives the comparison from the exact summary rows. The saved-row audit independently expands the polynomial, checks the support condition, and recalculates original accuracy, donor accuracy, support violations, total probability, and absolute score change. The audit calls the shared resource guard. The updated code passes Python compilation. The scientific calculation has not run at this review stage.

The report must distinguish a label-matched donor from a target-removal intervention. The former conditions on the answer and serves as a control. The study establishes a property of an explicit constructed model. It does not estimate the behavior of trained networks.
