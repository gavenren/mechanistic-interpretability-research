# Fixed plan: spread of a local edit through layer normalization

Date: 22 October 2026. Formal and constructed numerical study. No trained model. No scientific run has started.

Question: how much of a one-coordinate edit can appear in other coordinates after layer normalization, and which part can be computed exactly from the shared mean and scale?

Use widths 4, 8, 16, 64. For each width draw 32 standard-normal vectors with NumPy seed 451000+width. Normalize with epsilon 0 or 0.00001, without learned gain or bias. Edit coordinate zero by delta equal to baseline standard deviation times each of -4,-1,-0.1,-0.01,0.01,0.1,1,4. This gives 2,048 fixed cases. Save every input, output, and case score.

Let z=x-mean(x), s=sqrt(mean(z*z)+epsilon), x'=x+delta*e0, and s' be its scale. Test the exact identity: norm(x')-norm(x) = delta*e0/s' + (1/s'-1/s)*z - delta*1/(d*s'). For coordinates outside zero the effect lies in the span of the constant vector and the old centered vector. This is an algebraic statement under the declared normalization. Derive it in the paper. No claim of a new general normalization theorem is made.

Measure changed-coordinate fraction, fraction of squared output change outside coordinate zero, and error of the local Jacobian approximation. The Jacobian is (I-11T/d)/s - zzT/(d*s^3). Use the entire output-change norm in the relative error denominator, bounded below by 1e-12. Report width and edit-size groups. Do not select a favorable input.

Controls: zero edit, constant input shift, native PyTorch layer_norm comparison, independent scalar calculation using math.fsum, and exact decomposition. Require maximum formula/native error at most 5e-12. No requirement that off-target effects be small; they are the outcome. Preserve all failed controls. The frozen-mean/frozen-scale comparator is a different operation, not the network's natural behavior.

Two CPU threads, 120-second cap including the separate calculation audit. Root controls the private usage permit and serial execution. Do not run before the public plan/code lock. No data/model downloads or paid compute. The paper must distinguish these random constructed states from natural model activations. This question differs from earlier basis transport and language patch calibration: it quantifies finite-coordinate spread through a fixed normalization operator.
