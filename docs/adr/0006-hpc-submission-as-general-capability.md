# HPC Submission is a general capability, not a fine-tuning-specific backend

Pawsey access only makes sense for End Users who already hold an HPC allocation — a different, Advanced-Mode-only audience from the default non-technical flow. It would have been simplest to bolt Pawsey on as another `TrainerBackend` variant, matching the old pipeline's dual-backend design. We instead model it as a general **HPC Submission** capability that fine-tuning is the first consumer of, because dataset enrichment is a likely second consumer (large raw-data corpora may need HPC-scale processing too) and we don't want that to require reworking the abstraction later.

Consequences: submitting to Pawsey sends the End User's raw data to a shared academic cluster, a real departure from this product's "fully local" pitch. The wizard requires explicit one-time consent the first time a user selects it, and assumes credentials/allocation are already configured — provisioning HPC access is out of scope.
