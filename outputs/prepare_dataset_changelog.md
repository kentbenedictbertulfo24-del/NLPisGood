# `prepare_dataset.py` Changelog

- **Previous (`mainProj`):** Read a local `datasets/cebuaner_final_annotated.txt` file (which was missing from `datasets/`) and processed only `CebuaNER`.
- **Current (`step-1-2-lora-decoding`):** Loads `josephimperial/CebuaNER` (reconstructing plain Cebuano text per passage without entity tags or `-DOCSTART-`) and `filbench/cebuano-readability` directly from Hugging Face, writing 4,608 combined passages to `datasets/cebuano_combined.txt`.
- **How to restore the previous version if needed:** Run `git checkout mainProj -- prepare_dataset.py` from the repository root.