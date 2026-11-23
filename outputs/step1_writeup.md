# Step 1 Write-Up: Scaling Laws & LoRA Fine-Tuning on Cebuano

## Scaling Laws (Kaplan et al., 2020 & Hoffmann et al., 2022)
Language model cross-entropy loss follows predictable power-law relationships with model parameter count ($N$), dataset size ($D$), and compute budget ($C$), provided none of these become the limiting bottleneck. Kaplan et al. (2020) established power-law scaling across compute, dataset size, and model parameters, while Hoffmann et al. (2022) ("Chinchilla") showed that model size and training tokens should scale together in balanced proportions for compute-optimal training.

## Why LoRA Fits `distilgpt2` Adaptation
Updating every parameter during full fine-tuning increases memory and compute demands even on a compact model like `distilgpt2`. Low-Rank Adaptation (LoRA) freezes the pretrained `distilgpt2` weights and trains small low-rank decomposition matrices on target layers (`c_attn`), adapting the model to Cebuano text from `josephimperial/CebuaNER` and `filbench/cebuano-readability` while keeping only a small trainable parameter footprint and saving a lightweight adapter (`checkpoints/lora_cebuano_adapter`) for downstream use.