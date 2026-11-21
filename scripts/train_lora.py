import math
from pathlib import Path
import matplotlib.pyplot as plt
import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

MODEL_NAME = "distilgpt2"
CORPUS_PATH = Path("datasets/cebuano_combined.txt")
ADAPTER_DIR = Path("checkpoints/lora_cebuano_adapter")
OUTPUT_DIR = Path("outputs")

# Exact words and prompts from baseline.py for direct before/after comparison
WORDS_BY_LANG = {
    "English": ["Terrace", "Arcade", "Casino", "bombing", "oversea", "nervous"],
    "Cebuano": ["gutom", "malibang", "maglakaw", "kumpare", "kan-anan", "sakit"],
}
PROMPTS = ["Kumusta ka", "Help me I", "Nag unsa"]


def evaluate_model(model, tokenizer, label: str) -> list[str]:
  model.eval()
  device = next(model.parameters()).device
  lines = [f"=== {label} ==="]

  lines.append("\n--- TOKENIZER TEST ---")
  for lang, group in WORDS_BY_LANG.items():
    counts = []
    for word in group:
      tokens = tokenizer.tokenize(word)
      counts.append(len(tokens))
      lines.append(f"{word}: {tokens} ({len(tokens)} tokens)")
    avg = sum(counts) / len(counts)
    lines.append(f"Average {lang}: {avg:.2f} tokens per word")

  lines.append("\n--- TEXT GENERATION ---")
  for prompt in PROMPTS:
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
      outputs = model.generate(
          **inputs,
          max_new_tokens=50,
          do_sample=False,
          pad_token_id=tokenizer.eos_token_id,
      )
    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    lines.append(f"[{prompt}] -> {text}")

  return lines


def main():
  if not CORPUS_PATH.exists():
    raise FileNotFoundError(
        f"Missing {CORPUS_PATH}. Run `python prepare_dataset.py` first."
    )

  OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
  ADAPTER_DIR.mkdir(parents=True, exist_ok=True)

  tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
  tokenizer.pad_token = tokenizer.eos_token

  base_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
  before_lines = evaluate_model(base_model, tokenizer, "BEFORE LORA (BASE DISTILGPT2)")

  lora_config = LoraConfig(
      r=16,
      lora_alpha=32,
      target_modules=["c_attn"],
      lora_dropout=0.05,
      bias="none",
      task_type="CAUSAL_LM",
      fan_in_fan_out=True,
  )
  model = get_peft_model(base_model, lora_config)
  model.print_trainable_parameters()

  raw_ds = load_dataset("text", data_files={"train": str(CORPUS_PATH)})
  block_size = 128

  def chunk_examples(examples):
    tokens = tokenizer(examples["text"])
    concatenated = sum(tokens["input_ids"], [])
    total_len = (len(concatenated) // block_size) * block_size
    chunks = [
        concatenated[i : i + block_size] for i in range(0, total_len, block_size)
    ]
    return {"input_ids": chunks, "labels": chunks.copy()}

  lm_ds = raw_ds["train"].map(
      chunk_examples,
      batched=True,
      remove_columns=raw_ds["train"].column_names,
  )
  split_ds = lm_ds.train_test_split(test_size=0.05, seed=42)

  training_args = TrainingArguments(
      output_dir="./trainer_runs",
      per_device_train_batch_size=4,
      gradient_accumulation_steps=4,
      num_train_epochs=2,
      learning_rate=5e-4,
      logging_steps=25,
      save_strategy="no",
      fp16=torch.cuda.is_available(),
      report_to="none",
  )

  trainer = Trainer(
      model=model,
      args=training_args,
      train_dataset=split_ds["train"],
      eval_dataset=split_ds["test"],
      data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
  )

  train_result = trainer.train()
  eval_metrics = trainer.evaluate()
  perplexity = math.exp(eval_metrics["eval_loss"])

  model.save_pretrained(str(ADAPTER_DIR))
  tokenizer.save_pretrained(str(ADAPTER_DIR))

  after_lines = evaluate_model(model, tokenizer, "AFTER LORA (CEBUANO FINE-TUNED)")
  summary_lines = [
      f"Final training loss: {train_result.training_loss:.4f}",
      f"Eval loss: {eval_metrics['eval_loss']:.4f}",
      f"Held-out perplexity: {perplexity:.2f}",
      "",
  ]
  comp_path = OUTPUT_DIR / "step1_comparison.txt"
  comp_path.write_text("\n".join(summary_lines + before_lines + [""] + after_lines), encoding="utf-8")

  losses = [
      (entry["step"], entry["loss"])
      for entry in trainer.state.log_history
      if "loss" in entry
  ]
  if losses:
    steps, vals = zip(*losses)
    plt.figure(figsize=(7, 4))
    plt.plot(steps, vals, marker="o", linewidth=1.5)
    plt.title("Step 1: DistilGPT2 Cebuano LoRA Training Loss")
    plt.xlabel("Step")
    plt.ylabel("Training Loss")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "training_loss_curve.png", dpi=150)
    plt.close()

  print(f"Final training loss: {train_result.training_loss:.4f}")
  print(f"Held-out perplexity: {perplexity:.2f}")
  print(f"Saved adapter to: {ADAPTER_DIR}")
  print(f"Saved comparison to: {comp_path}")


if __name__ == "__main__":
  main()