from pathlib import Path
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

BASE_MODEL = "distilgpt2"
ADAPTER_DIR = Path("checkpoints/lora_cebuano_adapter")
OUT_FILE = Path("outputs/step2_decoding_kvcache.txt")

PROMPTS = [
    "Ang komunidad sa Sugbo nagtinabangay aron",
    "Sa panahon sa kusog nga ulan, kinahanglan nga",
    "Language adaptation on small models helps",
]

DECODING_CONFIGS = [
    {"name": "greedy", "do_sample": False},
    {"name": "top_k_50_temp_0.8", "do_sample": True, "top_k": 50, "temperature": 0.8},
    {"name": "top_p_0.9_temp_0.8", "do_sample": True, "top_p": 0.9, "temperature": 0.8},
]


def benchmark_cache(model, tokenizer, prompt: str, use_cache: bool, max_new_tokens: int = 40):
    inputs = tokenizer(prompt, return_tensors="pt")
    input_len = inputs["input_ids"].shape[1]

    with torch.no_grad():
        _ = model.generate(
            **inputs,
            max_new_tokens=10,
            do_sample=False,
            use_cache=use_cache,
            pad_token_id=tokenizer.eos_token_id,
        )

    start = time.perf_counter()
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=use_cache,
            pad_token_id=tokenizer.eos_token_id,
        )
    elapsed = time.perf_counter() - start
    new_tokens = out.shape[1] - input_len
    tok_per_sec = new_tokens / elapsed if elapsed > 0 else 0.0
    text = tokenizer.decode(out[0], skip_special_tokens=True)
    return elapsed, tok_per_sec, text


def main():
    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(str(ADAPTER_DIR))
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL)
    model = PeftModel.from_pretrained(base_model, str(ADAPTER_DIR))
    model.eval()

    lines = ["=== STEP 2: DECODING STRATEGIES & KV-CACHE BENCHMARK ===\n"]

    for prompt in PROMPTS:
        lines.append(f"\nPROMPT: {prompt}")
        inputs = tokenizer(prompt, return_tensors="pt")
        for cfg in DECODING_CONFIGS:
            kwargs = {k: v for k, v in cfg.items() if k != "name"}
            with torch.no_grad():
                out = model.generate(
                    **inputs,
                    max_new_tokens=40,
                    use_cache=True,
                    pad_token_id=tokenizer.eos_token_id,
                    **kwargs,
                )
            decoded = tokenizer.decode(out[0], skip_special_tokens=True)
            lines.append(f"  [{cfg['name']}] -> {decoded}")

    lines.append("\n=== KV-CACHE BENCHMARK (use_cache=True vs use_cache=False) ===")
    sample_prompt = PROMPTS[0]
    for use_cache in [True, False]:
        elapsed, tps, sample_out = benchmark_cache(model, tokenizer, sample_prompt, use_cache=use_cache)
        lines.append(
            f"use_cache={use_cache}: elapsed={elapsed:.3f}s | throughput={tps:.2f} tok/s | sample={sample_out}"
        )

    report = "\n".join(lines) + "\n"
    OUT_FILE.write_text(report, encoding="utf-8")
    print(report)
    print(f"Saved Step 2 report to {OUT_FILE}")


if __name__ == "__main__":
    main()