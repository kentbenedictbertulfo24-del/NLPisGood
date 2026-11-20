from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase
from pathlib import Path
result=[]
# The type hints and noinspection comments only exist to quiet PyCharm's checker.
# noinspection PyNoneFunctionAssignment

def log(message):
    print(message)
    result.append(message)
tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained("distilgpt2")
# noinspection PyNoneFunctionAssignment
model: PreTrainedModel = AutoModelForCausalLM.from_pretrained("distilgpt2")
log("DistilGPT2 loaded successfully!")

words = {
    "English": ["Terrace", "Arcade", "Casino", "bombing", "oversea", "nervous"],
    "Cebuano": ["gutom", "malibang", "maglakaw", "kumpare", "kan-anan", "sakit"],
}

log("\n--- TOKENIZER TEST ---")
for language, group in words.items():
    counts = []
    for word in group:
        tokens = tokenizer.tokenize(word)
        counts.append(len(tokens))
        log(f"{word}: {tokens} ({len(tokens)} tokens)")
    log(f"Average {language}: {sum(counts) / len(counts):.2f} tokens per word\n")

log("--- TEXT GENERATION ---")
for prompt in ["Kumusta ka", "Help me I", "Nag unsa"]:
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(
        **inputs,
        max_new_tokens=50,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )
    log(f"Prompt: {prompt}")
    log(f"Generated: {tokenizer.decode(outputs[0], skip_special_tokens=True)}")

    output_folder = Path("outputs")
    output_folder.mkdir(exist_ok=True)
    output_file = output_folder / "baseline_results.txt"
    with output_file.open("a", encoding="utf-8") as file:
        file.write("\n".join(result))
        file.write("\n\n")