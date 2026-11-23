from pathlib import Path
import re
from datasets import load_dataset


def clean_punctuation(text: str) -> str:
  text = re.sub(r"\s+([.,!?;:%])", r"\1", text)
  text = re.sub(r"\s+", " ", text)
  return text.strip()


def extract_ner_passages(ner_ds) -> list[str]:
  passages = []

  for split_name in ner_ds.keys():
    current_tokens = []

    for row in ner_ds[split_name]:
      # Case 1: Dataset rows already contain a list of tokens per sentence/example
      if "tokens" in row and isinstance(row["tokens"], list):
        toks = [str(t).strip() for t in row["tokens"] if str(t).strip()]
        if not toks:
          if current_tokens:
            passages.append(clean_punctuation(" ".join(current_tokens)))
            current_tokens = []
          continue

        if toks[0] == "-DOCSTART-":
          if current_tokens:
            passages.append(clean_punctuation(" ".join(current_tokens)))
            current_tokens = []
          continue

        current_tokens.extend(toks)

      # Case 2: Dataset rows are CoNLL/IOB lines like "word O" or blank lines
      else:
        first_val = next(iter(row.values()), "")
        line = str(first_val).strip() if first_val is not None else ""

        if not line:
          if current_tokens:
            passages.append(clean_punctuation(" ".join(current_tokens)))
            current_tokens = []
          continue

        parts = line.split()
        token = parts[0]
        if token == "-DOCSTART-":
          if current_tokens:
            passages.append(clean_punctuation(" ".join(current_tokens)))
            current_tokens = []
          continue

        current_tokens.append(token)

    if current_tokens:
      passages.append(clean_punctuation(" ".join(current_tokens)))

  return [p for p in passages if p]


def extract_readability_passages(read_ds) -> list[str]:
  passages = []

  for split_name in read_ds.keys():
    for row in read_ds[split_name]:
      raw_text = row.get("text")
      if raw_text is not None:
        cleaned = clean_punctuation(str(raw_text))
        if cleaned:
          passages.append(cleaned)

  return passages


def main():
  output_dir = Path("datasets")
  output_dir.mkdir(parents=True, exist_ok=True)
  output_file = output_dir / "cebuano_combined.txt"

  print("Downloading and loading josephimperial/CebuaNER...")
  ner_ds = load_dataset("josephimperial/CebuaNER")
  ner_passages = extract_ner_passages(ner_ds)
  print(f"Reconstructed CebuaNER passages: {len(ner_passages)}")

  print("Downloading and loading filbench/cebuano-readability...")
  read_ds = load_dataset("filbench/cebuano-readability")
  read_passages = extract_readability_passages(read_ds)
  print(f"Loaded cebuano-readability passages: {len(read_passages)}")

  combined_passages = ner_passages + read_passages

  with output_file.open("w", encoding="utf-8") as f:
    for passage in combined_passages:
      f.write(passage + "\n")

  print(f"Total combined passages: {len(combined_passages)}")
  print(f"Saved combined corpus to: {output_file}")


if __name__ == "__main__":
  main()