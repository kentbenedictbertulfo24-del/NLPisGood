from pathlib import Path
import re

input_file = Path("datasets/cebuaner_final_annotated.txt")

print("File exists:", input_file.exists())


with input_file.open("r", encoding="utf-8") as file:
    lines = file.readlines()

print("Total lines:", len(lines))


words = []

for line in lines:
    parts = line.split()

    if len(parts) > 0 and parts[0] != "-DOCSTART-":
        words.append(parts[0])

print("Total extracted words:", len(words))
print("First 20 words:", words[:20])


article_count = 0

for line in lines:
    if line.strip() == "":
        article_count += 1

print("Article separators:", article_count)

def clean_punctuation(text):
    text = re.sub(r"\s+([.,!?;:%])", r"\1", text)
    return text

articles = []
current_article = []

for line in lines:
    parts = line.split()

    if len(parts) == 0:
        if current_article:
            articles.append(clean_punctuation(" ".join(current_article)))
            current_article = []

    elif parts[0] != "-DOCSTART-":
        current_article.append(parts[0])

if current_article:
    articles.append(" ".join(current_article))

print("Total articles:", len(articles))
print("First article:")
print(articles[0])

output_file = Path("datasets/cebuaner_cleaned.txt")

with output_file.open("w", encoding="utf-8") as file:
    for article in articles:
        file.write(article + "\n")

print("Cleaned dataset saved successfully!")
print("Saved to:", output_file)