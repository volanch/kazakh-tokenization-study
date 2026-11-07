from pathlib import Path
import csv
import sentencepiece as spm

root = Path(__file__).resolve().parents[1]
model_file = root / "data_tokenizer" / "models" / "kk_unigram_2048.model"
input_file = root / "data_tokenizer" / "test.txt"
output_file = root / "data_tokenizer" / "custom_tokenizer_results.csv"

tokenizer = spm.SentencePieceProcessor(model_file=str(model_file))

with input_file.open("r", encoding="utf-8-sig") as source, \
     output_file.open("w", newline="", encoding="utf-8-sig") as result:

    writer = csv.writer(result)
    writer.writerow(["sentence_id", "text", "token_count", "tokens"])

    for sentence_id, line in enumerate(source, start=1):
        text = line.strip()
        if not text:
            continue

        tokens = tokenizer.encode(text, out_type=str)
        writer.writerow([sentence_id, text, len(tokens), " ".join(tokens)])

print(f"Готово. Результаты сохранены в: {output_file}")