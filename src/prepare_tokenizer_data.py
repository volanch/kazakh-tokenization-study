import csv
import random
import unicodedata
from pathlib import Path

root = Path(__file__).resolve().parents[1]
source = root / "data" / "used_texts.csv"
out_dir = root / "data_tokenizer"
out_dir.mkdir(exist_ok=True)

with source.open(encoding="utf-8-sig", newline="") as f:
    rows = [r for r in csv.DictReader(f) if r.get("kazakh", "").strip()]

random.Random(42).shuffle(rows)
cut = int(len(rows) * 0.8)
train_rows, test_rows = rows[:cut], rows[cut:]

for name, subset in (("train.txt", train_rows), ("test.txt", test_rows)):
    with (out_dir / name).open("w", encoding="utf-8", newline="\n") as f:
        for row in subset:
            text = unicodedata.normalize("NFC", row["kazakh"]).strip()
            f.write(text + "\n")

print(f"Train: {len(train_rows)} sentences; test: {len(test_rows)} sentences")