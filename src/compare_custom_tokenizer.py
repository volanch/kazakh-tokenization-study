"""Compare the Qwen tokenizer and the custom Kazakh SentencePiece tokenizer."""
from __future__ import annotations

import csv
import statistics
import unicodedata
from pathlib import Path

import sentencepiece as spm
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
DATA_TOKENIZER = ROOT / "data_tokenizer"
RESULTS = ROOT / "results"
QWEN_ID = "Qwen/Qwen3-8B"
SP_MODEL = DATA_TOKENIZER / "models" / "kk_unigram_2048.model"
SOURCE = DATA / "used_texts.csv"
PER_ITEM = RESULTS / "custom_qwen_comparison.csv"
SUMMARY = RESULTS / "custom_qwen_summary.csv"


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    sentencepiece = spm.SentencePieceProcessor(model_file=str(SP_MODEL))
    qwen = AutoTokenizer.from_pretrained(QWEN_ID)
    rows: list[dict[str, object]] = []

    with SOURCE.open("r", encoding="utf-8-sig", newline="") as source:
        for row in csv.DictReader(source):
            text = unicodedata.normalize("NFC", row["kazakh"].strip())
            if not text:
                continue
            qwen_count = len(qwen.encode(text, add_special_tokens=False))
            custom_count = len(sentencepiece.encode(text, out_type=int))
            rows.append({
                "item_id": row["item_id"],
                "qwen_tokens": qwen_count,
                "custom_sentencepiece_tokens": custom_count,
                "token_change_percent": round((qwen_count - custom_count) / qwen_count * 100, 4) if qwen_count else "",
            })

    if not rows:
        raise ValueError(f"No non-empty Kazakh rows found in {SOURCE}")

    with PER_ITEM.open("w", encoding="utf-8-sig", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    qwen_counts = [int(row["qwen_tokens"]) for row in rows]
    custom_counts = [int(row["custom_sentencepiece_tokens"]) for row in rows]
    qwen_total = sum(qwen_counts)
    custom_total = sum(custom_counts)
    summary = [
        {"metric": "sentences", "qwen": "", "custom_sentencepiece": len(rows), "change_percent": ""},
        {"metric": "total_tokens", "qwen": qwen_total, "custom_sentencepiece": custom_total,
         "change_percent": round((qwen_total - custom_total) / qwen_total * 100, 4)},
        {"metric": "mean_tokens_per_sentence", "qwen": round(statistics.mean(qwen_counts), 4),
         "custom_sentencepiece": round(statistics.mean(custom_counts), 4),
         "change_percent": round((statistics.mean(qwen_counts) - statistics.mean(custom_counts)) / statistics.mean(qwen_counts) * 100, 4)},
        {"metric": "sd_tokens_per_sentence", "qwen": round(statistics.stdev(qwen_counts), 4),
         "custom_sentencepiece": round(statistics.stdev(custom_counts), 4), "change_percent": ""},
        {"metric": "median_tokens_per_sentence", "qwen": statistics.median(qwen_counts),
         "custom_sentencepiece": statistics.median(custom_counts), "change_percent": ""},
    ]
    with SUMMARY.open("w", encoding="utf-8-sig", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)

    print(f"Processed {len(rows)} Kazakh FLORES+ rows.")
    print(f"Qwen: {qwen_total} tokens, mean {statistics.mean(qwen_counts):.2f} per sentence.")
    print(f"SentencePiece 2048: {custom_total} tokens, mean {statistics.mean(custom_counts):.2f} per sentence.")
    print(f"Difference: {(qwen_total-custom_total)/qwen_total*100:.2f}% fewer SentencePiece tokens.")
    print(f"Wrote {PER_ITEM} and {SUMMARY}.")


if __name__ == "__main__":
    main()
