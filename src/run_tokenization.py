"""Compare Qwen3 and Llama 3.1 tokenizer counts on FLORES+ dev sentences.

This script stores item IDs and derived counts, not dataset sentence text.
"""

from __future__ import annotations

import argparse
import csv
import importlib.metadata
import json
import platform
import statistics
import unicodedata
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from datasets import load_dataset
from transformers import AutoTokenizer


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
LANGUAGES = {
    "eng_Latn": "English",
    "kaz_Cyrl": "Kazakh",
    "rus_Cyrl": "Russian",
}
MODELS = {
    "Qwen/Qwen3-8B": "Qwen3-8B",
    "meta-llama/Llama-3.1-8B-Instruct": "Llama-3.1-8B-Instruct",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=200, help="Number of aligned IDs to process; use 20 for the pilot.")
    parser.add_argument("--dataset-revision", default="main", help="FLORES+ revision; pin a commit for the final run.")
    parser.add_argument("--qwen-revision", default="main", help="Qwen tokenizer revision; pin a commit for the final run.")
    parser.add_argument("--llama-revision", default="main", help="Llama tokenizer revision; pin a commit for the final run.")
    return parser.parse_args()


def load_aligned_rows(revision: str, limit: int):
    by_language: dict[str, dict[str, str]] = {}
    for code in LANGUAGES:
        dataset = load_dataset(
            "openlanguagedata/flores_plus",
            code,
            split="dev",
            revision=revision,
        )
        by_language[code] = {
            str(row["id"]): str(row["text"])
            for row in dataset
            if row.get("id") is not None and row.get("text") is not None
        }

    all_ids = set.union(*(set(by_language[code]) for code in LANGUAGES))
    complete_id_sets = {
        code: {item_id for item_id, text in by_language[code].items() if text.strip()}
        for code in LANGUAGES
    }
    shared_ids = set.intersection(*(complete_id_sets[code] for code in LANGUAGES))

    def sort_key(item_id: str) -> tuple[int, int | str]:
        return (0, int(item_id)) if item_id.isdigit() else (1, item_id)

    ordered_shared = sorted(shared_ids, key=sort_key)
    selected = ordered_shared[:limit]
    selected_set = set(selected)
    audit = []
    for item_id in sorted(all_ids, key=sort_key):
        present = {code: item_id in by_language[code] and bool(by_language[code][item_id].strip()) for code in LANGUAGES}
        audit.append({
            "item_id": item_id,
            **{f"present_{code}": present[code] for code in LANGUAGES},
            "complete_triplet": all(present.values()),
            "selected_for_run": item_id in selected_set,
            "exclusion_reason": "" if all(present.values()) else "missing_or_blank_language_text",
        })
    aligned = []
    for item_id in selected:
        texts = {code: by_language[code][item_id] for code in LANGUAGES}
        if all(text.strip() for text in texts.values()):
            aligned.append((item_id, texts))
    if not aligned:
        raise RuntimeError("No complete aligned rows were loaded. Check access, dataset revision, and language configs.")
    return aligned, audit


def get_tokenizers(qwen_revision: str, llama_revision: str):
    revisions = {
        "Qwen/Qwen3-8B": qwen_revision,
        "meta-llama/Llama-3.1-8B-Instruct": llama_revision,
    }
    tokenizers = {}
    for model_id, revision in revisions.items():
        tokenizers[model_id] = AutoTokenizer.from_pretrained(model_id, revision=revision)
    return tokenizers


def measure(aligned_rows, tokenizers):
    records = []
    for item_id, texts in aligned_rows:
        for code, raw_text in texts.items():
            text = unicodedata.normalize("NFC", raw_text)
            words = len(text.split())
            chars = len(text)
            for model_id, tokenizer in tokenizers.items():
                token_ids = tokenizer.encode(text, add_special_tokens=False)
                count = len(token_ids)
                records.append({
                    "item_id": item_id,
                    "language_code": code,
                    "language": LANGUAGES[code],
                    "model_id": model_id,
                    "word_count_whitespace": words,
                    "character_count_nfc": chars,
                    "token_count_no_special_tokens": count,
                    "tokens_per_word": count / words if words else "",
                    "characters_per_token": chars / count if count else "",
                })
    return records


def summarize(records):
    grouped = defaultdict(list)
    for row in records:
        grouped[(row["language_code"], row["language"], row["model_id"])].append(float(row["tokens_per_word"]))
    summaries = []
    for (code, language, model_id), values in sorted(grouped.items()):
        summaries.append({
            "language_code": code,
            "language": language,
            "model_id": model_id,
            "n_items": len(values),
            "mean_tokens_per_word": statistics.mean(values),
            "median_tokens_per_word": statistics.median(values),
            "stdev_tokens_per_word": statistics.stdev(values) if len(values) > 1 else 0.0,
        })
    return summaries


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"No rows to write to {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    if args.limit < 1:
        raise SystemExit("--limit must be at least 1")

    RESULTS.mkdir(parents=True, exist_ok=True)
    aligned_rows, audit = load_aligned_rows(args.dataset_revision, args.limit)
    tokenizers = get_tokenizers(args.qwen_revision, args.llama_revision)
    records = measure(aligned_rows, tokenizers)
    summaries = summarize(records)

    write_csv(RESULTS / "token_counts.csv", records)
    write_csv(RESULTS / "summary.csv", summaries)
    write_csv(RESULTS / "alignment_audit.csv", audit)
    manifest = {
        "run_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": "openlanguagedata/flores_plus",
        "split": "dev",
        "dataset_revision_requested": args.dataset_revision,
        "languages": LANGUAGES,
        "requested_limit": args.limit,
        "aligned_items_processed": len(aligned_rows),
        "incomplete_or_blank_ids": sum(not row["complete_triplet"] for row in audit),
        "item_ids": [item_id for item_id, _ in aligned_rows],
        "models": {
            "Qwen/Qwen3-8B": args.qwen_revision,
            "meta-llama/Llama-3.1-8B-Instruct": args.llama_revision,
        },
        "normalization": "Unicode NFC",
        "word_count": "Python whitespace split",
        "special_tokens": "excluded",
        "software_versions": {
            "python": platform.python_version(),
            "datasets": importlib.metadata.version("datasets"),
            "transformers": importlib.metadata.version("transformers"),
            "tokenizers": importlib.metadata.version("tokenizers"),
            "huggingface_hub": importlib.metadata.version("huggingface_hub"),
        },
        "outputs": ["token_counts.csv", "summary.csv", "alignment_audit.csv"],
        "note": "Requested refs are recorded; pin immutable commits for final reproducibility. No source sentence text is written to outputs.",
    }
    (RESULTS / "run_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Processed {len(aligned_rows)} aligned IDs; wrote count tables and manifest under {RESULTS}.")


if __name__ == "__main__":
    main()
