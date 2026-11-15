# Kazakh Tokenization Efficiency Study

A reproducible course project comparing token counts for aligned Kazakh, English, and Russian text. The experiment uses the Qwen3-8B and Llama 3.1 8B Instruct tokenizers, and compares Kazakh Qwen counts with a separately trained SentencePiece Unigram tokenizer.

The instructions below reproduce the CSV files in `results/`. They do not generate figures. The tokenizer-only experiment does not load 8B model weights, call an inference provider, or measure answer quality.

## What is measured

- `token_count`: content tokens, with special tokens disabled.
- `tokens_per_word`: token count divided by whitespace-delimited word count.
- `chars_per_token`: NFC-normalized Unicode character count divided by token count.
- `estimated_input_cost`: raw sentence token count multiplied by the input price supplied at run time. This is an estimate, not an API invoice.

FLORES+ sentences are NFC-normalized before counting. Chat templates, prompts, generated output, and provider-specific billing rules are not included. The Llama tokenizer is a comparison baseline; its results should not be interpreted as evidence of equal Kazakh language capability.

## Requirements and dataset access

- Python 3.10 or newer.
- Hugging Face account and access to the [FLORES+ dataset](https://huggingface.co/datasets/openlanguagedata/flores_plus) and gated [Llama 3.1 8B Instruct model](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct).
- Local access to the Kazakh CC-100 text if retraining the custom tokenizer.

Accept the relevant access conditions on Hugging Face. From PowerShell, at the repository root, create an environment, install dependencies, and log in:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
hf auth login
```

The scripts load tokenizer files only; an 8B model download and GPU are not required for token counting.

## Reproduce the baseline CSV files

Run all 997 aligned sentence IDs in the FLORES+ `dev` split for English (`eng_Latn`), Kazakh (`kaz_Cyrl`), and Russian (`rus_Cyrl`):

```powershell
python src/run_tokenization.py --limit 997 --dataset-revision main --qwen-revision main --llama-revision main --qwen-input-price-per-million 0.07 --llama-input-price-per-million 0.06 --currency USD --pricing-provider Nscale
```

This creates:

- `results/token_counts.csv`
- `results/summary.csv`
- `results/cost_estimates.csv`
- `results/alignment_audit.csv`
- `results/run_manifest.json` (run metadata; not a CSV)
- `data/used_texts.csv` (exact selected source sentences; local only, do not upload)

The input prices above are the historical rates recorded for this project on 25 October 2026. Cost estimates cover raw sentence text only. The revisions are set to `main`, which can change; pin immutable dataset and tokenizer commit IDs for strict reproducibility.

## Reproduce the custom Kazakh tokenizer CSV files

The custom model is a SentencePiece Unigram tokenizer with a requested vocabulary size of 2,048. The model file `data_tokenizer/models/kk_unigram_2048.model` must be present. If it is already present, skip training and run the comparison:

```powershell
python src/compare_custom_tokenizer.py
```

To train the model again, obtain the Kazakh CC-100 text separately and place the extracted file at `data_tokenizer/kk.txt/kk.txt`. Then run:

```powershell
python src/train_sentencepiece.py
python src/compare_custom_tokenizer.py
```

The comparison uses local `data/used_texts.csv` from the baseline run and creates:

- `results/custom_qwen_comparison.csv`
- `results/custom_qwen_summary.csv`

SentencePiece is an independent tokenizer; it is not integrated into Qwen or Llama. Its token counts do not demonstrate an actual reduction in Qwen API usage or provider costs.

## Create the per-tokenizer count CSV files

After running the baseline and custom comparison, use these PowerShell commands to derive the compact Qwen and SentencePiece CSVs from their per-item source results:

```powershell
Import-Csv .\results\token_counts.csv |
    Where-Object { $_.model_id -eq 'Qwen/Qwen3-8B' } |
    Select-Object item_id, model_id, token_count_no_special_tokens |
    Export-Csv .\results\token_counts_qwen.csv -NoTypeInformation -Encoding utf8

Import-Csv .\results\custom_qwen_comparison.csv |
    ForEach-Object {
        [PSCustomObject]@{
            item_id = $_.item_id
            model_id = 'SentencePiece-kk-unigram-2048'
            token_count_no_special_tokens = $_.custom_sentencepiece_tokens
        }
    } |
    Export-Csv .\results\token_counts_custom.csv -NoTypeInformation -Encoding utf8
```

## CSV files in `results/`

After completing the steps above, the eight experiment CSV files are:

| File | Contents |
| --- | --- |
| `alignment_audit.csv` | Presence and alignment checks for each English–Kazakh–Russian item ID. |
| `cost_estimates.csv` | Estimated raw-text input cost by language and model. |
| `custom_qwen_comparison.csv` | Per-item Qwen and SentencePiece Kazakh counts and percentage change. |
| `custom_qwen_summary.csv` | Summary statistics for the Kazakh Qwen–SentencePiece comparison. |
| `summary.csv` | Mean, median, and standard deviation of tokens per word by language and model. |
| `token_counts.csv` | Per-item Qwen and Llama counts plus text-length measures. |
| `token_counts_custom.csv` | Per-item SentencePiece counts in a compact three-column format. |
| `token_counts_qwen.csv` | Per-item Qwen counts in a compact three-column format. |

Check the CSVs and alignment audit with:

```powershell
Get-ChildItem .\results -Filter '*.csv'
Import-Csv .\results\alignment_audit.csv | Where-Object { $_.complete_triplet -ne 'True' }
```

The audit command should return no rows when all selected items have complete triplets.

## Data and repository hygiene

Do not upload `data/used_texts.csv`, FLORES+ or CC-100 source text, local model caches, credentials, or Hugging Face tokens. Follow the FLORES+ access conditions and course sharing rules when handling its source text.

## Sources

- [FLORES+ dataset card](https://huggingface.co/datasets/openlanguagedata/flores_plus)
- [Qwen3-8B model card](https://huggingface.co/Qwen/Qwen3-8B)
- [Meta Llama 3.1 8B Instruct model card](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct)
- [CC-100 dataset card](https://huggingface.co/datasets/statmt/cc100)
- [CC-100 original documentation](https://data.statmt.org/cc-100/)
