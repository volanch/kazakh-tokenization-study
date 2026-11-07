# Kazakh Tokenization Efficiency Study

Reproducible course project comparing token counts for aligned Kazakh, English, and Russian text with Qwen3-8B and Llama 3.1 8B Instruct tokenizers.

## Dataset handling

This repository does not contain FLORES+ sentence text. The script loads the official `openlanguagedata/flores_plus` dataset after you accept its access conditions on Hugging Face. The dataset card identifies the license as CC BY-SA 4.0 and says not to redistribute exported dataset files publicly unless protected from automatic scraping.

## Step-by-step setup

1. Create or sign in to a Hugging Face account. Open the [FLORES+ dataset card](https://huggingface.co/datasets/openlanguagedata/flores_plus), accept its access conditions, and read its license and citation guidance.
2. Install Python 3.10 or newer. Create and activate a virtual environment in this folder. Only tokenizers are loaded for this practical component, so the 8B model weights and a GPU are not required.
3. Install dependencies:

   ```powershell
   python -m pip install -r requirements.txt
   ```

4. Authenticate to Hugging Face without putting a token in this repository:

   ```powershell
   hf auth login
   ```

5. Run the full baseline comparison over all 997 FLORES+ dev sentence IDs with the documented input rates:

   ```powershell
   python src/run_tokenization.py --limit 997 --qwen-input-price-per-million 0.07 --llama-input-price-per-million 0.06 --currency USD --pricing-provider Nscale
   ```

6. Inspect `results/token_counts.csv`, `results/summary.csv`, `results/cost_estimates.csv`, `results/alignment_audit.csv`, and `results/run_manifest.json`. Check the audit for missing or blank rows.

7. Keep raw text and local dataset/model caches out of GitHub. Commit only code, documentation, the run manifest, and derived count tables, following the dataset terms and course sharing rules.

## Models

- Primary tokenizer/model: `Qwen/Qwen3-8B`.
- Comparison tokenizer/model: `meta-llama/Llama-3.1-8B-Instruct`.

The script loads tokenizers only, so you do not need to load 8B model weights for token counts. Llama access require accepting Meta's model terms. Meta's Llama 3.1 model card lists eight supported languages besides English and does not include Kazakh. Treat it as a tokenizer comparison and state this limitation; we do not describe it as an equally Kazakh-capable baseline.

## Measures

- `tokens_per_word`: tokens divided by whitespace-delimited words.
- `chars_per_token`: normalized Unicode code points divided by tokens.
- `token_count`: content tokens with special tokens disabled.

Texts are NFC-normalized before counting. Chat templates are not applied to benchmark sentences. These measures describe tokenizer output; they do not measure answer quality.

## Sources

- [FLORES+ dataset card](https://huggingface.co/datasets/openlanguagedata/flores_plus)
- [Qwen3-8B model card](https://huggingface.co/Qwen/Qwen3-8B)
- [Meta Llama 3.1 8B Instruct model card](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct)

## Estimated processing cost

The script estimates input cost by multiplying raw sentence token counts by input rates you supply for your chosen inference provider. It does not call an inference provider or measure LLM performance. Use rates from the provider and model version you would actually use.

For the input rates recorded in the project methodology on 25 October 2026, run:

    python src/run_tokenization.py --limit 997 --qwen-input-price-per-million 0.07 --llama-input-price-per-million 0.06 --currency USD --pricing-provider Nscale

If a rate is omitted, its estimated cost is left blank in results/cost_estimates.csv. The estimate covers raw FLORES+ sentence text only. It excludes system/user prompt text, chat-template tokens, generated output, and provider-specific overhead, so it is not an actual API bill. The source text remains local under data/used_texts.csv; do not commit it to GitHub.


## Custom Kazakh tokenizer comparison

The additional SentencePiece Unigram tokenizer was trained with a 2,048-piece vocabulary on the locally obtained Kazakh CC-100 corpus. The corpus and its extracted text are not stored in this repository. Place the extracted file at `data_tokenizer/kk.txt/kk.txt` before retraining. Check CC-100 access and licensing terms before use or redistribution; never commit the corpus.

From the project root in PowerShell, train the tokenizer and compare its counts with Qwen on the Kazakh sentences in the local `data/used_texts.csv`:

```powershell
& "$venv\Scripts\python.exe" src\train_sentencepiece.py
& "$venv\Scripts\python.exe" src\compare_custom_tokenizer.py
```

The comparison writes `results/custom_qwen_comparison.csv` and `results/custom_qwen_summary.csv`. Both tokenizers process the same NFC-normalized Kazakh text without special tokens. The SentencePiece tokenizer is an independent tokenizer; it is not plugged into Qwen, so its lower count is not an observed reduction in Qwen API usage or an actual provider bill. Keep `data/used_texts.csv`, the CC-100 corpus, and any restricted source text out of public Git history.




