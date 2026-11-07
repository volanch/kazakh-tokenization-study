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

5. Run the pilot over 20 aligned sentence IDs:

   ```powershell
   python src/run_tokenization.py --limit 20
   ```

6. Inspect `results/token_counts.csv`, `results/summary.csv`, `results/cost_estimates.csv`, `results/alignment_audit.csv`, and `results/run_manifest.json`. Check the audit for missing or blank rows.

7. If the pilot works, you can make higher limit:

   ```powershell
   python src/run_tokenization.py --limit 200
   ```

8. Added only the code, documentation, manifest, and derived count tables to the project repo after checking the current dataset terms and course sharing rules. Kept raw text and local dataset/model caches out of Git.

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

The script estimates input cost by multiplying raw sentence token counts by input rates you supply for your chosen inference provider. It does not run either LLM or measure translation quality. Use rates from the provider and model version you would actually use.

Example command (replace 0.XX and 0.YY with actual provider rates per million input tokens):

    python src/run_tokenization.py --limit 200 --qwen-input-price-per-million 0.XX --llama-input-price-per-million 0.YY --currency USD --pricing-provider "Provider name"

If a rate is omitted, its estimated cost is left blank in results/cost_estimates.csv. The estimate covers raw FLORES+ sentence text only. It excludes system/user prompt text, chat-template tokens, generated output, and provider-specific overhead, so it is not an actual API bill. The source text remains local under data/used_texts.csv; do not commit it to GitHub.


## Kazakh-to-English translation and token-count pilot

`src/translate_kazakh_google.py` uses Google Cloud Translation Basic NMT (`kk` to `en`) for the Kazakh sentences in local `data/used_texts.csv`, saves the translation results under `data_translate/`, then counts translated English tokens with the Qwen3-8B and Llama 3.1 8B Instruct tokenizers. It also counts the existing aligned FLORES+ English reference text for comparison. This measures English translation output token counts, not Kazakh tokenization efficiency. It estimates translation charges from input characters and an assumed remaining monthly free character allowance; verify actual project usage and current billing.

Enable Cloud Translation in a Google Cloud project, create and restrict an API key to the Translation API, and keep it out of the repository. In the current PowerShell window, set it without writing it into source code:

```powershell
$env:GOOGLE_TRANSLATE_API_KEY = Read-Host "Paste Google API key"
& "$venv\Scripts\python.exe" src\translate_kazakh_google.py --limit 20 --qwen-input-price-per-million 0.07 --llama-input-price-per-million 0.06
```

The default translation estimate assumes 500,000 free NMT characters remain for the current month and uses $20 per million characters above that allowance. Change `--free-characters-remaining` to reflect other usage on the Google Cloud project. If the output already exists, use `--resume` to skip completed IDs or `--overwrite` to translate again. The output CSV contains restricted dataset text and stays in the ignored `data_translate/` folder; do not publish it.



## Custom Kazakh tokenizer comparison

The additional SentencePiece Unigram tokenizer was trained with a 2,048-piece vocabulary on the locally obtained Kazakh CC-100 corpus. The corpus and its extracted text are not stored in this repository. Place the extracted file at `data_tokenizer/kk.txt/kk.txt` before retraining. Check CC-100 access and licensing terms before use or redistribution; never commit the corpus.

From the project root in PowerShell, train the tokenizer and compare its counts with Qwen on the Kazakh sentences in the local `data/used_texts.csv`:

```powershell
& "$venv\Scripts\python.exe" src\train_sentencepiece.py
& "$venv\Scripts\python.exe" src\compare_custom_tokenizer.py
```

The comparison writes `results/custom_qwen_comparison.csv` and `results/custom_qwen_summary.csv`. Both tokenizers process the same NFC-normalized Kazakh text without special tokens. The SentencePiece tokenizer is an independent tokenizer; it is not plugged into Qwen, so its lower count is not an observed reduction in Qwen API usage or an actual provider bill. Keep `data/used_texts.csv`, the CC-100 corpus, and any restricted source text out of public Git history.
