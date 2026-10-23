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

6. Inspect `results/token_counts.csv`, `results/summary.csv`, `results/alignment_audit.csv`, and `results/run_manifest.json`. Check the audit for missing or blank rows.

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
