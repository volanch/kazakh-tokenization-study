import sentencepiece as spm
from pathlib import Path

root = Path(__file__).resolve().parents[1]
train_file = root / "data_tokenizer" / "kk.txt" / "kk.txt"
model_dir = root / "data_tokenizer" / "models"
model_dir.mkdir(exist_ok=True)

spm.SentencePieceTrainer.train(
    input=str(train_file),
    model_prefix=str(model_dir / "kk_unigram_2048"),
    model_type="unigram",
    vocab_size=2048,
    character_coverage=1.0,
    hard_vocab_limit=False,
)