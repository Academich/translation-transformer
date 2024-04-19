from pathlib import Path

from data_wrappers import Seq2SeqDM
from synthetic_tasks.copy_sequence.tokenizer import AsciiTokenizer


class CopySequenceDM(Seq2SeqDM):

    def __init__(self, vocab_path: str | None = None, *args, **kwargs):
        self.vocab_path = vocab_path
        super().__init__(*args, **kwargs)

    def create_tokenizers(self):
        if self.vocab_path is None:
            self.vocab_path = self.data_dir / "vocabs" / "vocab.json"
        else:
            self.vocab_path = Path(self.vocab_path).resolve()

        tokenizer = AsciiTokenizer()
        try:
            tokenizer.load_vocab(self.vocab_path)
            print(f"Loaded tokenizer vocabulary from {self.vocab_path}")
        except FileNotFoundError:
            print("Training tokenizer...")
            with open(self.src_train_path) as f:
                tokenizer.train_tokenizer(f)
            tokenizer.save_vocab(self.vocab_path)
            print(f"Saved tokenizer vocab to: {self.vocab_path}")
        finally:
            return tokenizer, tokenizer
