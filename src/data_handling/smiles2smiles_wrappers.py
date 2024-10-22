from pathlib import Path
from itertools import chain

from data_handling.seq2seq_wrappers import Seq2SeqDM
from data_handling.tokenizer_smiles import ChemSMILESTokenizer


class ReactionPredictionDM(Seq2SeqDM):


    def create_tokenizers(self, vocab_path: str | None) -> tuple[ChemSMILESTokenizer, ChemSMILESTokenizer]:
        if vocab_path is None:
            vocab_path = self.data_dir / "vocabs" / "vocab.json"
        else:
            vocab_path = Path(vocab_path).resolve()

        tokenizer = ChemSMILESTokenizer()
        try:
            tokenizer.load_vocab(vocab_path)
            print(f"Loaded tokenizer vocabulary from {self.vocab_path}")
        except FileNotFoundError:
            print("Training tokenizer...")
            with open(self.src_train_path) as f, open(self.tgt_train_path) as g:
                tokenizer.train_tokenizer(chain(f, g))
            tokenizer.save_vocab(vocab_path)
            print(f"Saved tokenizer vocab to: {vocab_path}")
        finally:
            return tokenizer, tokenizer
