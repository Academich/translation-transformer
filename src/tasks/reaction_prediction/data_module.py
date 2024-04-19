from pathlib import Path
from itertools import chain

from data_wrappers import Seq2SeqDM
from tasks.reaction_prediction.tokenizer import ChemSMILESTokenizer


class ReactionPredictionDM(Seq2SeqDM):

    def __init__(self, vocab_path=None, *args, **kwargs):
        self.vocab_path = vocab_path
        super().__init__(*args, **kwargs)

    def create_tokenizers(self):
        tokenizer = ChemSMILESTokenizer()
        if self.vocab_path is not None:
            print(f"Loading tokenizer vocabulary from {self.vocab_path}...")
            tokenizer.load_vocab(self.vocab_path)
            return tokenizer, tokenizer

        data_dir = Path(self.data_dir).resolve()
        save_vocab_dir = data_dir / "vocabs"
        save_vocab_dir.mkdir(parents=True, exist_ok=True)
        print("Training tokenizer...")
        with open(data_dir / "src-train.txt") as f, open(data_dir / "tgt-train.txt") as g:
            tokenizer.train_tokenizer(chain(f, g))
        tokenizer.save_vocab(save_vocab_dir / "vocab.json")
        return tokenizer, tokenizer
