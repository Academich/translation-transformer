from typing import Iterable
from tokenization import GenericTokenizer, reverse_dict


class AsciiTokenizer(GenericTokenizer):
    """
    A simple tokenizer that treats every character in a string as a token
    """

    def train_tokenizer(self, train_data: Iterable[str]):
        for line in train_data:
            _line = line.strip()
            for s in _line:
                if s not in self.encoder_dict:
                    self.encoder_dict[s] = len(self.encoder_dict)
        self.decoder_dict = reverse_dict(self.encoder_dict)

    def encode(self, seq: str) -> list[int]:
        unk = self.encoder_dict[self.unk_token]
        token_ids = [self.encoder_dict[i] if i in self.encoder_dict else unk for i in seq]
        token_ids = [self.bos_token_idx, *token_ids, self.eos_token_idx]
        return token_ids
