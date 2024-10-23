import re
from typing import Iterable
from collections import Counter
from itertools import chain

from data_handling.tokenizer_base import GenericTokenizer, reverse_dict

REGEX = "(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"


class SimpleSmilesTokenizer:
    pattern = re.compile(REGEX)

    @classmethod
    def split_into_tokens(cls, smi: str, check_reconstruction=False) -> list[str]:
        tokens = [token for token in cls.pattern.findall(smi)]
        if check_reconstruction:
            assert smi == ''.join(tokens)
        return tokens


class ChemSMILESTokenizer(GenericTokenizer):

    def train_tokenizer(self, train_data: Iterable[str]) -> None:
        tokenized_data = [SimpleSmilesTokenizer.split_into_tokens(line.strip(),
                                                                  check_reconstruction=True) for line in train_data]
        token_counts = Counter(chain(*tokenized_data))

        for token, _ in token_counts.most_common():
            self.encoder_dict[token] = len(self.encoder_dict)

        self.decoder_dict = reverse_dict(self.encoder_dict)

    def encode(self, seq: str) -> list[int]:
        unk = self.encoder_dict[self.unk_token]
        smi_tokens = SimpleSmilesTokenizer.split_into_tokens(seq, check_reconstruction=False)
        token_ids = [self.encoder_dict[i] if i in self.encoder_dict else unk for i in smi_tokens]
        token_ids = [self.bos_token_idx, *token_ids, self.eos_token_idx]
        return token_ids
