import json
from pathlib import Path
from typing import Iterable
from string import ascii_lowercase, ascii_uppercase

BOS_TOKEN = "<BOS>"
EOS_TOKEN = "<EOS>"
PAD_TOKEN = "<PAD>"
UNK_TOKEN = "?"


def reverse_dict(d: dict[int | str, str | int]) -> dict[str | int, int | str]:
    return {v: k for k, v in d.items()}


class AsciiTokenizer:
    chars = list(ascii_lowercase) + list(ascii_uppercase)

    def __init__(self,
                 bos_token: str = BOS_TOKEN,
                 eos_token: str = EOS_TOKEN,
                 pad_token: str = PAD_TOKEN,
                 unk_token: str = UNK_TOKEN):
        self.bos_token = bos_token
        self.eos_token = eos_token
        self.pad_token = pad_token
        self.unk_token = unk_token

        self.pad_token_idx = 0
        self.bos_token_idx = 1
        self.eos_token_idx = 2
        self.unk_token_idx = 3

        self.encoder_dict, self.decoder_dict = self._make_dictionaries()

    @property
    def n_tokens(self):
        return len(self.encoder_dict)

    def _make_dictionaries(self) -> tuple[dict[str, int], dict[int, str]]:
        encoder_dict = {self.pad_token: self.pad_token_idx,
                        self.bos_token: self.bos_token_idx,
                        self.eos_token: self.eos_token_idx,
                        self.unk_token: self.unk_token_idx}
        decoder_dict = reverse_dict(encoder_dict)
        return encoder_dict, decoder_dict

    def train_tokenizer(self, train_data_path):
        with open(train_data_path) as f:
            train_data = [line.strip() for line in f.readlines()]

        for line in train_data:
            for s in line:
                if s not in self.encoder_dict:
                    self.encoder_dict[s] = len(self.encoder_dict)
        self.decoder_dict = reverse_dict(self.encoder_dict)

    def save_vocab(self, voc_save_path: Path | str) -> None:
        with open(voc_save_path, "w") as f:
            json.dump(self.decoder_dict, f)

    def load_vocab(self, voc_load_path: str) -> None:
        """
        Loads vocabulary dictionary which maps indices to strings
        """
        with open(voc_load_path) as f:
            self.decoder_dict = {int(k): v for k, v in json.load(f).items()}
            self.encoder_dict = reverse_dict(self.decoder_dict)

    def encode(self, seq: str) -> list[int]:
        unk = self.encoder_dict[self.unk_token]
        token_ids = [self.encoder_dict[i] if i in self.encoder_dict else unk for i in seq]
        token_ids = [self.bos_token_idx, *token_ids, self.eos_token_idx]
        return token_ids

    def decode(self, tokens: Iterable[int], skip_service_tokens: bool = True) -> str:
        service_tokens = (self.bos_token_idx, self.eos_token_idx, self.pad_token_idx)
        if not skip_service_tokens:
            return "".join([self.decoder_dict[i] for i in tokens])
        decoded_chars = []
        for i in tokens:
            if i not in service_tokens:
                decoded_chars.append(self.decoder_dict[i])
            if i == self.eos_token_idx:
                break
        decoded_string = "".join(decoded_chars)
        return decoded_string

    def decode_batch(self, tokens: list[Iterable[int]]) -> list[str]:
        return [self.decode(i) for i in tokens]
