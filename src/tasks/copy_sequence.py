from typing import Union
from pathlib import Path
import json
from string import ascii_lowercase, ascii_uppercase
from typing import Tuple, Dict, Iterable

import numpy as np
import torch


class AsciiTokenizer:
    chars = list(ascii_lowercase) + list(ascii_uppercase)

    def __init__(self, bos_token: str = "<BOS>", eos_token: str = "<EOS>", pad_token: str = "<PAD>"):
        self.bos_token = bos_token
        self.eos_token = eos_token
        self.pad_token = pad_token
        self.encoder_dict, self.decoder_dict = self._make_dictionaries()
        self.n_tokens = len(self.encoder_dict)

    def _make_dictionaries(self) -> Tuple[Dict[str, int], Dict[int, str]]:
        encoder_dict = {ch: i + 3 for i, ch in enumerate(self.chars)}
        encoder_dict[self.pad_token] = 0
        encoder_dict[self.bos_token] = 1
        encoder_dict[self.eos_token] = 2

        decoder_dict = {v: k for k, v in encoder_dict.items()}
        return encoder_dict, decoder_dict

    def save_vocabulary(self, voc_save_path: str) -> None:
        with open(voc_save_path, "w") as f:
            json.dump(self.decoder_dict, f)

    def encode(self, seq: str):
        token_ids = [self.encoder_dict[i] for i in [self.bos_token] + list(seq) + [self.eos_token]]
        return torch.tensor(token_ids).long()

    def decode(self, tokens: torch.LongTensor):
        service_tokens = (self.bos_token_idx, self.eos_token_idx, self.pad_token_idx)
        decoded_chars = [self.decoder_dict[i] for i in tokens.numpy() if i not in service_tokens]
        decoded_string = "".join(decoded_chars)
        return decoded_string

    @property
    def bos_token_idx(self):
        return self.encoder_dict[self.bos_token]

    @property
    def eos_token_idx(self):
        return self.encoder_dict[self.eos_token]

    @property
    def pad_token_idx(self):
        return self.encoder_dict[self.pad_token]


class CopySequenceDataSampler:
    """
    Class that generates sequences of lowercase and uppercase ASCII characters.
    """
    chars = list(ascii_lowercase) + list(ascii_uppercase)

    def __init__(self, min_length: int, max_length: int):
        self.min_length = min_length
        self.max_length = max_length

    def sample(self):
        src_sequence = "".join(np.random.choice(self.chars, np.random.randint(self.min_length,
                                                                              self.max_length + 1)))
        tgt_sequence = src_sequence
        return src_sequence, tgt_sequence


def _save_subset(name: str,
                 data_strings: Iterable[str],
                 save_directory: Path,
                 sort_by_length: bool,
                 tokenizer: AsciiTokenizer,
                 save_csv: bool = True) -> None:
    """
    Generates a collection of sequences for the copy-sequence task, tokenizes and saves them.
    :param sort_by_length:
    :param name: Name of the subset. Expected to be 'train', 'val' or 'test'.
    :param data_strings: Generated strings
    :param save_directory: Path to the directory where files will be saved.
    :param tokenizer: The tokenized class instance.
    :param save_csv: A flag whether to save a .csv file with sequences for visual assessment alongside the .pt file.
    :return:
    """
    # Tokens
    if sort_by_length:
        data_strings = sorted([i for i in data_strings], key=len)

    domain_tokens, target_tokens = [], []
    for s in data_strings:
        src, tgt = s.split(",")
        domain_tokens.append(tokenizer.encode(src))
        target_tokens.append(tokenizer.encode(tgt))

    if save_csv:
        with open(save_directory / f"{name}.csv", "w") as f:
            f.write("\n".join(data_strings))

    with open(save_directory / f"src-{name}-tokens.pt", "wb") as f:
        torch.save(domain_tokens, f)
    with open(save_directory / f"tgt-{name}-tokens.pt", "wb") as f:
        torch.save(target_tokens, f)


class CopySequenceGenerator:

    def __init__(self,
                 data_dir: Union[Path, str],
                 train_size: int = 40_000,
                 train_min_len: int = 1,
                 train_max_len: int = 20,
                 val_size: int = 100,
                 val_min_len: int = 1,
                 val_max_len: int = 20,
                 test_size: int = 1000,
                 test_min_len: int = 1,
                 test_max_len: int = 20,
                 seed: int = 123456):
        self.__tokenizer = AsciiTokenizer()

        self.data_dir = Path(data_dir).resolve()
        self.vocab_dir = self.data_dir / "vocabs"
        self.vocab_path = self.vocab_dir / "ascii_vocab.json"
        self.seed = seed
        self.train_size = train_size
        self.train_min_len, self.train_max_len = train_min_len, train_max_len
        self.val_size = val_size
        self.val_min_len, self.val_max_len = val_min_len, val_max_len
        self.test_size = test_size
        self.test_min_len, self.test_max_len = test_min_len, test_max_len

    def generate(self) -> None:
        try:
            self.vocab_dir.mkdir(parents=True, exist_ok=False)
        except FileExistsError:
            return

        np.random.seed(self.seed)
        self.__tokenizer.save_vocabulary(self.vocab_path)
        print("Generating data...")

        data_generator_train = CopySequenceDataSampler(min_length=self.train_min_len,
                                                       max_length=self.train_max_len)
        data_generator_val = CopySequenceDataSampler(min_length=self.val_min_len,
                                                     max_length=self.val_max_len)
        data_generator_test = CopySequenceDataSampler(min_length=self.test_min_len,
                                                      max_length=self.test_max_len)
        train_strings = list({",".join(data_generator_train.sample()) for _ in range(self.train_size)})
        val_strings = list({",".join(data_generator_val.sample()) for _ in range(self.val_size)})
        test_strings = list({",".join(data_generator_test.sample()) for _ in range(self.test_size)})

        print("Saving train data...")
        _save_subset("train",
                     train_strings,
                     self.data_dir,
                     sort_by_length=False,
                     tokenizer=self.__tokenizer)

        print("Saving validation data...")
        _save_subset("val",
                     val_strings,
                     self.data_dir,
                     sort_by_length=True,
                     tokenizer=self.__tokenizer)

        print("Saving test data...")
        _save_subset("test",
                     test_strings,
                     self.data_dir,
                     sort_by_length=True,
                     tokenizer=self.__tokenizer)
