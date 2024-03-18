from argparse import ArgumentParser
from pathlib import Path
from string import ascii_lowercase, ascii_uppercase
from typing import Iterable

import numpy as np


class CopySequenceDataSampler:
    """
    Class that generates sequences of lowercase and uppercase ASCII characters.
    """
    chars = list(ascii_lowercase) + list(ascii_uppercase)

    def __init__(self, min_length: int, max_length: int, seed=1234):
        self.min_length = min_length
        self.max_length = max_length
        self.rng = np.random.RandomState(seed)  # TODO Still not reproducible

    def sample(self):
        src_sequence = "".join(self.rng.choice(self.chars, self.rng.randint(self.min_length,
                                                                            self.max_length + 1)))
        tgt_sequence = src_sequence
        return src_sequence, tgt_sequence


class CopySequenceGenerator:

    def __init__(self,
                 data_dir: Path | str,
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

        self.data_dir = Path(data_dir).resolve()
        self.seed = seed
        self.train_size = train_size
        self.train_min_len, self.train_max_len = train_min_len, train_max_len
        self.val_size = val_size
        self.val_min_len, self.val_max_len = val_min_len, val_max_len
        self.test_size = test_size
        self.test_min_len, self.test_max_len = test_min_len, test_max_len

    def _save_subset(self,
                     subset: str,
                     src_data_strings: Iterable[str],
                     tgt_data_strings: Iterable[str],
                     sort_by_length: bool) -> None:
        """
        Generates a collection of sequences for the copy-sequence task, tokenizes and saves them.
        """
        # Tokens
        if sort_by_length:
            data_strings = [(s, t) for s, t in zip(src_data_strings, tgt_data_strings)]
            data_strings = sorted([ts for ts in data_strings], key=lambda x: len(x[0]))
            src_data_strings, tgt_data_strings = zip(*data_strings)

        with open(self.data_dir / f"src-{subset}.txt", "w") as fs, open(self.data_dir / f"tgt-{subset}.txt", "w") as ft:
            fs.write("\n".join(src_data_strings))
            ft.write("\n".join(tgt_data_strings))

    def generate(self) -> None:

        print("Generating data...")

        data_generator_train = CopySequenceDataSampler(min_length=self.train_min_len,
                                                       max_length=self.train_max_len)
        data_generator_val = CopySequenceDataSampler(min_length=self.val_min_len,
                                                     max_length=self.val_max_len)
        data_generator_test = CopySequenceDataSampler(min_length=self.test_min_len,
                                                      max_length=self.test_max_len)
        src_train_strings, tgt_train_strings = zip(*{data_generator_train.sample() for _ in range(self.train_size)})
        src_val_strings, tgt_val_strings = zip(*{data_generator_val.sample() for _ in range(self.val_size)})
        src_test_strings, tgt_test_strings = zip(*{data_generator_test.sample() for _ in range(self.test_size)})

        print("Saving train data...")
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self._save_subset("train",
                          src_train_strings,
                          tgt_train_strings,
                          sort_by_length=False)

        print("Saving validation data...")
        self._save_subset("val",
                          src_val_strings,
                          tgt_val_strings,
                          sort_by_length=True)

        print("Saving test data...")
        self._save_subset("test",
                          src_test_strings,
                          tgt_test_strings,
                          sort_by_length=True)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Path to the directory to store generated copy-sequence data to", )
    parser.add_argument("--train_size", type=int, default=40_000, help="Size of the training dataset")
    parser.add_argument("--train_min_len", type=int, default=1, help="Minimum length of training sequences")
    parser.add_argument("--train_max_len", type=int, default=20, help="Maximum length of training sequences")
    parser.add_argument("--val_size", type=int, default=100, help="Size of the validation dataset")
    parser.add_argument("--val_min_len", type=int, default=1, help="Minimum length of validation sequences")
    parser.add_argument("--val_max_len", type=int, default=20, help="Maximum length of validation sequences")
    parser.add_argument("--test_size", type=int, default=1000, help="Size of the test dataset")
    parser.add_argument("--test_min_len", type=int, default=1, help="Minimum length of test sequences")
    parser.add_argument("--test_max_len", type=int, default=20, help="Maximum length of test sequences")
    parser.add_argument("--seed", type=int, default=123456, help="Seed for random number generation")
    args = parser.parse_args()

    data_generator = CopySequenceGenerator(args.data_dir,
                                           args.train_size, args.train_min_len, args.train_max_len,
                                           args.val_size, args.val_min_len, args.val_max_len,
                                           args.test_size, args.test_min_len, args.test_max_len,
                                           args.seed)
    data_generator.generate()
