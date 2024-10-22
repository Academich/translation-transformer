from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

from pytorch_lightning import LightningDataModule

from data_handling.batching import TokenSampler
from data_handling.tokenizer_base import GenericTokenizer


class Seq2SeqDataset(Dataset):

    def __init__(self,
                 src_path: Path | str,
                 tgt_path: Path | str,
                 src_tokenizer: GenericTokenizer,
                 tgt_tokenizer: GenericTokenizer):
        self.src_tokenizer = src_tokenizer
        self.tgt_tokenizer = tgt_tokenizer
        with open(src_path) as fs, open(tgt_path) as ft:
            self.source: list[str] = [s.strip() for s in fs.readlines()]
            self.target: list[str] = [s.strip() for s in ft.readlines()]
        assert len(self.source) == len(
            self.target), f"The source and target data at {src_path} and {tgt_path} have different lenghts"
        self.source_tokens = [self.src_tokenizer.encode(s) for s in self.source]
        self.target_tokens = [self.tgt_tokenizer.encode(s) for s in self.target]

        self.source_lenghts = [len(i) for i in self.source_tokens]
        self.target_lenghts = [len(i) for i in self.target_tokens]

    def __len__(self):
        return len(self.source_tokens)

    def __getitem__(self, item):
        return torch.tensor(self.source_tokens[item]).long(), torch.tensor(self.target_tokens[item]).long()


class Seq2SeqDM(LightningDataModule):
    def __init__(self,
                 data_dir: str | None = None,  # Data location arguments
                 src_train_path: str | None = None,
                 tgt_train_path: str | None = None,
                 src_val_path: str | None = None,
                 tgt_val_path: str | None = None,
                 src_test_path: str | None = None,
                 tgt_test_path: str | None = None,
                 vocab_path: str | None = None,

                 batch_size: int = 1,  # Batching arguments
                 tokens_in_batch: int | None = None,
                 num_workers: int = 0,
                 persistent_workers=False,
                 pin_memory: bool = False,
                 shuffle_train: bool = False):
        super().__init__()

        self.data_dir = Path(data_dir).resolve()
        self.src_train_path = src_train_path or self.data_dir / "src-train.txt"
        self.tgt_train_path = tgt_train_path or self.data_dir / "tgt-train.txt"
        self.src_val_path = src_val_path or self.data_dir / "src-val.txt"
        self.tgt_val_path = tgt_val_path or self.data_dir / "tgt-val.txt"
        self.src_test_path = src_test_path or self.data_dir / "src-test.txt"
        self.tgt_test_path = tgt_test_path or self.data_dir / "tgt-test.txt"
        self.vocab_path = vocab_path

        self.batch_size = batch_size
        self.shuffle_train = shuffle_train
        self.num_workers = num_workers
        self.persistent_workers = persistent_workers
        self.pin_memory = pin_memory
        self.tokens_in_batch = tokens_in_batch

        self.src_tokenizer, self.tgt_tokenizer = self.create_tokenizers(self.vocab_path)

    def create_tokenizers(self, vocab_path: str | None = None) -> tuple[GenericTokenizer, GenericTokenizer]:
        """
        Create tokenizers for a particular task and assign
        them to self.src_tokenizer and self.tgt_tokenizer
        """
        raise NotImplementedError

    def setup(self, stage: str | None = None) -> None:
        if stage == "fit" or stage is None:
            self.train = Seq2SeqDataset(self.src_train_path, self.tgt_train_path, self.src_tokenizer,
                                        self.tgt_tokenizer)
            self.val = Seq2SeqDataset(self.src_val_path, self.tgt_val_path, self.src_tokenizer, self.tgt_tokenizer)

        if stage == "validate":
            self.val = Seq2SeqDataset(self.src_val_path, self.tgt_val_path, self.src_tokenizer, self.tgt_tokenizer)

        if stage == "test" or stage is None:
            self.test = Seq2SeqDataset(self.src_test_path, self.tgt_test_path, self.src_tokenizer, self.tgt_tokenizer)

        if stage == "predict" or stage is None:
            self.prd = Seq2SeqDataset(self.src_test_path, self.tgt_test_path, self.src_tokenizer, self.tgt_tokenizer)

    def collate_fn(self, batch: list[tuple[str, str]]) -> dict[str, torch.Tensor]:
        src_tokens, tgt_tokens = zip(*batch)
        src_tokens = pad_sequence(src_tokens,
                                  padding_value=self.src_tokenizer.pad_token_idx, batch_first=True)
        tgt_tokens = pad_sequence(tgt_tokens,
                                  padding_value=self.tgt_tokenizer.pad_token_idx, batch_first=True)
        return {"src_tokens": src_tokens, "tgt_tokens": tgt_tokens}

    def train_dataloader(self):
        if self.tokens_in_batch is None:
            return DataLoader(self.train,
                              batch_size=self.batch_size,
                              num_workers=self.num_workers,
                              collate_fn=self.collate_fn,
                              persistent_workers=self.persistent_workers,
                              pin_memory=self.pin_memory,
                              shuffle=self.shuffle_train)
        return DataLoader(
            self.train,
            batch_sampler=TokenSampler(
                self.train.target_lenghts,
                self.tokens_in_batch,
                shuffle=self.shuffle_train
            ),
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
            pin_memory=self.pin_memory
        )

    def val_dataloader(self):
        return DataLoader(self.val,
                          batch_size=self.batch_size,
                          num_workers=self.num_workers,
                          collate_fn=self.collate_fn,
                          persistent_workers=self.persistent_workers,
                          pin_memory=self.pin_memory,
                          shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.test,
                          batch_size=self.batch_size,
                          num_workers=self.num_workers,
                          collate_fn=self.collate_fn,
                          persistent_workers=self.persistent_workers,
                          pin_memory=self.pin_memory,
                          shuffle=False)

    def predict_dataloader(self):
        return DataLoader(self.prd,
                          batch_size=self.batch_size,
                          num_workers=self.num_workers,
                          collate_fn=self.collate_fn,
                          persistent_workers=self.persistent_workers,
                          pin_memory=self.pin_memory,
                          shuffle=False)
