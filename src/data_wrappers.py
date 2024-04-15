from typing import Any
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

from pytorch_lightning import LightningDataModule


class Seq2SeqDataset(Dataset):

    def __init__(self, src_path: Path | str, tgt_path: Path | str):
        with open(src_path) as fs, open(tgt_path) as ft:
            self.source = [s.strip() for s in fs.readlines()]
            self.target = [s.strip() for s in ft.readlines()]
        assert len(self.source) == len(
            self.target), f"The source and target data at {src_path} and {tgt_path} have different lenghts"

    def __len__(self):
        return len(self.source)

    def __getitem__(self, item) -> tuple[str, str]:
        return self.source[item], self.target[item]


class Seq2SeqDM(LightningDataModule):
    def __init__(self,
                 data_dir: str | None = None,
                 src_tokenizer: Any | None = None,
                 tgt_tokenizer: Any | None = None,
                 batch_size: int = 1,
                 num_workers: int = 0,
                 shuffle_train: bool = False):
        super().__init__()

        self.data_dir = Path(data_dir).resolve()
        self.src_train_path = self.data_dir / "src-train.txt"
        self.tgt_train_path = self.data_dir / "tgt-train.txt"
        self.src_val_path = self.data_dir / "src-val.txt"
        self.tgt_val_path = self.data_dir / "tgt-val.txt"
        self.src_test_path = self.data_dir / "src-test.txt"
        self.tgt_test_path = self.data_dir / "tgt-test.txt"

        self.src_tokenizer = src_tokenizer
        self.tgt_tokenizer = tgt_tokenizer

        self.batch_size = batch_size
        self.shuffle_train = shuffle_train
        self.num_workers = num_workers

    def setup(self, stage: str | None = None) -> None:
        if stage == "fit" or stage is None:
            self.train = Seq2SeqDataset(self.src_train_path, self.tgt_train_path)
            self.val = Seq2SeqDataset(self.src_val_path, self.tgt_val_path)

        if stage == "validate":
            self.val = Seq2SeqDataset(self.src_val_path, self.tgt_val_path)

        if stage == "test" or stage is None:
            self.test = Seq2SeqDataset(self.src_test_path, self.tgt_test_path)

        if stage == "predict" or stage is None:
            self.prd = Seq2SeqDataset(self.src_test_path, self.tgt_test_path)

    def _prepare_tokens(self, src_strings, tgt_strings) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        src_tokens = []
        tgt_tokens = []
        for src_string, tgt_string in zip(src_strings, tgt_strings):
            src_tokens.append(self.src_tokenizer.encode(src_string))
            tgt_tokens.append(self.tgt_tokenizer.encode(tgt_string))
        src_tokens = [torch.tensor(i).long() for i in src_tokens]
        tgt_tokens = [torch.tensor(i).long() for i in tgt_tokens]
        return src_tokens, tgt_tokens

    def collate_fn(self, batch: list[tuple[str, str]]) -> dict[str, torch.Tensor]:
        src_strings, tgt_strings = zip(*batch)
        src_tokens, tgt_tokens = self._prepare_tokens(src_strings, tgt_strings)
        src_tokens = pad_sequence(src_tokens,
                                  padding_value=self.src_tokenizer.pad_token_idx, batch_first=True)
        src_pad_mask = (src_tokens == self.src_tokenizer.pad_token_idx).bool()
        tgt_tokens = pad_sequence(tgt_tokens,
                                  padding_value=self.tgt_tokenizer.pad_token_idx, batch_first=True)
        tgt_pad_mask = (tgt_tokens == self.tgt_tokenizer.pad_token_idx).bool()
        return {"src_tokens": src_tokens,
                "tgt_tokens": tgt_tokens,
                "src_pad_mask": src_pad_mask,
                "tgt_pad_mask": tgt_pad_mask}

    def train_dataloader(self):
        return DataLoader(self.train,
                          batch_size=self.batch_size,
                          num_workers=self.num_workers,
                          collate_fn=self.collate_fn,
                          shuffle=self.shuffle_train)

    def val_dataloader(self):
        return DataLoader(self.val,
                          batch_size=self.batch_size,
                          num_workers=self.num_workers,
                          collate_fn=self.collate_fn,
                          shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.test,
                          batch_size=self.batch_size,
                          num_workers=self.num_workers,
                          collate_fn=self.collate_fn,
                          shuffle=False)

    def predict_dataloader(self):
        return DataLoader(self.prd,
                          batch_size=self.batch_size,
                          num_workers=self.num_workers,
                          collate_fn=self.collate_fn,
                          shuffle=False)
