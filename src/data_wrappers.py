# TODO A file with only base classes to inherit from?

from typing import List, Tuple, Optional, Union
from multiprocessing import cpu_count
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


class Generic(LightningDataModule):
    def __init__(self,
                 data_dir: str = None,
                 batch_size: int = 1,
                 num_workers: int = cpu_count(),
                 shuffle_train: bool = False,
                 pad_idx: int = 0):
        super().__init__()

        self.data_dir = Path(data_dir).resolve()
        self.batch_size = batch_size
        self.shuffle_train = shuffle_train
        self.num_workers = num_workers
        self.pad_idx = pad_idx

    def setup(self, stage: 'Optional[str]' = None) -> None:
        train_src_path = self.data_dir / "src-train-tokens.pt"
        train_tgt_path = self.data_dir / "tgt-train-tokens.pt"
        val_src_path = self.data_dir / "src-val-tokens.pt"
        val_tgt_path = self.data_dir / "tgt-val-tokens.pt"
        test_src_path = self.data_dir / "src-test-tokens.pt"
        test_tgt_path = self.data_dir / "tgt-test-tokens.pt"
        if stage == "fit" or stage is None:
            self.train = Seq2SeqDataset(train_src_path, train_tgt_path)
            self.val = Seq2SeqDataset(val_src_path, val_tgt_path)

        if stage == "validate":
            self.val = Seq2SeqDataset(val_src_path, val_tgt_path)

        if stage == "test" or stage is None:
            self.test = Seq2SeqDataset(test_src_path, test_tgt_path)

        if stage == "predict" or stage is None:
            self.prd = Seq2SeqDataset(test_src_path)

    def collate_fn(self, batch: 'List[Tuple[torch.LongTensor, torch.LongTensor]]'
                   ) -> 'Tuple[torch.LongTensor, torch.LongTensor]':
        src_batch, tgt_batch = [], []
        for src_sample, tgt_sample in batch:
            src_batch.append(src_sample)
            tgt_batch.append(tgt_sample)
        src_batch = pad_sequence(src_batch, padding_value=self.pad_idx, batch_first=True)
        tgt_batch = pad_sequence(tgt_batch, padding_value=self.pad_idx, batch_first=True)
        return src_batch.long(), tgt_batch.long()

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
