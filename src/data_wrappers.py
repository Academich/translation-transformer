# TODO A file with only base classes to inherit from?

from typing import List, Tuple, Optional, Mapping, Union
from multiprocessing import cpu_count
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

from pytorch_lightning import LightningDataModule


class TokenVocabulary:

    def __init__(self,
                 vocab: Mapping[Union[int, str], str],
                 bos_token_idx: int = 1,
                 eos_token_idx: int = 2,
                 pad_token_idx: int = 0):
        self.vocab = {int(k): v for k, v in vocab.items()}
        self.bos_token_idx = bos_token_idx
        self.eos_token_idx = eos_token_idx
        self.pad_token_idx = pad_token_idx
        self.service_tokens = (bos_token_idx, eos_token_idx, pad_token_idx)
        self.n_tokens = len(self.vocab)

    def decode(self, tokens: torch.LongTensor):
        decoded_chars = []
        for i in tokens.numpy():
            if i == self.eos_token_idx:
                break
            if i not in self.service_tokens:
                decoded_chars.append(self.vocab[i])

        decoded_string = "".join(decoded_chars)
        return decoded_string

    def decode_batch(self, tokens: torch.LongTensor):
        return [self.decode(tokens[i]) for i in range(tokens.size()[0])]


class Seq2SeqDataset(Dataset):

    def __init__(self, src_path: Union[Path, str], tgt_path: Optional[Union[Path, str]] = None):
        with open(src_path, "rb") as f:
            self.domain = torch.load(f)
            self.target = None
        if tgt_path is not None:
            with open(tgt_path, "rb") as g:
                self.target = torch.load(g)
        else:
            self.target = self.domain

    def __len__(self):
        return len(self.domain)

    def __getitem__(self, item):
        src_tokens = self.domain[item]
        tgt_tokens = self.target[item]
        return src_tokens, tgt_tokens


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