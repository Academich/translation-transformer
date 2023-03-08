from typing import List, Tuple, Optional, Mapping, Union
from multiprocessing import cpu_count

import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

import pytorch_lightning as pl


class TokenVocabulary:

    def __init__(self,
                 vocab: 'Mapping[Union[int, str], str]',
                 bos_token_idx: 'int' = 1,
                 eos_token_idx: 'int' = 2,
                 pad_token_idx: 'int' = 0):
        self.vocab = {int(k): v for k, v in vocab.items()}
        self.bos_token_idx = bos_token_idx
        self.eos_token_idx = eos_token_idx
        self.pad_token_idx = pad_token_idx
        self.service_tokens = (bos_token_idx, eos_token_idx, pad_token_idx)
        self.n_tokens = len(self.vocab)

    def decode(self, tokens: 'torch.LongTensor'):
        decoded_chars = []
        for i in tokens.numpy():
            if i == self.eos_token_idx:
                break
            if i not in self.service_tokens:
                decoded_chars.append(self.vocab[i])

        decoded_string = "".join(decoded_chars)
        return decoded_string


class Seq2SeqDataset(Dataset):

    def __init__(self, src_path: 'str', tgt_path: 'str'):
        with open(src_path, "rb") as f, open(tgt_path, "rb") as g:
            self.domain = torch.load(f)
            self.target = torch.load(g)

    def __len__(self):
        return len(self.domain)

    def __getitem__(self, item):
        src_tokens = self.domain[item]
        tgt_tokens = self.target[item]
        return src_tokens, tgt_tokens


class Seq2SeqDM(pl.LightningDataModule):

    def __init__(self,
                 train_src_path: Optional[str] = None,
                 train_tgt_path: Optional[str] = None,
                 val_src_path: Optional[str] = None,
                 val_tgt_path: Optional[str] = None,
                 test_src_path: Optional[str] = None,
                 test_tgt_path: Optional[str] = None,
                 batch_size: int = 1,
                 num_workers: int = cpu_count(),
                 shuffle_train: bool = False,
                 padding_idx: int = 0):
        super().__init__()
        self.train_src_path = train_src_path
        self.val_src_path = val_src_path
        self.test_src_path = test_src_path

        self.train_tgt_path = train_tgt_path
        self.val_tgt_path = val_tgt_path
        self.test_tgt_path = test_tgt_path

        self.batch_size = batch_size
        self.shuffle_train = shuffle_train
        self.num_workers = num_workers
        self.padding_idx = padding_idx

    def setup(self, stage: 'Optional[str]' = None) -> None:
        if stage == "fit" or stage is None:
            self.train = Seq2SeqDataset(self.train_src_path, self.train_tgt_path)
            self.val = Seq2SeqDataset(self.val_src_path, self.val_tgt_path)

        if stage == "validate":
            self.val = Seq2SeqDataset(self.val_src_path, self.val_tgt_path)

        if stage in ("test", "predict") or stage is None:
            self.test = Seq2SeqDataset(self.test_src_path, self.test_tgt_path)

    def collate_fn(self, batch: 'List[Tuple[torch.LongTensor, torch.LongTensor]]'
                   ) -> 'Tuple[torch.LongTensor, torch.LongTensor]':
        src_batch, tgt_batch = [], []
        for src_sample, tgt_sample in batch:
            src_batch.append(src_sample)
            tgt_batch.append(tgt_sample)
        src_batch = pad_sequence(src_batch, padding_value=self.padding_idx, batch_first=True)
        tgt_batch = pad_sequence(tgt_batch, padding_value=self.padding_idx, batch_first=True)
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
        return DataLoader(self.test,
                          batch_size=self.batch_size,
                          num_workers=self.num_workers,
                          collate_fn=self.collate_fn,
                          shuffle=False)
