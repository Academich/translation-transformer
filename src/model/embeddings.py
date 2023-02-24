import math

import torch
from torch import nn
from torch.nn.functional import one_hot


# class TokenEmbedding(nn.Module):
#     def __init__(self, vocab_size: int, emb_size: int, paddind_idx):
#         super().__init__()
#         self.embedding = nn.Embedding(vocab_size, emb_size, padding_idx=paddind_idx)
#         self.emb_size = emb_size
#
#     def forward(self, tokens):
#         return self.embedding(tokens.long()) * math.sqrt(self.emb_size)

class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size: int, emb_size: int):
        super(TokenEmbedding, self).__init__()
        self.vocab_size = vocab_size
        self.embedding = nn.Linear(vocab_size, emb_size, bias=False)

    def forward(self, tokens):
        return self.embedding(
            one_hot(tokens, num_classes=self.vocab_size).float()
        )


class PositionalEncoding(nn.Module):
    def __init__(self, emb_size, max_len=5000):
        """
        Args
            d_model: Hidden dimensionality of the input.
            max_len: Maximum length of a sequence to expect.
        """
        super().__init__()

        # Create matrix of [SeqLen, HiddenDim] representing the positional encoding for max_len inputs
        pe = torch.zeros(max_len, emb_size)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, emb_size, 2).float() * (-math.log(10000.0) / emb_size))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)

        # register_buffer => Tensor which is not a parameter, but should be part of the modules state.
        # Used for tensors that need to be on the same device as the module.
        # persistent=False tells PyTorch to not add the buffer to the state dict (e.g. when we save the model)
        self.register_buffer("pe", pe, persistent=False)

    def forward(self, x):
        x = x + self.pe[:, : x.size(1)]
        return x
