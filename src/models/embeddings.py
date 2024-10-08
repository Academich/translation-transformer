import math

import torch
from torch import nn
from torch.nn.functional import one_hot


class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size: int, emb_size: int, padding_idx=0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_size, padding_idx=padding_idx)
        self.emb_size = emb_size

    def forward(self, tokens):
        return self.embedding(tokens)


# class ManualTokenEmbedding(nn.Module):
#     def __init__(self, vocab_size: int, emb_size: int):
#         super(ManualTokenEmbedding, self).__init__()
#         self.vocab_size = vocab_size
#         self.embedding = nn.Linear(vocab_size, emb_size, bias=False)
#
#     def forward(self, tokens):
#         return self.embedding(
#             one_hot(tokens, num_classes=self.vocab_size).float()
#         )


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
        pe = torch.vstack((torch.zeros(1, emb_size), pe))

        # register_buffer => Tensor which is not a parameter, but should be part of the modules state.
        # Used for tensors that need to be on the same device as the module.
        # persistent=False tells PyTorch to not add the buffer to the state dict (e.g. when we save the model)
        self.register_buffer("pe", pe, persistent=False)

    def forward(self, x: torch.Tensor, offset: torch.LongTensor = torch.LongTensor([0])):
        """
        If a sequence is padded from the left, can assign correct positional embeddings
        for meaningful tokens at the right positions.
        x: B, L, D
        offset: B, 1
        self.pe: 1, L, D
        """
        seq_len = x.size(1)
        shifts = torch.relu(
            torch.arange(1, seq_len + 1).type_as(x) - offset.type_as(x)).long()  # amounts to self.pe[1: seq_len + 1] if offset is 0
        pe = self.pe[shifts]
        return x + pe


if __name__ == '__main__':

    B, L, D = 2, 5, 6
    emb_layer = nn.Embedding(L + 1, D)
    pos = PositionalEncoding(emb_size=D, max_len=L)
    src = torch.tensor([[1, 2, 3, 4, 0], [1, 2, 0, 0, 0]])
    # src_emb = emb_layer(src)
    # src_emb_pos = pos(src_emb)
    # print(src_emb_pos)

    src_2 = torch.tensor([[0, 1, 2, 3, 4], [0, 0, 0, 1, 2]])
    src_emb_2 = emb_layer(src_2)
    src_emb_pos_2 = pos(src_emb_2, offset=torch.tensor([[1], [3]]))
    print(src_emb_pos_2)
