import torch


class MockCopySequence:
    """
    A mock hardcoded model to imitate a transformer for copy-sequence
    """

    def __init__(self):
        self.pad_token = 0
        self.bos_token = 1
        self.eos_token = 10

        self.weights = torch.tensor([
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0.4, 0.38, 0.12, 0.09, 0.01, 0, 0, 0, 0],
            [0, 0, 0, 0.4, 0.37, 0.13, 0.08, 0.02, 0, 0, 0],
            [0, 0, 0, 0, 0.4, 0.36, 0.14, 0.07, 0.03, 0, 0],
            [0, 0, 0, 0, 0, 0.4, 0.35, 0.15, 0.06, 0.04, 0],
            [0, 0, 0, 0, 0, 0, 0.4, 0.34, 0.16, 0.1, 0],
            [0, 0, 0, 0, 0, 0, 0, 0.4, 0.33, 0.17, 0.1],
            [0, 0, 0, 0, 0, 0, 0, 0, 0.5, 0.3, 0.2],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0.6, 0.4],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
        ])

        self.emb_matrix = torch.eye(11)

    def embedding(self, x: 'torch.LongTensor') -> 'torch.Tensor':
        return torch.cat(
            [torch.index_select(self.emb_matrix, dim=0, index=x[i, ...]).unsqueeze(0) for i in range(x.shape[0])],
            dim=0
        )

    def forward(self, src: 'torch.LongTensor', tgt: 'torch.LongTensor') -> 'torch.Tensor':
        _, decoded_length = tgt.shape
        return (self.embedding(src) @ self.weights)[:, 1:decoded_length + 1, :]

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)
