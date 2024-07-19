"""
Taken from https://github.com/johnma2006/mamba-minimal.git
"""

import math

import torch
from torch import nn
import torch.nn.functional as F

from einops import repeat, rearrange, einsum

from models.embeddings import TokenEmbedding


class RMSNorm(nn.Module):
    def __init__(self,
                 d_model: int,
                 eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x):
        output = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight

        return output


def selective_scan(u, delta, A, B, C, D):
    """Does selective scan algorithm. See:
        - Section 2 State Space Models in the Mamba paper [1]
        - Algorithm 2 in Section 3.2 in the Mamba paper [1]
        - run_SSM(A, B, C, u) in The Annotated S4 [2]

    This is the classic discrete state space formula:
        x(t + 1) = Ax(t) + Bu(t)
        y(t)     = Cx(t) + Du(t)
    except B and C (and the step size delta, which is used for discretization) are dependent on the input x(t).

    Args:
        u: shape (b, l, d_in)    (See Glossary at top for definitions of b, l, d_in, n...)
        delta: shape (b, l, d_in)
        A: shape (d_in, n)
        B: shape (b, l, n)
        C: shape (b, l, n)
        D: shape (d_in,)

    Returns:
        output: shape (b, l, d_in)

    Official Implementation:
        selective_scan_ref(), https://github.com/state-spaces/mamba/blob/main/mamba_ssm/ops/selective_scan_interface.py#L86
        Note: I refactored some parts out of `selective_scan_ref` out, so the functionality doesn't match exactly.

    """
    (b, l, d_in) = u.shape
    n = A.shape[1]

    # Discretize continuous parameters (A, B)
    # - A is discretized using zero-order hold (ZOH) discretization (see Section 2 Equation 4 in the Mamba paper [1])
    # - B is discretized using a simplified Euler discretization instead of ZOH. From a discussion with authors:
    #   "A is the more important term and the performance doesn't change much with the simplification on B"
    deltaA = torch.exp(einsum(delta, A, 'b l d_in, d_in n -> b l d_in n'))
    deltaB_u = einsum(delta, B, u, 'b l d_in, b l n, b l d_in -> b l d_in n')

    # Perform selective scan (see scan_SSM() in The Annotated S4 [2])
    # Note that the below is sequential, while the official implementation does a much faster parallel scan that
    # is additionally hardware-aware (like FlashAttention).
    x = torch.zeros((b, d_in, n), device=deltaA.device)
    ys = []
    for i in range(l):
        x = deltaA[:, i] * x + deltaB_u[:, i]
        y = einsum(x, C[:, i, :], 'b d_in n, b n -> b d_in')
        ys.append(y)
    y = torch.stack(ys, dim=1)  # shape (b, l, d_in)

    y = y + u * D

    return y


class MambaBlock(nn.Module):
    def __init__(self,
                 d_model: int,
                 d_inner: int,
                 d_conv: int,
                 d_state: int,
                 dt_rank: int,
                 bias: bool,
                 conv_bias: bool):
        """A single Mamba block, as described in Figure 3 in Section 3.4 in the Mamba paper [1]."""
        super().__init__()
        self.d_model = d_model
        self.d_inner = d_inner
        self.d_conv = d_conv
        self.d_state = d_state
        self.dt_rank = dt_rank
        self.bias = bias
        self.conv_bias = conv_bias

        self.in_proj = nn.Linear(d_model, d_inner * 2, bias=bias)

        self.conv1d = nn.Conv1d(
            in_channels=d_inner,
            out_channels=d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            groups=d_inner,
            padding=d_conv - 1,
        )

        # x_proj takes in `x` and outputs the input-specific Δ, B, C
        self.x_proj = nn.Linear(d_inner, dt_rank + d_state * 2, bias=False)

        # dt_proj projects Δ from dt_rank to d_in
        self.dt_proj = nn.Linear(dt_rank, d_inner, bias=True)

        A = repeat(torch.arange(1, d_state + 1), 'n -> d n', d=d_inner)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(d_inner))
        self.out_proj = nn.Linear(d_inner, d_model, bias=bias)

    def forward(self, x):
        """Mamba block forward. This looks the same as Figure 3 in Section 3.4 in the Mamba paper [1].

        Args:
            x: shape (b, l, d)    (See Glossary at top for definitions of b, l, d_in, n...)

        Returns:
            output: shape (b, l, d)

        Official Implementation:
            class Mamba, https://github.com/state-spaces/mamba/blob/main/mamba_ssm/modules/mamba_simple.py#L119
            mamba_inner_ref(), https://github.com/state-spaces/mamba/blob/main/mamba_ssm/ops/selective_scan_interface.py#L311

        """
        (b, l, d) = x.shape

        x_and_res = self.in_proj(x)  # shape (b, l, 2 * d_in)
        (x, res) = x_and_res.split(split_size=[self.d_inner, self.d_inner], dim=-1)

        x = rearrange(x, 'b l d_in -> b d_in l')
        x = self.conv1d(x)[:, :, :l]
        x = rearrange(x, 'b d_in l -> b l d_in')

        x = F.silu(x)

        y = self.ssm(x)

        y = y * F.silu(res)

        output = self.out_proj(y)

        return output

    def ssm(self, x):
        """Runs the SSM. See:
            - Algorithm 2 in Section 3.2 in the Mamba paper [1]
            - run_SSM(A, B, C, u) in The Annotated S4 [2]

        Args:
            x: shape (b, l, d_in)    (See Glossary at top for definitions of b, l, d_in, n...)

        Returns:
            output: shape (b, l, d_in)

        Official Implementation:
            mamba_inner_ref(), https://github.com/state-spaces/mamba/blob/main/mamba_ssm/ops/selective_scan_interface.py#L311

        """
        (d_in, n) = self.A_log.shape

        # Compute ∆ A B C D, the state space parameters.
        #     A, D are input independent (see Mamba paper [1] Section 3.5.2 "Interpretation of A" for why A isn't selective)
        #     ∆, B, C are input-dependent (this is a key difference between Mamba and the linear time invariant S4,
        #                                  and is why Mamba is called **selective** state spaces)

        A = -torch.exp(self.A_log.float())  # shape (d_in, n)
        D = self.D.float()

        x_dbl = self.x_proj(x)  # (b, l, dt_rank + 2*n)

        (delta, B, C) = x_dbl.split(split_size=[self.dt_rank, n, n],
                                    dim=-1)  # delta: (b, l, dt_rank). B, C: (b, l, n)
        delta = F.softplus(self.dt_proj(delta))  # (b, l, d_in)

        y = selective_scan(x, delta, A, B, C, D)  # This is similar to run_SSM(A, B, C, u) in The Annotated S4 [2]

        return y


class BidirectionalMambaBlock(nn.Module):

    def __init__(self,
                 d_model: int,
                 d_inner: int,
                 d_conv: int,
                 d_state: int,
                 dt_rank: int,
                 bias: bool,
                 conv_bias: bool):
        super().__init__()
        self.d_model = d_model
        self.d_inner = d_inner
        self.d_conv = d_conv
        self.d_state = d_state
        self.dt_rank = dt_rank
        self.bias = bias
        self.conv_bias = conv_bias

        # Same for both directions
        self.in_proj = nn.Linear(d_model, d_inner * 2, bias=bias)
        self.out_proj = nn.Linear(d_inner, d_model, bias=bias)

        # Different for both directions
        self.conv1d = nn.Conv1d(
            in_channels=d_inner,
            out_channels=d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            groups=d_inner,
            padding=d_conv - 1,
        )

        self.conv1d_flipped = nn.Conv1d(
            in_channels=d_inner,
            out_channels=d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            groups=d_inner,
            padding=d_conv - 1,
        )

        # x_proj takes in `x` and outputs the input-specific Δ, B, C
        self.x_proj = nn.Linear(d_inner, dt_rank + d_state * 2, bias=False)
        self.x_proj_flipped = nn.Linear(d_inner, dt_rank + d_state * 2, bias=False)

        # dt_proj projects Δ from dt_rank to d_in
        self.dt_proj = nn.Linear(dt_rank, d_inner, bias=True)
        self.dt_proj_flipped = nn.Linear(dt_rank, d_inner, bias=True)

        A = repeat(torch.arange(1, d_state + 1), 'n -> d n', d=d_inner)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(d_inner))
        A_flipped = repeat(torch.arange(1, d_state + 1), 'n -> d n', d=d_inner)
        self.A_log_flipped = nn.Parameter(torch.log(A_flipped))
        self.D_flipped = nn.Parameter(torch.ones(d_inner))

    def forward(self, x, x_flip_ids=None):
        (b, l, d) = x.shape

        x_and_res = self.in_proj(x)  # shape (b, l, 2 * d_in)
        (x, res) = x_and_res.split(split_size=[self.d_inner, self.d_inner], dim=-1)
        res = F.silu(res)

        x = rearrange(x, 'b l d_in -> b d_in l')
        x = self.conv1d(x)[:, :, :l]
        x = rearrange(x, 'b d_in l -> b l d_in')

        x = F.silu(x)

        y = self.ssm(x)

        y = y * res

        if x_flip_ids is not None:
            perm = torch.eye(l)[x_flip_ids]
            x_flipped = torch.bmm(perm, x)
            x_flipped = rearrange(x_flipped, 'b l d_in -> b d_in l')
            x_flipped = self.conv1d_flipped(x_flipped)[:, :, :l]
            x_flipped = rearrange(x_flipped, 'b d_in l -> b l d_in')

            x_flipped = F.silu(x_flipped)

            y_flipped = self.ssm_flipped(x_flipped)

            y_flipped = y_flipped * res

            y = y + torch.bmm(perm, y_flipped)

        output = self.out_proj(y)
        return output

    def ssm(self, x):
        """Runs the SSM. See:
            - Algorithm 2 in Section 3.2 in the Mamba paper [1]
            - run_SSM(A, B, C, u) in The Annotated S4 [2]

        Args:
            x: shape (b, l, d_in)    (See Glossary at top for definitions of b, l, d_in, n...)

        Returns:
            output: shape (b, l, d_in)

        Official Implementation:
            mamba_inner_ref(), https://github.com/state-spaces/mamba/blob/main/mamba_ssm/ops/selective_scan_interface.py#L311

        """
        (d_in, n) = self.A_log.shape

        # Compute ∆ A B C D, the state space parameters.
        #     A, D are input independent (see Mamba paper [1] Section 3.5.2 "Interpretation of A" for why A isn't selective)
        #     ∆, B, C are input-dependent (this is a key difference between Mamba and the linear time invariant S4,
        #                                  and is why Mamba is called **selective** state spaces)

        A = -torch.exp(self.A_log.float())  # shape (d_in, n)
        D = self.D.float()

        x_dbl = self.x_proj(x)  # (b, l, dt_rank + 2*n)

        (delta, B, C) = x_dbl.split(split_size=[self.dt_rank, n, n],
                                    dim=-1)  # delta: (b, l, dt_rank). B, C: (b, l, n)
        delta = F.softplus(self.dt_proj(delta))  # (b, l, d_in)

        y = selective_scan(x, delta, A, B, C, D)  # This is similar to run_SSM(A, B, C, u) in The Annotated S4 [2]

        return y

    def ssm_flipped(self, x):
        (d_in, n) = self.A_log_flipped.shape

        # Compute ∆ A B C D, the state space parameters.
        #     A, D are input independent (see Mamba paper [1] Section 3.5.2 "Interpretation of A" for why A isn't selective)
        #     ∆, B, C are input-dependent (this is a key difference between Mamba and the linear time invariant S4,
        #                                  and is why Mamba is called **selective** state spaces)

        A = -torch.exp(self.A_log_flipped.float())  # shape (d_in, n)
        D = self.D_flipped.float()

        x_dbl = self.x_proj_flipped(x)  # (b, l, dt_rank + 2*n)

        (delta, B, C) = x_dbl.split(split_size=[self.dt_rank, n, n],
                                    dim=-1)  # delta: (b, l, dt_rank). B, C: (b, l, n)
        delta = F.softplus(self.dt_proj_flipped(delta))  # (b, l, d_in)

        y = selective_scan(x, delta, A, B, C, D)  # This is similar to run_SSM(A, B, C, u) in The Annotated S4 [2]

        return y


class ResidualBlock(nn.Module):
    def __init__(self,
                 d_model: int,
                 **block_kwargs
                 ):
        """Simple block wrapping Mamba block with normalization and residual connection."""
        super().__init__()
        self.mixer = MambaBlock(d_model,
                                **block_kwargs)
        self.norm = RMSNorm(d_model)

    def forward(self, x):
        """
        Args:
            x: shape (b, l, d)    (See Glossary at top for definitions of b, l, d_in, n...)

        Returns:
            output: shape (b, l, d)

        Official Implementation:
            Block.forward(), https://github.com/state-spaces/mamba/blob/main/mamba_ssm/modules/mamba_simple.py#L297

            Note: the official repo chains residual blocks that look like
                [Add -> Norm -> Mamba] -> [Add -> Norm -> Mamba] -> [Add -> Norm -> Mamba] -> ...
            where the first Add is a no-op. This is purely for performance reasons as this
            allows them to fuse the Add->Norm.

            We instead implement our blocks as the more familiar, simpler, and numerically equivalent
                [Norm -> Mamba -> Add] -> [Norm -> Mamba -> Add] -> [Norm -> Mamba -> Add] -> ....

        """
        output = self.mixer(self.norm(x)) + x

        return output


class ResidualBidirectionalBlock(nn.Module):
    def __init__(self,
                 d_model: int,
                 **block_kwargs):
        """Simple block wrapping Mamba block with normalization and residual connection."""
        super().__init__()
        self.mixer = BidirectionalMambaBlock(d_model,
                                             **block_kwargs)
        self.norm = RMSNorm(d_model)

    def forward(self, src_embs, src_flip_ids):
        output = self.mixer(self.norm(src_embs), src_flip_ids) + src_embs

        return output


class EncoderSelectiveSSM(nn.Module):

    def __init__(self,
                 n_layer: int,
                 d_model: int,
                 **block_kwargs):
        super().__init__()

        self.layers = nn.ModuleList(
            [ResidualBidirectionalBlock(d_model, **block_kwargs) for _ in range(n_layer)])
        self.norm_f = RMSNorm(d_model)

    def forward(self, src_embs, src_flip_ids=None):
        """
        src_embs: B L_src D
        src_flip_ids: B L_src
        return: B L_src D
        """
        for layer in self.layers:
            src_embs = layer(src_embs, src_flip_ids)
        src_embs = self.norm_f(src_embs)
        return src_embs


class DecoderSelectiveSSM(nn.Module):

    def __init__(self,
                 n_layer: int,
                 d_model: int,
                 **block_kwargs):
        super().__init__()
        self.layers = nn.ModuleList([ResidualBlock(d_model, **block_kwargs) for _ in range(n_layer)])
        self.norm_f = RMSNorm(d_model)

    def forward(self, tgt_embs, memory):
        """
        tgt_embs: B L_tgt D
        memory: B L_src D
        return: B L_tgt D
        """
        src_tgt_embs = torch.cat([memory, tgt_embs], dim=1)
        for layer in self.layers:
            src_tgt_embs = layer(src_tgt_embs)
        _, tgt_embs = torch.split(src_tgt_embs, [memory.size()[1], tgt_embs.size()[1]], dim=1)
        tgt_embs = self.norm_f(tgt_embs)
        return tgt_embs


class PlainSelectiveSSM(nn.Module):

    def __init__(self,
                 n_layer: int,
                 d_model: int,
                 **block_kwargs):
        super().__init__()
        self.layers = nn.ModuleList([ResidualBlock(d_model, **block_kwargs) for _ in range(n_layer)])
        self.norm_f = RMSNorm(d_model)

    def forward(self, tgt_embs):
        """
        tgt_embs: B L_tgt D
        memory: B L_src D
        return: B L_tgt D
        """
        x = tgt_embs
        for layer in self.layers:
            x = layer(x)
        x = self.norm_f(x)
        return x


class EncoderDecoderMamba(nn.Module):

    def __init__(self,
                 src_vocab_size: int,
                 tgt_vocab_size: int,
                 num_encoder_layers: int,
                 num_decoder_layers: int,
                 embedding_dim: int,

                 expand: int = 2,
                 d_conv: int = 4,
                 d_state: int = 16,
                 dt_rank: int | str = "auto",
                 bias: bool = False,
                 conv_bias: bool = True
                 ):
        super().__init__()
        self.src_vocab_size = src_vocab_size
        self.tgt_vocab_size = tgt_vocab_size
        self.num_encoder_layers = num_encoder_layers
        self.num_decoder_layers = num_decoder_layers
        self.embedding_dim = embedding_dim
        self.expand = expand
        self.d_conv = d_conv
        self.d_state = d_state
        self.dt_rank = dt_rank
        self.bias = bias
        self.conv_bias = conv_bias

    def create(self):
        d_inner = int(self.expand * self.embedding_dim)

        assert self.dt_rank == "auto" or isinstance(self.dt_rank,
                                                    int), "The value for dt_rank cat be either an integer or 'auto'"
        if self.dt_rank == 'auto':
            self.dt_rank = math.ceil(self.embedding_dim / 16)
        layer_kwargs = dict(
            d_inner=d_inner,
            d_conv=self.d_conv,
            d_state=self.d_state,
            dt_rank=self.dt_rank,
            bias=self.bias,
            conv_bias=self.conv_bias)
        self.norm_f = RMSNorm(self.embedding_dim)

        # Embedding constructor
        self.src_token_featurizer = TokenEmbedding(self.src_vocab_size,
                                                   self.embedding_dim)

        self.tgt_token_featurizer = self.src_token_featurizer

        # Embedding updater
        self.encoder = EncoderSelectiveSSM(self.num_encoder_layers, self.embedding_dim, **layer_kwargs)
        self.decoder = DecoderSelectiveSSM(self.num_decoder_layers, self.embedding_dim, **layer_kwargs)

        # Decision function

        self.next_token_classifier = nn.Linear(self.embedding_dim, self.tgt_vocab_size, bias=False)

    def forward(self, src: torch.LongTensor, tgt: torch.LongTensor, src_flip_ids: torch.LongTensor | None = None):
        src_embs = self.src_token_featurizer(src)  # B L -> B L D
        tgt_embs = self.tgt_token_featurizer(tgt)  # B L -> B L D

        memory = self.encoder(src_embs, src_flip_ids)  # B L D -> B L D
        output = self.decoder(tgt_embs, memory)  # B L D, B L D -> B L D

        logits = self.next_token_classifier(output)  # B L D -> B L V

        return logits

    # def forward(self, src: torch.LongTensor, tgt: torch.LongTensor, src_flip_ids: torch.LongTensor | None = None):
    #     src_embs = self.src_token_featurizer(src)  # B L -> B L D
    #     tgt_embs = self.tgt_token_featurizer(tgt)  # B L -> B L D
    #     src_tgt_embs = torch.cat((src_embs, tgt_embs), dim=1)
    #
        # output = self.encoder(src_tgt_embs, src_flip_ids=None)  # B L D -> B L D
        # output = self.decoder(src_tgt_embs)  # B L D, B L D -> B L D
        # _, output = torch.split(output, [src_embs.shape[1], tgt_embs.shape[1]], dim=1)
        #
        # logits = self.next_token_classifier(output)  # B L D -> B L V
        #
        # return logits


class DecoderOnlyMamba(nn.Module):

    def __init__(self,
                 src_vocab_size: int,
                 tgt_vocab_size: int,
                 num_encoder_layers: int,
                 num_decoder_layers: int,
                 embedding_dim: int,

                 expand: int = 2,
                 d_conv: int = 4,
                 d_state: int = 16,
                 dt_rank: int | str = "auto",
                 bias: bool = False,
                 conv_bias: bool = True
                 ):
        super().__init__()
        self.src_vocab_size = src_vocab_size
        self.tgt_vocab_size = tgt_vocab_size
        self.num_encoder_layers = num_encoder_layers
        self.num_decoder_layers = num_decoder_layers
        self.embedding_dim = embedding_dim
        self.expand = expand
        self.d_conv = d_conv
        self.d_state = d_state
        self.dt_rank = dt_rank
        self.bias = bias
        self.conv_bias = conv_bias

    def create(self):
        d_inner = int(self.expand * self.embedding_dim)

        assert self.dt_rank == "auto" or isinstance(self.dt_rank,
                                                    int), "The value for dt_rank cat be either an integer or 'auto'"
        if self.dt_rank == 'auto':
            self.dt_rank = math.ceil(self.embedding_dim / 16)
        layer_kwargs = dict(
            d_inner=d_inner,
            d_conv=self.d_conv,
            d_state=self.d_state,
            dt_rank=self.dt_rank,
            bias=self.bias,
            conv_bias=self.conv_bias)
        self.norm_f = RMSNorm(self.embedding_dim)

        # Embedding constructor
        self.src_token_featurizer = TokenEmbedding(self.src_vocab_size,
                                                   self.embedding_dim)

        self.tgt_token_featurizer = self.src_token_featurizer

        # Embedding updater
        self.decoder = PlainSelectiveSSM(self.num_decoder_layers, self.embedding_dim, **layer_kwargs)

        # Decision function

        self.next_token_classifier = nn.Linear(self.embedding_dim, self.tgt_vocab_size, bias=False)

    def forward(self, src: torch.LongTensor, tgt: torch.LongTensor, src_flip_ids: torch.LongTensor | None = None):
        src_embs = self.src_token_featurizer(src)  # B L -> B L D
        tgt_embs = self.tgt_token_featurizer(tgt)  # B L -> B L D
        src_tgt_embs = torch.cat((src_embs, tgt_embs), dim=1)

        output = self.decoder(src_tgt_embs)  # B L D, B L D -> B L D
        _, output = torch.split(output, [src_embs.shape[1], tgt_embs.shape[1]], dim=1)

        logits = self.next_token_classifier(output)  # B L D -> B L V

        return logits
