from typing import Optional, Union, Callable
from functools import partial

import torch
from torch import nn
from torch import LongTensor, BoolTensor, Tensor
import torch.nn.functional as F

from mamba_ssm.modules.mamba_simple import Mamba
from mamba_ssm.models.mixer_seq_simple import create_block, RMSNorm, layer_norm_fn, rms_norm_fn, _init_weights

from models.embeddings import TokenEmbedding, PositionalEncoding


class MambaTransformerDecoderLayer(nn.Module):

    def __init__(self,
                 d_model: int,
                 nhead: int,
                 dim_feedforward: int = 2048,
                 dropout: float = 0.1,
                 activation: Union[str, Callable[[Tensor], Tensor]] = "relu",
                 layer_norm_eps: float = 1e-5,
                 batch_first: bool = True,
                 norm_first: bool = False,
                 bias: bool = True) -> None:
        super().__init__()
        self.mamba = Mamba(d_model,
                           d_state=16,
                           d_conv=4,
                           expand=2,
                           dt_rank="auto",
                           dt_min=0.001,
                           dt_max=0.1,
                           dt_init="random",
                           dt_scale=1.0,
                           dt_init_floor=1e-4,
                           conv_bias=True,
                           bias=False,
                           use_fast_path=True  # Fused kernel options
                           )
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first,
                                                    bias=bias)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward, bias=bias)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model, bias=bias)

        self.norm_first = norm_first
        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps, bias=bias)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps, bias=bias)
        self.norm3 = nn.LayerNorm(d_model, eps=layer_norm_eps, bias=bias)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        # Legacy string support for activation function.
        if activation == "relu":
            self.activation = F.relu
        elif activation == "gelu":
            self.activation = F.gelu
        else:
            raise RuntimeError(f"activation should be relu/gelu, not {activation}")

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.relu
        super().__setstate__(state)

    def forward(
            self,
            tgt: Tensor,
            memory: Tensor,
            memory_key_padding_mask: Optional[Tensor] = None
    ) -> Tensor:
        r"""Pass the inputs (and mask) through the decoder layer.

        Args:
            tgt: the sequence to the decoder layer (required).
            memory: the sequence from the last layer of the encoder (required).
            tgt_mask: the mask for the tgt sequence (optional).
            memory_mask: the mask for the memory sequence (optional).
            tgt_key_padding_mask: the mask for the tgt keys per batch (optional).
            memory_key_padding_mask: the mask for the memory keys per batch (optional).
            tgt_is_causal: If specified, applies a causal mask as ``tgt mask``.
                Default: ``False``.
                Warning:
                ``tgt_is_causal`` provides a hint that ``tgt_mask`` is
                the causal mask. Providing incorrect hints can result in
                incorrect execution, including forward and backward
                compatibility.
            memory_is_causal: If specified, applies a causal mask as
                ``memory mask``.
                Default: ``False``.
                Warning:
                ``memory_is_causal`` provides a hint that
                ``memory_mask`` is the causal mask. Providing incorrect
                hints can result in incorrect execution, including
                forward and backward compatibility.

        Shape:
            see the docs in Transformer class.
        """
        # see Fig. 1 of https://arxiv.org/pdf/2002.04745v1.pdf

        x = tgt
        if self.norm_first:
            x = x + self._mamba_block(self.norm1(x))
            x = x + self._mha_block(self.norm2(x), memory, memory_key_padding_mask)
            x = x + self._ff_block(self.norm3(x))
        else:
            x = self.norm1(x + self._mamba_block(x))
            x = self.norm2(x + self._mha_block(x, memory, memory_key_padding_mask))
            x = self.norm3(x + self._ff_block(x))

        return x

    # self-attention block
    def _mamba_block(self, x: Tensor) -> Tensor:
        x = self.mamba(x)
        return self.dropout1(x)

    # multihead attention block
    def _mha_block(self, x: Tensor, mem: Tensor, key_padding_mask: Optional[Tensor]) -> Tensor:
        x = self.multihead_attn(x, mem, mem,
                                key_padding_mask=key_padding_mask,
                                is_causal=False,
                                need_weights=False)[0]
        return self.dropout2(x)

    # feed forward block
    def _ff_block(self, x: Tensor) -> Tensor:
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout3(x)


class TransformerEncMambaTransformerDec(nn.Module):
    def __init__(self,
                 src_vocab_size: int,
                 tgt_vocab_size: int,
                 num_encoder_layers: int = 3,
                 num_decoder_layers: int = 3,
                 embedding_dim: int = 128,
                 num_heads: int = 4,
                 feedforward_dim: int = 256,
                 dropout_rate: float = 0.0,
                 activation: str = "relu",
                 share_embeddings: bool = False,
                 src_pad_token_idx: int = 0,
                 tgt_pad_token_idx: int = 0):
        super().__init__()
        self.src_vocab_size = src_vocab_size
        self.tgt_vocab_size = tgt_vocab_size
        self.src_pad_token_i = src_pad_token_idx
        self.tgt_pad_token_i = tgt_pad_token_idx

        self.num_enc_layers = num_encoder_layers
        self.num_dec_layers = num_decoder_layers
        self.emb_dim = embedding_dim
        self.num_heads = num_heads
        self.ff_dim = feedforward_dim
        self.dropout_rate = dropout_rate
        self.activation = activation
        self.share_embeddings = share_embeddings

        self.layer_norm_eps = 1e-5

    def create(self):
        self.src_token_featurizer = TokenEmbedding(self.src_vocab_size,
                                                   self.emb_dim, padding_idx=self.src_pad_token_i)
        if self.share_embeddings:
            self.tgt_token_featurizer = self.src_token_featurizer
            assert self.src_vocab_size == self.tgt_vocab_size
        else:
            self.tgt_token_featurizer = TokenEmbedding(self.tgt_vocab_size,
                                                       self.emb_dim, padding_idx=self.tgt_pad_token_i)

        self.positional_encoding = PositionalEncoding(self.emb_dim)

        # Embedding updater

        self.encoder = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=self.emb_dim,
                                                                        nhead=self.num_heads,
                                                                        dim_feedforward=self.ff_dim,
                                                                        dropout=self.dropout_rate,
                                                                        activation=self.activation,
                                                                        layer_norm_eps=self.layer_norm_eps,
                                                                        batch_first=True,
                                                                        norm_first=False,
                                                                        bias=True),
                                             self.num_enc_layers,
                                             nn.LayerNorm(self.emb_dim, eps=1e-5, bias=True))
        self.decoder = nn.ModuleList([
            MambaTransformerDecoderLayer(d_model=self.emb_dim,
                                         nhead=self.num_heads,
                                         dim_feedforward=self.ff_dim,
                                         dropout=self.dropout_rate,
                                         activation=self.activation,
                                         layer_norm_eps=self.layer_norm_eps,
                                         batch_first=True,
                                         norm_first=True,
                                         bias=True) for _ in
            range(self.num_dec_layers)
        ])

        # Decision function
        self.next_token_classifier = nn.Linear(self.emb_dim, self.tgt_vocab_size)

    def forward(self, src: LongTensor, tgt: LongTensor):
        _, tgt_seq_len = tgt.size()

        # Embed tokens
        src_emb = self.positional_encoding(self.src_token_featurizer(src))
        tgt_emb = self.positional_encoding(self.tgt_token_featurizer(tgt))

        # Update embeddings
        src_pad_mask = (src == self.src_pad_token_i).bool()
        memory = self.encoder(src_emb, src_key_padding_mask=src_pad_mask)
        for dec_layer in self.decoder:
            tgt_emb = dec_layer(tgt_emb, memory,
                                memory_key_padding_mask=src_pad_mask)

        # Propose the next token
        logits = self.next_token_classifier(tgt_emb)
        return logits


class MambaMixerModel(nn.Module):
    def __init__(
            self,
            d_model: int,
            n_layer: int,
            ssm_cfg=None,
            norm_epsilon: float = 1e-5,
            rms_norm: bool = False,
            initializer_cfg=None,
            fused_add_norm=False,
            residual_in_fp32=False,
    ) -> None:
        super().__init__()
        self.residual_in_fp32 = residual_in_fp32

        # We change the order of residual and layer norm:
        # Instead of LN -> Attn / MLP -> Add, we do:
        # Add -> LN -> Attn / MLP / Mixer, returning both the residual branch (output of Add) and
        # the main branch (output of MLP / Mixer). The model definition is unchanged.
        # This is for performance reason: we can fuse add + layer_norm.
        self.fused_add_norm = fused_add_norm
        if self.fused_add_norm:
            if layer_norm_fn is None or rms_norm_fn is None:
                raise ImportError("Failed to import Triton LayerNorm / RMSNorm kernels")

        self.layers = nn.ModuleList(
            [
                create_block(
                    d_model,
                    ssm_cfg=ssm_cfg,
                    norm_epsilon=norm_epsilon,
                    rms_norm=rms_norm,
                    residual_in_fp32=residual_in_fp32,
                    fused_add_norm=fused_add_norm,
                    layer_idx=i,
                )
                for i in range(n_layer)
            ]
        )

        self.norm_f = (nn.LayerNorm if not rms_norm else RMSNorm)(
            d_model, eps=norm_epsilon
        )

        self.apply(
            partial(
                _init_weights,
                n_layer=n_layer,
                **(initializer_cfg if initializer_cfg is not None else {}),
            )
        )

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return {
            i: layer.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)
            for i, layer in enumerate(self.layers)
        }

    def forward(self, hidden_states, inference_params=None):
        residual = None
        for layer in self.layers:
            hidden_states, residual = layer(
                hidden_states, residual, inference_params=inference_params
            )
        if not self.fused_add_norm:
            residual = (hidden_states + residual) if residual is not None else hidden_states
            hidden_states = self.norm_f(residual.to(dtype=self.norm_f.weight.dtype))
        else:
            # Set prenorm=False here since we don't need the residual
            fused_add_norm_fn = rms_norm_fn if isinstance(self.norm_f, RMSNorm) else layer_norm_fn
            hidden_states = fused_add_norm_fn(
                hidden_states,
                self.norm_f.weight,
                self.norm_f.bias,
                eps=self.norm_f.eps,
                residual=residual,
                prenorm=False,
                residual_in_fp32=self.residual_in_fp32,
            )
        return hidden_states


class BidirMambaEncoder(nn.Module):
    def __init__(self, emb_dim, n_layer, pad_token_idx):
        super().__init__()
        self.pad_token_idx = pad_token_idx
        self.mamba_1 = MambaMixerModel(emb_dim, n_layer)
        self.mamba_2 = MambaMixerModel(emb_dim, n_layer)

    def forward(self, src_emb, src_emb_flipped):
        x = self.mamba_1(src_emb)
        x_flipped = self.mamba_2(src_emb_flipped)
        return (x + x_flipped) / 2


class MambaEncMambaTransformerDec(nn.Module):
    def __init__(self,
                 src_vocab_size: int,
                 tgt_vocab_size: int,
                 num_encoder_layers: int = 3,
                 num_decoder_layers: int = 3,
                 embedding_dim: int = 128,
                 num_heads: int = 4,
                 feedforward_dim: int = 256,
                 dropout_rate: float = 0.0,
                 activation: str = "relu",
                 share_embeddings: bool = False,
                 src_pad_token_idx: int = 0,
                 tgt_pad_token_idx: int = 0):
        super().__init__()
        self.src_vocab_size = src_vocab_size
        self.tgt_vocab_size = tgt_vocab_size
        self.src_pad_token_i = src_pad_token_idx
        self.tgt_pad_token_i = tgt_pad_token_idx

        self.num_enc_layers = num_encoder_layers
        self.num_dec_layers = num_decoder_layers
        self.emb_dim = embedding_dim
        self.num_heads = num_heads
        self.ff_dim = feedforward_dim
        self.dropout_rate = dropout_rate
        self.activation = activation
        self.share_embeddings = share_embeddings

        self.layer_norm_eps = 1e-5

    def create(self):
        self.src_token_featurizer = TokenEmbedding(self.src_vocab_size,
                                                   self.emb_dim, padding_idx=self.src_pad_token_i)
        if self.share_embeddings:
            self.tgt_token_featurizer = self.src_token_featurizer
            assert self.src_vocab_size == self.tgt_vocab_size
        else:
            self.tgt_token_featurizer = TokenEmbedding(self.tgt_vocab_size,
                                                       self.emb_dim, padding_idx=self.tgt_pad_token_i)

        # Embedding updater
        self.encoder = BidirMambaEncoder(self.emb_dim, self.num_enc_layers, self.src_pad_token_i)

        self.decoder = nn.ModuleList([
            MambaTransformerDecoderLayer(d_model=self.emb_dim,
                                         nhead=self.num_heads,
                                         dim_feedforward=self.ff_dim,
                                         dropout=self.dropout_rate,
                                         activation=self.activation,
                                         layer_norm_eps=self.layer_norm_eps,
                                         batch_first=True,
                                         norm_first=True,
                                         bias=True) for _ in
            range(self.num_dec_layers)
        ])

        # Decision function
        self.next_token_classifier = nn.Linear(self.emb_dim, self.tgt_vocab_size)

    def flip_keep_pad(self, arr):
        # TODO You should flip 3d embeddings, not 2d sequences
        dim_indices = torch.arange(arr.shape[1]).type_as(arr).long().repeat(arr.shape[0]).reshape(arr.shape[0], -1)
        eos_index = (arr == self.src_pad_token_i).sum(1)
        indices = (dim_indices - eos_index.unsqueeze(1)) % arr.shape[1]
        return torch.flip(torch.gather(arr, dim=1, index=indices), [1])

    def forward(self, src: LongTensor, tgt: LongTensor):
        _, tgt_seq_len = tgt.size()
        flipped_src = self.flip_keep_pad(src)

        # Embed tokens
        src_emb = self.src_token_featurizer(src)
        src_flipped_emb = self.src_token_featurizer(flipped_src)  # TODO It should be inside the encoder class
        tgt_emb = self.tgt_token_featurizer(tgt)

        # Update embeddings
        src_pad_mask = (src == self.src_pad_token_i).bool()
        memory = self.encoder(src_emb, src_flipped_emb)
        for dec_layer in self.decoder:
            tgt_emb = dec_layer(tgt_emb, memory,
                                memory_key_padding_mask=src_pad_mask)

        # Propose the next token
        logits = self.next_token_classifier(tgt_emb)
        return logits
