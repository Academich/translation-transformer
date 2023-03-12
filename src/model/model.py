from typing import Tuple, Optional

from torch import LongTensor, Tensor
from torch import nn

from src.model.featurization.embeddings import TokenEmbedding
from src.model.featurization.embeddings import PositionalEncoding


class VanillaTransformer(nn.Module):

    def __init__(self,
                 num_encoder_layers: int = 3,
                 num_decoder_layers: int = 3,
                 embedding_dim: int = 128,
                 num_heads: int = 4,
                 feedforward_dim: int = 256,
                 dropout_rate: float = 0.0,
                 activation: str = "relu"):
        super().__init__()
        self.src_vocab_len: Optional[int] = None
        self.tgt_vocab_len: Optional[int] = None
        self.pad_token_idx: Optional[int] = None

        self.num_enc_layers = num_encoder_layers
        self.num_dec_layers = num_decoder_layers
        self.emb_dim = embedding_dim
        self.num_heads = num_heads
        self.ff_dim = feedforward_dim
        self.dropout_rate = dropout_rate
        self.activation = activation

    def create(self):
        # Embedding constructor
        self.src_token_featurizer = TokenEmbedding(self.src_vocab_len,
                                                   self.emb_dim)

        self.tgt_token_featurizer = TokenEmbedding(self.tgt_vocab_len,
                                                   self.emb_dim)

        self.positional_encoding = PositionalEncoding(self.emb_dim)

        # Embedding updater

        self.transformer = nn.Transformer(d_model=self.emb_dim,
                                          nhead=self.num_heads,
                                          num_encoder_layers=self.num_enc_layers,
                                          num_decoder_layers=self.num_dec_layers,
                                          dim_feedforward=self.ff_dim,
                                          dropout=self.dropout_rate,
                                          activation=self.activation,
                                          batch_first=True)

        # Decision function
        self.next_token_classifier = nn.Linear(self.emb_dim, self.tgt_vocab_len)

    def _featurize(self, src: LongTensor, tgt: LongTensor) -> Tuple[Tensor, Tensor]:
        src_emb = self.positional_encoding(self.src_token_featurizer(src))
        tgt_emb = self.positional_encoding(self.tgt_token_featurizer(tgt))
        return src_emb, tgt_emb

    def _update_embeddings(self,
                           src_emb: Tensor,
                           tgt_emb: Tensor,
                           src: LongTensor,
                           tgt: LongTensor) -> Tensor:
        _, tgt_seq_len, _ = tgt_emb.size()
        tgt_mask = self.transformer.generate_square_subsequent_mask(tgt_seq_len).type_as(tgt_emb)
        src_pad_mask = (src == self.pad_token_idx).bool()
        tgt_pad_mask = (tgt == self.pad_token_idx).bool()

        target_token_updated_emb = self.transformer(src_emb,
                                                    tgt_emb,
                                                    src_mask=None,
                                                    tgt_mask=tgt_mask,
                                                    memory_mask=None,
                                                    src_key_padding_mask=src_pad_mask,
                                                    tgt_key_padding_mask=tgt_pad_mask,
                                                    memory_key_padding_mask=src_pad_mask)
        return target_token_updated_emb

    def _decision(self, tgt_emb: Tensor) -> Tensor:
        logits = self.next_token_classifier(tgt_emb)
        return logits

    def forward(self, src: LongTensor, tgt: LongTensor):
        src_emb, tgt_emb = self._featurize(src, tgt)
        upd_tgt_emb = self._update_embeddings(src_emb, tgt_emb, src, tgt)
        return self._decision(upd_tgt_emb)
