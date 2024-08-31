from torch import LongTensor, BoolTensor, Tensor
from torch import nn

from models.embeddings import TokenEmbedding, PositionalEncoding


class VanillaTransformer(nn.Module):

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
                 tgt_pad_token_idx: int = 0
                 ):
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

    def create(self):
        # Embedding constructor
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

        self.transformer = nn.Transformer(d_model=self.emb_dim,
                                          nhead=self.num_heads,
                                          num_encoder_layers=self.num_enc_layers,
                                          num_decoder_layers=self.num_dec_layers,
                                          dim_feedforward=self.ff_dim,
                                          dropout=self.dropout_rate,
                                          activation=self.activation,
                                          batch_first=True)

        # Decision function
        self.next_token_classifier = nn.Linear(self.emb_dim, self.tgt_vocab_size)

    def forward(self, src: LongTensor, tgt: LongTensor):
        _, tgt_seq_len = tgt.size()

        # Embed tokens
        src_emb = self.positional_encoding(self.src_token_featurizer(src))
        tgt_emb = self.positional_encoding(self.tgt_token_featurizer(tgt))

        # Update embeddings
        src_pad_mask: torch.Tensor = torch.where(src != self.src_pad_token_i, torch.tensor(0.0), torch.tensor(float('-inf')))
        tgt_pad_mask: torch.Tensor = torch.where(tgt != self.tgt_pad_token_i, torch.tensor(0.0), torch.tensor(float('-inf')))
        tgt_mask = self.transformer.generate_square_subsequent_mask(tgt_seq_len).type_as(tgt_emb)
        tgt_emb = self.transformer(src_emb,
                                   tgt_emb,
                                   src_mask=None,
                                   tgt_mask=tgt_mask,
                                   memory_mask=None,
                                   src_key_padding_mask=src_pad_mask,
                                   tgt_key_padding_mask=tgt_pad_mask,
                                   memory_key_padding_mask=src_pad_mask)

        # Propose the next token
        logits = self.next_token_classifier(tgt_emb)
        return logits

    def encode_src(self, src: LongTensor, src_pad_mask: BoolTensor):
        # Embed tokens
        src_emb = self.positional_encoding(self.src_token_featurizer(src))

        # Update embeddings
        src_emb = self.transformer.encoder(src_emb, src_key_padding_mask=src_pad_mask)
        return src_emb

    def decode_tgt(self, tgt: LongTensor, memory: Tensor, memory_pad_mask: BoolTensor,
                   pos_enc_offset: torch.LongTensor = torch.LongTensor([0])):
        _, tgt_seq_len = tgt.size()

        # Embed tokens
        tgt_emb = self.tgt_token_featurizer(tgt)
        tgt_emb = self.positional_encoding(tgt_emb, offset=pos_enc_offset)

        # Update embeddings
        tgt_pad_mask = torch.where(tgt != self.tgt_pad_token_i, torch.tensor(0.0), torch.tensor(float('-inf')))
        tgt_mask = self.transformer.generate_square_subsequent_mask(tgt_seq_len).type_as(tgt_emb)
        tgt_emb = self.transformer.decoder(tgt_emb,
                                           memory,
                                           tgt_mask=tgt_mask,
                                           tgt_key_padding_mask=tgt_pad_mask,
                                           memory_key_padding_mask=memory_pad_mask
                                           )

        # Propose the next token
        logits = self.next_token_classifier(tgt_emb)
        return logits
