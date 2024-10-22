import torch

from models.modules import VanillaTransformer
from lightning_model_wrappers import TranslationModel


class VanillaTransformerTranslationLightningModule(TranslationModel):

    def __init__(self,
                 embedding_dim: int = 128,
                 feedforward_dim: int = 256,
                 num_encoder_layers: int = 3,
                 num_decoder_layers: int = 3,
                 num_heads: int = 4,
                 dropout_rate: float = 0.0,
                 activation: str = "relu",
                 share_embeddings: bool = False,

                 **kwargs  # Default arguments of the base class
                 ):
        super().__init__(**kwargs)

    def _create_model(self) -> torch.nn.Module:
        model = VanillaTransformer(self.src_vocab_size,
                                   self.tgt_vocab_size,
                                   self.hparams.num_encoder_layers,
                                   self.hparams.num_decoder_layers,
                                   self.hparams.embedding_dim,
                                   self.hparams.num_heads,
                                   self.hparams.feedforward_dim,
                                   self.hparams.dropout_rate,
                                   self.hparams.activation,
                                   self.hparams.share_embeddings,
                                   self.src_pad_token_i,
                                   self.tgt_pad_token_i)
        return model

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        return self.model(batch["src_tokens"],
                          batch["tgt_tokens"][:, :-1])
