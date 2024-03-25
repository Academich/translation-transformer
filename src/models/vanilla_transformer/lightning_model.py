from torch import Tensor

from models.vanilla_transformer import VanillaTransformer
from lightning_model_wrappers import TranslationModel


class VanillaTransformerTranslationLightningModule(TranslationModel):

    def __init__(self,
                 src_vocab_size: int | None = None,  # Model size and architecture arguments
                 tgt_vocab_size: int | None = None,
                 embedding_dim: int = 128,
                 feedforward_dim: int = 256,
                 num_encoder_layers: int = 3,
                 num_decoder_layers: int = 3,
                 num_heads: int = 4,
                 dropout_rate: float = 0.0,
                 activation: str = "relu",

                 **kwargs  # Default arguments of the base class
                 ):
        super().__init__(**kwargs)

    def _create_model(self) -> None:
        self.model = VanillaTransformer(self.hparams.src_vocab_size,
                                        self.hparams.tgt_vocab_size,
                                        self.hparams.num_encoder_layers,
                                        self.hparams.num_decoder_layers,
                                        self.hparams.embedding_dim,
                                        self.hparams.num_heads,
                                        self.hparams.feedforward_dim,
                                        self.hparams.dropout_rate,
                                        self.hparams.activation,
                                        self.hparams.pad_token_idx)
        self.model.create()

    def forward(self, batch: dict[str, Tensor]) -> Tensor:
        source_tokens = batch["src_tokens"]
        target_tokens = batch["tgt_tokens"]
        target_given = target_tokens[:, :-1]
        return self.model(source_tokens, target_given)
