from torch import Tensor

from models.mamba_minimal.modules import EncoderDecoderMamba
from models.mamba_minimal.modules import DecoderOnlyMamba
from lightning_model_wrappers import TranslationModel


class EncoderDecoderMambaNaiveTranslationLightningModule(TranslationModel):

    def __init__(self,
                 src_vocab_size: int | None = None,  # Model size and architecture arguments
                 tgt_vocab_size: int | None = None,
                 num_encoder_layers: int = 3,
                 num_decoder_layers: int = 3,
                 embedding_dim: int = 128,
                 expand: int = 2,
                 d_conv: int = 4,
                 d_state: int = 16,
                 dt_rank: int | str = "auto",
                 bias: bool = False,
                 conv_bias: bool = True,

                 **kwargs  # Default arguments of the base class
                 ):
        super().__init__(**kwargs)

    def _create_model(self):
        self.model = EncoderDecoderMamba(self.hparams.src_vocab_size,
                                         self.hparams.tgt_vocab_size,
                                         self.hparams.num_encoder_layers,
                                         self.hparams.num_decoder_layers,
                                         self.hparams.embedding_dim,
                                         self.hparams.expand,
                                         self.hparams.d_conv,
                                         self.hparams.d_state,
                                         self.hparams.dt_rank,
                                         self.hparams.bias,
                                         self.hparams.conv_bias)
        self.model.create()

    def forward(self, batch: dict[str, Tensor]) -> Tensor:
        source_tokens = batch["src_tokens"]
        target_tokens = batch["tgt_tokens"]
        src_flip_ids = batch["src_flip_ids"]
        target_given = target_tokens[:, :-1]
        return self.model(source_tokens, target_given, src_flip_ids)
