from typing import Optional, List, Any, Type
import json

import torch
from torch import nn
from torch import optim

from pytorch_lightning import LightningModule
from pytorch_lightning.utilities.types import STEP_OUTPUT

from models import VanillaTransformer
from translators import TranslationInferenceBeamSearch, TranslationInferenceGreedy


class VanillaTextTranslationTransformer(LightningModule):

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

                 pad_token_idx: int = 0,  # Service token indices
                 bos_token_idx: int = 1,
                 eos_token_idx: int = 2,

                 learning_rate: float = 0.1,  # Optimization arguments
                 warmup_steps: int | None = None,

                 generation: str = "beam_search",  # Prediction generation arguments
                 beam_size: int = 1,
                 max_len: int = 100
                 ):
        super().__init__()
        self.save_hyperparameters()
        self._create_model()
        self._create_generator()

        self.validation_step_outputs = []

    def _create_model(self):
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
        self.criterion = nn.CrossEntropyLoss(reduction="mean")

    def _create_generator(self):
        self.generator = TranslationInferenceBeamSearch(self.model,
                                                        self.hparams.beam_size,
                                                        self.hparams.max_len,
                                                        self.hparams.pad_token_idx,
                                                        self.hparams.bos_token_idx,
                                                        self.hparams.eos_token_idx)

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        return self.model(*args, **kwargs)

    def training_step(self, batch, batch_idx) -> STEP_OUTPUT:
        source, target = batch

        # We predict the next token given the previous ones
        target_behind = target[:, :-1]
        target_ahead = target[:, 1:]
        pred_logits = self(source, target_behind)

        loss = self.criterion(pred_logits.reshape(-1, self.hparams.tgt_vocab_size),
                              target_ahead.reshape(-1))
        self._log_training_step(loss, pred_logits, target_ahead)
        return loss

    def _log_training_step(self, loss, predicted_logits, target_sequence):
        self.log(f"train/loss", loss, on_step=True, on_epoch=True, reduce_fx='mean', prog_bar=True)

        # Single token prediction accuracy
        pred_tokens = torch.argmax(predicted_logits, dim=2)
        single_tokens_predicted_right = (pred_tokens == target_sequence).float()  # TODO Beware of EOS != PAD
        single_token_pred_acc = single_tokens_predicted_right.mean()
        self.log(f"train/acc_single_tok",
                 single_token_pred_acc, on_step=True, on_epoch=True, reduce_fx='mean', prog_bar=True)

        # Mean number of pad tokens in a batch
        pad_tokens_in_batch_target = (target_sequence == self.hparams.pad_token_idx)
        self.log(f"train/pads_in_batch_tgt", pad_tokens_in_batch_target.float().mean(), on_step=True, on_epoch=True,
                 reduce_fx='mean')

        # Mean number of EOS tokens in predictions in a batch
        eos_in_pred_batch = (pred_tokens == self.hparams.eos_token_idx)
        self.log(f"train/eos_in_batch_pred", eos_in_pred_batch.float().mean(), on_step=True, on_epoch=True,
                 reduce_fx='mean')

    def _log_validation_step(self, val_loss, predicted_tokens, target_sequence):
        self.log(f"val/loss", val_loss, on_step=True, on_epoch=True, reduce_fx='mean')

        # Single token prediction accuracy
        single_tokens_predicted_right = (predicted_tokens == target_sequence).float()  # TODO Beware of EOS != PAD
        single_token_pred_acc = single_tokens_predicted_right.mean()  # TODO Correct by pad tokens
        self.log(f"train/acc_single_tok", single_token_pred_acc, on_step=True, on_epoch=True, reduce_fx='mean')

        # Number of tokens in batch
        self.log(f"val/tokens_in_batch", target_sequence.shape[0] * (target_sequence.shape[1] + 1), on_step=True,
                 on_epoch=False, reduce_fx='mean')

        # Mean number of pad tokens in a batch
        pad_tokens_in_batch_target = (target_sequence == self.hparams.pad_token_idx)
        self.log(f"val/pads_in_batch_tgt", pad_tokens_in_batch_target.float().mean(), on_step=True, on_epoch=True,
                 reduce_fx='mean')

        # Mean number of EOS tokens in predictions in a batch
        eos_in_pred_batch = (predicted_tokens == self.hparams.eos_token_idx)
        self.log(f"val/eos_in_batch_pred", eos_in_pred_batch.float().mean(), on_step=True, on_epoch=True,
                 reduce_fx='mean')

    def validation_step(self, batch, batch_idx) -> Optional[STEP_OUTPUT]:
        source, target = batch

        target_behind = target[:, :-1]
        target_ahead = target[:, 1:]
        pred_logits = self(source, target_behind)

        val_loss = self.criterion(pred_logits.reshape(-1, self.hparams.tgt_vocab_size),
                                  target_ahead.reshape(-1))
        pred_tokens = torch.argmax(pred_logits, dim=2)
        self._log_validation_step(val_loss, pred_tokens, target_ahead)
        self.validation_step_outputs.append({"pred_tokens": pred_tokens, "target_ahead": target_ahead})

    def test_step(self, batch, batch_idx) -> Optional[STEP_OUTPUT]:
        source, target = batch

        target_behind = target[:, :-1]
        target_ahead = target[:, 1:]
        pred_logits = self(source, target_behind)

        test_loss = self.criterion(pred_logits.reshape(-1, self.hparams.tgt_vocab_size),
                                   target_ahead.reshape(-1))
        self.log(f"test/loss", test_loss, on_step=False, on_epoch=True, reduce_fx='mean')

        return {"source_token_ids": source, "pred_logits": pred_logits, "target_token_ids": target}

    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> Any:
        source, _ = batch
        generated = self.generator.generate(source, n_best=self.hparams.beam_size)
        return generated

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        if self.hparams.warmup_steps is None:
            return optimizer

        d = self.model.emb_dim
        scheduler = {
            "scheduler": optim.lr_scheduler.LambdaLR(
                optimizer,
                lambda i: d ** (-0.5) * min((i + 1) ** (-0.5),
                                            (i + 1) * self.hparams.warmup_steps ** (-1.5))
            ),
            "name": "Noam scheduler",
            "interval": "step",
            "frequency": 1,
        }
        return [optimizer], [scheduler]
