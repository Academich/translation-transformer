from typing import Optional, List, Any, Type
import json

import torch
from torch import nn
from torch import optim

from pytorch_lightning import LightningModule
from pytorch_lightning.utilities.types import STEP_OUTPUT

from models import VanillaTransformer
from translators import TranslationInferenceBeamSearch, TranslationInferenceGreedy
from utils import NoamLRSchedule, ConstantLRSchedule, calc_token_acc, calc_sequence_acc


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

                 learning_rate: float = 3e-4,  # Optimization arguments
                 weight_decay: float = 0.,
                 scheduler: str = "const",
                 warmup_steps: int = 0,

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

    def _calc_loss(self, logits, tgt_ids):
        return self.criterion(logits.reshape(-1, self.hparams.tgt_vocab_size),
                              tgt_ids.reshape(-1))

    def training_step(self, batch, batch_idx) -> STEP_OUTPUT:
        source, target = batch

        # We predict the next token given the previous ones
        target_given = target[:, :-1]
        target_future = target[:, 1:]
        pred_logits = self.__call__(source, target_given)
        loss = self._calc_loss(pred_logits, target_future)

        pred_tokens = torch.argmax(pred_logits, dim=2)
        token_acc = calc_token_acc(pred_tokens, target_future)
        sequence_acc = calc_sequence_acc(pred_tokens, target_future, self.hparams.eos_token_idx)
        mean_pad_tokens_in_target = (target_future == self.hparams.pad_token_idx).float().mean()

        self.log(f"train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log(f"train/acc_single_tok", token_acc, on_step=True, on_epoch=True, prog_bar=True)
        self.log(f"train/acc_sequence", sequence_acc, on_step=True, on_epoch=True, prog_bar=False)
        self.log(f"train/pads_in_batch_tgt", mean_pad_tokens_in_target, on_step=True, on_epoch=True, prog_bar=False)
        return loss

    def validation_step(self, batch, batch_idx):
        source, target = batch

        target_given = target[:, :-1]
        target_future = target[:, 1:]
        pred_logits = self.__call__(source, target_given)

        loss = self._calc_loss(pred_logits, target_future)

        pred_tokens = torch.argmax(pred_logits, dim=2)
        token_acc = calc_token_acc(pred_tokens, target_future)
        sequence_acc = calc_sequence_acc(pred_tokens, target_future, self.hparams.eos_token_idx)

        self.log(f"val/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log(f"val/acc_single_tok", token_acc, on_step=False, on_epoch=True, prog_bar=False)
        self.log(f"val/acc_sequence", sequence_acc, on_step=False, on_epoch=True, prog_bar=False)
        self.validation_step_outputs.append({"pred_tokens": pred_tokens, "target_ahead": target_future})

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
        optimizer = optim.Adam(self.parameters(),
                               lr=self.hparams.learning_rate,
                               weight_decay=self.hparams.weight_decay,
                               betas=(0.9, 0.999))

        sched_name = self.hparams.scheduler
        lr = self.hparams.learning_rate
        ws = self.hparams.warmup_steps
        if sched_name == "const":
            if ws == 0:  # TODO Ugly crutch
                return optimizer
            scheduler = {
                "scheduler": optim.lr_scheduler.LambdaLR(
                    optimizer,
                    ConstantLRSchedule(lr, ws)  # TODO does not work idk
                ),
                "name": "Constant LR scheduler",
                "interval": "step",
                "frequency": 1,
            }
        elif sched_name == "noam":
            scheduler = {
                "scheduler": optim.lr_scheduler.LambdaLR(
                    optimizer,
                    NoamLRSchedule(self.hparams.embedding_dim, lr, ws)
                ),
                "name": "Noam scheduler",
                "interval": "step",
                "frequency": 1,
            }
        else:
            raise ValueError(f'Unknown scheduler name {self.hparams.scheduler}. Options are "const", "noam".')

        return [optimizer], [scheduler]
