from typing import Any

import torch
from torch import nn
from torch import optim

from pytorch_lightning import LightningModule
from pytorch_lightning.utilities.types import STEP_OUTPUT

from translators import TranslationInferenceBeamSearch, TranslationInferenceGreedy
from utils import NoamLRSchedule, ConstantLRSchedule, calc_token_acc, calc_sequence_acc


class TranslationModel(LightningModule):

    def __init__(self,
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
        self.criterion = nn.CrossEntropyLoss(reduction="mean")

        self.model: nn.Module | None = None
        self._create_model()
        assert self.model is not None, \
            f"Override the _create_model method in {self.__class__} to assign an nn.Module to self.model"

        self._create_generator()

        self.validation_step_outputs = []

    def _create_model(self):
        raise NotImplementedError

    def _create_generator(self):
        if self.hparams.generation == "beam_search":
            self.generator = TranslationInferenceBeamSearch(self.model,
                                                            self.hparams.beam_size,
                                                            self.hparams.max_len,
                                                            self.hparams.pad_token_idx,
                                                            self.hparams.bos_token_idx,
                                                            self.hparams.eos_token_idx)
        elif self.hparams.generation == "greedy":
            self.generator = TranslationInferenceGreedy(self.model,
                                                        self.hparams.max_len,
                                                        self.hparams.pad_token_idx,
                                                        self.hparams.bos_token_idx,
                                                        self.hparams.eos_token_idx)
        else:
            raise ValueError(
                f'Unknown generation option {self.hparams.generation}. Options are "beam_search", "greedy".')

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """
        The model receives source token indices and target token indices,
        and possibly some masks or any other necessary information.
        The output is the predicted next token probability distribution,
        a tensor of shape BATCH_SIZE x SEQUENCE_LENGTH x TARGET_VOCABULARY_SIZE
        """
        raise NotImplementedError

    def _calc_loss(self, logits, tgt_ids):
        _, _, tgt_vocab_size = logits.size()
        return self.criterion(logits.reshape(-1, tgt_vocab_size),
                              tgt_ids.reshape(-1))

    def training_step(self, batch, batch_idx) -> STEP_OUTPUT:
        pred_logits = self.__call__(batch)

        # We predict the next token given the previous ones
        target_future = batch["tgt_tokens"][:, 1:]

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
        pred_logits = self.__call__(batch)

        # We predict the next token given the previous ones
        target_future = batch["tgt_tokens"][:, 1:]

        loss = self._calc_loss(pred_logits, target_future)

        pred_tokens = torch.argmax(pred_logits, dim=2)
        token_acc = calc_token_acc(pred_tokens, target_future)
        sequence_acc = calc_sequence_acc(pred_tokens, target_future, self.hparams.eos_token_idx)

        self.log(f"val/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log(f"val/acc_single_tok", token_acc, on_step=False, on_epoch=True, prog_bar=False)
        self.log(f"val/acc_sequence", sequence_acc, on_step=False, on_epoch=True, prog_bar=False)
        self.validation_step_outputs.append({"pred_tokens": pred_tokens, "target_ahead": target_future})

    def test_step(self, batch, batch_idx) -> STEP_OUTPUT | None:
        pred_logits = self.__call__(batch)

        source = batch["src_tokens"]
        target = batch["tgt_tokens"]
        target_future = batch["tgt_tokens"][:, 1:]

        loss = self._calc_loss(pred_logits, target_future)
        self.log(f"test/loss", loss, on_step=False, on_epoch=True)

        return {"source_token_ids": source, "pred_logits": pred_logits, "target_token_ids": target}

    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> Any:
        source = batch["src_tokens"]
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
            scheduler = {
                "scheduler": optim.lr_scheduler.LambdaLR(
                    optimizer,
                    ConstantLRSchedule(lr, ws)
                ),
                "name": "Constant LR scheduler",
                "interval": "step",
                "frequency": 1,
            }
        elif sched_name == "noam":
            d = self.model.emb_dim  # May fail if the model does not have an 'emb_dim' attribute
            scheduler = {
                "scheduler": optim.lr_scheduler.LambdaLR(
                    optimizer,
                    NoamLRSchedule(d, lr, ws)
                ),
                "name": "Noam scheduler",
                "interval": "step",
                "frequency": 1,
            }
        else:
            raise ValueError(f'Unknown scheduler name {self.hparams.scheduler}. Options are "const", "noam".')

        return [optimizer], [scheduler]
