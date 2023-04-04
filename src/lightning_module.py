from typing import Optional, List, Any, Type
import json

import torch
from torch import nn
from torch import optim

from pytorch_lightning import LightningModule
from pytorch_lightning.utilities.types import STEP_OUTPUT

from src.wrappers import TokenVocabulary
from src.translation.translators import TranslationInferenceBeamSearch, TranslationInferenceGreedy


class TextTranslationTransformer(LightningModule):
    module_class: Type[nn.Module] = None

    def __init__(self,
                 src_vocab_path: str,
                 tgt_vocab_path: str,
                 learning_rate: float = 3e-4,
                 warmup_steps: int = 200,
                 generation: str = "beam_search",
                 beam_size: int = 1,
                 max_len: int = 100,
                 num_encoder_layers: int = 3,
                 num_decoder_layers: int = 3,
                 embedding_dim: int = 128,
                 num_heads: int = 4,
                 feedforward_dim: int = 256,
                 dropout_rate: float = 0.0,
                 activation: str = "relu"):
        super().__init__()
        self.save_hyperparameters()  # The hyperparameters are saved to the “hyper_parameters” key in the checkpoint
        self.validation_step_outputs = []

    def setup(self, stage: str) -> None:
        # Creating the model here instead of in __init__ to allow
        # the data module to execute .prepare_data first
        self._load_vocabularies()
        self._create_model(**self.hparams)
        self._create_generator()

    def _load_vocabularies(self):
        with open(self.hparams.src_vocab_path) as fs, open(self.hparams.tgt_vocab_path) as ft:
            self.src_vocab = TokenVocabulary(json.load(fs))
            self.tgt_vocab = TokenVocabulary(json.load(ft))

    def _create_model(self, *args, **kwargs):
        self.model = self.module_class(*args, **kwargs)
        self.model.src_vocab_len = self.src_vocab.n_tokens
        self.model.tgt_vocab_len = self.tgt_vocab.n_tokens
        self.model.pad_token_idx = self.src_vocab.pad_token_idx
        self.model.create()
        self.criterion = nn.CrossEntropyLoss(reduction="mean")

    def _create_generator(self):
        self.generator = TranslationInferenceBeamSearch(self.model,
                                                        self.hparams.beam_size,
                                                        self.hparams.max_len,
                                                        self.tgt_vocab.pad_token_idx,
                                                        self.tgt_vocab.bos_token_idx,
                                                        self.tgt_vocab.eos_token_idx)

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        return self.model(*args, **kwargs)

    def training_step(self, batch, batch_idx) -> STEP_OUTPUT:
        source, target = batch

        # We predict the next token given the previous ones
        target_behind = target[:, :-1]
        target_ahead = target[:, 1:]
        pred_logits = self(source, target_behind)

        loss = self.criterion(pred_logits.reshape(-1, self.tgt_vocab.n_tokens),
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
        pad_tokens_in_batch_target = (target_sequence == self.tgt_vocab.pad_token_idx)
        self.log(f"train/pads_in_batch_tgt", pad_tokens_in_batch_target.float().mean(), on_step=True, on_epoch=True,
                 reduce_fx='mean')

        # Mean number of EOS tokens in predictions in a batch
        eos_in_pred_batch = (pred_tokens == self.tgt_vocab.eos_token_idx)
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
        pad_tokens_in_batch_target = (target_sequence == self.tgt_vocab.pad_token_idx)
        self.log(f"val/pads_in_batch_tgt", pad_tokens_in_batch_target.float().mean(), on_step=True, on_epoch=True,
                 reduce_fx='mean')

        # Mean number of EOS tokens in predictions in a batch
        eos_in_pred_batch = (predicted_tokens == self.tgt_vocab.eos_token_idx)
        self.log(f"val/eos_in_batch_pred", eos_in_pred_batch.float().mean(), on_step=True, on_epoch=True,
                 reduce_fx='mean')

    def validation_step(self, batch, batch_idx) -> Optional[STEP_OUTPUT]:
        source, target = batch

        target_behind = target[:, :-1]
        target_ahead = target[:, 1:]
        pred_logits = self(source, target_behind)

        val_loss = self.criterion(pred_logits.reshape(-1, self.tgt_vocab.n_tokens),
                                  target_ahead.reshape(-1))
        pred_tokens = torch.argmax(pred_logits, dim=2)
        self._log_validation_step(val_loss, pred_tokens, target_ahead)
        self.validation_step_outputs.append({"pred_tokens": pred_tokens, "target_ahead": target_ahead})

    def on_validation_epoch_end(self) -> None:
        total_correct, total = 0, 0
        for o in self.validation_step_outputs:
            pred_tokens = o["pred_tokens"].cpu()
            target_ahead = o["target_ahead"].cpu()
            b_size = pred_tokens.size()[0]
            for i in range(b_size):
                target_str = self.tgt_vocab.decode(target_ahead[i])
                predicted_str = self.tgt_vocab.decode(pred_tokens[i])
                total_correct += int(predicted_str == target_str)
                total += 1
        self.log("val/whole_seq_exact_match_acc_total", total_correct / total)
        self.validation_step_outputs.clear()

    def test_step(self, batch, batch_idx) -> Optional[STEP_OUTPUT]:
        source, target = batch

        target_behind = target[:, :-1]
        target_ahead = target[:, 1:]
        pred_logits = self(source, target_behind)

        test_loss = self.criterion(pred_logits.reshape(-1, self.tgt_vocab.n_tokens),
                                   target_ahead.reshape(-1))
        self.log(f"test/loss", test_loss, on_step=False, on_epoch=True, reduce_fx='mean')

        return {"source_token_ids": source, "pred_logits": pred_logits, "target_token_ids": target}

    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> Any:
        source, _ = batch
        generated = self.generator.generate(source)
        return generated

    def configure_optimizers(self):
        d = self.model.emb_dim
        optimizer = optim.Adam(self.parameters(),
                               lr=self.hparams.learning_rate)
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
