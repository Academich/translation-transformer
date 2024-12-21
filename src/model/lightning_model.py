import datetime
from typing import Any
from timeit import default_timer as timer
import json
from pathlib import Path

import torch
from torch import nn
from torch import optim

from pytorch_lightning import LightningModule
from pytorch_lightning.utilities.types import STEP_OUTPUT

from data_handling.tokenizer_base import GenericTokenizer
from decoding.standard_decoding import TranslationInferenceGreedy, TranslationInferenceBeamSearch
from decoding.speculative_decoding import TranslationInferenceGreedySpeculative, TranslationInferenceBeamSearchSpeculativeBatchedWithoutLeftPads
from model.modules import VanillaTransformer
from utils.lr_schedules import NoamLRSchedule, ConstantLRSchedule
from utils.metrics import calc_token_acc, calc_sequence_acc


class VanillaEncoderDecoderTransformerLightning(LightningModule):

    def __init__(self,
                 src_tokenizer: GenericTokenizer | None = None,  # Tokenizer objects
                 tgt_tokenizer: GenericTokenizer | None = None,

                 embedding_dim: int = 128,  # Model arguments
                 feedforward_dim: int = 256,
                 num_encoder_layers: int = 3,
                 num_decoder_layers: int = 3,
                 num_heads: int = 4,
                 dropout_rate: float = 0.0,
                 activation: str = "relu",
                 share_embeddings: bool = False,

                 learning_rate: float = 3e-4,  # Optimization arguments
                 weight_decay: float = 0.,
                 scheduler: str = "const",
                 warmup_steps: int = 0,

                 generation: str = "beam_search",  # Prediction generation arguments
                 beam_size: int = 1,
                 n_best: int = 1,
                 max_len: int = 100,
                 n_drafts: int = 0,
                 draft_len: int = 0,
                 report_prediction_time: bool = True,
                 report_prediction_file: str | None = None
                 ):
        super().__init__()
        self.save_hyperparameters(ignore=["src_tokenizer", "tgt_tokenizer"])

        assert src_tokenizer is not None, "source tokenizer not provided"
        assert tgt_tokenizer is not None, "target tokenizer not provided"
        self.src_tokenizer = src_tokenizer
        self.tgt_tokenizer = tgt_tokenizer
        self.src_vocab_size: int = src_tokenizer.n_tokens
        self.tgt_vocab_size: int = tgt_tokenizer.n_tokens
        self.src_pad_token_i: int = src_tokenizer.pad_token_idx
        self.src_bos_token_i: int = src_tokenizer.bos_token_idx
        self.src_eos_token_i: int = src_tokenizer.eos_token_idx
        self.tgt_pad_token_i: int = tgt_tokenizer.pad_token_idx
        self.tgt_bos_token_i: int = tgt_tokenizer.bos_token_idx
        self.tgt_eos_token_i: int = tgt_tokenizer.eos_token_idx

        self.criterion = nn.CrossEntropyLoss(reduction="mean")

        self.model = self._create_model()

        self.generator = self._create_generator()
        print(self.generator)

        self.report_prediction_time = report_prediction_time
        self.prediction_start_time = None

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
    def _create_generator(self):
        if self.hparams.generation == "greedy":
            return TranslationInferenceGreedy(self.model,
                                                        max_len=self.hparams.max_len,
                                                        pad_token=self.tgt_pad_token_i,
                                                        bos_token=self.tgt_bos_token_i,
                                                        eos_token=self.tgt_eos_token_i)
        elif self.hparams.generation == "beam_search":
            return TranslationInferenceBeamSearch(self.model,
                                                            beam_size=self.hparams.beam_size,
                                                            max_len=self.hparams.max_len,
                                                            pad_token=self.tgt_pad_token_i,
                                                            bos_token=self.tgt_bos_token_i,
                                                            eos_token=self.tgt_eos_token_i)

        elif self.hparams.generation == "greedy_speculative":
            assert self.hparams.draft_len > 0, "Number of speculative tokens must be a positive integer."
            return TranslationInferenceGreedySpeculative(
                self.model,
                max_len=self.hparams.max_len,
                draft_len=self.hparams.draft_len,
                n_drafts=self.hparams.n_drafts,
                pad_token=self.tgt_pad_token_i,
                bos_token=self.tgt_bos_token_i,
                eos_token=self.tgt_eos_token_i,
                replace_token=self.tgt_tokenizer.encoder_dict["c"]
            )
        elif self.hparams.generation == "beam_search_speculative":
            return TranslationInferenceBeamSearchSpeculativeBatchedWithoutLeftPads(
                self.model,
                vocab_size=self.tgt_vocab_size,
                max_len=self.hparams.max_len,
                n_best=self.hparams.n_best,
                draft_len=self.hparams.draft_len,
                n_drafts=self.hparams.n_drafts,
                pad_token=self.tgt_pad_token_i,
                bos_token=self.tgt_bos_token_i,
                eos_token=self.tgt_eos_token_i,
                C_token=self.tgt_tokenizer.encoder_dict["c"]
            )

        else:
            options = ", ".join(["beam_search", "greedy", "greedy_speculative", "beam_search_speculative"])
            raise ValueError(
                f'Unknown generation option {self.hparams.generation}. Options are {options}.')

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """
        The model receives source token indices and target token indices,
        and possibly some masks or any other necessary information.
        The output is the predicted next token probability distribution,
        a tensor of shape BATCH_SIZE x SEQUENCE_LENGTH x TARGET_VOCABULARY_SIZE
        """
        return self.model(batch["src_tokens"],
                          batch["tgt_tokens"][:, :-1])


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
        sequence_acc = calc_sequence_acc(pred_tokens, target_future, self.tgt_eos_token_i)
        mean_pad_tokens_in_target = (target_future == self.tgt_pad_token_i).float().mean()

        self.log(f"train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log(f"train/acc_single_tok", token_acc, on_step=True, on_epoch=True, prog_bar=True)
        self.log(f"train/acc_sequence", sequence_acc, on_step=True, on_epoch=True, prog_bar=False)
        self.log(f"train/pads_in_batch_tgt", mean_pad_tokens_in_target, on_step=True, on_epoch=True, prog_bar=False)
        return loss

    def validation_step(self, batch, batch_idx) -> STEP_OUTPUT:
        pred_logits = self.__call__(batch)

        # We predict the next token given the previous ones
        target_future = batch["tgt_tokens"][:, 1:]

        loss = self._calc_loss(pred_logits, target_future)

        pred_tokens = torch.argmax(pred_logits, dim=2)
        token_acc = calc_token_acc(pred_tokens, target_future)
        sequence_acc = calc_sequence_acc(pred_tokens, target_future, self.tgt_eos_token_i)

        self.log(f"val/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log(f"val/acc_single_tok", token_acc, on_step=False, on_epoch=True, prog_bar=False)
        self.log(f"val/acc_sequence", sequence_acc, on_step=False, on_epoch=True, prog_bar=False)
        return {"pred_tokens": pred_tokens, "target_ahead": target_future}

    def test_step(self, batch, batch_idx) -> STEP_OUTPUT:
        pred_logits = self.__call__(batch)

        source = batch["src_tokens"]
        target = batch["tgt_tokens"]
        target_future = batch["tgt_tokens"][:, 1:]

        loss = self._calc_loss(pred_logits, target_future)
        pred_tokens = torch.argmax(pred_logits, dim=2)
        token_acc = calc_token_acc(pred_tokens, target_future)
        sequence_acc = calc_sequence_acc(pred_tokens, target_future, self.tgt_eos_token_i)

        self.log(f"test/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log(f"test/acc_single_tok", token_acc, on_step=False, on_epoch=True, prog_bar=False)
        self.log(f"test/acc_sequence", sequence_acc, on_step=False, on_epoch=True, prog_bar=False)

        return {"source_token_ids": source, "pred_logits": pred_logits, "target_token_ids": target}

    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> Any:
        source = batch["src_tokens"]
        generated = self.generator.generate(source)
        return generated

    def on_predict_start(self) -> None:
        if self.report_prediction_time:
            self.prediction_start_time = timer()
    
    def on_predict_end(self) -> None:
        elapsed = datetime.timedelta(seconds=timer() - self.prediction_start_time)
        report = {
                "algorithm": self.hparams.generation,
                "batch_size": self.trainer.datamodule.batch_size,
                "tgt_test_path": str(self.trainer.datamodule.tgt_test_path),
                "max_len": self.hparams.max_len,
                "total_seconds": round(elapsed.total_seconds(), 4),
                "model_calls": self.generator.model_calls_num,
                "seconds_per_model_call": round(elapsed.total_seconds() / self.generator.model_calls_num, 4)
            }
        if self.hparams.generation in ("greedy_speculative", "beam_search_speculative"):
            report["n_drafts"] = self.hparams.n_drafts
            report["draft_len"] = self.hparams.draft_len
            if self.hparams.generation == "beam_search_speculative":
                report["accepted_tokens"] = self.generator.accepted_tokens_num
                report["acceptance_rate"] = round(self.generator.accepted_tokens_num / self.generator.produced_non_pad_tokens, 4)
        report = json.dumps(report)

        if self.report_prediction_time:
            print(report)
            if self.hparams.report_prediction_file is not None:
                report_prediction_dir = Path(self.hparams.report_prediction_file).parent
                report_prediction_dir.mkdir(exist_ok=True)
                with open(self.hparams.report_prediction_file, "a") as f:
                    print(report, file=f)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(),
                               lr=self.hparams.learning_rate,
                               weight_decay=self.hparams.weight_decay,
                               betas=(0.9, 0.999))

        sched_name = self.hparams.scheduler
        ws = self.hparams.warmup_steps
        if sched_name == "const":
            scheduler = {
                "scheduler": optim.lr_scheduler.LambdaLR(
                    optimizer,
                    ConstantLRSchedule(ws)
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
                    NoamLRSchedule(d, ws)
                ),
                "name": "Noam scheduler",
                "interval": "step",
                "frequency": 1,
            }
        else:
            raise ValueError(f'Unknown scheduler name {self.hparams.scheduler}. Options are "const", "noam".')

        return [optimizer], [scheduler]
