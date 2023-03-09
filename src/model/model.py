import inspect
from argparse import ArgumentParser
from typing import Optional, List, Tuple

import torch
from torch import LongTensor, Tensor
from torch import nn
from torch import optim

import pytorch_lightning as pl

from pytorch_lightning.utilities.types import STEP_OUTPUT

from src.model.featurization.embeddings import TokenEmbedding
from src.model.featurization.embeddings import PositionalEncoding

from src.wrappers import TokenVocabulary


class TextTranslationTransformer(pl.LightningModule):
    """
    A module for translation - it learns to turn one sequence of integers to another sequence of integers.
    """

    def __init__(self,
                 learning_rate: 'float',
                 beta1: 'float',
                 beta2: 'float',
                 weight_decay: 'float',
                 num_encoder_layers: 'int',
                 num_decoder_layers: 'int',
                 embedding_dim: 'int',
                 num_heads: 'int',
                 feedforward_dim: 'int',
                 dropout_rate: 'float',
                 activation: 'str',
                 src_vocab: 'TokenVocabulary',
                 tgt_vocab: 'TokenVocabulary',
                 **kwargs):  # Only serves a purpose of ignoring unexpected arguments
        super().__init__()
        self.save_hyperparameters()
        self._create_model()
        self.criterion = nn.CrossEntropyLoss(ignore_index=self.hparams.src_vocab.pad_token_idx,
                                             reduction='mean')

    def _create_model(self):
        raise NotImplementedError

    def _featurize(self, src: LongTensor, tgt: LongTensor) -> Tuple[Tensor, Tensor]:
        """
        Constructs the embeddings of tokens of both the source sequence and the target sequence.
        Input shape: (batch size X sequence length)
        :param src: A batch of source sequences. Tokens are represented by their indices in the source dictionary.
        :param tgt: A batch of target sequences. Tokens are represented by their indices in the target dictionary.
        :return: Embeddings of each token in both the src sequence and the target sequence.
        Shape - (batch size X sequence length X embedding dimensionality)
        """
        raise NotImplementedError

    def _update_embeddings(self,
                           src_emb: Tensor,
                           tgt_emb: Tensor,
                           src: LongTensor,
                           tgt: LongTensor) -> Tensor:
        """
        Updates the embeddings of the tokens before supplying the relevant ones to the decision subnetwork.
        :param src_emb: Embeddings of tokens of the source sequence.
        Used as conditions for the generation of the target sequence.
        :param tgt_emb: Embeddings of the target sequence tokens.
        :param src: Indices of tokens of the source sequence. Used to construct masks.
        :param tgt: Indices of tokens of the target sequence. Used to construct masks.
        :return: Updated embeddings of the target sequence tokens - those that come from the decoder.
        """
        raise NotImplementedError

    def _decision(self, tgt_emb: Tensor) -> Tensor:
        """
        A subnetwork that makes a decision about the next token.
        Basically a classifier, where each token in the target vocabulary is a class.
        :param tgt_emb: Updated embeddings of the target sequence.
        :return: The logits of the next token for each token in the target sequence.
        They will be supplied in the softmax layer to calculate the loss.
        """
        raise NotImplementedError

    def forward(self, src: LongTensor, tgt: LongTensor):
        src_emb, tgt_emb = self._featurize(src, tgt)
        upd_tgt_emb = self._update_embeddings(src_emb, tgt_emb, src, tgt)
        return self._decision(upd_tgt_emb)

    def training_step(self, batch, batch_idx) -> STEP_OUTPUT:
        source, target = batch

        # We predict the next token given the previous ones
        target_behind = target[:, :-1]
        target_ahead = target[:, 1:]
        pred_logits = self(source, target_behind)

        loss = self.criterion(pred_logits.reshape(-1, self.hparams.tgt_vocab.n_tokens),
                              target_ahead.reshape(-1))
        self._log_training_step(loss, pred_logits, target_ahead)
        return loss

    def _log_training_step(self, loss, predicted_logits, target_sequence):
        self.log(f"train/loss", loss, on_step=True, on_epoch=True, reduce_fx='mean')

        # Single token prediction accuracy
        pred_tokens = torch.argmax(predicted_logits, dim=2)
        single_tokens_predicted_right = (pred_tokens == target_sequence).float()  # TODO Beware of EOS != PAD
        single_token_pred_acc = single_tokens_predicted_right.mean()
        self.log(f"train/acc_single_tok", single_token_pred_acc, on_step=True, on_epoch=True, reduce_fx='mean')

        # Mean number of pad tokens in a batch
        pad_tokens_in_batch_target = (target_sequence == self.hparams.tgt_vocab.pad_token_idx)
        self.log(f"train/pads_in_batch_tgt", pad_tokens_in_batch_target.float().mean(), on_step=True, on_epoch=True,
                 reduce_fx='mean')

        # Mean number of EOS tokens in predictions in a batch
        eos_in_pred_batch = (pred_tokens == self.hparams.tgt_vocab.eos_token_idx)
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
        pad_tokens_in_batch_target = (target_sequence == self.hparams.tgt_vocab.pad_token_idx)
        self.log(f"val/pads_in_batch_tgt", pad_tokens_in_batch_target.float().mean(), on_step=True, on_epoch=True,
                 reduce_fx='mean')

        # Mean number of EOS tokens in predictions in a batch
        eos_in_pred_batch = (predicted_tokens == self.hparams.tgt_vocab.eos_token_idx)
        self.log(f"val/eos_in_batch_pred", eos_in_pred_batch.float().mean(), on_step=True, on_epoch=True,
                 reduce_fx='mean')

    def validation_step(self, batch, batch_idx) -> Optional[STEP_OUTPUT]:
        source, target = batch

        target_behind = target[:, :-1]
        target_ahead = target[:, 1:]
        pred_logits = self(source, target_behind)

        val_loss = self.criterion(pred_logits.reshape(-1, self.hparams.tgt_vocab.n_tokens),
                                  target_ahead.reshape(-1))
        pred_tokens = torch.argmax(pred_logits, dim=2)
        self._log_validation_step(val_loss, pred_tokens, target_ahead)
        return {"pred_tokens": pred_tokens, "target_ahead": target_ahead}

    def validation_epoch_end(self, outputs: List[STEP_OUTPUT]) -> None:
        total_correct, total = 0, 0
        for o in outputs:
            pred_tokens = o["pred_tokens"].cpu()
            target_ahead = o["target_ahead"].cpu()
            b_size = pred_tokens.size()[0]
            for i in range(b_size):
                target_str = self.hparams.tgt_vocab.decode(target_ahead[i])
                predicted_str = self.hparams.tgt_vocab.decode(pred_tokens[i])
                total_correct += int(predicted_str == target_str)
                total += 1
        self.log("val/whole_seq_exact_match_acc_total", total_correct / total)

    def test_step(self, batch, batch_idx) -> Optional[STEP_OUTPUT]:
        source, target = batch

        target_behind = target[:, :-1]
        target_ahead = target[:, 1:]
        pred_logits = self(source, target_behind)

        test_loss = self.criterion(pred_logits.reshape(-1, self.hparams.tgt_vocab.n_tokens),
                                   target_ahead.reshape(-1))
        self.log(f"test/loss", test_loss, on_step=False, on_epoch=True, reduce_fx='mean')

        return {"source_token_ids": source, "pred_logits": pred_logits, "target_token_ids": target}

    # === Optimization ===
    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(),
                                lr=self.hparams.learning_rate,
                                betas=(self.hparams.beta1, self.hparams.beta2),
                                weight_decay=self.hparams.weight_decay)
        return optimizer

    # === Argparse-related methods ===
    @classmethod
    def add_model_specific_args(cls, parent_parser):
        parser = parent_parser.add_argument_group(f"pl.{cls.__name__}")
        # Model
        parser.add_argument("--learning_rate", type=float, default=0.0004,
                            help="Constant learning rate for the AdamW optimizer.")
        parser.add_argument("--beta1", type=float, default=0.9,
                            help="Beta 1 for the AdamW optimizer")
        parser.add_argument("--beta2", type=float, default=0.999,
                            help="Beta 2 for the AdamW optimizer")
        parser.add_argument("--weight_decay", type=float, default=0,
                            help="Weight decay for the AdamW optimizer. Zero by default.")
        parser.add_argument("--num_encoder_layers", type=int, default=1,
                            help="Number of encoder layers in the transformer."),
        parser.add_argument("--num_decoder_layers", type=int, default=1,
                            help="Number of encoder layers in the transformer."),
        parser.add_argument("--embedding_dim", type=int, default=32,
                            help="Model dimension of the transformer: of token embeddings and k/q/v embeddings.")
        parser.add_argument("--num_heads", type=int, default=1,
                            help="Number of heads in multi-head attention.")
        parser.add_argument("--feedforward_dim", type=int, default=64,
                            help="Dimensionality of the hidden layers of the MLPs in decoder and encoder layers.")
        parser.add_argument("--dropout_rate", type=float, default=0.0,
                            help="Dropout rate in the transformer.")
        parser.add_argument("--activation", type=str, default='relu',
                            help="Activation function to use: 'relu' or 'gelu'.")
        return parent_parser

    @classmethod
    def from_argparse_args(cls, args, **kwargs):
        if isinstance(args, ArgumentParser):
            args = cls.parse_argparser(args)

        params = vars(args)

        # We only want to pass in valid class args, the rest may be user specific
        valid_kwargs = inspect.signature(TextTranslationTransformer.__init__).parameters
        trainer_kwargs = {name: params[name] for name in valid_kwargs if name in params}
        trainer_kwargs.update(**kwargs)

        return cls(**trainer_kwargs)


class VanillaTransformer(TextTranslationTransformer):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _create_model(self):
        # Embedding constructor
        self.src_token_featurizer = TokenEmbedding(self.hparams.src_vocab.n_tokens,
                                                   self.hparams.embedding_dim)

        self.tgt_token_featurizer = TokenEmbedding(self.hparams.tgt_vocab.n_tokens,
                                                   self.hparams.embedding_dim)

        self.positional_encoding = PositionalEncoding(self.hparams.embedding_dim)

        # Embedding updater

        self.transformer = nn.Transformer(d_model=self.hparams.embedding_dim,
                                          nhead=self.hparams.num_heads,
                                          num_encoder_layers=self.hparams.num_encoder_layers,
                                          num_decoder_layers=self.hparams.num_decoder_layers,
                                          dim_feedforward=self.hparams.feedforward_dim,
                                          dropout=self.hparams.dropout_rate,
                                          activation=self.hparams.activation,
                                          batch_first=True)

        # Decision function
        self.next_token_classifier = nn.Linear(self.hparams.embedding_dim, self.hparams.tgt_vocab.n_tokens)

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
        src_pad_mask = (src == self.hparams.src_vocab.pad_token_idx).bool()
        tgt_pad_mask = (tgt == self.hparams.tgt_vocab.pad_token_idx).bool()

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
