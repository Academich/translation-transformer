import inspect
import math
import sys
from argparse import ArgumentParser
from typing import Optional, List, Any, Union
from collections import defaultdict

import torch
from torch import LongTensor
from torch import nn
from torch import optim
from torch.nn.functional import log_softmax, softmax

import pytorch_lightning as pl

from pytorch_lightning.utilities.types import STEP_OUTPUT

from src.model.embeddings import TokenEmbedding, PositionalEncoding
from src.wrappers import TokenVocabulary


class Transformer(nn.Module):
    r"""A transformer model. User is able to modify the attributes as needed. The architecture
    is based on the paper "Attention Is All You Need". Ashish Vaswani, Noam Shazeer,
    Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Lukasz Kaiser, and
    Illia Polosukhin. 2017. Attention is all you need. In Advances in Neural Information
    Processing Systems, pages 6000-6010. Users can build the BERT(https://arxiv.org/abs/1810.04805)
    model with corresponding parameters.

    Args:
        d_model: the number of expected features in the encoder/decoder inputs (default=512).
        nhead: the number of heads in the multiheadattention models (default=8).
        num_encoder_layers: the number of sub-encoder-layers in the encoder (default=6).
        num_decoder_layers: the number of sub-decoder-layers in the decoder (default=6).
        dim_feedforward: the dimension of the feedforward network model (default=2048).
        dropout: the dropout value (default=0.1).
        activation: the activation function of encoder/decoder intermediate layer, relu or gelu (default=relu).
        custom_encoder: custom encoder (default=None).
        custom_decoder: custom decoder (default=None).
        layer_norm_eps: the eps value in layer normalization components (default=1e-5).
        batch_first: If ``True``, then the input and output tensors are provided
            as (batch, seq, feature). Default: ``False`` (seq, batch, feature).

    Examples::
        >>> transformer_model = nn.Transformer(nhead=16, num_encoder_layers=12)
        >>> src = torch.rand((10, 32, 512))
        >>> tgt = torch.rand((20, 32, 512))
        >>> out = transformer_model(src, tgt)

    Note: A full example to apply nn.Transformer module for the word language model is available in
    https://github.com/pytorch/examples/tree/master/word_language_model
    """

    def __init__(self, d_model: int = 512, nhead: int = 8, num_encoder_layers: int = 6,
                 num_decoder_layers: int = 6, dim_feedforward: int = 2048, dropout: float = 0.1,
                 activation: str = "relu", custom_encoder: Optional[Any] = None, custom_decoder: Optional[Any] = None,
                 layer_norm_eps: float = 1e-5, batch_first: bool = False,
                 device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(Transformer, self).__init__()

        if custom_encoder is not None:
            self.encoder = custom_encoder
        else:
            encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout,
                                                    activation, layer_norm_eps, batch_first,
                                                    **factory_kwargs)
            encoder_norm = LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
            self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

        if custom_decoder is not None:
            self.decoder = custom_decoder
        else:
            decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout,
                                                    activation, layer_norm_eps, batch_first,
                                                    **factory_kwargs)
            decoder_norm = LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
            self.decoder = TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm)

        self._reset_parameters()

        self.d_model = d_model
        self.nhead = nhead

        self.batch_first = batch_first

    def forward(self, src: Tensor, tgt: Tensor, src_mask: Optional[Tensor] = None, tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None, src_key_padding_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        r"""Take in and process masked source/target sequences.

        Args:
            src: the sequence to the encoder (required).
            tgt: the sequence to the decoder (required).
            src_mask: the additive mask for the src sequence (optional).
            tgt_mask: the additive mask for the tgt sequence (optional).
            memory_mask: the additive mask for the encoder output (optional).
            src_key_padding_mask: the ByteTensor mask for src keys per batch (optional).
            tgt_key_padding_mask: the ByteTensor mask for tgt keys per batch (optional).
            memory_key_padding_mask: the ByteTensor mask for memory keys per batch (optional).

        Shape:
            - src: :math:`(S, N, E)`, `(N, S, E)` if batch_first.
            - tgt: :math:`(T, N, E)`, `(N, T, E)` if batch_first.
            - src_mask: :math:`(S, S)`.
            - tgt_mask: :math:`(T, T)`.
            - memory_mask: :math:`(T, S)`.
            - src_key_padding_mask: :math:`(N, S)`.
            - tgt_key_padding_mask: :math:`(N, T)`.
            - memory_key_padding_mask: :math:`(N, S)`.

            Note: [src/tgt/memory]_mask ensures that position i is allowed to attend the unmasked
            positions. If a ByteTensor is provided, the non-zero positions are not allowed to attend
            while the zero positions will be unchanged. If a BoolTensor is provided, positions with ``True``
            are not allowed to attend while ``False`` values will be unchanged. If a FloatTensor
            is provided, it will be added to the attention weight.
            [src/tgt/memory]_key_padding_mask provides specified elements in the key to be ignored by
            the attention. If a ByteTensor is provided, the non-zero positions will be ignored while the zero
            positions will be unchanged. If a BoolTensor is provided, the positions with the
            value of ``True`` will be ignored while the position with the value of ``False`` will be unchanged.

            - output: :math:`(T, N, E)`, `(N, T, E)` if batch_first.

            Note: Due to the multi-head attention architecture in the transformer model,
            the output sequence length of a transformer is same as the input sequence
            (i.e. target) length of the decode.

            where S is the source sequence length, T is the target sequence length, N is the
            batch size, E is the feature number

        Examples:
            >>> output = transformer_model(src, tgt, src_mask=src_mask, tgt_mask=tgt_mask)
        """

        if not self.batch_first and src.size(1) != tgt.size(1):
            raise RuntimeError("the batch number of src and tgt must be equal")
        elif self.batch_first and src.size(0) != tgt.size(0):
            raise RuntimeError("the batch number of src and tgt must be equal")

        if src.size(2) != self.d_model or tgt.size(2) != self.d_model:
            raise RuntimeError("the feature number of src and tgt must be equal to d_model")

        memory = self.encoder(src, mask=src_mask, src_key_padding_mask=src_key_padding_mask)
        output = self.decoder(tgt, memory, tgt_mask=tgt_mask, memory_mask=memory_mask,
                              tgt_key_padding_mask=tgt_key_padding_mask,
                              memory_key_padding_mask=memory_key_padding_mask)
        return output

    def generate_square_subsequent_mask(self, sz: int) -> Tensor:
        r"""Generate a square mask for the sequence. The masked positions are filled with float('-inf').
            Unmasked positions are filled with float(0.0).
        """
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def _reset_parameters(self):
        r"""Initiate parameters in the transformer model."""

        for p in self.parameters():
            if p.dim() > 1:
                xavier_uniform_(p)


class TextTranslationTransformer(pl.LightningModule):

    def __init__(self,
                 learning_rate: 'float',
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

    def _create_model(self):
        # Model layers
        src_vocab_size = self.hparams.src_vocab.n_tokens
        tgt_vocab_size = self.hparams.tgt_vocab.n_tokens
        self.transformer = nn.Transformer(d_model=self.hparams.embedding_dim,
                                          nhead=self.hparams.num_heads,
                                          num_encoder_layers=self.hparams.num_encoder_layers,
                                          num_decoder_layers=self.hparams.num_decoder_layers,
                                          dim_feedforward=self.hparams.feedforward_dim,
                                          dropout=self.hparams.dropout_rate,
                                          activation=self.hparams.activation,
                                          batch_first=True)
        self.generator = nn.Linear(self.hparams.embedding_dim, tgt_vocab_size)
        self.src_tok_emb = TokenEmbedding(src_vocab_size,
                                          self.hparams.embedding_dim)

        self.tgt_tok_emb = TokenEmbedding(tgt_vocab_size,
                                          self.hparams.embedding_dim)

        self.positional_encoding = PositionalEncoding(self.hparams.embedding_dim)

        # Loss function
        self.criterion = nn.CrossEntropyLoss(ignore_index=self.hparams.src_vocab.pad_token_idx,
                                             reduction='mean')

    def forward(self, src: LongTensor, tgt: LongTensor):
        # TODO Specify shapes
        src_emb = self.positional_encoding(self.src_tok_emb(src))
        tgt_emb = self.positional_encoding(self.tgt_tok_emb(tgt))

        _, tgt_seq_len, _ = tgt_emb.size()
        tgt_mask = self.transformer.generate_square_subsequent_mask(tgt_seq_len).type_as(tgt_emb)
        src_pad_mask = (src == self.hparams.src_vocab.pad_token_idx).bool()
        tgt_pad_mask = (tgt == self.hparams.tgt_vocab.pad_token_idx).bool()

        transformer_output = self.transformer(src_emb,
                                              tgt_emb,
                                              src_mask=None,
                                              tgt_mask=tgt_mask,
                                              memory_mask=None,
                                              src_key_padding_mask=src_pad_mask,
                                              tgt_key_padding_mask=tgt_pad_mask,
                                              memory_key_padding_mask=src_pad_mask)

        logits = self.generator(transformer_output)
        return logits

    def training_step(self, batch, batch_idx) -> STEP_OUTPUT:
        source, target = batch

        # We need to predict the next token given the previous ones
        # Target is label-encoded
        target_behind = target[:, :-1]
        target_ahead = target[:, 1:]
        pred_logits = self(source, target_behind)

        loss = self.criterion(pred_logits.reshape(-1, self.hparams.tgt_vocab.n_tokens),
                              target_ahead.reshape(-1))
        # === Logging ===
        self.log(f"train/loss", loss, on_step=True, on_epoch=True, reduce_fx='mean')
        # Perplexity
        perplexity = torch.exp(loss)
        self.log(f"train/perpl", perplexity, on_step=True, on_epoch=True, reduce_fx='mean')

        # Single token prediction accuracy
        pred_tokens = torch.argmax(pred_logits, dim=2)
        n_pred_tokens_in_batch = math.prod(pred_tokens.shape)
        pad_tokens_in_batch_target = (target_ahead == self.hparams.tgt_vocab.pad_token_idx)
        single_tokens_predicted_right = (pred_tokens == target_ahead)
        single_token_pred_acc = single_tokens_predicted_right.sum() / (
                n_pred_tokens_in_batch - pad_tokens_in_batch_target.sum())
        self.log(f"train/acc_single_tok", single_token_pred_acc, on_step=True, on_epoch=False, reduce_fx='mean')

        # Whole sequence prediction accuracy
        sequence_pred_acc = torch.all(
            torch.logical_or(
                pred_tokens == target_ahead,
                pad_tokens_in_batch_target
            ),
            dim=-1
        ).float().mean()
        self.log(f"train/acc_whole_seq", sequence_pred_acc, on_step=True, on_epoch=False, reduce_fx='mean')

        # Number of tokens in batch
        self.log(f"train/tokens_in_batch", target_ahead.shape[0] * (target_ahead.shape[1] + 1), on_step=True,
                 on_epoch=False, reduce_fx='mean')

        # Mean number of pad tokens in a batch
        self.log(f"train/pads_in_batch_tgt", pad_tokens_in_batch_target.float().mean(), on_step=True, on_epoch=True,
                 reduce_fx='mean')

        # Mean number of EOS tokens in predictions in a batch
        eos_in_pred_batch = (pred_tokens == self.hparams.tgt_vocab.eos_token_idx)
        self.log(f"train/eos_in_batch_pred", eos_in_pred_batch.float().mean(), on_step=True, on_epoch=True,
                 reduce_fx='mean')

        return loss

    def validation_step(self, batch, batch_idx) -> Optional[STEP_OUTPUT]:
        source, target = batch

        target_behind = target[:, :-1]
        target_ahead = target[:, 1:]
        pred_logits = self(source, target_behind)

        val_loss = self.criterion(pred_logits.reshape(-1, self.hparams.tgt_vocab.n_tokens),
                                  target_ahead.reshape(-1))
        # === Logging ===
        self.log(f"val/loss", val_loss, on_step=True, on_epoch=True, reduce_fx='mean')
        # Perplexity
        perplexity = torch.exp(val_loss)
        self.log(f"val/perpl", perplexity, on_step=True, on_epoch=True, reduce_fx='mean')

        # Single token prediction accuracy
        pred_tokens = torch.argmax(pred_logits, dim=2)
        n_pred_tokens_in_batch = math.prod(pred_tokens.shape)
        pad_tokens_in_batch_target = (target_ahead == self.hparams.tgt_vocab.pad_token_idx)
        single_tokens_predicted_right = (pred_tokens == target_ahead)
        single_token_pred_acc = single_tokens_predicted_right.sum() / (
                n_pred_tokens_in_batch - pad_tokens_in_batch_target.sum())
        self.log(f"val/acc_single_tok", single_token_pred_acc, on_step=True, on_epoch=False, reduce_fx='mean')

        # Whole sequence prediction accuracy
        sequence_pred_acc = torch.all(
            torch.logical_or(
                pred_tokens == target_ahead,
                pad_tokens_in_batch_target
            ),
            dim=-1
        ).float().mean()
        self.log(f"val/acc_whole_seq", sequence_pred_acc, on_step=True, on_epoch=False, reduce_fx='mean')

        # Number of tokens in batch
        self.log(f"val/tokens_in_batch", target_ahead.shape[0] * (target_ahead.shape[1] + 1), on_step=True,
                 on_epoch=False, reduce_fx='mean')

        # Mean number of pad tokens in a batch
        self.log(f"val/pads_in_batch_tgt", pad_tokens_in_batch_target.float().mean(), on_step=True, on_epoch=True,
                 reduce_fx='mean')

        # Mean number of EOS tokens in predictions in a batch
        eos_in_pred_batch = (pred_tokens == self.hparams.tgt_vocab.eos_token_idx)
        self.log(f"val/eos_in_batch_pred", eos_in_pred_batch.float().mean(), on_step=True, on_epoch=True,
                 reduce_fx='mean')
        return

    def test_step(self, batch, batch_idx) -> Optional[STEP_OUTPUT]:
        source, target = batch

        target_behind = target[:, :-1]
        target_ahead = target[:, 1:]
        pred_logits = self(source, target_behind)

        test_loss = self.criterion(pred_logits.reshape(-1, self.hparams.tgt_vocab.n_tokens),
                                   target_ahead.reshape(-1))
        self.log(f"test/loss", test_loss, on_step=True, on_epoch=True, reduce_fx='mean')

        return {"source_token_ids": source, "pred_logits": pred_logits, "target_token_ids": target}

    def test_epoch_end(self, outputs: List[STEP_OUTPUT]) -> None:
        correct = defaultdict(list)
        total_correct, total = 0, 0
        for t in outputs:
            source = t["source_token_ids"]
            pred_logits = t["pred_logits"]
            target = t["target_token_ids"]

            pred_token_ids = torch.argmax(pred_logits, dim=2)
            b_size = pred_token_ids.size()[0]

            # self.generate_translation_beam_search(source, beam_size=2)
            generated_sequences = self.generate_translation_greedy(source)
            for i in range(b_size):
                input_sequence = source[i]
                target_sequence = target[i]
                pred_tokens = pred_token_ids[i]
                gen_tokens_greedy = generated_sequences[i]

                input_str = self.hparams.src_vocab.decode(input_sequence.cpu())
                target_str = self.hparams.tgt_vocab.decode(target_sequence.cpu())
                predicted_str = self.hparams.tgt_vocab.decode(pred_tokens.cpu())
                gen_str_greedy = self.hparams.tgt_vocab.decode(gen_tokens_greedy.cpu())
                print("    INPUT SEQUENCE:", input_str)
                print("   TARGET SEQUENCE:", target_str)
                print("   OUTPUT SEQUENCE:", predicted_str)
                print("GENERATED SEQUENCE:", gen_str_greedy)
                hit = int(predicted_str == target_str)
                total_correct += hit
                total += 1
                correct[len(target_str)].append(hit)

        for k in correct:
            self.log(f"test/acc_len_{k}", sum(correct[k]) / len(correct[k]))
        self.log("test/total_acc", total_correct / total)

    # === Decoding ===
    def generate_translation_greedy(self, src: 'LongTensor'):
        """
        :param src: (B, L, D)
        :return:
        """
        b_size = src.size()[0]
        max_len = 200

        bos_token_idx = self.hparams.tgt_vocab.bos_token_idx
        eos_token_idx = self.hparams.tgt_vocab.eos_token_idx
        pad_token_idx = self.hparams.tgt_vocab.pad_token_idx
        generated_tokens = torch.full((b_size, 1), pad_token_idx)
        generated_tokens[:, 0] = bos_token_idx
        generated_tokens = generated_tokens.type_as(src).long()

        for _ in range(max_len):
            pred_logits = self(src, generated_tokens)
            pred_token = torch.argmax(pred_logits, dim=2)[:, -1:]
            generated_tokens = torch.cat(
                (generated_tokens,
                 pred_token),
                dim=1
            )
            if (pred_token == eos_token_idx).sum().item() == b_size:
                break

        return generated_tokens

    def generate_translation_beam_search(self, src: 'LongTensor', beam_size: int):
        """
        :param beam_size:
        :param src: (B, L, D)
        :return:
        """
        batch_size = src.size()[0]
        max_len = 200

        bos_token_idx = self.hparams.tgt_vocab.bos_token_idx
        eos_token_idx = self.hparams.tgt_vocab.eos_token_idx
        pad_token_idx = self.hparams.tgt_vocab.pad_token_idx
        generated_tokens = torch.full((beam_size, batch_size, 1), pad_token_idx)
        generated_tokens[:, :, 0] = bos_token_idx
        generated_tokens = generated_tokens.type_as(src).long()

        # TODO Make log confidence
        confidence = torch.ones((beam_size, batch_size, 1)).type_as(src)  # log(1) is zero
        # print(generated_tokens)

        initial_pred_probs = softmax(self(src, generated_tokens[0, :, :]),
                                     dim=2)
        initial_pred_token = torch.topk(initial_pred_probs, k=beam_size, dim=2)
        print("111")
        print(initial_pred_token.indices)

        generated_tokens = torch.cat(
            (generated_tokens,
             initial_pred_token.indices.permute(2, 0, 1)),
            dim=-1
        )
        print(generated_tokens)
        confidence = confidence * initial_pred_token.values.permute(2, 0, 1)
        print(confidence)
        # Before the cycle everything is right

        for _ in range(max_len):
            branches_tokens: Union[List['torch.Tensor'], 'torch.Tensor'] = []
            branches_log_conf: Union[List['torch.Tensor'], 'torch.Tensor'] = []
            for i in range(beam_size):
                pred_probs = softmax(self(src, generated_tokens[i, :, :]),
                                     dim=2)
                pred_token = torch.topk(pred_probs, k=beam_size, dim=2)
                branches_tokens.append(pred_token.indices[:, -1:, :].permute(2, 0, 1))
                print("Appending")
                print(pred_token.values[:, -1:, :].permute(2, 0, 1))
                branches_log_conf.append(pred_token.values[:, -1:, :].permute(2, 0, 1))

            print("--")
            print(branches_tokens)
            print(branches_log_conf)
            branches_tokens = torch.cat(branches_tokens, dim=-1)
            branches_log_conf = torch.cat(branches_log_conf, dim=-1)
            print(branches_log_conf)
            print(confidence)
            sys.exit()

            sorted_log_conf, tokens_to_pick = torch.sort(branches_log_conf, dim=-1, descending=True, stable=True)
            print(branches_tokens)
            print(sorted_log_conf)
            print(tokens_to_pick)
            print(torch.index_select(branches_tokens, index=tokens_to_pick, dim=-1))

            sys.exit()

        for i in range(1, max_len - 1):
            pred_logits = self(src, generated_tokens)
            pred_token = torch.argmax(pred_logits, dim=2)[:, -1:]
            generated_tokens = torch.cat(
                (generated_tokens,
                 pred_token),
                dim=1
            )
            if (pred_token == eos_token_idx).sum().item() == batch_size:
                break

        return generated_tokens

    # === Optimization ===
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        return optimizer

    # === Argparse-related methods ===
    @classmethod
    def add_model_specific_args(cls, parent_parser):
        parser = parent_parser.add_argument_group(f"pl.{cls.__name__}")
        # Model
        parser.add_argument("--learning_rate", type=float, default=0.0004,
                            help="Constant learning rate for the Adam optimizer.")
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
        valid_kwargs = inspect.signature(cls.__init__).parameters
        trainer_kwargs = {name: params[name] for name in valid_kwargs if name in params}
        trainer_kwargs.update(**kwargs)

        return cls(**trainer_kwargs)


class BeamSearchDecoder(pl.LightningModule):

    def __init__(self, model: 'pl.LightningModule',
                 max_len: int,
                 beam_size: int,
                 tgt_vocab: 'TokenVocabulary'):
        super().__init__()
        self.model = model
        self.max_len = max_len
        self.beam_size = beam_size
        self.vocab = tgt_vocab
        self.pad_idx = tgt_vocab.pad_token_idx
        self.eos_idx = tgt_vocab.eos_token_idx
        self.bos_idx = tgt_vocab.bos_token_idx

    def create_padding_mask(self, tgt: 'torch.Tensor') -> 'torch.Tensor':
        mask = torch.where(
            (tgt == self.pad_idx) | (tgt == self.eos_idx), True, False
        )
        return mask.transpose(0, 1)

    def eos_score_mask(self, max_score_k_masked):
        max_score_k_masked[:, 1:] = -torch.abs(max_score_k_masked[:, 1:]) * float('inf')
        return max_score_k_masked

    def _predict_k(self, ll, eos_padding_mask, k_):
        topk = torch.topk(ll, self.beam_size)
        k_idx_ = topk.indices
        k_idx_[eos_padding_mask] = self.pad_idx
        k_idx = torch.full(topk.indices.shape, k_).type_as(ll).long()
        k_idx = torch.stack([k_idx_, k_idx], dim=-1)
        k_score = topk.values
        k_score[eos_padding_mask] = 0
        return k_idx, k_score

    def _get_max_score_k(self, max_score, eos_padding_mask, k_):
        max_score_k = max_score[:, k_]
        max_score_k = max_score_k.unsqueeze(-1).expand(max_score.shape)
        max_score_k = max_score_k.clone()
        max_score_k[eos_padding_mask] = self.eos_score_mask(max_score_k[eos_padding_mask])
        return max_score_k

    def reorder_scores(self, interim_score_, interim_idx_):
        topk = torch.topk(interim_score_, self.beam_size)
        topk_idx_ = topk.indices.unsqueeze(-1).expand([topk.indices.shape[0], self.beam_size, 2])
        topk_idx_ = torch.gather(interim_idx_, 1, topk_idx_)
        return topk.values, topk_idx_

    def update_beams(self, beam_tokens, idx, i):
        beam_tokens[:, :i, :] = torch.gather(beam_tokens[:, :i, :], 0,
                                             idx[:, :, 1].transpose(0, 1).unsqueeze(1).repeat(1, i, 1))
        beam_tokens[:, i, :] = idx[:, :, 0].transpose(0, 1)
        return beam_tokens

    def _beam_step(self, src, beam_tokens, i, max_score, batch_size):
        score = torch.ones(
            (batch_size, self.beam_size),
            # TODO: initializing with -inf to avoid if else of k = 0; its quite slow so best to optimize
        ).type_as(src) * float('-inf')
        idx = torch.full(
            (batch_size, self.beam_size, 2), self.pad_idx,
        ).type_as(src)
        for k in range(self.beam_size):
            tgt = beam_tokens[k]
            tgt_padding_mask = self.create_padding_mask(tgt[:i, :])
            eos_padding_mask = torch.any(tgt_padding_mask, dim=1)

            ll = self.model(src, tgt=tgt[:i, :].t())

            ll = ll[:, -1, :]
            k_idx, k_score = self._predict_k(ll, eos_padding_mask, k)
            max_score_k = self._get_max_score_k(max_score, eos_padding_mask, k)

            interim_score = torch.cat((score, max_score_k + k_score), dim=-1)
            interim_idx = torch.cat((idx, k_idx), dim=1)
            score, idx = self.reorder_scores(interim_score, interim_idx)
            if i == 1:
                break  # Initialize beams before diverging
        beam_tokens = self.update_beams(beam_tokens, idx, i)
        return beam_tokens, score

    def beam_search(self, src: 'torch.LongTensor'):
        batch_size, _ = src.size()
        score = torch.zeros([batch_size, self.beam_size]).type_as(src)
        beam_tokens = torch.full(
            (self.beam_size, self.max_len, batch_size), self.pad_idx,
        ).type_as(src)
        beam_tokens[:, 0, :] = self.bos_idx
        for i in range(1, self.max_len):
            beam_tokens, score = self._beam_step(src, beam_tokens, i, score, batch_size)
            eos = torch.where(beam_tokens == self.eos_idx, True, False)
            eos = torch.any(eos, dim=1)
            if torch.all(eos):
                break
        return beam_tokens, score

    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        src, tgt = batch

        beam_tokens, score = self.beam_search(src)
        beam_tokens = beam_tokens.permute(2, 0, 1)
        batch_size = beam_tokens.size()[0]
        for i in range(batch_size):
            print("BATCH", i)
            for k in range(self.beam_size):
                print(beam_tokens[i, k].cpu())
                print(self.vocab.decode(beam_tokens[i, k].cpu()))
        # return preds
