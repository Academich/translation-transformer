from typing import Union, List, Callable

import torch


# Beam size: K
# Batch size: B
# Current length: L


class TranslationInferenceGreedy:

    def __init__(self,
                 model: 'torch.nn.Module',  # TODO Or its subclasses
                 max_len: int,
                 pad_token: int,
                 bos_token: int,
                 eos_token: int) -> None:
        self.model = model
        self.max_len = max_len
        self.pad_token = pad_token
        self.bos_token = bos_token
        self.eos_token = eos_token

    def generate(self, src: 'torch.LongTensor') -> 'torch.LongTensor':
        b_size = src.size()[0]
        generated_tokens = torch.full((b_size, 1), self.pad_token)
        generated_tokens[:, 0] = self.bos_token
        generated_tokens = generated_tokens.type_as(src).long()

        for _ in range(self.max_len):
            pred_logits = self.model(src, generated_tokens)
            pred_token = torch.argmax(pred_logits, dim=2)[:, -1:]
            generated_tokens = torch.cat(
                (generated_tokens,
                 pred_token),
                dim=1
            )
            if (torch.logical_or(pred_token == self.eos_token,
                                 pred_token == self.pad_token)).sum().item() == b_size:
                break

        return generated_tokens


class TranslationInferenceBeamSearch:

    def __init__(self,
                 model: Callable[['torch.Tensor', 'torch.Tensor'], 'torch.Tensor'],
                 beam_size: int,
                 max_len: int,
                 pad_token: int,
                 bos_token: int,
                 eos_token: int):
        self.model = model
        self.beam_size = beam_size
        self.max_len = max_len
        self.pad_token = pad_token
        self.bos_token = bos_token
        self.eos_token = eos_token

    def generate(self, src: 'torch.LongTensor') -> 'torch.LongTensor':
        batch_size = src.size()[0]

        generated_tokens = torch.full((batch_size, self.beam_size, 1), self.bos_token)
        confidence = torch.ones((batch_size, self.beam_size, 1))
        finished = torch.zeros((batch_size, self.beam_size, 1)).bool()

        output_tokens = []
        output_conf = []

        # First tokens
        logits = self.model(src, generated_tokens[:, 0, :])
        initial_pred_token = torch.topk(logits, k=self.beam_size, dim=2)
        generated_tokens = torch.cat(
            (generated_tokens,
             initial_pred_token.indices.permute(0, 2, 1)),
            dim=-1
        )
        confidence = confidence * initial_pred_token.values.permute(0, 2, 1)
        finished = torch.logical_or(finished, generated_tokens[:, :, -1:] == self.eos_token)

        # Other tokens
        for _ in range(self.max_len):
            if torch.all(finished):
                break
            branches_tokens: Union[List['torch.Tensor'], 'torch.Tensor'] = []
            branches_log_conf: Union[List['torch.Tensor'], 'torch.Tensor'] = []
            for i in range(self.beam_size):
                _b = generated_tokens[:, i, :]
                if finished[:, i, :]:
                    output_tokens.append(_b)
                    output_conf.append(confidence[:, i, :])
                    continue

                pred_probs = self.model(src, _b)
                pred_token = torch.topk(pred_probs[:, -1:, :], k=self.beam_size, dim=2)
                _a = pred_token.indices.permute(0, 2, 1)
                new_branch = torch.cat((
                    _b.unsqueeze(1).expand(-1, self.beam_size, -1),
                    _a
                ), dim=2)
                branches_tokens.append(new_branch)
                branches_log_conf.append(confidence[:, i, :].unsqueeze(1) * pred_token.values.permute(0, 2, 1))

            branches_tokens = torch.cat(branches_tokens, dim=1)
            branches_log_conf = torch.cat(branches_log_conf, dim=1)
            sorted_log_conf, tokens_to_pick = torch.sort(branches_log_conf, dim=1, descending=True, stable=True)
            n_take = self.beam_size - len(output_tokens)
            confidence = sorted_log_conf[:, :n_take, :]
            generated_tokens = torch.gather(branches_tokens,
                                            index=tokens_to_pick.expand(*branches_tokens.size()),
                                            dim=1)[:, :n_take, :]
            finished = torch.logical_or(finished, generated_tokens[:, :, -1:] == self.eos_token)

        return generated_tokens
