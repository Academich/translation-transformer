from typing import Union, List, Callable

import torch
from torch.nn.utils.rnn import pad_sequence


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

        # Unified output format with TranslationInferenceBeamSearch
        return [i for i in generated_tokens.unsqueeze(1)]


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
        batch_size, _ = src.size()

        results = []
        for seq_ in src:
            seq = seq_.unsqueeze(0)
            logits = self.model(seq, torch.tensor([[self.bos_token]]).type_as(src))
            initial_pred_token = torch.topk(logits[0], k=self.beam_size, dim=-1)
            generated_tokens = torch.cat((torch.full((self.beam_size, 1),
                                                     self.bos_token).type_as(src),
                                          initial_pred_token.indices.t()), dim=-1)
            confidence = initial_pred_token.values.t()
            finished_beams = []
            finished_beams_conf = []
            # Other tokens
            for _ in range(self.max_len - 1):
                branches_tokens: Union[List['torch.Tensor'], 'torch.Tensor'] = []
                branches_confidence: Union[List['torch.Tensor'], 'torch.Tensor'] = []

                finished = generated_tokens[..., -1] == self.eos_token
                if torch.all(finished):
                    break
                if torch.any(finished):
                    finished_beams += ([i for i in generated_tokens[finished]])
                    finished_beams_conf += ([i for i in confidence[finished]])
                    generated_tokens = generated_tokens[~finished]
                    confidence = confidence[~finished]

                for gen_beam, gen_conf in zip(generated_tokens, confidence):
                    branch = gen_beam.unsqueeze(0)
                    pred_probs = self.model(seq, branch)
                    pred_token = torch.topk(pred_probs[0, -1:, :], k=self.beam_size, dim=-1)
                    new_branch = torch.cat((
                        branch.expand(self.beam_size, -1),
                        pred_token.indices.t()
                    ), dim=-1)
                    branches_tokens.append(new_branch)
                    branches_confidence.append(gen_conf.unsqueeze(1) * pred_token.values.t())

                branches_tokens = torch.cat(branches_tokens, dim=0)
                branches_confidence = torch.cat(branches_confidence, dim=0)
                sorted_log_conf, tokens_to_pick = torch.sort(branches_confidence, dim=0, descending=True, stable=True)
                n_take = self.beam_size - len(finished_beams)
                confidence = sorted_log_conf[:n_take]
                generated_tokens = torch.index_select(branches_tokens,
                                                      index=tokens_to_pick[:n_take].flatten(),
                                                      dim=0)
            confidence = torch.cat([i for i in finished_beams_conf] + [i for i in confidence])
            confidence, order = confidence.sort(dim=0, descending=True)
            generated_tokens = pad_sequence(
                [i for i in finished_beams] + [i for i in generated_tokens],
                padding_value=self.pad_token,
                batch_first=True)
            generated_tokens = torch.index_select(generated_tokens, dim=0, index=order)
            results.append(generated_tokens)

        return results


if __name__ == '__main__':
    from tests.mock_model import MockCopySequence

    tr = TranslationInferenceBeamSearch(model=MockCopySequence(),
                                        max_len=10,
                                        beam_size=4,
                                        pad_token=MockCopySequence.pad_token,
                                        bos_token=MockCopySequence.bos_token,
                                        eos_token=MockCopySequence.eos_token)
    src = torch.tensor([[1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 10],
                        [1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 10]]).long()
    print(tr.generate(src))
