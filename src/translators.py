from typing import Union, List, Callable

import torch
from torch.nn.utils.rnn import pad_sequence
import copy
from heapq import heappush, heappop


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


class BeamSearchNode:
    def __init__(self, decoder_inp, logprob, length):
        self.decoder_inp = decoder_inp
        self.logprob = logprob
        self.length = length  # nonpad_length

    def eval(self):
        return self.logprob / float(self.length - 1 + 1e-6)

    # overload < operator
    def __lt__(self, other):
        return self.eval() < other.eval()


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

    def generate(self, src: 'torch.LongTensor', n_best) -> 'torch.LongTensor':
        """Batch Beam Seach Decoding for RNN
              Args:
                  src: (bs, max_len)
                  n_best: The number of output sequences for each input

              Returns:
                  n_best_list: Decoded N-best results. (bs, T)
              """

        assert self.beam_size >= n_best
        assert self.max_len > 1

        n_best_list = []
        bs = src.shape[0]

        # Prepare first tokens for decoder (bs, max_len)
        batch_decoder_input = torch.tensor([self.bos_token] + [self.pad_token] * (self.max_len - 1)).repeat(bs,
                                                                                                            1).long()

        # list of eosnodes for each sentence in the batch
        eosnodes_per_sid = [[] for _ in range(bs)]

        # whole beam search node tree
        nodetree_per_sid = [[] for _ in range(bs)]

        # Start beam search
        done_sids_set = set()

        nonpad_length_per_sid = [1 for _ in range(bs)]
        curr_logprob_per_sid = [0. for _ in range(bs)]

        while len(done_sids_set) < bs:
            # Fetch the best node
            sid = 0
            while sid < bs:
                if sid in done_sids_set:
                    done_sids_set.add(sid)
                    # we don't need to work with such a sentence anymore
                    sid += 1
                elif len(nodetree_per_sid[sid]) == 0:
                    # nothing to do if the tree of the given node is empty
                    sid += 1
                else:
                    score, n = heappop(nodetree_per_sid[sid])  # pop the best node for the given sentence id
                    if n.decoder_inp[n.length - 1] == self.eos_token or n.length == self.max_len:
                        heappush(eosnodes_per_sid[sid], (-n.eval(), n))
                        # If we reached n_best finished predictions for the given sentence
                        if len(eosnodes_per_sid[sid]) >= n_best:
                            done_sids_set.add(sid)
                            sid += 1
                    else:
                        nonpad_length_per_sid[sid] = n.length
                        curr_logprob_per_sid[sid] = n.logprob
                        batch_decoder_input[sid] = torch.tensor(n.decoder_inp)
                        sid += 1

            batch_decoder_input = batch_decoder_input.type_as(src)  # (bs,max_len)

            # Decode for one step using decoder
            decoder_output = self.model(src, batch_decoder_input)  # (bs, max_len, dict_len)
            decoder_output = torch.log(torch.softmax(decoder_output, dim=-1))

            # check shape of the prediction
            dictnr_len = decoder_output.shape[2]
            assert self.beam_size <= dictnr_len
            assert decoder_output.shape == (bs, self.max_len, dictnr_len)

            # Get top-k from this decoded result
            topk_log_prob, topk_indexes = torch.topk(decoder_output, self.beam_size,
                                                     dim=-1)  # (bs, max_len, beam_size), (bs, max_len, beam_size)

            # Then, register new top-k nodes
            for sid in range(bs):
                if sid in done_sids_set:
                    continue
                last_t_before_pad_id = nonpad_length_per_sid[sid] - 1

                for new_cnddt in range(self.beam_size):
                    decoded_t = topk_indexes[sid][last_t_before_pad_id][new_cnddt].item()  # int64
                    logp = topk_log_prob[sid][last_t_before_pad_id][new_cnddt].item()  # float log probability val
                    batch_decoder_input[sid][last_t_before_pad_id + 1] = decoded_t

                    node = BeamSearchNode(decoder_inp=torch.tensor(batch_decoder_input[sid]),
                                          logprob=curr_logprob_per_sid[sid] + logp,
                                          length=nonpad_length_per_sid[sid] + 1)
                    if decoded_t == self.eos_token and node.length <= 2:  # if predicted [bos,eos]
                        continue
                    else:
                        heappush(nodetree_per_sid[sid], (-node.eval(), node))

        # Construct sequences from end_nodes
        # if there are no end_nodes, retrieve best nodes (they are probably truncated)
        n_best_seq_list = torch.tensor([self.pad_token] * self.max_len).repeat(bs, n_best, 1)  # (bs, n_best, max_len)
        for sid in range(bs):
            # if this sentence hasn't come to its dot
            if len(eosnodes_per_sid[sid]) == 0:
                eosnodes_per_sid[sid] = [heappop(nodetree_per_sid[sid]) for _ in range(self.beam_size)]

            for decoded_seq_id, score_n_tuple in enumerate(eosnodes_per_sid[sid]):
                n = score_n_tuple[1]
                decoded_seq = n.decoder_inp

                n_best_seq_list[sid][decoded_seq_id] = torch.tensor(decoded_seq)

        return n_best_seq_list


if __name__ == '__main__':
    from tests.mock_model import MockCopySequence

    tr = TranslationInferenceBeamSearch(model=MockCopySequence(),
                                        max_len=4,
                                        beam_size=3,
                                        pad_token=MockCopySequence.pad_token,
                                        bos_token=MockCopySequence.bos_token,
                                        eos_token=MockCopySequence.eos_token)
    src = torch.tensor([[1, 2, 3, 4, 10]])
    print(tr.generate(src, n_best=3))
