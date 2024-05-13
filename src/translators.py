from heapq import heappush, heappop

import torch
from torch.nn.utils.rnn import pad_sequence


# Beam size: K
# Batch size: B
# Current length: L


class TranslationInferenceGreedy:

    def __init__(self,
                 model,  # TranslationModel
                 max_len: int,
                 pad_token: int,
                 bos_token: int,
                 eos_token: int) -> None:
        self.model = model
        self.max_len = max_len
        self.pad_token = pad_token
        self.bos_token = bos_token
        self.eos_token = eos_token

    def __str__(self):
        return f"Greedy decoding (max_len={self.max_len})"

    def sample(self, pred_logits):
        return torch.argmax(pred_logits, dim=2)[:, -1:]

    def generate(self, src: 'torch.LongTensor') -> 'torch.LongTensor':
        b_size = src.size()[0]
        generated_tokens = torch.full((b_size, 1), self.pad_token)
        generated_tokens[:, 0] = self.bos_token
        generated_tokens = generated_tokens.type_as(src).long()

        src_pad_mask = (src == self.model.src_pad_token_i).bool()
        memory = self.model.encode_src(src, src_pad_mask)

        for _ in range(self.max_len):
            pred_logits = self.model.decode_tgt(generated_tokens, memory, memory_pad_mask=src_pad_mask)
            pred_token = self.sample(pred_logits)
            generated_tokens = torch.cat(
                (generated_tokens,
                 pred_token),
                dim=1
            )
            if (torch.logical_or(pred_token == self.eos_token,
                                 pred_token == self.pad_token)).sum().item() == b_size:
                break
        return torch.cat([i.unsqueeze(0) for i in generated_tokens.unsqueeze(1)], dim=0)


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
                 model,  # TranslationModel
                 beam_size: int,
                 n_best: int,
                 max_len: int,
                 pad_token: int,
                 bos_token: int,
                 eos_token: int):
        self.model = model
        self.beam_size = beam_size
        self.n_best = n_best
        self.max_len = max_len
        self.pad_token = pad_token
        self.bos_token = bos_token
        self.eos_token = eos_token

    def __str__(self):
        return f"Beam search decoding (beam_size={self.beam_size}, n_best={self.n_best}, max_len={self.max_len})"

    def generate(self, src: 'torch.LongTensor') -> 'torch.LongTensor':
        """Batch Beam Seach Decoding for RNN
              Args:
                  src: (bs, max_len)
                  n_best: The number of output sequences for each input

              Returns:
                  n_best_list: Decoded N-best results. (bs, T)
              """

        assert self.beam_size >= self.n_best
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

        src_pad_mask = (src == self.model.src_pad_token_i).bool()
        memory = self.model.encode_src(src, src_pad_mask)

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
                        if len(eosnodes_per_sid[sid]) >= self.n_best:
                            done_sids_set.add(sid)
                            sid += 1
                    else:
                        nonpad_length_per_sid[sid] = n.length
                        curr_logprob_per_sid[sid] = n.logprob
                        batch_decoder_input[sid] = torch.tensor(n.decoder_inp)
                        sid += 1

            batch_decoder_input = batch_decoder_input.type_as(src)  # (bs,max_len)

            # Decode for one step using decoder
            decoder_output = self.model.decode_tgt(batch_decoder_input,
                                                   memory,
                                                   memory_pad_mask=src_pad_mask)  # (bs, max_len, dict_len)
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
        # if there are no end_nodes, retrieve the best nodes (they are probably truncated)
        n_best_seq_list = torch.tensor([self.pad_token] * self.max_len).repeat(bs, self.n_best,
                                                                               1)  # (bs, n_best, max_len)
        for sid in range(bs):
            # if this sentence hasn't come to its dot
            if len(eosnodes_per_sid[sid]) == 0:
                eosnodes_per_sid[sid] = [heappop(nodetree_per_sid[sid]) for _ in range(self.beam_size)]

            for decoded_seq_id, score_n_tuple in enumerate(eosnodes_per_sid[sid]):
                n = score_n_tuple[1]
                decoded_seq = n.decoder_inp

                n_best_seq_list[sid][decoded_seq_id] = torch.tensor(decoded_seq)

        return n_best_seq_list


def num_speculative_tokens_to_accept(arr):
    _range = arr.cumsum(-1)
    return (torch.arange(1, arr.size(1) + 1).type_as(_range) == _range).sum(-1)


class TranslationInferenceGreedySpeculativeUnbatched:

    def __init__(self,
                 model,  # TranslationModel
                 max_len: int,
                 n_speculative_tokens: int,
                 pad_token: int,
                 bos_token: int,
                 eos_token: int) -> None:
        self.model = model
        self.max_len = max_len
        self.pad_token = pad_token
        self.bos_token = bos_token
        self.eos_token = eos_token

        self.n_speculative_tokens = n_speculative_tokens

    def __str__(self):
        return f"Greedy speculative decoding (n_speculative_tokens={self.n_speculative_tokens}, max_len={self.max_len})"

    def generate(self, src: 'torch.LongTensor') -> 'torch.LongTensor':
        b_size, src_len = src.size()

        src_pad_mask = (src == self.model.src_pad_token_i).bool()
        memory = self.model.encode_src(src, src_pad_mask)

        src_unbatched = src.unsqueeze(1)
        src_pad_mask_unbatched = src_pad_mask.unsqueeze(1)
        memory_unbatched = memory.unsqueeze(1)

        result = []
        for i in range(b_size):
            generated_tokens = torch.full((1, 1), self.bos_token).type_as(src).long()
            draft_tokens = src_unbatched[i, :, 1:].unfold(-1, self.n_speculative_tokens, 1).squeeze(0)
            n_drafts = draft_tokens.size(0)
            iters = 0
            while generated_tokens.size(1) < self.max_len:
                iters += 1
                draft_sequence = torch.cat([generated_tokens.repeat(n_drafts, 1), draft_tokens], dim=-1)
                pred_logits = self.model.decode_tgt(draft_sequence,
                                                    memory_unbatched[i].repeat(n_drafts, 1, 1),
                                                    memory_pad_mask=src_pad_mask_unbatched[i].repeat(n_drafts, 1))
                pred_tokens = torch.argmax(pred_logits, dim=2)
                pred_tokens = pred_tokens[:, -(draft_tokens.size(1) + 1):]
                verification = draft_tokens == pred_tokens[:, :-1]
                _range = verification.cumsum(-1)
                accepted_in_drafts = (torch.arange(1, verification.size(1) + 1).type_as(_range) == _range)
                n_accepted_in_drafts = accepted_in_drafts.sum(-1)
                n_accepted_in_drafts = n_accepted_in_drafts.topk(1, -1)
                draft_i = n_accepted_in_drafts.indices
                n_accepted = n_accepted_in_drafts.values

                pred_tokens = pred_tokens[draft_i, :n_accepted + 1]

                generated_tokens = torch.cat(
                    (generated_tokens,
                     pred_tokens),
                    dim=1
                )
                if (pred_tokens == self.eos_token).sum(-1).item() > 0:
                    break
            result.append(generated_tokens.squeeze(0))
        return pad_sequence(result, padding_value=self.pad_token, batch_first=True).unsqueeze(1)


if __name__ == '__main__':
    from tests.mock_model import MockCopySequence

    tr = TranslationInferenceBeamSearch(model=MockCopySequence(),
                                        max_len=4,
                                        beam_size=3,
                                        n_best=3,
                                        pad_token=MockCopySequence.pad_token,
                                        bos_token=MockCopySequence.bos_token,
                                        eos_token=MockCopySequence.eos_token)
    src = torch.tensor([[1, 2, 3, 4, 10]])
    print(tr.generate(src))
