from heapq import heappush, heappop
import torch.nn.functional as F
import torch


# Beam size: K
# Batch size: B
# Current length: L

class TranslationInferenceBeamSearchOurs:

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
        self.memory = None

    def __str__(self):
        return f"Beam search decoding (beam_size={self.beam_size}, n_best={self.n_best}, max_len={self.max_len})"

    def generate(self, src: 'torch.LongTensor') -> 'torch.LongTensor':
        """Batch Beam Seach Decoding for RNN
              Args:
                  src: (bs, src_max_len)

              Returns:
                  n_best_list: Decoded N-best results. (bs, T)
              """

        b_sz, max_len = src.size()
        assert b_sz == 1
        # Prepare first tokens for decoder (bs, max_len)
        batch_decoder_input = torch.tensor([self.bos_token] + [self.pad_token] * (self.max_len - 1)).repeat(b_sz,
                                                                                                            1).long()
        src_pad_mask = (src == self.model.src_pad_token_i).bool()  # (b_sz, max_len)
        memory = self.model.encode_src(src, src_pad_mask)  # (b_sz, max_len, dim)
        logits = self.model.decode_tgt(batch_decoder_input,
                                       memory,
                                       memory_pad_mask=src_pad_mask)[:, 0, :]  # (bsz, max_len, dict_len) -> (bsz,
        # dict_len)
        probs = torch.softmax(logits, dim=-1)
        curr_probs, seqs = probs.topk(self.beam_size, dim=-1)  # b_sz, beam_width
        bos = torch.tensor([self.bos_token]).repeat(self.beam_size, 1).long()  # (beam_size,1)
        pred_lsts = []

        for i in range(b_sz):
            curr_seqs = torch.cat((bos, seqs[i].unsqueeze(1)), dim=1)  # beam_width, 2
            memory_i = memory[i].unsqueeze(0).expand(self.beam_size, -1, -1)
            src_pad_mask_i = src_pad_mask[i].unsqueeze(0).expand(self.beam_size, -1)
            best_candidates = []
            num_done = 0
            n_best_list = self.bm_search(memory_i, src_pad_mask_i, curr_probs[i], curr_seqs, best_candidates,
                                         num_done)
            tmp = torch.cat([F.pad(t.unsqueeze(0), pad=(0, self.max_len - t.shape[0], 0, 0)) for t in n_best_list],
                            dim=0)  # (n_best,max_len)
            pred_lsts.append(tmp.unsqueeze(0))
        return torch.cat(pred_lsts, dim=0)  # (b_sz,n_best,max_len)

    def bm_search(self, memory, src_pad_mask, curr_probs, inds_curr_seq, best_candidates, num_done):
        """Batch Beam Seach Decoding
              Args:
                  memory: tensor (beam_width, src_max_len, dim)
                  src_pad_mask: tensor (beam_width, src_max_len)
                  curr_probs: tensor (beam_width)
                  inds_curr_seq (beam_width, curr_len)
                  best_candidates: list

              Returns:
                  best_candidates: list of self.n_best tensors of different lengths
              """

        beam_width, curr_len = inds_curr_seq.size()
        if curr_len == self.max_len:
            return best_candidates  # (n_best, self.max_len)

        prelim_probs = curr_probs.unsqueeze(-1).expand(beam_width, beam_width)  # beam_width,beam_width

        # Decode for one step using decoder
        new_logits = self.model.decode_tgt(inds_curr_seq,
                                           memory,
                                           memory_pad_mask=src_pad_mask)[:, -1,
                     :]  # (beam_width, curr_len, dict_len)->(beam_width, dict_len)
        new_probs = torch.softmax(new_logits, dim=-1)

        predicted_probs, predicted_inds = new_probs.topk(beam_width,
                                                         dim=-1)  # (beam_width, beam_width)

        new_seqs_probs = prelim_probs * predicted_probs  # beam_width, beam_width

        best_probs, flattened_best_ids = new_seqs_probs.flatten().topk(beam_width,
                                                                       dim=-1)  # (beam_width)  we chose
        # the best beam_width seqs and their probabilities from beam_width * beam_width variants

        best_inds_for_old_seq = (flattened_best_ids / beam_width).long()  # beam_width

        best_old_seqs = inds_curr_seq[
            best_inds_for_old_seq]  # (beam_width, curr_len)[beam_width] ->(beam_width, curr_len)

        best_predicted_rg_ids = predicted_inds.flatten()[flattened_best_ids]  # beam_width

        best_seqs = torch.cat([best_old_seqs, best_predicted_rg_ids.unsqueeze(-1)],
                              dim=-1)  # (beam_width, curr_len + 1)
        num_finished = sum((best_seqs == self.eos_token)[:, -1])

        if num_finished:
            best_candidates += [t.cpu() for t in
                                best_seqs[(best_seqs == self.eos_token)[:, -1]]]  # num_finished, curr_len + 1
            best_probs = best_probs[(best_seqs != self.eos_token)[:, -1]]
            best_seqs[(best_seqs != self.eos_token)[:, -1]]

            num_done += num_finished
            if num_done >= self.n_best:
                # To do: what if num_done > n_best?
                return best_candidates  # list of num_done tensors of different lengths

        rg_ids_chains = self.bm_search(memory, src_pad_mask, best_probs, best_seqs, best_candidates, num_done)
        return rg_ids_chains


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

    def generate(self, src: 'torch.LongTensor') -> 'torch.LongTensor':
        b_size = src.size()[0]
        generated_tokens = torch.full((b_size, 1), self.pad_token)
        generated_tokens[:, 0] = self.bos_token
        generated_tokens = generated_tokens.type_as(src).long()

        src_pad_mask = (src == self.model.src_pad_token_i).bool()
        memory = self.model.encode_src(src, src_pad_mask)

        for _ in range(self.max_len):
            pred_logits = self.model.decode_tgt(generated_tokens, memory, memory_pad_mask=src_pad_mask)
            pred_token = torch.argmax(pred_logits, dim=2)[:, -1:]
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


class TranslationInferenceBeamSearchJarobyte:
    """
    Inspired by https://github.com/jarobyte91/pytorch_beam_search
    """

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


def num_speculative_tokens_to_accept(arr: torch.BoolTensor):
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
        _, L = src.size()
        generated_tokens = torch.full((1, 1), self.bos_token).type_as(src).long()

        src_pad_mask = (src == self.model.src_pad_token_i).bool()
        memory = self.model.encode_src(src, src_pad_mask)

        src_idx_to_copy_from = 1
        while generated_tokens.size(1) < self.max_len:
            draft_tokens = src[:, src_idx_to_copy_from:src_idx_to_copy_from + self.n_speculative_tokens]
            draft_sequence = torch.cat([
                generated_tokens,
                draft_tokens
            ], dim=-1)
            pred_logits = self.model.decode_tgt(draft_sequence, memory, memory_pad_mask=src_pad_mask)
            pred_tokens = torch.argmax(pred_logits, dim=2)
            pred_tokens = pred_tokens[:, -(draft_tokens.size(1) + 1):]
            n_accepted = num_speculative_tokens_to_accept(draft_tokens == pred_tokens[:, :-1])
            pred_tokens = pred_tokens[:, :n_accepted + 1]
            src_idx_to_copy_from = min(src_idx_to_copy_from + n_accepted + 1, L - 1)

            generated_tokens = torch.cat(
                (generated_tokens,
                 pred_tokens),
                dim=1
            )
            if (pred_tokens == self.eos_token).sum(-1).item() > 0:
                break
        return torch.cat([i.unsqueeze(0) for i in generated_tokens.unsqueeze(1)], dim=0)


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
