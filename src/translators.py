import torch.nn.functional as F
import torch


class TranslationInferenceNucleusSpeculativeUnbatchedNoCyclesLogProbHistory:

    def __init__(self,
                 model,  # TranslationModel
                 max_len: int,
                 n_speculative_tokens: int,
                 pad_token: int,
                 bos_token: int,
                 eos_token: int,
                 n_best: int = 5,
                 temperature: float = 1.,
                 nucleus: float = 0.995
                 ) -> None:
        self.model = model
        self.max_len = max_len
        self.pad_token = pad_token
        self.bos_token = bos_token
        self.eos_token = eos_token

        self.n_speculative_tokens = n_speculative_tokens
        self.nucleus = nucleus
        self.n_best = n_best
        self.extra_pad = -1
        self.max_num_positions_for_sampling = 5
        self.log_prob_pad = 1
        self.log_prob_extra_pad = 2
        self.max_drafts_num = 100

    def __str__(self):
        return f"NucleusSpeculativeUnbatched decoding (max_len={self.max_len}, nucleus={self.nucleus})"

    def sample(self, curr_lines, curr_log_probs_history, pred_logits, n_accepted, chosen_drafts):
        """
        :param curr_lines: tensor (n_candidates, len_),
        :param curr_log_probs_history: tensor (n_candidates, len_),
        :param pred_logits: tensor (n_candidates, draft_len + 1, vocab_size),
        :param n_accepted: tensor (n_candidates),
        :param chosen_drafts: tensor (n_candidates, draft_len)
        :return:
          ->  new_lines: tensor (num_lines, len),
              new_log_probs_history: tensor (num_lines, len)
        """

        n_candidates, draft_len_plus_one, vocab_size = pred_logits.size()
        masked_logits = self.mask_with_num_logits_according_nucleus(pred_logits, num=0.)
        # -> (n_candidates, draft_len + 1, vocab_size)

        tmp_range = torch.arange(draft_len_plus_one).type_as(curr_lines).unsqueeze(0)
        #   -> (1, draft_len + 1)
        mask_for_unaccepted_draft_tokens = tmp_range.repeat(n_candidates, 1) <= n_accepted.unsqueeze(-1)
        #   -> (n_candidates, draft_len + 1)
        masked_logits *= mask_for_unaccepted_draft_tokens.unsqueeze(-1)
        masked_logits[:, :-1, :].scatter_(index=chosen_drafts.unsqueeze(-1), dim=2, value=0.)

        candts_inds, token_postn, token_inds = torch.nonzero(masked_logits, as_tuple=True)  # (num)
        num = token_inds.size()[0]
        previous_roots = curr_lines[candts_inds]  # (num, len_)
        log_prob_history_of_roots = curr_log_probs_history[candts_inds]  # (num, len_)
        drafts = chosen_drafts[candts_inds]  # (num, draft_len)
        tail = torch.full((num, 1), 0.).type_as(drafts)  # -> (num, 1)
        new_seqs = torch.cat((drafts, tail), dim=-1)  # (num, draft_len+1)
        new_seqs.scatter_(1, index=token_postn.unsqueeze(-1), src=token_inds.unsqueeze(-1))
        #   -> (num, draft_len + 1)

        mask_for_tokens_after_the_sampled = tmp_range > token_postn.unsqueeze(-1)
        #   -> (num, draft_len + 1)
        predicted_log_probs = pred_logits.softmax(-1).log()[candts_inds]  # -> (num, draft_len + 1, vocab_size)

        new_seqs_log_probs = torch.gather(predicted_log_probs, dim=2, index=new_seqs.unsqueeze(-1)).squeeze(-1)
        #    -> (num, draft_len + 1)
        new_seqs_log_probs = new_seqs_log_probs.cumsum(dim=-1)  # -> (num, draft_len + 1)

        new_seqs_log_probs = log_prob_history_of_roots[:, -1].unsqueeze(-1) + new_seqs_log_probs

        new_seqs.masked_fill_(mask_for_tokens_after_the_sampled, self.pad_token)
        #    -> (num, draft_len + 1)

        previous_roots.masked_fill_(previous_roots == self.pad_token, self.extra_pad)
        # now all the left usual pads are changed by the extra pad tokens
        log_prob_history_of_roots.masked_fill_(log_prob_history_of_roots == self.log_prob_pad, self.log_prob_extra_pad)

        new_candidates = torch.cat((previous_roots, new_seqs), dim=-1)  # -> (num, len_ + draft_len + 1)
        log_prob_history = torch.cat((log_prob_history_of_roots, new_seqs_log_probs), dim=-1)

        # In case if the model generated the pad tokens:
        bool_inds = token_inds == self.pad_token
        pad_postns = token_postn[bool_inds].unsqueeze(-1)
        if pad_postns.shape[0] > 0:
            curr_len = log_prob_history_of_roots.size()[1]
            log_prob_history[bool_inds] = log_prob_history[bool_inds].scatter(1, index=curr_len + pad_postns - 1,
                                                                              src=torch.gather(
                                                                                  log_prob_history[bool_inds], 1,
                                                                                  curr_len + pad_postns))

        log_prob_history.masked_fill_(new_candidates == self.pad_token, self.log_prob_pad)

        new_candidates = move_pads_to_the_left(new_candidates, self.pad_token)
        log_prob_history = move_pads_to_the_left(log_prob_history, self.log_prob_pad)

        new_candidates.masked_fill_(new_candidates == self.extra_pad, self.pad_token)  # all pads will be at the left
        log_prob_history.masked_fill_(log_prob_history == self.log_prob_extra_pad, self.log_prob_pad)

        new_candidates = new_candidates[:, ((new_candidates == self.pad_token).sum(0) == num).sum():]
        log_prob_history = log_prob_history[:, ((log_prob_history == self.log_prob_pad).sum(0) == num).sum():]

        return new_candidates, log_prob_history

    def mask_with_num_logits_according_nucleus(self, pred_logits, num=0.):
        n, curr_len, vocab_size = pred_logits.size()  # (n_candidates, draft_len + 1, vocab_size)
        pred_logits = pred_logits.reshape(n * curr_len, vocab_size)  # -> (n * curr_len, vocab_size)

        sorted_logits, sorted_indices = torch.sort(pred_logits,
                                                   descending=True)  # -> (n * curr_len, vocab_size)
        cumulative_probs = torch.cumsum(sorted_logits.softmax(-1), dim=-1)  # -> (n * curr_len, vocab_size)

        # Remove tokens with cumulative probability above the threshold
        cumulative_probs = torch.roll(cumulative_probs, 1, dims=-1)
        cumulative_probs[:, 0] = 0
        keep_candidates_mask = cumulative_probs < self.nucleus  # -> (n * curr_len, vocab_size)

        keep_candidates_mask[:, self.max_num_positions_for_sampling:] = False
        # no more than self.max_num_positions_for_sampling

        sorted_logits.masked_fill_(~keep_candidates_mask, float(num))

        masked_logits_according_nucleus = torch.gather(sorted_logits, 1, sorted_indices.argsort(1))
        # -> (n * curr_len, vocab_size)
        return masked_logits_according_nucleus.reshape(n, curr_len, vocab_size)

    def generate(self, src: 'torch.LongTensor') -> list['torch.LongTensor']:
        b_size, src_len = src.size()

        src_pad_mask = (src == self.model.src_pad_token_i).bool()
        memory = self.model.encode_src(src, src_pad_mask)

        src_unbatched = src.unsqueeze(1)
        src_pad_mask_unbatched = src_pad_mask.unsqueeze(1)
        memory_unbatched = memory.unsqueeze(1)

        result = []

        for i in range(b_size):
            src_unbatched_i = src_unbatched[i, :, 1:]
            src_unbatched_i_pads = (src_unbatched_i == self.pad_token).int().sum(-1)
            n_tokens_without_pads = src_unbatched_i.size(1) - src_unbatched_i_pads
            src_unbatched_i_unpadded = src_unbatched_i[:, :n_tokens_without_pads]
            drafts = src_unbatched_i_unpadded.unfold(-1, self.n_speculative_tokens, 1).squeeze(0)[:self.max_drafts_num, :]
            # -> (n_drafts, draft_len)
            n_drafts, draft_len = drafts.size()
            iters = 0

            generated_tokens = torch.full((1, 1), self.bos_token).type_as(src).long()
            # -> (1, 1)
            memory_i = memory_unbatched[i].repeat(n_drafts, 1, 1)
            memory_pad_mask_i = src_pad_mask_unbatched[i].repeat(n_drafts, 1)

            finished_candidates_t = None

            log_probs_history = torch.full((1, 1), 0.).type_as(src).float()
            while (generated_tokens.size(1) + self.n_speculative_tokens + 1) < self.max_len and iters < self.max_len:
                iters += 1
                n_candidates, curr_len = generated_tokens.size()
                draft_tokens = drafts.repeat(n_candidates, 1)  # -> (n_candidates * n_drafts, draft_len)
                inp = generated_tokens.unsqueeze(1).expand(n_candidates, n_drafts, curr_len).reshape(
                    n_candidates * n_drafts,
                    curr_len)
                draft_sequence = torch.cat([inp, draft_tokens], dim=1)
                # (n_candidates * n_drafts, curr_len + draft_len)
                _, seq_len = draft_sequence.size()
                pos_enc_offset = (draft_sequence == self.pad_token).int().sum(-1).reshape(-1, 1)
                # For production, we use this model which supports positional encoding offset
                pred_logits = self.model.decode_tgt(draft_sequence, memory_i.repeat(n_candidates, 1, 1),
                                                    memory_pad_mask=memory_pad_mask_i.repeat(n_candidates, 1),
                                                    pos_enc_offset=pos_enc_offset)
                # # # This one we use only for debugging:
                # pred_logits = self.model.decode_tgt(draft_sequence, memory_i.repeat(n_candidates, 1, 1),
                #                                     memory_pad_mask=memory_pad_mask_i.repeat(n_candidates, 1))
                #  -> (n_candidates * n_drafts, curr_len + draft_len, vocab_size)
                vocab_size = pred_logits.shape[-1]
                pred_logits = pred_logits[:, -(draft_len + 1):, :]
                #   -> (n_candidates * n_drafts, draft_len + 1, vocab_size)
                masked_probs = self.mask_with_num_logits_according_nucleus(pred_logits, num="-inf").softmax(-1)
                #   -> (n_candidates * n_drafts, draft_len + 1, vocab_size)
                masked_probs = masked_probs.reshape(n_candidates, n_drafts, draft_len + 1, vocab_size)
                draft_tokens = draft_tokens.reshape(n_candidates, n_drafts, draft_len)

                tmp = torch.gather(masked_probs[:, :, :-1, :], dim=-1, index=draft_tokens.unsqueeze(-1)).squeeze(-1)
                #   -> (n_candidates, n_drafts, draft_len)
                verification = tmp != 0.
                # num = n_candidates

                _range = verification.cumsum(-1)  # (num, n_drafts, draft_len)
                accepted_in_drafts_bool = (torch.arange(1, verification.size(2) + 1).unsqueeze(0).unsqueeze(0).type_as(
                    _range) == _range)  # (num, n_drafts, draft_len)

                n_accepted_in_drafts = accepted_in_drafts_bool.sum(-1)  # (num, n_drafts, draft_len) -> (num, n_drafts)
                n_accepted, draft_i = n_accepted_in_drafts.topk(1, dim=-1)
                # (n_candidates, n_drafts) -> (n_candidates, 1)
                chosen_drafts = torch.gather(draft_tokens, dim=1,
                                             index=draft_i.unsqueeze(-1).expand(n_candidates, 1, draft_len)).squeeze(1)
                #   -> (n_candidates, draft_len)

                pred_logits = pred_logits.reshape(n_candidates, n_drafts, draft_len + 1, vocab_size)

                pred_logits = torch.gather(pred_logits, dim=1, index=draft_i.unsqueeze(-1).unsqueeze(-1).
                                           expand(n_candidates, 1, draft_len + 1, vocab_size)).squeeze(1)
                #   -> (n_candidates, draft_len + 1, vocab_size)

                new_candidates, new_log_probs_history = \
                    self.sample(generated_tokens, log_probs_history, pred_logits, n_accepted.squeeze(-1),
                                chosen_drafts)
                # generated_tokens: (n_candidates, curr_len),
                # log_probs_history: (n_candidates, curr_len),
                # pred_logits: (n_candidates, draft_len + 1, vocab_size),
                # n_accepted: (n_candidates),
                # chosen_drafts: (n_candidates, draft_len)
                #   ->  new_candidates: (num_lines, len),
                #       new_log_probs_history: (num_lines, len)

                new_candidates, new_log_probs_history = self.sort(new_candidates, new_log_probs_history,
                                                                  descending=True)
                if finished_candidates_t is None:
                    new_candidates = new_candidates[:self.n_best]
                    new_log_probs_history = new_log_probs_history[:self.n_best]
                else:
                    new_candidates = new_candidates[:(self.n_best - finished_candidates_t.shape[0])]
                    new_log_probs_history = new_log_probs_history[:(self.n_best - finished_candidates_t.shape[0])]

                finished_bool_ids = (new_candidates == self.eos_token).sum(-1).bool()
                #   -> (num_samples * n_candidates)

                num_new_finished = finished_bool_ids.sum().item()

                if num_new_finished > 0:
                    new_finished_candidates = cat_left_useless_pads(new_candidates[finished_bool_ids], self.pad_token)
                    _, tokens_num = new_finished_candidates.size()

                    pad_tail = torch.full((num_new_finished, self.max_len - tokens_num),
                                          self.pad_token).type_as(src)
                    # answers' pads will be on the left side of the row
                    new_finished_candidates = torch.cat((pad_tail, new_finished_candidates), dim=1)
                    #   -> (num_new_finished, max_len)
                    new_finished_log_probs_t = new_log_probs_history[finished_bool_ids][:,-1]
                    #   -> (num_new_finished)

                    if finished_candidates_t is None:
                        finished_candidates_t = new_finished_candidates

                        finished_candidates_log_probs_t = new_finished_log_probs_t
                    else:
                        finished_candidates_t = torch.cat((finished_candidates_t, new_finished_candidates), dim=0)

                        finished_candidates_log_probs_t = torch.cat((finished_candidates_log_probs_t,
                                                                     new_finished_log_probs_t), dim=0)

                    if num_new_finished == new_candidates.shape[0] or (finished_candidates_t.shape[0] >= self.n_best and \
                        new_log_probs_history[~finished_bool_ids][:, -1].max().item() < finished_candidates_log_probs_t.min().item()):
                        break

                generated_tokens = cat_left_useless_pads(new_candidates[~finished_bool_ids], self.pad_token)
                log_probs_history = cat_left_useless_pads(new_log_probs_history[~finished_bool_ids], self.log_prob_pad)

            if finished_candidates_t is None:
                print("there is no finished candidates for the src:", )
                result.append(generated_tokens)
            else:
                finished_candidates_t, finished_candidates_log_probs_t = \
                    sort(finished_candidates_t, finished_candidates_log_probs_t,
                         descending=True)
                finished_candidates_t = finished_candidates_t[:self.n_best]
                finished_candidates_log_probs_t = finished_candidates_log_probs_t[:self.n_best]
                result.append(finished_candidates_t)  # (n, max_len)
            return result

    def sort(self, candidates, candidates_log_probs_history, descending=True):
        non_pad_tokens_num = (candidates != self.pad_token).sum(-1)
        # -> (candidates_num)
        non_padded_log_progs_num = (candidates_log_probs_history <= 0.).sum(-1)
        # -> (candidates_num)
        candidates_num, max_len = candidates.size()
        assert (non_padded_log_progs_num == non_pad_tokens_num).sum().item() == candidates_num
        candidates_log_probs = candidates_log_probs_history[:, -1]

        sorted_log_probs, sorted_inds = candidates_log_probs.sort(descending=descending)
        return candidates[sorted_inds], candidates_log_probs_history[sorted_inds]

def cat_left_useless_pads(tensor_t, pad_id):
    # tensor_t is supposed to have pad ids only at the left
    rows_num, _ = tensor_t.size()
    # number of left columns filled with the pad id
    padded_columns_num = ((tensor_t == pad_id).sum(0) == rows_num).sum()
    return tensor_t[:, padded_columns_num:]

# Beam size: K
# Batch size: B
# Current length: L


def sort(candidates, candidates_log_probs, descending=True):
    sorted_log_probs, sorted_inds = candidates_log_probs.sort(descending=descending)
    return candidates[sorted_inds], sorted_log_probs


def move_pads_to_the_left(arr, pad_token=0):
    dim_indices = torch.arange(arr.shape[1]).type_as(arr).long().repeat(arr.shape[0]).reshape(arr.shape[0], -1)
    eos_index = (arr == pad_token).sum(1)
    indices = (dim_indices - eos_index.unsqueeze(1)) % arr.shape[1]
    return torch.gather(arr, dim=1, index=indices)

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




def num_speculative_tokens_to_accept(arr: torch.BoolTensor):
    _range = arr.cumsum(-1)
    return (torch.arange(1, arr.size(1) + 1).type_as(_range) == _range).sum(-1)


class TranslationInferenceBeamSearchDraftsLen2:

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
        self.drafts_len_2 = torch.tensor(self.calculate_drafts_len_2())
        self.n_drafts = self.drafts_len_2.size()[0]
        print("n_drafts ", self.n_drafts)

    def calculate_drafts_len_2(self):
        drafts = []
        # vocab_len = 5
        vocab = {i: i for i in range(9)}  # {0: 0, 1: 2, 3: 4, 4: 5, 5: 6, 6: 7, 7: 41}
        for n, i in vocab.items():
            if i == self.pad_token or i == self.bos_token:
                continue
            if i == self.eos_token:
                drafts.append([i, self.pad_token])
                for _, j in vocab.items():
                    if j == self.eos_token or j == self.bos_token or j == self.pad_token:
                        continue
                    drafts.append([j, i])
                continue
            drafts.append([i, i])
            for m in range(n + 1, len(vocab)):
                drafts.append([i, vocab[m]])
                drafts.append([vocab[m], i])
        return drafts

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

        b_sz, max_len = src.size()
        assert b_sz == 1
        # Prepare first tokens for decoder (bs, max_len)
        batch_decoder_input = torch.tensor([self.bos_token]).repeat(b_sz, 1).long()
        src_pad_mask = (src == self.model.src_pad_token_i).bool()  # (b_sz, max_len)
        memory = self.model.encode_src(src, src_pad_mask)  # (b_sz, max_len, dim)
        logits = self.model.decode_tgt(batch_decoder_input,
                                       memory,
                                       memory_pad_mask=src_pad_mask)[:, 0,
                 :]  # (bsz, max_len, dict_len) -> (bsz, dict_len)
        probs = torch.softmax(logits, dim=-1)
        probs[:, self.pad_token] = -1.
        curr_probs, seqs = probs.topk(self.beam_size, dim=-1)  # b_sz, beam_width
        bos = torch.tensor([self.bos_token]).repeat(self.beam_size, 1).long()  # (beam_size,1)
        pred_lsts = []

        for i in range(b_sz):
            curr_seqs = torch.cat((bos, seqs[i].unsqueeze(1)), dim=1)  # beam_width, 2
            memory_i = memory[i].unsqueeze(0).expand(self.beam_size * self.n_drafts, -1, -1)
            src_pad_mask_i = src_pad_mask[i].unsqueeze(0).expand(self.beam_size * self.n_drafts, -1)
            best_candidates = []
            num_done = 0
            curr_probs = torch.ones(self.beam_size)
            n_best_list = self.bm_search_drafts_len_2(memory_i, src_pad_mask_i, curr_probs, curr_seqs,
                                                      best_candidates, num_done)  # curr_probs[i]
            tmp = torch.cat([F.pad(t.unsqueeze(0), pad=(0, self.max_len - t.shape[0], 0, 0)) for t in n_best_list],
                            dim=0)  # (n_best,max_len)
            pred_lsts.append(tmp.unsqueeze(0))
        # print("pred_lsts", pred_lsts)
        return torch.cat(pred_lsts, dim=0)  # (b_sz,n_best,max_len)

    def bm_search_drafts_len_2(self, memory, src_pad_mask, curr_probs, inds_curr_seq, best_candidates,
                               num_done):
        # "recursive_bm_srch_eos"
        # memory:
        # src_pad_mask:
        # curr_probs: (beam_width)
        # inds_curr_seq (beam_width, curr_len)
        # best_candidates: list

        beam_width, curr_len = inds_curr_seq.size()
        if curr_len == self.max_len:
            return best_candidates[:self.n_best]  # (n_best, self.max_len)

        prelim_probs = curr_probs.unsqueeze(1).expand(beam_width, self.n_drafts).reshape(
            self.n_drafts * beam_width)  # (beam_width) -> (beam_width, 1) -> (beam_width, N_drafts) ->
        # -> (beam_width * N_drafts)

        # Decode for one step using decoder
        inp = torch.cat((inds_curr_seq.unsqueeze(1).expand(beam_width, self.n_drafts, -1),
                         self.drafts_len_2.unsqueeze(0).expand(beam_width, self.n_drafts, -1)),
                        dim=-1)  # (beam_width, 1, curr_len) (1, N_drafts, 2) -> (beam_width, N_drafts, curr_len + 2)
        inp = inp.reshape(self.n_drafts * beam_width, curr_len + 2)
        draft_logits = self.model.decode_tgt(inp,
                                             memory,
                                             memory_pad_mask=src_pad_mask)[:, :-1,
                       :]  # (beam_width * N_drafts, curr_len + 1, dict_len)
        draft_probs = torch.softmax(draft_logits, dim=-1)  # (beam_width * N_drafts, curr_len + 1, dict_len)
        ##probs = torch.prod(torch.prod(draft_probs, dim=-1), dim=-1)  # (beam_width * N_drafts, curr_len + 1, dict_len) ->
        ## -> (beam_width * N_drafts, curr_len + 1)  -> (beam_width * N_drafts)  ## ETO ZAGLUSHKA

        probs = torch.prod(torch.gather(draft_probs, 2, inp[:, 1:].unsqueeze(-1)).squeeze(-1),
                           dim=-1)  # для bos не надо считать вероятность
        # (self.n_drafts * beam_width, 1, curr_len + 2) -> (self.n_drafts * beam_width, curr_len + 2)

        probs *= prelim_probs  ###

        best_probs, best_seqs_inds = probs.topk(beam_width,
                                                dim=-1)  # (beam_width), (beam_width) (нагенерили n_drafts * beam_width вариантов)
        best_seqs = inp[best_seqs_inds]  # (beam_width, curr_len + 2)
        num_finished = torch.sum((best_seqs == self.eos_token)[:, -2:]).item()

        if num_finished:
            best_candidates += [t.cpu() for t in
                                best_seqs[torch.sum((best_seqs == self.eos_token)[:, -2:],
                                                    dim=-1).bool()]]  # num_finished, curr_len + 2
            curr_probs[torch.sum((best_seqs == self.eos_token)[:, -2:], dim=-1).bool()] = -1.

            num_done += num_finished
            if num_done >= self.n_best:
                # Что если num_done больше n_best?
                return best_candidates[:self.n_best]  # list of tensors(num_done, different_lens)

        rg_ids_chains = self.bm_search_drafts_len_2(memory, src_pad_mask, curr_probs, best_seqs,
                                                    best_candidates, num_done)
        return rg_ids_chains


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


class TranslationInferenceBeamSearchSpeculative:
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
        self.beam_width = beam_size
        self.n_best = n_best
        self.max_len = max_len
        self.pad_token = pad_token
        self.bos_token = bos_token
        self.eos_token = eos_token
        self.drafts_of_len_1 = torch.tensor(self.calculate_drafts_len_1())
        self.n_drafts = self.drafts_of_len_1.size()[0]
        print("n_drafts ", self.n_drafts)

    def calculate_drafts_len_1(self):
        drafts = []
        vocab = [i for i in range(20)]
        for i in vocab:
            if i == self.bos_token:
                continue
            drafts.append([i])
        return drafts

    def generate(self, src: 'torch.LongTensor') -> 'torch.LongTensor':
        assert self.max_len > 1
        b_sz = src.shape[0]

        # prepare src for the later cycle
        src_bw = src.repeat((self.beam_width * self.n_drafts, 1, 1)).transpose(0, 1).flatten(end_dim=1)
        # !(b_sz, src_len) -> (beam_width * n_drafts, b_sz, src_len) ->  (b_sz, beam_width * n_drafts, src_len) ->
        # -> (b_sz * beam_width * n_drafts, src_len)
        # src_bw: tensor(b_sz * beam_width * n_drafts, src_len)
        # Example. b_sz: 2, src_len: 5, beam_width: 3, n_drafts: 1
        #          src: tensor([[1, 5, 20, 27, 2],
        #                       [1, 31, 2, 0, 0]])       - is a tensor of size (b_sz=2, src_len=5).
        #    -> src_bw: tensor([[ 1,  5, 20, 27,  2],
        #                       [ 1,  5, 20, 27,  2],
        #                       [ 1,  5, 20, 27,  2],
        #                       [ 1, 31,  2,  0,  0],
        #                       [ 1, 31,  2,  0,  0],
        #                       [ 1, 31,  2,  0,  0]])  - is a tensor of size (b_sz * beam_width * n_drafts, src_len=5).

        # Prepare first tokens for decoder (bs, max_len)
        y = torch.tensor([self.bos_token]).repeat(b_sz, 1).long().type_as(src)  # -> (b_sz, init_seq_len=1)

        # Decode for one step using decoder
        decoder_output = self.model(src, y)  # -> (b_sz, init_seq_len, vocab_size)
        logprob_decoder_output = torch.log(torch.softmax(decoder_output, dim=-1))  # -> (b_sz, 1, vocab_size)

        # check shape of the prediction
        vocab_size = logprob_decoder_output.shape[-1]

        probabilities, next_chars = torch.topk(logprob_decoder_output, self.beam_width,
                                               dim=-1, sorted=True)  # -> (b_sz, 1, beam_width), (b_sz, 1, beam_width)
        probabilities = probabilities.squeeze(1)  # -> (b_sz, beam_width)

        y = y.unsqueeze(1).repeat((1, 1, self.beam_width)).reshape(-1, 1)
        # (b_sz, init_seq_len=1) -> (b_sz, 1, init_seq_len) ->  (b_sz, 1, init_seq_len * beam_width) ->
        # -> (b_sz * beam_width, 1)

        next_chars = next_chars.reshape(-1, 1)  # -> (b_sz * beam_width, seq_len=1)
        y = torch.cat((y, next_chars), axis=-1)  # -> (b_sz * beam_width, curr_len)

        predictions = self.max_len - 1
        n_drafts, draft_len = self.drafts_of_len_1.size()

        drafts = self.drafts_of_len_1.repeat(b_sz * self.beam_width, 1).type_as(src)  # (n_drafts, draft_len=1) ->
        # -> (b_sz * beam_width * n_drafts, 1)
        curr_len = 2

        for i in range(predictions - 1):
            y = y.reshape(b_sz, self.beam_width, curr_len).unsqueeze(2).repeat(1, 1, self.n_drafts, 1).reshape(
                b_sz * self.beam_width * self.n_drafts, curr_len)
            # (b_sz * beam_width, curr_len) -> (b_sz * self.beam_width * self.n_drafts, curr_len)

            y = torch.cat((y, drafts),
                          axis=1)  # (b_sz * beam_width * n_drafts, curr_len), (b_sz * beam_width * n_drafts, 1) ->
            # -> (b_sz * beam_width * n_drafts, curr_len + 1)
            curr_len += 1
            outp = torch.log(torch.softmax(self.model(src_bw, y), dim=-1))
            #   -> (b_sz * beam_width * n_drafts, curr_len, vocab_size)
            drafts_probs = torch.gather(outp[:, :-1, :], dim=2, index=y[:, 1:].unsqueeze(2)).squeeze(2)
            #   -> (b_sz * beam_width * n_drafts, curr_len)
            drafts_probs = torch.sum(drafts_probs, dim=1)  # -> (b_sz * beam_width * n_drafts)
            draft_probs = drafts_probs.reshape(b_sz, self.beam_width * self.n_drafts)  # ->(b_sz, beam_width * n_drafts)
            top_draft_probs, inds = torch.topk(draft_probs, self.beam_width, dim=1, sorted=True)
            # (b_sz, beam_width * n_drafts) -> (b_sz, beam_width), (b_sz, beam_width)

            last_pred = outp[:, -1, :]  # -> (b_sz * beam_width * n_drafts, vocab_size)
            last_pred = last_pred.reshape(b_sz, self.beam_width * self.n_drafts, vocab_size)
            inds_tmp = inds.unsqueeze(-1).expand(b_sz, self.beam_width, vocab_size)
            next_probabilities = torch.gather(last_pred, 1, inds_tmp)  # -> (b_sz, beam_width, vocab_size)

            inds = inds.unsqueeze(-1).expand(b_sz, self.beam_width, curr_len)
            y = torch.gather(y.reshape(b_sz, self.beam_width * self.n_drafts, curr_len), 1, inds)
            y = y.reshape(b_sz * self.beam_width,
                          curr_len)  # (b_sz, beam_width, curr_len) ->(b_sz * beam_width, curr_len)
            if curr_len == self.max_len:
                break

            probabilities = top_draft_probs  # (b_sz, beam_width)

            probabilities = probabilities.unsqueeze(-1) + next_probabilities
            # (examples,b_w,1) + (examples,b_w,vocab_size) ->(examples,b_w,vocab_size)

            probabilities = probabilities.flatten(start_dim=1)  # (examples,b_w * vocab_size)
            probabilities, idx = probabilities.topk(k=self.beam_width, axis=-1,
                                                    sorted=True)  # (examples,b_w), (examples,b_w)
            next_chars = torch.remainder(idx, vocab_size).flatten().unsqueeze(-1)  # (examples * b_w,1)
            best_candidates = (idx / vocab_size).long()  # (examples,b_w)
            best_candidates += torch.arange(y.shape[0] // self.beam_width, device=src.device).unsqueeze(
                -1) * self.beam_width  # (beam_width * bs, 1)
            y = y[best_candidates].flatten(end_dim=-2)  # (beam_width * bs, 2+i)
            y = torch.cat((y, next_chars), axis=1)  # (beam_width * bs, 2+i)
            curr_len += 1
            if curr_len == self.max_len:
                break
            if (y == self.eos_token).sum(-1).bool().sum() == y.size()[0]:
                break
        y = y.reshape(b_sz, self.beam_width, -1)
        return y  # , probabilities  # (examples,b_w, max_len), (examples,b_w)

    def __str__(self):
        return f"Beam search decoding (beam_size={self.beam_size}, n_best={self.n_best}, max_len={self.max_len})"


class TranslationInferenceBeamSearchSpeculativeSrcAtoms:
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
        self.beam_width = beam_size
        self.n_best = n_best
        self.max_len = max_len
        self.pad_token = pad_token
        self.bos_token = bos_token
        self.eos_token = eos_token

    def generate(self, src: 'torch.LongTensor') -> 'torch.LongTensor':
        assert self.max_len > 1
        b_sz = src.shape[0]
        assert b_sz == 1  # Now we try this algorithm only for one reaction

        # Length of drafts is one now. They all are based on src atoms.
        drafts_of_len_1 = torch.unique(src[:, 1:], dim=1).transpose(0, 1)  # -> (n_drafts, 1)
        n_drafts, draft_len = drafts_of_len_1.size()

        # prepare src for the later cycle
        src_bw = src.repeat((self.beam_width * n_drafts, 1, 1)).transpose(0, 1).flatten(end_dim=1)
        # !(b_sz, src_len) -> (beam_width * n_drafts, b_sz, src_len) ->  (b_sz, beam_width * n_drafts, src_len) ->
        # -> (b_sz * beam_width * n_drafts, src_len)
        # src_bw: tensor(b_sz * beam_width * n_drafts, src_len)
        # Example. b_sz: 2, src_len: 5, beam_width: 3, n_drafts: 1
        #          src: tensor([[1, 5, 20, 27, 2],
        #                       [1, 31, 2, 0, 0]])       - is a tensor of size (b_sz=2, src_len=5).
        #    -> src_bw: tensor([[ 1,  5, 20, 27,  2],
        #                       [ 1,  5, 20, 27,  2],
        #                       [ 1,  5, 20, 27,  2],
        #                       [ 1, 31,  2,  0,  0],
        #                       [ 1, 31,  2,  0,  0],
        #                       [ 1, 31,  2,  0,  0]])  - is a tensor of size (b_sz * beam_width * n_drafts, src_len=5).

        # Prepare first tokens for decoder (bs, max_len)
        y = torch.tensor([self.bos_token]).repeat(b_sz, 1).long().type_as(src)  # -> (b_sz, init_seq_len=1)

        # Decode for one step using decoder
        decoder_output = self.model(src, y)  # -> (b_sz, init_seq_len, vocab_size)
        logprob_decoder_output = torch.log(torch.softmax(decoder_output, dim=-1))  # -> (b_sz, 1, vocab_size)

        # check shape of the prediction
        vocab_size = logprob_decoder_output.shape[-1]

        curr_log_probs, next_chars = torch.topk(logprob_decoder_output, self.beam_width,
                                                dim=-1, sorted=True)  # -> (b_sz, 1, beam_width), (b_sz, 1, beam_width)
        curr_log_probs = curr_log_probs.squeeze(1)  # -> (b_sz, beam_width)

        y = y.unsqueeze(1).repeat((1, 1, self.beam_width)).reshape(-1, 1)
        # (b_sz, init_seq_len=1) -> (b_sz, 1, init_seq_len) ->  (b_sz, 1, init_seq_len * beam_width) ->
        # -> (b_sz * beam_width, 1)

        next_chars = next_chars.reshape(-1, 1)  # -> (b_sz * beam_width, seq_len=1)
        candidates = torch.cat((y, next_chars), axis=-1)  # -> (b_sz * beam_width, curr_len)

        predictions = self.max_len - 1

        drafts = drafts_of_len_1.repeat(b_sz * self.beam_width, 1).type_as(src)  # (n_drafts, draft_len=1) ->
        # -> (b_sz * beam_width * n_drafts, 1)
        curr_len = 2

        for i in range(predictions - 1):
            y = candidates.reshape(b_sz, self.beam_width, curr_len).unsqueeze(2).repeat(1, 1, n_drafts, 1).reshape(
                b_sz * self.beam_width * n_drafts, curr_len)
            # (b_sz * beam_width, curr_len) -> (b_sz * self.beam_width * self.n_drafts, curr_len)

            y = torch.cat((y, drafts),
                          axis=1)  # (b_sz * beam_width * n_drafts, curr_len), (b_sz * beam_width * n_drafts, 1) ->
            # -> (b_sz * beam_width * n_drafts, curr_len + 1)
            curr_len += 1
            outp = torch.log(torch.softmax(self.model(src_bw, y), dim=-1))
            #   -> (b_sz * beam_width * n_drafts, curr_len, vocab_size)

            tmp_inds = torch.arange(0, b_sz * self.beam_width * n_drafts, n_drafts).type_as(src)
            #   -> (b_sz * beam_width)
            next_log_probs_for_bw = outp[tmp_inds, -2].unsqueeze(1).reshape(b_sz, self.beam_width,
                                                                            vocab_size) + curr_log_probs.unsqueeze(-1)
            #   -> (b_sz, beam_width, vocab_size)
            curr_log_probs, best_raw_inds = next_log_probs_for_bw.reshape(b_sz, -1).topk(self.beam_width, dim=-1)
            #   -> (b_sz, beam_width)
            best_next_inds = best_raw_inds % vocab_size
            #   -> (b_sz, beam_width)

            initial_seqs = candidates[best_raw_inds[0] // vocab_size]  # -> (beam_width, curr_len)
            candidates = torch.cat((initial_seqs, best_next_inds[0].unsqueeze(-1)),
                                   dim=-1)  # -> (beam_width, curr_len)
            if (candidates == self.eos_token).sum(-1).bool().sum() == candidates.size()[0]:
                break
            if curr_len == self.max_len:
                break

            # If best_next_inds contains only src atoms than we can save 1 model call
            tmp = torch.cat((drafts_of_len_1.transpose(0, 1), best_next_inds), dim=-1)
            # -> (b_sz, n_drafts + beam_width)
            assert b_sz == 1
            if torch.unique(tmp, dim=-1).size()[1] == n_drafts:
                drafts_log_probs = torch.gather(outp[:, :-1, :], dim=2, index=y[:, 1:].unsqueeze(2)).squeeze(2)
                #   -> (b_sz * beam_width * n_drafts, curr_len)
                drafts_log_probs = torch.sum(drafts_log_probs, dim=1)  # -> (b_sz * beam_width * n_drafts)
                drafts_log_probs = drafts_log_probs.reshape(b_sz,
                                                            self.beam_width * n_drafts)  # ->(b_sz, beam_width * n_drafts)
                top_draft_log_probs, inds = torch.topk(drafts_log_probs, self.beam_width, dim=1, sorted=True)
                # (b_sz, beam_width * n_drafts) -> (b_sz, beam_width), (b_sz, beam_width)

                last_pred = outp[:, -1, :]  # -> (b_sz * beam_width * n_drafts, vocab_size)
                last_pred = last_pred.reshape(b_sz, self.beam_width * n_drafts, vocab_size)
                inds_tmp = inds.unsqueeze(-1).expand(b_sz, self.beam_width, vocab_size)
                next_log_probs = torch.gather(last_pred, 1, inds_tmp)  # -> (b_sz, beam_width, vocab_size)

                inds = inds.unsqueeze(-1).expand(b_sz, self.beam_width, curr_len)
                y = torch.gather(y.reshape(b_sz, self.beam_width * n_drafts, curr_len), 1, inds)
                y = y.reshape(b_sz * self.beam_width,
                              curr_len)  # (b_sz, beam_width, curr_len) ->(b_sz * beam_width, curr_len)

                probabilities = top_draft_log_probs.unsqueeze(-1) + next_log_probs
                # (examples,b_w,1) + (examples,b_w,vocab_size) ->(examples,b_w,vocab_size)

                probabilities = probabilities.flatten(start_dim=1)  # (examples,b_w * vocab_size)
                curr_log_probs, idx = probabilities.topk(k=self.beam_width, axis=-1,
                                                         sorted=True)  # (examples,b_w), (examples,b_w)
                next_chars = torch.remainder(idx, vocab_size).flatten().unsqueeze(-1)  # (examples * b_w,1)
                best_candidates = (idx / vocab_size).long()  # (examples,b_w)
                best_candidates += torch.arange(y.shape[0] // self.beam_width, device=src.device).unsqueeze(
                    -1) * self.beam_width  # (beam_width * bs, 1)
                y = y[best_candidates].flatten(end_dim=-2)  # (beam_width * bs, 2+i)
                candidates = torch.cat((y, next_chars), axis=1)  # (beam_width * bs, 2+i)
                curr_len += 1
                if (candidates == self.eos_token).sum(-1).bool().sum() == candidates.size()[0]:
                    break
                if curr_len == self.max_len:
                    break
        candidates = candidates.reshape(b_sz, self.beam_width, -1)
        return candidates  # , probabilities  # (examples,b_w, max_len), (examples,b_w)

    def __str__(self):
        return f"Beam search decoding (beam_size={self.beam_size}, n_best={self.n_best}, max_len={self.max_len})"


class TranslationInferenceNucleusSpeculativeUnbatchedMinAccepted:

    def __init__(self,
                 model,  # TranslationModel
                 max_len: int,
                 n_speculative_tokens: int,
                 pad_token: int,
                 bos_token: int,
                 eos_token: int,
                 n_best: int = 5,
                 temperature: float = 500.,
                 nucleus: float = 0.995
                 ) -> None:
        self.model = model
        self.max_len = max_len
        self.pad_token = pad_token
        self.bos_token = bos_token
        self.eos_token = eos_token

        self.n_speculative_tokens = n_speculative_tokens
        self.nucleus = nucleus
        self.temperature = temperature
        self.n_best = n_best

    def __str__(self):
        return f"SpeculativeUnbatchedNucleusNew decoding (max_len={self.max_len}, nucleus={self.nucleus}, temperature={self.temperature})"

    def multinomial(self, probs, samples_num_coeff):
        num_samples = 12 * samples_num_coeff
        mask = probs <= 0.0001
        probs_num, vocab_size = mask.size()
        inds_num_t = torch.ones(probs_num, vocab_size).masked_fill_(mask, -float("inf")).softmax(-1) * num_samples
        inds_num_t = inds_num_t.int()

        tmp_t = inds_num_t.unsqueeze(-1).expand(probs_num, vocab_size, num_samples)
        mask_for_all_inds = (tmp_t - torch.arange(num_samples).unsqueeze(0).unsqueeze(
            0)) > 0  # (probs_num, vocab_size, num_samples)
        all_inds_for_all = torch.arange(1, vocab_size + 1).unsqueeze(0).unsqueeze(-1)  # -> (1, vocab_size, 1)
        masked_inds_for_all = (all_inds_for_all * mask_for_all_inds)  # ->(probs_num, vocab_size, num_samples)
        inds_for_all_probs = masked_inds_for_all[masked_inds_for_all > 0].reshape(probs_num,
                                                                                  num_samples) - 1  # Надо, чтобы функция это выдавала

        tmp_indexes = torch.multinomial(torch.ones(num_samples).softmax(-1).repeat(probs_num, 1), num_samples,
                                        replacement=False)  # (probs_num, num_samples)
        # А если вероятности не равны?
        return torch.gather(inds_for_all_probs, 1, tmp_indexes)  # (probs_num, num_samples)

    def sample(self, pred_logits):
        n_drafts, curr_len, vocab_size = pred_logits.size()
        pred_logits = pred_logits.reshape(n_drafts * curr_len, vocab_size)  # -> (n_drafts * curr_len, vocab_size)

        sorted_logits, sorted_indices = torch.sort(pred_logits, descending=True)  # -> (n_drafts * curr_len, vocab_size)
        cumulative_probs = torch.cumsum(sorted_logits.softmax(-1), dim=-1)  # -> (n_drafts * curr_len, vocab_size)

        # Remove tokens with cumulative probability above the threshold
        cumulative_probs = torch.roll(cumulative_probs, 1, dims=-1)
        cumulative_probs[:, 0] = 0
        keep_candidates_mask = cumulative_probs < self.nucleus  # -> (n_drafts * curr_len, vocab_size)
        max_num_of_positions = 4
        keep_candidates_mask[:, max_num_of_positions:] = False

        sorted_logits.masked_fill_(~keep_candidates_mask, float("-inf"))

        best_candidates_logits = torch.gather(sorted_logits, 1, sorted_indices.argsort(1))
        best_probs = (best_candidates_logits / self.temperature).softmax(-1)

        samples_num_coeff = 2
        sampled_tokens = self.multinomial(best_probs, samples_num_coeff)  # -> (n_drafts * curr_len, samples_num)

        # sampled_tokens = torch.multinomial(best_probs, replacement=True, num_sampleas=samples_num).squeeze(1)  # -> (n_drafts * curr_len, 1)
        # # -> (n_drafts * curr_len)
        #
        # sampled_tokens = sampled_tokens.reshape(n_drafts, curr_len)
        return sampled_tokens

    # def generate_with_offset(self, src: 'torch.LongTensor') -> 'torch.LongTensor':
    #     b_size, src_len = src.size()
    #
    #     src_pad_mask = (src == self.model.src_pad_token_i).bool()
    #     memory = self.model.encode_src(src, src_pad_mask)
    #
    #     src_unbatched = src.unsqueeze(1)
    #     src_pad_mask_unbatched = src_pad_mask.unsqueeze(1)
    #     memory_unbatched = memory.unsqueeze(1)
    #
    #     result = []
    #
    #     for i in range(b_size):
    #         src_unbatched_i = src_unbatched[i, :, 1:]
    #         src_unbatched_i_pads = (src_unbatched_i == self.pad_token).int().sum(-1)
    #         n_tokens_without_pads = src_unbatched_i.size(1) - src_unbatched_i_pads
    #         src_unbatched_i_unpadded = src_unbatched_i[:, :n_tokens_without_pads]
    #         drafts = src_unbatched_i_unpadded.unfold(-1, self.n_speculative_tokens, 1).squeeze(0)
    #         # -> (n_drafts, draft_len)
    #         n_drafts = drafts.size(0)
    #         iters = 0
    #
    #         generated_tokens = torch.full((1, 1), self.bos_token).type_as(src).long().repeat(self.n_best, 1)
    #         # -> (n_best, 1)
    #         memory_i = memory_unbatched[i].repeat(n_drafts * self.n_best, 1, 1)
    #         memory_pad_mask_i = src_pad_mask_unbatched[i].repeat(n_drafts * self.n_best, 1)
    #         finished_candidates = []
    #         while generated_tokens.size(1) < self.max_len:
    #             iters += 1
    #             n_best, curr_len = generated_tokens.size()
    #             draft_tokens = drafts.repeat(n_best, 1)  # -> (n_best * n_drafts, 1)
    #             inp = generated_tokens.unsqueeze(1).expand(n_best, n_drafts, curr_len).reshape(n_best * n_drafts,
    #                                                                                            curr_len)
    #             draft_sequence = torch.cat([inp, draft_tokens], dim=1)
    #             # (n_best * n_drafts, curr_len), (n_best * n_drafts, draft_len) -> (n_best * n_drafts, curr_len + draft_len)
    #             _, seq_len = draft_sequence.size()
    #             pos_enc_offset = (draft_sequence == self.pad_token).int().sum(-1).reshape(-1, 1)
    #             pred_logits = self.model.decode_tgt(draft_sequence,
    #                                                 memory_i[:n_best * n_drafts, :, :],
    #                                                 memory_pad_mask=memory_pad_mask_i[:n_best * n_drafts, :],
    #                                                 pos_enc_offset=pos_enc_offset)
    #             #   -> (n_best * n_drafts, curr_len + draft_len, vocab_size)
    #
    #             pred_tokens = self.sample(
    #                 pred_logits)  # (n_best * n_drafts, curr_len + draft_len, vocab_size) -> (n_best * n_drafts, curr_len + draft_len)
    #
    #             pred_tokens = pred_tokens.reshape(n_best, n_drafts, seq_len)
    #             pred_tokens = pred_tokens[:, :, -(draft_tokens.size(
    #                 1) + 1):]  # (n_best, n_drafts, curr_len + draft_len) -> (n_best, n_drafts, draft_len + 1)
    #             verification = draft_tokens.reshape(n_best, n_drafts, -1) == pred_tokens[:, :,
    #                                                                          :-1]  # (n_best, n_drafts, draft_len + 1) -> (n_best, n_drafts, draft_len)
    #             _range = verification.cumsum(-1)  # (n_best, n_drafts, draft_len)
    #             accepted_in_drafts = (torch.arange(1, verification.size(2) + 1).unsqueeze(0).unsqueeze(0).type_as(
    #                 _range) == _range)  # (n_best, n_drafts, draft_len)
    #             n_accepted_in_drafts = accepted_in_drafts.sum(-1)  # (n_best, n_drafts, draft_len) -> (n_best, n_drafts)
    #             n_accepted_in_drafts = n_accepted_in_drafts.topk(1, dim=-1)  # (n_best, n_drafts) -> (n_best, 1)
    #             draft_i = n_accepted_in_drafts.indices  # (n_best, 1)
    #             n_accepted = n_accepted_in_drafts.values  # (n_best, 1)
    #
    #             # min_num_accepted = torch.min(n_accepted).item()
    #             best_candidates = []
    #             for j in range(n_best):
    #                 min_accepted_in_j = n_accepted[j]
    #                 draft_id = draft_i[j]
    #                 pred_tokens_for_j = pred_tokens[j, draft_id, :min_accepted_in_j + 1]  # -> (min_accepted_in_i + 1)
    #                 initial = generated_tokens[j].unsqueeze(0)  # (n_best, curr_len) -> (curr_len)
    #                 pads_in_initial = (initial == self.pad_token).int().sum(-1)
    #                 initial = initial[:, pads_in_initial:]
    #                 candidate = torch.cat((initial, pred_tokens_for_j), dim=-1).squeeze(0)
    #                 if (candidate == self.eos_token).sum().item() > 0:
    #                     finished_candidates.append(candidate)
    #                 else:
    #                     best_candidates.append(candidate)
    #
    #             if len(best_candidates) == 0:
    #                 break
    #             generated_tokens = pad_sequence(best_candidates, padding_value=self.pad_token, batch_first=True)
    #             generated_tokens = move_pads_to_the_left(generated_tokens, self.pad_token)
    #
    #         result.append(pad_sequence(finished_candidates, padding_value=self.pad_token, batch_first=True))
    #     return result

    def generate(self, src: 'torch.LongTensor') -> 'torch.LongTensor':
        b_size, src_len = src.size()

        src_pad_mask = (src == self.model.src_pad_token_i).bool()
        memory = self.model.encode_src(src, src_pad_mask)

        src_unbatched = src.unsqueeze(1)
        src_pad_mask_unbatched = src_pad_mask.unsqueeze(1)
        memory_unbatched = memory.unsqueeze(1)

        result = [[] for _ in range(b_size)]

        for i in range(b_size):
            draft_tokens = src_unbatched[i, :, 1:].unfold(-1, self.n_speculative_tokens, 1).squeeze(0)
            # -> (n_drafts, draft_len)
            n_drafts = draft_tokens.size(0)
            draft_tokens = draft_tokens.repeat(self.n_best, 1)  # -> (n_best * n_drafts, 1)
            iters = 0

            generated_tokens = torch.full((1, 1), self.bos_token).type_as(src).long().repeat(self.n_best, 1)
            # -> (n_best, 1)
            memory_i = memory_unbatched[i].repeat(n_drafts * self.n_best, 1, 1)
            memory_pad_mask_i = src_pad_mask_unbatched[i].repeat(n_drafts * self.n_best, 1)
            while generated_tokens.size(1) < self.max_len:
                iters += 1
                n_best, curr_len = generated_tokens.size()
                inp = generated_tokens.unsqueeze(1).expand(n_best, n_drafts, curr_len).reshape(n_best * n_drafts,
                                                                                               curr_len)
                draft_sequence = torch.cat([inp, draft_tokens], dim=1)
                # (n_best * n_drafts, curr_len), (n_best * n_drafts, draft_len) -> (n_best * n_drafts, curr_len + draft_len)
                _, seq_len = draft_sequence.size()
                pred_logits = self.model.decode_tgt(draft_sequence,
                                                    memory_i,
                                                    memory_pad_mask=memory_pad_mask_i)
                #   -> (n_best * n_drafts, curr_len + draft_len, vocab_size)

                pred_tokens = self.sample(
                    pred_logits)  # (n_best * n_drafts, curr_len + draft_len, vocab_size) -> (n_best * n_drafts, curr_len + draft_len)

                pred_tokens = pred_tokens.reshape(self.n_best, n_drafts, seq_len)
                pred_tokens = pred_tokens[:, :, -(draft_tokens.size(
                    1) + 1):]  # (n_best, n_drafts, curr_len + draft_len) -> (n_best, n_drafts, draft_len + 1)
                verification = draft_tokens.reshape(n_best, n_drafts, -1) == pred_tokens[:, :,
                                                                             :-1]  # (n_best, n_drafts, draft_len + 1) -> (n_best, n_drafts, draft_len)
                _range = verification.cumsum(-1)  # (n_best, n_drafts, draft_len)
                accepted_in_drafts = (torch.arange(1, verification.size(2) + 1).unsqueeze(0).unsqueeze(0).type_as(
                    _range) == _range)  # (n_best, n_drafts, draft_len)
                n_accepted_in_drafts = accepted_in_drafts.sum(2)  # (n_best, n_drafts, draft_len) -> (n_best, n_drafts)
                n_accepted_in_drafts = n_accepted_in_drafts.topk(1, dim=1)  # (n_best, n_drafts) -> (n_best, 1)
                draft_i = n_accepted_in_drafts.indices  # (n_best, 1)
                n_accepted = n_accepted_in_drafts.values  # (n_best, 1)

                min_num_accepted = torch.min(n_accepted).item()

                # pred_tokens = pred_tokens[draft_i, :n_accepted + 1]  # (n_drafts, curr_len + draft_len, n_best)
                n_best, n_drafts, pred_len = pred_tokens.size()  # (n_best, n_drafts, draft_len + 1)
                draft_i = draft_i.unsqueeze(-1).expand(n_best, 1, pred_len)
                pred_tokens = torch.gather(pred_tokens, dim=1, index=draft_i)  # -> (n_best, 1, draft_len + 1)
                pred_tokens = pred_tokens.squeeze(1)  # -> (n_best, draft_len + 1)
                pred_tokens = pred_tokens[:, :min_num_accepted + 1]  # -> (n_best, min_num_accepted + 1)

                generated_tokens = torch.cat(
                    (generated_tokens,
                     pred_tokens),
                    dim=1
                )  # (n_best, curr_len), (n_best, min_num_accepted + 1) -> (n_best, new_curr_len)
                if (generated_tokens == self.eos_token).sum(-1).bool().sum().item() == n_best:
                    break
            result[i] = generated_tokens
            print("src[i] ", src[i])
            print("result[i] ", result[i])
        return result


def multinomial(probs, samples_num_coeff):
    num_samples = 12 * samples_num_coeff
    mask = ~(probs > 0.)
    probs_num, vocab_size = mask.size()
    inds_num_t = torch.ones(probs_num, vocab_size).masked_fill_(mask, -float("inf")).softmax(-1) * num_samples
    inds_num_t = inds_num_t.int()

    tmp_t = inds_num_t.unsqueeze(-1).expand(probs_num, vocab_size, num_samples)
    mask_for_all_inds = (tmp_t - torch.arange(num_samples).unsqueeze(0).unsqueeze(
        0)) > 0  # (probs_num, vocab_size, num_samples)
    all_inds_for_all = torch.arange(1, vocab_size + 1).unsqueeze(0).unsqueeze(-1)  # -> (1, vocab_size, 1)
    masked_inds_for_all = (all_inds_for_all * mask_for_all_inds)  # ->(probs_num, vocab_size, num_samples)
    inds_for_all_probs = masked_inds_for_all[masked_inds_for_all > 0].reshape(probs_num,
                                                                              num_samples) - 1  # Надо, чтобы функция это выдавала

    tmp_indexes = torch.multinomial(torch.ones(num_samples).softmax(-1).repeat(probs_num, 1), num_samples,
                                    replacement=False)  # (probs_num, num_samples)
    return torch.gather(inds_for_all_probs, 1, tmp_indexes)  # (probs_num, num_samples)


class TranslationInferenceNucleusClassic:

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
        self.max_len = max_len
        self.pad_token = pad_token
        self.bos_token = bos_token
        self.eos_token = eos_token
        self.nucleus = 0.995
        assert self.max_len > 1

    def sample_0(self, next_token_pred_logits, current_lines, current_log_probs):
        """
        """
        b_sz, n_candidates, curr_len = current_lines.size()
        b_sz, n_candidates = current_log_probs.size()
        b_sz, n_candidates, vocab_size = next_token_pred_logits.size()  # (b_sz, n_candidates, vocab_size)

        potential_log_probs = next_token_pred_logits.softmax(-1).log() + current_log_probs.unsqueeze(-1)
        #    -> (b_sz, n_candidates, vocab_size)

        logits_for_sampling = next_token_pred_logits.reshape(-1, vocab_size)  # -> (b_sz * n_candidates, vocab_size)

        # actually we don't need to generate the pad token but it can change final
        # log_prob for the lines that already have eos
        # logits_for_sampling[:, self.pad_token] = -float("inf")

        sorted_logits, sorted_indices = torch.sort(logits_for_sampling,
                                                   descending=True)  # -> (n, vocab_size)
        cumulative_probs = torch.cumsum(sorted_logits.softmax(-1), dim=-1)  # -> (n, vocab_size)

        cumulative_probs = torch.roll(cumulative_probs, 1, dims=-1)
        # Remove tokens with cumulative probability above the threshold
        keep_candidates_mask = cumulative_probs < self.nucleus  # -> (n, vocab_size)
        # To sample at least one token
        keep_candidates_mask[:, 0] = True
        # To avoid sampling more than beam_size indexes
        keep_candidates_mask[:, 2 * self.beam_size:] = False

        best_candidates_inds_for_sampling_bool = torch.gather(keep_candidates_mask, 1, sorted_indices.argsort(1))
        # -> (b_sz * n_candidates, vocab_size)

        assert b_sz == 1
        best_log_probs = potential_log_probs.reshape(-1, vocab_size)[best_candidates_inds_for_sampling_bool]
        # -> (num_samples)
        corresponding_lines_inds, best_next_inds = best_candidates_inds_for_sampling_bool.nonzero(as_tuple=True)
        #   -> (num_samples)

        sorted_log_probs, sorted_indices = torch.sort(best_log_probs, descending=True)  # -> (num_samples)
        # We keep only the beam_width best probs:
        best_log_probs = sorted_log_probs[:self.beam_size]  # -> (1>=new_n_candidates<=beam_width)
        new_n_candidates = best_log_probs.size()[0]
        best_indices = best_next_inds[sorted_indices[:self.beam_size]]  # -> (new_n_candidates)
        corresponding_lines_inds = corresponding_lines_inds[sorted_indices[:self.beam_size]]
        #   -> (new_n_candidates)
        new_lines = torch.cat((current_lines[0][corresponding_lines_inds], best_indices.reshape(new_n_candidates, 1)),
                              dim=-1)
        #   -> (new_n_candidates, curr_len+1)

        # To imitate b_sz at this stage of code development
        new_lines = new_lines.unsqueeze(0)
        best_log_probs = best_log_probs.unsqueeze(0)
        return new_lines, best_log_probs  # (1, new_n_candidates, curr_len+1), (1, new_n_candidates)

    def sample(self, next_token_pred_logits, current_lines, current_log_probs):
        """
        """
        # sample with renorm
        b_sz, n_candidates, curr_len = current_lines.size()
        b_sz, n_candidates = current_log_probs.size()
        b_sz, n_candidates, vocab_size = next_token_pred_logits.size()  # (b_sz, n_candidates, vocab_size)

        logits_for_sampling = next_token_pred_logits.reshape(-1, vocab_size)  # -> (b_sz * n_candidates, vocab_size)

        sorted_logits, sorted_indices = torch.sort(logits_for_sampling,
                                                   descending=True)  # -> (n, vocab_size)
        cumulative_probs = torch.cumsum(sorted_logits.softmax(-1), dim=-1)  # -> (n, vocab_size)

        # Remove tokens with cumulative probability above the threshold
        keep_candidates_mask = cumulative_probs < self.nucleus  # -> (n, vocab_size)
        # To sample at least one token
        keep_candidates_mask[:, 0] = True

        # To avoid sampling more than beam_size indexes
        keep_candidates_mask[:, 2 * self.beam_size:] = False

        sorted_logits.masked_fill_(~keep_candidates_mask, float("-inf"))

        best_candidates_logits = torch.gather(sorted_logits, 1, sorted_indices.argsort(1))
        next_log_probs_masked = best_candidates_logits.softmax(-1).log()  # (n, vocab_size)

        best_candidates_inds_for_sampling_bool = torch.gather(keep_candidates_mask, 1, sorted_indices.argsort(1))
        # -> (b_sz * n_candidates, vocab_size)

        # potential_log_probs = next_token_pred_logits.softmax(-1).log().flatten(end_dim=1) + current_log_probs.flatten(end_dim=1).unsqueeze(-1)
        # #    -> (b_sz * n_candidates, vocab_size)
        potential_log_probs = next_log_probs_masked + current_log_probs.flatten(end_dim=1).unsqueeze(-1)
        #    -> (b_sz * n_candidates, vocab_size)

        assert b_sz == 1
        best_log_probs = potential_log_probs[best_candidates_inds_for_sampling_bool]
        # -> (num_samples)
        corresponding_lines_inds, best_next_inds = best_candidates_inds_for_sampling_bool.nonzero(as_tuple=True)
        #   -> (num_samples)

        sorted_log_probs, sorted_indices = torch.sort(best_log_probs, descending=True)  # -> (num_samples)
        # We keep only the beam_width best probs:
        best_log_probs = sorted_log_probs[:self.beam_size]  # -> (1>=new_n_candidates<=beam_width)
        new_n_candidates = best_log_probs.size()[0]
        best_indices = best_next_inds[sorted_indices[:self.beam_size]]  # -> (new_n_candidates)
        corresponding_lines_inds = corresponding_lines_inds[sorted_indices[:self.beam_size]]
        #   -> (new_n_candidates)
        new_lines = torch.cat((current_lines[0][corresponding_lines_inds], best_indices.reshape(new_n_candidates, 1)),
                              dim=-1)
        #   -> (new_n_candidates, curr_len+1)

        # To imitate b_sz at this stage of code development
        new_lines = new_lines.unsqueeze(0)
        best_log_probs = best_log_probs.unsqueeze(0)
        return new_lines, best_log_probs  # (1, new_n_candidates, curr_len+1), (1, new_n_candidates)

    def __str__(self):
        return f"Beam search decoding (beam_size={self.beam_size}, n_best={self.n_best}, max_len={self.max_len})"

    def generate(self, src: 'torch.LongTensor') -> 'torch.LongTensor':
        bs, _ = src.size()
        assert bs == 1
        # Prepare first tokens for decoder (bs, max_len)
        n_candidates = 1
        y = torch.tensor([self.bos_token]).repeat(bs, 1).long().type_as(src)  # -> (bs,n_candidates=1)
        current_log_probs = torch.tensor([0]).repeat(bs, 1).type_as(src).float()  # -> (bs, n_candidates=1)

        # Decode for one step using decoder
        decoder_output = self.model(src, y)  # -> (bs, 1, dict_len)
        logprob_decoder_output = torch.log(torch.softmax(decoder_output, dim=-1))

        # check shape of the prediction
        vocab_size = logprob_decoder_output.shape[-1]

        curr_lines, current_log_probs = self.sample(decoder_output, y.reshape(bs, 1, -1), current_log_probs)
        # -> (bs=1, 1 <= n_candidates <= beam_width, curr_len+1), (bs=1, n_candidates)

        predictions = self.max_len - 1

        for i in range(predictions - 1):
            bs, n_candidates, curr_len = curr_lines.size()
            src_bw = src.repeat((n_candidates, 1, 1)).transpose(0, 1).flatten(end_dim=1)
            # -> (b_s * n_candidates, length)
            next_logits = self.model(src_bw, curr_lines.reshape(-1, curr_len))[:, -1, :]
            #   -> (bs * n_candidates, vocab_size)
            curr_lines, current_log_probs = self.sample(next_logits.reshape(bs, n_candidates, -1), curr_lines,
                                                        current_log_probs)
            # -> (bs=1, 1 <= n_candidates <= beam_width, curr_len+1), (bs=1, n_candidates)
            if (curr_lines.flatten(end_dim=1) == self.eos_token).sum(-1).bool().sum().item() == bs * curr_lines.shape[
                1]:
                break
        return curr_lines  # (bs=1, 1 <= n_candidates <= beam_width, len)


if __name__ == '__main__':
    from models import VanillaTransformerTranslationLightningModule
    from synthetic_tasks.copy_sequence.tokenizer import AsciiTokenizer
    from torch.nn.utils.rnn import pad_sequence

    torch.manual_seed(0)


    def check_example(model: VanillaTransformerTranslationLightningModule,
                      tokenizer: AsciiTokenizer,
                      examples: list[str]):
        res = model.generator.generate(
            pad_sequence([torch.tensor(tokenizer.encode(s)) for s in examples], padding_value=0,
                         batch_first=True)
        )
        print(res)
        print(tokenizer.decode_batch(res.squeeze(1).numpy()))


    tkz = AsciiTokenizer()
    tkz.load_vocab("/home/ma/work/translation-transformer/data/copy_sequence/vocabs/vocab.json")
    model = VanillaTransformerTranslationLightningModule(src_tokenizer=tkz, tgt_tokenizer=tkz,
                                                         generation="greedy_speculative",
                                                         beam_size=2,
                                                         max_len=20,
                                                         n_speculative_tokens=2
                                                         )
    ckpt = torch.load("/home/ma/work/translation-transformer/lightning_logs/copy_sequence/checkpoints/last.ckpt",
                      map_location=torch.device("cpu"))
    model.load_state_dict(ckpt["state_dict"])
    check_example(
        model,
        tkz,
        examples=["B"]
    )

#
#
#
# class TranslationInferenceBeamSearchDraft1:
#
#     def __init__(self,
#                  model,  # TranslationModel
#                  beam_size: int,
#                  n_best: int,
#                  max_len: int,
#                  pad_token: int,
#                  bos_token: int,
#                  eos_token: int):
#         self.model = model
#         self.beam_size = beam_size
#         self.n_best = n_best
#         self.max_len = max_len
#         self.pad_token = pad_token
#         self.bos_token = bos_token
#         self.eos_token = eos_token
#         self.drafts_of_len_1 = torch.tensor(self.calculate_drafts_len_1())
#         self.n_drafts = self.drafts_of_len_1.size()[0]
#         print("n_drafts ", self.n_drafts)
#
#     def calculate_drafts_len_1(self):
#         drafts = []
#         # vocab_len = 5
#         vocab = [i for i in range(56)]  # {0: 0, 1: 2, 3: 4, 4: 5, 5: 6, 6: 7, 7: 41}
#         for i in vocab:
#             if i == self.pad_token or i == self.bos_token:
#                 continue
#             drafts.append([i])
#         return drafts
#
#     def __str__(self):
#         return f"Beam search decoding (beam_size={self.beam_size}, n_best={self.n_best}, max_len={self.max_len})"
#
#     def probs_of_drafts(self, src, src_pad_mask, batch_decoder_input):
#         pass
#
#     def generate(self, src: 'torch.LongTensor') -> 'torch.LongTensor':
#         """Batch Beam Seach Decoding for RNN
#               Args:
#                   src: (bs, max_len)
#                   n_best: The number of output sequences for each input
#
#               Returns:
#                   n_best_list: Decoded N-best results. (bs, T)
#               """
#
#         b_sz, max_len = src.size()
#         assert b_sz == 1
#         # Prepare first tokens for decoder (bs, max_len)
#         batch_decoder_input = torch.tensor([self.bos_token]).repeat(b_sz, 1).long()
#         src_pad_mask = (src == self.model.src_pad_token_i).bool()  # (b_sz, max_len)
#         memory = self.model.encode_src(src, src_pad_mask)  # (b_sz, max_len, dim)
#         logits = self.model.decode_tgt(batch_decoder_input,
#                                        memory,
#                                        memory_pad_mask=src_pad_mask)[:, 0,
#                  :]  # (bsz, max_len, dict_len) -> (bsz, dict_len)
#         probs = torch.softmax(logits, dim=-1)
#         probs[:, self.pad_token] = -1.
#         curr_probs, seqs = probs.topk(self.beam_size, dim=-1)  # b_sz, beam_width
#         bos = torch.tensor([self.bos_token]).repeat(self.beam_size, 1).long()  # (beam_size,1)
#         pred_lsts = []
#
#         for i in range(b_sz):
#             curr_seqs = torch.cat((bos, seqs[i].unsqueeze(1)), dim=1)  # beam_width, 2
#             memory_i = memory[i].unsqueeze(0).expand(self.beam_size * self.n_drafts, -1, -1)
#             src_pad_mask_i = src_pad_mask[i].unsqueeze(0).expand(self.beam_size * self.n_drafts, -1)
#             best_candidates = []
#             num_done = 0
#             curr_probs = torch.ones(self.beam_size)
#             n_best_list = self.bm_search_drafts_len_1(memory_i, src_pad_mask_i, curr_probs, curr_seqs,
#                                                       best_candidates, num_done)  # curr_probs[i]
#             tmp = torch.cat([F.pad(t.unsqueeze(0), pad=(0, self.max_len - t.shape[0], 0, 0)) for t in n_best_list],
#                             dim=0)  # (n_best,max_len)
#             pred_lsts.append(tmp.unsqueeze(0))
#         # print("pred_lsts", pred_lsts)
#         return torch.cat(pred_lsts, dim=0)  # (b_sz,n_best,max_len)
#
#     def bm_search_drafts_len_1(self, memory, src_pad_mask, curr_probs, inds_curr_seq, best_candidates, num_done):
#         # "recursive_bm_srch_eos"
#         # memory:
#         # src_pad_mask:
#         # curr_probs: (beam_width)
#         # inds_curr_seq (beam_width, curr_len)
#         # best_candidates: list
#
#         beam_width, curr_len = inds_curr_seq.size()
#         if curr_len == self.max_len:
#             return best_candidates[:self.n_best]  # (n_best, self.max_len)
#
#         prelim_probs = curr_probs.unsqueeze(1).expand(beam_width, self.n_drafts).reshape(
#             self.n_drafts * beam_width)  # (beam_width) -> (beam_width, 1) -> (beam_width, N_drafts) ->
#         # -> (beam_width * N_drafts)
#
#         inp = torch.cat((inds_curr_seq.unsqueeze(1).expand(beam_width, self.n_drafts, -1),
#                          self.drafts_of_len_1.unsqueeze(0).expand(beam_width, self.n_drafts, -1)), dim=-1)
#         # (beam_width, N_drafts, curr_len) (beam_width, N_drafts, 1) ->
#         # -> (beam_width, N_drafts, curr_len + 1)
#         inp = inp.reshape(beam_width * self.n_drafts, curr_len + 1)
#         draft_logits = self.model.decode_tgt(inp,
#                                              memory,
#                                              memory_pad_mask=src_pad_mask)  # (beam_width * N_drafts, curr_len + 1, dict_len)
#         draft_probs = torch.softmax(draft_logits, dim=-1)  # (beam_width * N_drafts, curr_len + 1, dict_len)
#
#         # ! We calculate probs of drafts:
#         inp_probs = torch.prod(torch.gather(draft_probs[:, :-1, :], 2, inp[:, 1:].unsqueeze(-1)).squeeze(-1),
#                                dim=1)  # we don't need to predict bos
#         # draft_probs[:, :-1, :]: tensor of size (beam_width * self.n_drafts, curr_len, dict_len)
#         # inp[:, 1:].unsqueeze(-1): tensor of size (beam_width * self.n_drafts, curr_len, 1)
#
#         # (self.n_drafts * beam_width, curr_len + 1) -> (self.n_drafts * beam_width)
#         # inp_probs: tensor of size (self.n_drafts * beam_width)
#
#         inp_probs *= prelim_probs  ###
#
#         best_drafts_probs, best_drafts_inds = inp_probs.topk(beam_width, dim=-1)
#         best_inp_seqs = inp[best_drafts_inds]  # -> (beam_width, curr_len + 1)
#
#         last_pred_for_best_inp_seqs = draft_probs[best_drafts_inds][-1:, :]  # (beam_width, dict)
#         new_probs = best_drafts_probs.unsqueeze(1) * last_pred_for_best_inp_seqs  # -> (beam_width, dict)
#         beam_width, dict_len = new_probs.size()
#         curr_probs, flattened_best_ids = new_probs.reshape(beam_width * dict_len).topk(beam_width,
#                                                                                        dim=-1)  # -> (beam_width)
#
#         best_inds_for_old_seq = (flattened_best_ids / beam_width).long()  # beam_width
#
#         best_old_seqs = best_inp_seqs[
#             best_inds_for_old_seq]  # (beam_width, curr_len)[beam_width] ->(beam_width, curr_len)
#
#         best_predicted_rg_ids = last_pred_for_best_inp_seqs.flatten()[flattened_best_ids]  # beam_width
#
#         best_seqs = torch.cat([best_old_seqs, best_predicted_rg_ids.unsqueeze(-1)],
#                               dim=-1)  # (beam_width, curr_len + 2)
#         num_finished = torch.sum((best_seqs == self.eos_token)[:, -2:]).item()
#
#         if num_finished:
#             best_candidates += [t.cpu() for t in
#                                 best_seqs[torch.sum((best_seqs == self.eos_token)[:, -2:],
#                                                     dim=-1).bool()]]  # num_finished, curr_len + 2
#             curr_probs[torch.sum((best_seqs == self.eos_token)[:, -2:], dim=-1).bool()] = -1.
#
#             num_done += num_finished
#             if num_done >= self.n_best:
#                 # Что если num_done больше n_best?
#                 return best_candidates[:self.n_best]  # list of tensors(num_done, different_lens)
#
#         rg_ids_chains = self.bm_search_drafts_len_1(memory, src_pad_mask, curr_probs, best_seqs,
#                                                     best_candidates, num_done)
#         return rg_ids_chains
