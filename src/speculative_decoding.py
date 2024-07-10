import torch


def move_pads_to_the_right(arr: torch.Tensor, pad_token: int = 0) -> torch.Tensor:
    """
    Moves pad tokens "pad_tokens" from the left side of the tensor to the right.
    """
    n_rows, n_cols = arr.size()
    dim_indices = torch.arange(n_cols).type_as(arr).long().repeat(n_rows).reshape(n_rows, -1)
    pad_count = (arr == pad_token).sum(1)
    indices = (dim_indices + pad_count.unsqueeze(1)) % n_cols
    return torch.gather(arr, dim=1, index=indices)


def move_pads_to_the_left(arr: torch.Tensor, pad_token: int = 0) -> torch.Tensor:
    """
    Moves pad tokens "pad_tokens" from the right side of the tensor to the left.
    """
    n_rows, n_cols = arr.size()
    dim_indices = torch.arange(n_cols).type_as(arr).long().repeat(n_rows).reshape(n_rows, -1)
    eos_index = (arr == pad_token).sum(1)
    indices = (dim_indices - eos_index.unsqueeze(1)) % n_cols
    return torch.gather(arr, dim=1, index=indices)


def trim_left_pads(tensor_t, pad_id: int):
    """
    Remove columns from the left that contain only PAD tokens.
    tensor_t is supposed to have PAD tokens only on the left
    """
    rows_num, _ = tensor_t.size()
    # number of left columns filled with the pad id
    padded_columns_num = ((tensor_t == pad_id).sum(0) == rows_num).sum()
    return tensor_t[:, padded_columns_num:]


class TranslationInferenceBeamSearchSpeculativeUnbatched:

    def __init__(self,
                 model,  # TranslationModel
                 max_len: int,
                 n_speculative_tokens: int,
                 pad_token: int,
                 bos_token: int,
                 eos_token: int,
                 n_best: int,
                 nucleus: float,
                 max_num_of_drafts: int = 23,
                 draft_mode: bool = True
                 ) -> None:
        self.model = model
        self.max_len = max_len
        self.pad_token = pad_token
        self.bos_token = bos_token
        self.eos_token = eos_token
        self.draft_mode = draft_mode
        if self.draft_mode:
            self.draft_len = n_speculative_tokens
            self.max_drafts_num = max_num_of_drafts
        else:
            self.draft_len = 1
            self.max_drafts_num = 1
        self.nucleus_for_sampling = nucleus
        self.n_best = n_best
        self.extra_pad = -1
        self.max_num_positions_for_sampling = 1 * n_best
        self.log_prob_pad = 1
        self.log_prob_extra_pad = 2

    def __str__(self):
        return f"NucleusSpeculativeUnbatched decoding (max_len={self.max_len}, nucleus={self.nucleus_for_sampling})"

    def sample(self, curr_lines, curr_log_probs_history, pred_logits, chosen_drafts, n_accepted=None):
        """
        :param curr_lines: tensor (n_candidates, len_),
        :param curr_log_probs_history: tensor (n_candidates, len_),
        :param pred_logits: tensor (n_candidates, draft_len + 1, vocab_size),
        :param chosen_drafts: tensor (n_candidates, draft_len)
        :param n_accepted: tensor (n_candidates) or None = None
        :return:
          ->  new_lines: tensor (num_lines, len),
              new_log_probs_history: tensor (num_lines, len)
        """

        n_candidates, draft_len_plus_one, vocab_size = pred_logits.size()
        assert draft_len_plus_one - 1 == self.draft_len
        masked_logits = mask_with_num_logits_according_nucleus(pred_logits, nucleus=self.nucleus_for_sampling,
                                                               max_num_of_unmasked_positions=self.max_num_positions_for_sampling,
                                                               num=0.)
        # -> (n_candidates, draft_len + 1, vocab_size)

        tmp_range = torch.arange(draft_len_plus_one).type_as(curr_lines).unsqueeze(0)
        #   -> (1, draft_len + 1)
        if n_accepted is None:
            n_accepted = self.calculate_n_accepted_in_drafts(chosen_drafts.unsqueeze(1),
                                                             masked_logits.unsqueeze(1)).squeeze(-1)
        else:
            not_fully_accepted_inds_bool = n_accepted != self.draft_len
            if not_fully_accepted_inds_bool.sum().item() != 0:
                chosen_drafts[not_fully_accepted_inds_bool] = \
                    chosen_drafts[not_fully_accepted_inds_bool].scatter_(
                        index=n_accepted[not_fully_accepted_inds_bool].unsqueeze(-1),
                        dim=1, value=self.pad_token)
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

        new_candidates = trim_left_pads(new_candidates, self.pad_token)
        log_prob_history = trim_left_pads(log_prob_history, self.log_prob_pad)

        return new_candidates, log_prob_history

    def generate(self, src: 'torch.LongTensor') -> list['torch.LongTensor']:
        b_size, src_len = src.size()

        src_pad_mask = (src == self.model.src_pad_token_i).bool()
        memory = self.model.encode_src(src, src_pad_mask)

        src_unbatched = src.unsqueeze(1)
        src_pad_mask_unbatched = src_pad_mask.unsqueeze(1)
        memory_unbatched = memory.unsqueeze(1)

        result = []

        for i in range(b_size):
            drafts = self.build_drafts(src_unbatched[i], self.draft_mode)
            # -> (self.max_drafts_num, self.draft_len)

            n_drafts, draft_len = drafts.size()
            assert draft_len == self.draft_len
            iters = 0

            generated_tokens = torch.full((1, 1), self.bos_token).type_as(src).long()
            # -> (1, 1)
            memory_i = memory_unbatched[i].repeat(n_drafts, 1, 1)
            memory_pad_mask_i = src_pad_mask_unbatched[i].repeat(n_drafts, 1)

            finished_candidates_t = None

            log_probs_history = torch.full((1, 1), 0.).type_as(src).float()
            while (generated_tokens.size(1) + self.draft_len + 1) < self.max_len and iters < self.max_len:
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
                masked_probs = mask_with_num_logits_according_nucleus(pred_logits, nucleus=0.9975,
                                                                      max_num_of_unmasked_positions=5,
                                                                      num="-inf").softmax(-1)
                #   -> (n_candidates * n_drafts, draft_len + 1, vocab_size)
                masked_probs = masked_probs.reshape(n_candidates, n_drafts, draft_len + 1, vocab_size)
                draft_tokens = draft_tokens.reshape(n_candidates, n_drafts, draft_len)

                n_accepted_in_drafts = self.calculate_n_accepted_in_drafts(draft_tokens, masked_probs)
                #   ->(num, n_drafts)

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
                    self.sample(generated_tokens, log_probs_history, pred_logits,
                                chosen_drafts, n_accepted.squeeze(-1))
                # generated_tokens: (n_candidates, curr_len),
                # log_probs_history: (n_candidates, curr_len),
                # pred_logits: (n_candidates, draft_len + 1, vocab_size),
                # chosen_drafts: (n_candidates, draft_len)
                # n_accepted: (n_candidates) or None = None
                #   ->  new_candidates: (num_lines, len),
                #       new_log_probs_history: (num_lines, len)

                if finished_candidates_t is None:
                    new_candidates, new_log_probs_history = self.sort(new_candidates, new_log_probs_history,
                                                                      descending=True)
                    finished_bool_ids = (new_candidates == self.eos_token).sum(-1).bool()
                    num_new_finished = finished_bool_ids.sum().item()
                    if num_new_finished > 0:
                        finished_candidates_t = self.make_left_pad_tail(new_candidates[finished_bool_ids][:self.n_best])
                        #   -> (min(num_new_finished, n_best), max_len)
                        finished_candidates_log_probs_t = new_log_probs_history[finished_bool_ids][:self.n_best][:, -1]
                        #   -> (min(num_new_finished, n_best))

                    finished_bool_ids = finished_bool_ids[:self.n_best]
                    if finished_bool_ids.sum().item() == finished_bool_ids.shape[0]:
                        break
                    generated_tokens = trim_left_pads(new_candidates[:self.n_best][~finished_bool_ids],
                                                      self.pad_token)
                    log_probs_history = trim_left_pads(new_log_probs_history[:self.n_best][~finished_bool_ids],
                                                       self.log_prob_pad)
                else:
                    finished_bool_ids = torch.cat(((new_candidates == self.eos_token).sum(-1).bool(),
                                                   (finished_candidates_t == self.eos_token).sum(-1).bool()), dim=0)
                    num, _ = new_candidates.size()
                    new_log_probs = new_log_probs_history[:, -1]
                    log_probs = torch.cat((new_log_probs, finished_candidates_log_probs_t), dim=0)
                    _, inds = torch.sort(log_probs, descending=True)

                    finished_bool_ids = finished_bool_ids[inds]
                    inds_of_finished = inds[finished_bool_ids][:self.n_best]

                    num_of_old_finished = (inds_of_finished >= num).sum().item()
                    num_of_new_finished = (inds_of_finished < num).sum().item()

                    if num_of_old_finished > 0 and num_of_new_finished > 0:
                        inds_of_new_finished = inds_of_finished[inds_of_finished < num]
                        inds_of_old_finished = inds_of_finished[inds_of_finished >= num] - num

                        new_finished_candidates = self.make_left_pad_tail(new_candidates[inds_of_new_finished])
                        finished_candidates_t = torch.cat((new_finished_candidates,
                                                           finished_candidates_t[inds_of_old_finished]), dim=0)
                        finished_candidates_log_probs_t = torch.cat((new_log_probs[inds_of_new_finished],
                                                                     finished_candidates_log_probs_t[
                                                                         inds_of_old_finished]),
                                                                    dim=0)
                    elif num_of_old_finished == 0 and num_of_new_finished > 0:
                        inds_of_new_finished = inds_of_finished[inds_of_finished < num]
                        finished_candidates_t = self.make_left_pad_tail(new_candidates[inds_of_new_finished])
                        finished_candidates_log_probs_t = new_log_probs[inds_of_new_finished]
                    else:
                        pass

                    finished_bool_ids = finished_bool_ids[:self.n_best]
                    next_circle_inds = inds[:self.n_best][~finished_bool_ids]
                    if next_circle_inds.shape[0] == 0:
                        break
                    generated_tokens = trim_left_pads(new_candidates[next_circle_inds], self.pad_token)
                    log_probs_history = trim_left_pads(new_log_probs_history[next_circle_inds],
                                                       self.log_prob_pad)

                best_current_log_prob = log_probs_history[:, -1].max().item()
                if finished_candidates_t is not None and \
                        (finished_candidates_t.shape[0] >= self.n_best and \
                         best_current_log_prob < finished_candidates_log_probs_t.min().item()):
                    break

            if finished_candidates_t is None:
                print("there is no finished candidates for the src")
                result.append(generated_tokens)
            else:
                finished_candidates_t, finished_candidates_log_probs_t = \
                    sort(finished_candidates_t, finished_candidates_log_probs_t,
                         descending=True)
                finished_candidates_t = finished_candidates_t[:self.n_best]
                finished_candidates_log_probs_t = finished_candidates_log_probs_t[:self.n_best]
                result.append(finished_candidates_t)  # (n, max_len)
            return result

    def calculate_n_accepted_in_drafts(self, draft_tokens, masked_probs):
        # masked_probs: tensor of size (n_candidates, n_drafts, draft_len + 1, vocab_size)
        # draft_tokens: tensor of size (n_candidates, n_drafts, draft_len)
        draft_tokens_probs = torch.gather(masked_probs[:, :, :-1, :], dim=-1, index=draft_tokens.unsqueeze(-1)).squeeze(
            -1)
        #   -> (n_candidates, n_drafts, draft_len)
        verification = draft_tokens_probs != 0.
        # num = n_candidates

        _range = verification.cumsum(-1)  # (num, n_drafts, draft_len)
        accepted_in_drafts_bool = (torch.arange(1, verification.size(2) + 1).unsqueeze(0).unsqueeze(0).type_as(
            _range) == _range)  # (num, n_drafts, draft_len)

        return accepted_in_drafts_bool.sum(-1)  # (num, n_drafts, draft_len) -> (num, n_drafts)

    def make_left_pad_tail(self, t):
        candidates_num, curr_len = t.size()
        if curr_len > self.max_len:
            t = trim_left_pads(t, self.pad_token)
        assert t.shape[1] <= self.max_len
        pad_tail = torch.full((t.shape[0], self.max_len - t.shape[1]), self.pad_token).type_as(t)
        return torch.cat((pad_tail, t), dim=1)

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

    def build_drafts(self, src_unbatched: 'torch.LongTensor', draft_mode: bool):
        if draft_mode:
            src_unbatched_i = src_unbatched[:, 1:]
            src_unbatched_i_pads = (src_unbatched_i == self.pad_token).int().sum(-1)
            n_tokens_without_pads = src_unbatched_i.size(1) - src_unbatched_i_pads
            src_unbatched_i_unpadded = src_unbatched_i[:, :n_tokens_without_pads]
            drafts = src_unbatched_i_unpadded.unfold(-1, self.draft_len, 1).squeeze(0)[:self.max_drafts_num, :]
            # -> (n_drafts, draft_len)
        else:
            drafts = src_unbatched[:, 0].unsqueeze(0).expand(self.max_drafts_num, 1)  # just [bos]
            # -> (n_drafts, draft_len=1)
        return drafts


def mask_with_num_logits_according_nucleus(pred_logits, nucleus, max_num_of_unmasked_positions, num=0.):
    n, curr_len, vocab_size = pred_logits.size()  # (n_candidates, draft_len + 1, vocab_size)
    pred_logits = pred_logits.reshape(n * curr_len, vocab_size)  # -> (n * curr_len, vocab_size)

    sorted_logits, sorted_indices = torch.sort(pred_logits,
                                               descending=True)  # -> (n * curr_len, vocab_size)
    cumulative_probs = torch.cumsum(sorted_logits.softmax(-1), dim=-1)  # -> (n * curr_len, vocab_size)

    # Remove tokens with cumulative probability above the threshold
    cumulative_probs = torch.roll(cumulative_probs, 1, dims=-1)
    cumulative_probs[:, 0] = nucleus - 1
    keep_candidates_mask = cumulative_probs < nucleus  # -> (n * curr_len, vocab_size)

    keep_candidates_mask[:, max_num_of_unmasked_positions:] = False
    # no more than max_num_of_unmasked_positions

    sorted_logits.masked_fill_(~keep_candidates_mask, float(num))

    masked_logits_according_nucleus = torch.gather(sorted_logits, 1, sorted_indices.argsort(1))
    # -> (n * curr_len, vocab_size)
    return masked_logits_according_nucleus.reshape(n, curr_len, vocab_size)


def sort(candidates, candidates_log_probs, descending=True):
    sorted_log_probs, sorted_inds = candidates_log_probs.sort(descending=descending)
    return candidates[sorted_inds], sorted_log_probs


class TranslationInferenceGreedySpeculative:
    """
    Supposed to be faster than TranslationInferenceGreedySpeculativeUnbatched because it supports batching.
    But isn't for some reason.
    """

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
        voc_size = memory.size(2)

        generated_tokens = torch.full((b_size, 1), self.bos_token).type_as(src).long()
        draft_tokens = src[:, 1:].unfold(-1, self.n_speculative_tokens, 1)
        n_drafts = draft_tokens.size(1)
        iters = 0
        finished_predictions = torch.full((b_size, self.max_len), self.pad_token)
        batch_indices = torch.arange(b_size).type_as(src)
        while generated_tokens.size(1) < self.max_len:
            iters += 1
            b_size = batch_indices.size(0)
            draft_sequence = torch.cat([generated_tokens.unsqueeze(1).repeat(1, n_drafts, 1), draft_tokens],
                                       dim=-1).reshape(b_size * n_drafts, -1)
            pos_enc_offset = (draft_sequence == -1).int().sum(-1).reshape(-1, 1)
            generated_tokens = generated_tokens.masked_fill(generated_tokens == -1, self.pad_token)
            draft_sequence = draft_sequence.masked_fill(draft_sequence == -1, self.pad_token)

            pred_logits = self.model.decode_tgt(draft_sequence,
                                                memory.unsqueeze(1).repeat(1, n_drafts, 1, 1).reshape(b_size * n_drafts,
                                                                                                      -1, voc_size),
                                                memory_pad_mask=src_pad_mask.unsqueeze(1).repeat(1, n_drafts, 1).view(
                                                    b_size * n_drafts, -1),
                                                pos_enc_offset=pos_enc_offset)
            pred_tokens = torch.argmax(pred_logits, dim=2)
            pred_tokens = pred_tokens[:, -(self.n_speculative_tokens + 1):]
            verification = draft_tokens.reshape(b_size * n_drafts, -1) == pred_tokens[:, :-1]
            _range = verification.cumsum(-1)
            accepted_in_drafts = (torch.arange(1, verification.size(1) + 1).type_as(_range) == _range)
            n_accepted_in_drafts = accepted_in_drafts.sum(-1)
            n_accepted_in_drafts = n_accepted_in_drafts.reshape(b_size, n_drafts)
            n_accepted_in_drafts = n_accepted_in_drafts.topk(1, -1)
            draft_i = n_accepted_in_drafts.indices
            n_accepted = n_accepted_in_drafts.values
            pred_tokens = pred_tokens.reshape(b_size, n_drafts, -1)

            chosen = torch.gather(pred_tokens, 1,
                                  draft_i.unsqueeze(-1).expand(b_size, 1, pred_tokens.size(-1))).squeeze(1)
            pred_tokens = chosen.masked_fill(torch.arange(pred_tokens.size(-1)).type_as(n_accepted) > n_accepted, -1)

            generated_tokens = torch.cat(
                (generated_tokens,
                 pred_tokens),
                dim=1
            )

            current_finished_mask = (generated_tokens == self.eos_token).sum(-1).bool()  # (<=b_sz)
            current_finished_ids = current_finished_mask.nonzero().ravel()
            if current_finished_ids.nelement() > 0:
                current_continuing_mask = ~current_finished_mask
                batch_finished_indices = batch_indices[current_finished_mask]

                current_finished_tokens = generated_tokens[current_finished_ids]
                current_finished_tokens = move_pads_to_the_right(current_finished_tokens)
                current_finished_tokens = current_finished_tokens.masked_fill(current_finished_tokens == -1,
                                                                              self.pad_token)
                current_finished_tokens = current_finished_tokens[:, :-(
                        (current_finished_tokens == self.pad_token).sum(dim=0) == current_finished_tokens.size(
                    0)).sum()]

                finished_predictions[batch_finished_indices, :current_finished_tokens.size(1)] = current_finished_tokens

                batch_indices = batch_indices[current_continuing_mask]

                generated_tokens = generated_tokens[current_continuing_mask]
                draft_tokens = draft_tokens[current_continuing_mask]
                memory = memory[current_continuing_mask]
                src_pad_mask = src_pad_mask[current_continuing_mask]
            if batch_indices.nelement() == 0:
                break

            generated_tokens = move_pads_to_the_left(generated_tokens, -1)
            generated_tokens = generated_tokens.masked_fill(generated_tokens == self.pad_token, -1)
            generated_tokens = generated_tokens[:, ((generated_tokens == -1).sum(
                dim=0) == b_size).sum():]  # Strip columns on the left that are all pads

        return finished_predictions.unsqueeze(1)


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


def move_pads_to_the_right(arr, pad_token=0):
    n_rows, n_cols = arr.size()
    dim_indices = torch.arange(n_cols).type_as(arr).long().repeat(n_rows).reshape(n_rows, -1)
    pad_count = (arr == pad_token).sum(1)
    indices = (dim_indices + pad_count.unsqueeze(1)) % n_cols
    return torch.gather(arr, dim=1, index=indices)


def move_pads_to_the_left(arr, pad_token=0):
    n_rows, n_cols = arr.size()
    dim_indices = torch.arange(n_cols).type_as(arr).long().repeat(n_rows).reshape(n_rows, -1)
    eos_index = (arr == pad_token).sum(1)
    indices = (dim_indices - eos_index.unsqueeze(1)) % n_cols
    return torch.gather(arr, dim=1, index=indices)


class TranslationInferenceNucleusSpeculativeUnbatched:

    def __init__(self,
                 model,  # TranslationModel
                 max_len: int,
                 n_speculative_tokens: int,
                 n_best: int,
                 pad_token: int,
                 bos_token: int,
                 eos_token: int,
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
        self.temperature = temperature
        self.n_best = n_best

    def __str__(self):
        return f"NucleusSpeculativeUnbatched decoding (max_len={self.max_len}, nucleus={self.nucleus}, temperature={self.temperature})"

    def sample(self, pred_logits):
        n_drafts, curr_len, vocab_size = pred_logits.size()
        pred_logits = pred_logits.reshape(n_drafts * curr_len, vocab_size)  # -> (n_drafts * curr_len, vocab_size)

        sorted_logits, sorted_indices = torch.sort(pred_logits, descending=True)  # -> (n_drafts * curr_len, vocab_size)
        cumulative_probs = torch.cumsum(sorted_logits.softmax(-1), dim=-1)  # -> (n_drafts * curr_len, vocab_size)

        # Remove tokens with cumulative probability above the threshold
        cumulative_probs = torch.roll(cumulative_probs, 1, dims=-1)
        cumulative_probs[:, 0] = 0
        keep_candidates_mask = cumulative_probs < self.nucleus  # -> (n_drafts * curr_len, vocab_size)

        sorted_logits.masked_fill_(~keep_candidates_mask, float("-inf"))

        best_candidates_logits = torch.gather(sorted_logits, 1, sorted_indices.argsort(1))
        best_probs = (best_candidates_logits / self.temperature).softmax(-1)

        sampled_tokens = torch.multinomial(best_probs, 1).squeeze(1)  # -> (n_drafts * curr_len, 1)
        # -> (n_drafts * curr_len)

        sampled_tokens = sampled_tokens.reshape(n_drafts, curr_len)
        return sampled_tokens

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
            drafts = src_unbatched_i_unpadded.unfold(-1, self.n_speculative_tokens, 1).squeeze(0)
            # -> (n_drafts, draft_len)
            n_drafts = drafts.size(0)
            iters = 0

            generated_tokens = torch.full((1, 1), self.bos_token).type_as(src).long().repeat(self.n_best, 1)
            # -> (n_best, 1)
            memory_i = memory_unbatched[i].repeat(n_drafts * self.n_best, 1, 1)
            memory_pad_mask_i = src_pad_mask_unbatched[i].repeat(n_drafts * self.n_best, 1)
            finished_candidates = []
            while generated_tokens.size(1) < self.max_len:
                iters += 1
                n_best, curr_len = generated_tokens.size()
                draft_tokens = drafts.repeat(n_best, 1)  # -> (n_best * n_drafts, 1)
                inp = generated_tokens.unsqueeze(1).expand(n_best, n_drafts, curr_len).reshape(n_best * n_drafts,
                                                                                               curr_len)
                draft_sequence = torch.cat([inp, draft_tokens], dim=1)
                # (n_best * n_drafts, curr_len), (n_best * n_drafts, draft_len) -> (n_best * n_drafts, curr_len + draft_len)
                _, seq_len = draft_sequence.size()
                pos_enc_offset = (draft_sequence == self.pad_token).int().sum(-1).reshape(-1, 1)
                pred_logits = self.model.decode_tgt(draft_sequence,
                                                    memory_i[:n_best * n_drafts, :, :],
                                                    memory_pad_mask=memory_pad_mask_i[:n_best * n_drafts, :],
                                                    pos_enc_offset=pos_enc_offset)
                #   -> (n_best * n_drafts, curr_len + draft_len, vocab_size)

                pred_tokens = self.sample(
                    pred_logits)  # (n_best * n_drafts, curr_len + draft_len, vocab_size) -> (n_best * n_drafts, curr_len + draft_len)

                pred_tokens = pred_tokens.reshape(n_best, n_drafts, seq_len)
                pred_tokens = pred_tokens[:, :, -(draft_tokens.size(
                    1) + 1):]  # (n_best, n_drafts, curr_len + draft_len) -> (n_best, n_drafts, draft_len + 1)
                verification = draft_tokens.reshape(n_best, n_drafts, -1) == pred_tokens[:, :,
                                                                             :-1]  # (n_best, n_drafts, draft_len + 1) -> (n_best, n_drafts, draft_len)
                _range = verification.cumsum(-1)  # (n_best, n_drafts, draft_len)
                accepted_in_drafts = (torch.arange(1, verification.size(2) + 1).unsqueeze(0).unsqueeze(0).type_as(
                    _range) == _range)  # (n_best, n_drafts, draft_len)
                n_accepted_in_drafts = accepted_in_drafts.sum(-1)  # (n_best, n_drafts, draft_len) -> (n_best, n_drafts)
                n_accepted_in_drafts = n_accepted_in_drafts.topk(1, dim=-1)  # (n_best, n_drafts) -> (n_best, 1)
                draft_i = n_accepted_in_drafts.indices  # (n_best, 1)
                n_accepted = n_accepted_in_drafts.values  # (n_best, 1)

                # min_num_accepted = torch.min(n_accepted).item()
                best_candidates = []
                for j in range(n_best):
                    min_accepted_in_j = n_accepted[j]
                    draft_id = draft_i[j]
                    pred_tokens_for_j = pred_tokens[j, draft_id, :min_accepted_in_j + 1]  # -> (min_accepted_in_i + 1)
                    initial = generated_tokens[j].unsqueeze(0)  # (n_best, curr_len) -> (curr_len)
                    pads_in_initial = (initial == self.pad_token).int().sum(-1)
                    initial = initial[:, pads_in_initial:]
                    candidate = torch.cat((initial, pred_tokens_for_j), dim=-1).squeeze(0)
                    if (candidate == self.eos_token).sum().item() > 0:
                        finished_candidates.append(candidate)
                    else:
                        best_candidates.append(candidate)

                if len(best_candidates) == 0:
                    break
                generated_tokens = pad_sequence(best_candidates, padding_value=self.pad_token, batch_first=True)
                generated_tokens = move_pads_to_the_left(generated_tokens, self.pad_token)

            result.append(pad_sequence(finished_candidates, padding_value=self.pad_token, batch_first=True))
        return result


class TranslationInferenceNucleusSpeculativeUnbatchedNoCycles:

    def __init__(self,
                 model,  # TranslationModel
                 max_len: int,
                 n_speculative_tokens: int,
                 n_best: int,
                 pad_token: int,
                 bos_token: int,
                 eos_token: int,
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
        self.temperature = temperature
        self.n_best = n_best

    def __str__(self):
        return f"NucleusSpeculativeUnbatched decoding (max_len={self.max_len}, nucleus={self.nucleus}, temperature={self.temperature})"

    def sample(self, pred_logits, num_samples):
        n, curr_len, vocab_size = pred_logits.size()  # (n_candidates * n_drafts, draft_len + 1, vocab_size)
        pred_logits = pred_logits.reshape(n * curr_len, vocab_size)  # -> (n * curr_len, vocab_size)

        sorted_logits, sorted_indices = torch.sort(pred_logits, descending=True)  # -> (n * curr_len, vocab_size)
        cumulative_probs = torch.cumsum(sorted_logits.softmax(-1), dim=-1)  # -> (n * curr_len, vocab_size)

        # Remove tokens with cumulative probability above the threshold
        cumulative_probs = torch.roll(cumulative_probs, 1, dims=-1)
        cumulative_probs[:, 0] = 0
        keep_candidates_mask = cumulative_probs < self.nucleus  # -> (n * curr_len, vocab_size)

        sorted_logits.masked_fill_(~keep_candidates_mask, float("-inf"))

        best_candidates_logits = torch.gather(sorted_logits, 1, sorted_indices.argsort(1))
        # -> (n * curr_len, vocab_size)

        best_probs = best_candidates_logits.softmax(-1)

        sampled_tokens = torch.multinomial(best_probs, num_samples, True)  # -> (n * curr_len, num_samples)
        line_log_probs = torch.gather(pred_logits.softmax(-1), dim=1,
                                      index=sampled_tokens).log()  # -> (n * curr_len, num_samples)

        sampled_tokens = sampled_tokens.reshape(n, curr_len, num_samples)  # ->(n, draft_len + 1, num_samples)
        line_log_probs = line_log_probs.reshape(n, curr_len, num_samples)  # ->(n, draft_len + 1, num_samples)
        return sampled_tokens, line_log_probs

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
            drafts = src_unbatched_i_unpadded.unfold(-1, self.n_speculative_tokens, 1).squeeze(0)[:100, :]
            # -> (n_drafts, draft_len)
            n_drafts, draft_len = drafts.size()
            iters = 0

            generated_tokens = torch.full((1, 1), self.bos_token).type_as(src).long()
            # -> (1, 1)
            memory_i = memory_unbatched[i].repeat(n_drafts, 1, 1)
            memory_pad_mask_i = src_pad_mask_unbatched[i].repeat(n_drafts, 1)

            finished_candidates_t = None

            candidates_log_probs = torch.zeros(1).to(src.device)  # (n_candidates)
            while generated_tokens.size(1) < self.max_len:
                iters += 1
                n_candidates, curr_len = generated_tokens.size()
                draft_tokens = drafts.repeat(n_candidates, 1)  # -> (n_candidates * n_drafts, 1)
                inp = generated_tokens.unsqueeze(1).expand(n_candidates, n_drafts, curr_len).reshape(
                    n_candidates * n_drafts,
                    curr_len)
                draft_sequence = torch.cat([inp, draft_tokens], dim=1)
                # (n_candidates * n_drafts, curr_len + draft_len)
                _, seq_len = draft_sequence.size()
                pos_enc_offset = (draft_sequence == self.pad_token).int().sum(-1).reshape(-1, 1)
                pred_logits = self.model.decode_tgt(draft_sequence, memory_i.repeat(n_candidates, 1, 1),
                                                    memory_pad_mask=memory_pad_mask_i.repeat(n_candidates, 1),
                                                    pos_enc_offset=pos_enc_offset)
                #   -> (n_candidates * n_drafts, curr_len + draft_len, vocab_size)

                num_samples = 24
                pred_tokens, draft_log_probs = self.sample(
                    pred_logits[:, -(draft_len +
                                     1):, :], num_samples)  # (n_candidates * n_drafts, draft_len + 1, vocab_size) ->
                # -> (n_candidates * n_drafts,  draft_len + 1, num_samples),
                # (n_candidates * n_drafts, draft_len + 1, num_samples)

                pred_tokens = pred_tokens.reshape(n_candidates, n_drafts, draft_len + 1, num_samples)
                # -> (n_candidates, n_drafts, draft_len + 1, num_samples)
                draft_log_probs = draft_log_probs.reshape(n_candidates, n_drafts, draft_len + 1, num_samples)

                pred_tokens = pred_tokens.transpose(2, 3).transpose(1, 2).transpose(0, 1)
                #   -> (num_samples, n_candidates, n_drafts, draft_len + 1)
                draft_log_probs = draft_log_probs.transpose(2, 3).transpose(1, 2).transpose(0, 1)
                #   -> (num_samples, n_candidates, n_drafts, draft_len + 1)

                pred_tokens = pred_tokens.reshape(num_samples * n_candidates, n_drafts, draft_len + 1)
                #   -> (num_samples * n_candidates, n_drafts, draft_len + 1)
                draft_log_probs = draft_log_probs.reshape(num_samples * n_candidates, n_drafts, draft_len + 1)
                #   -> (num_samples * n_candidates, n_drafts, draft_len + 1)

                num = num_samples * n_candidates

                verification = draft_tokens.reshape(
                    n_candidates,
                    n_drafts,
                    -1).unsqueeze(0).expand(num_samples,
                                            n_candidates,
                                            n_drafts,
                                            -1).reshape(num,
                                                        n_drafts,
                                                        -1) == pred_tokens[:, :, :-1]
                _range = verification.cumsum(-1)  # (num, n_drafts, draft_len)
                accepted_in_drafts = (torch.arange(1, verification.size(2) + 1).unsqueeze(0).unsqueeze(0).type_as(
                    _range) == _range)  # (num, n_drafts, draft_len)
                n_accepted_in_drafts = accepted_in_drafts.sum(-1)  # (num, n_drafts, draft_len) -> (num, n_drafts)
                n_accepted, draft_i = n_accepted_in_drafts.topk(1, dim=-1)  # (num, n_drafts) -> (num, 1)

                extra_pad = -1
                mask = torch.arange(draft_len + 1).to(src.device).unsqueeze(0) > n_accepted
                #   -> (num_samples * n_candidates, draft_len + 1)

                index = draft_i.unsqueeze(-1).expand(num, 1, draft_len + 1)
                #   (num_samples * n_candidates, 1) -> (num, 1, draft_len+1)
                pred_tokens = torch.gather(pred_tokens, dim=1, index=index).squeeze(1)
                #   (num_samples * n_candidates, n_drafts, draft_len + 1), (num, 1, draft_len+1) ->
                #    -> (num_samples * n_candidates, draft_len + 1)
                draft_log_probs = torch.gather(draft_log_probs, dim=1, index=index).squeeze(1)
                #    -> (num_samples * n_candidates, draft_len + 1)
                pred_tokens.masked_fill_(mask, extra_pad)
                draft_log_probs.masked_fill_(mask, 0.)

                tmp_ids = torch.arange(num).type_as(src) % n_candidates
                initial = generated_tokens[tmp_ids]
                #   -> (num_samples * n_candidates, curr_len)
                initial_log_probs = candidates_log_probs[tmp_ids]
                #   -> (num_samples * n_candidates)

                new_candidates = torch.cat((initial, pred_tokens), dim=-1)
                #   -> (num_samples * n_candidates, curr_len + draft_len + 1)
                new_log_probs = initial_log_probs + draft_log_probs.sum(-1)
                #   -> (num_samples * n_candidates)

                new_candidates = move_pads_to_the_left(new_candidates, extra_pad)
                #   -> (num_samples * n_candidates, curr_len + draft_len + 1)
                new_candidates.masked_fill_(new_candidates == extra_pad, self.pad_token)  # all pads will be at the left
                new_candidates = new_candidates[:, ((new_candidates == self.pad_token).sum(0) == num).sum():]

                new_candidates, new_log_probs = self.unique_and_sort(new_candidates, new_log_probs, descending=True)

                finished_bool_ids = (new_candidates == self.eos_token).sum(-1).bool()
                #   -> (num_samples * n_candidates)

                num_new_finished = finished_bool_ids.sum().item()

                if num_new_finished > 0:
                    new_finished_candidates = new_candidates[finished_bool_ids]
                    _, tokens_num = new_finished_candidates.size()
                    pad_tail = torch.full((num_new_finished, self.max_len - tokens_num),
                                          self.pad_token).type_as(src)
                    new_finished_candidates = torch.cat((pad_tail, new_finished_candidates), dim=1)
                    #   -> (num_new_finished, max_len)
                    new_finished_log_probs_t = new_log_probs[finished_bool_ids]
                    #   -> (num_new_finished)

                    if finished_candidates_t is None:
                        finished_candidates_t = new_finished_candidates

                        finished_candidates_log_probs_t = new_finished_log_probs_t
                    else:
                        finished_candidates_t = torch.cat((finished_candidates_t, new_finished_candidates), dim=0)

                        finished_candidates_log_probs_t = torch.cat((finished_candidates_log_probs_t,
                                                                     new_finished_log_probs_t), dim=0)

                    if finished_candidates_t.size()[0] >= self.n_best:
                        finished_candidates_t, finished_candidates_log_probs_t = \
                            self.unique_and_sort(finished_candidates_t, finished_candidates_log_probs_t,
                                                 descending=True)
                        finished_candidates_t = finished_candidates_t[:self.n_best]
                        finished_candidates_log_probs_t = finished_candidates_log_probs_t[:self.n_best]
                        break

                generated_tokens = new_candidates[~finished_bool_ids][:self.n_best]
                candidates_log_probs = new_log_probs[~finished_bool_ids][:self.n_best]
                if generated_tokens.size()[0] == 0:
                    break
                generated_tokens = generated_tokens[:,
                                   ((generated_tokens == self.pad_token).sum(0) == generated_tokens.size()[0]).sum():]

            result.append(finished_candidates_t)  # (n, max_len)
        return result

    def unique_and_sort(self, generated_tokens, best_candidates_log_probs, descending=True):
        generated_tokens_and_probs = torch.cat(
            [generated_tokens, best_candidates_log_probs.unsqueeze(1)], dim=-1)
        #   -> (m, new_curr_len+1)

        unique_generated_tokens_and_probs = torch.unique(generated_tokens_and_probs, dim=0)
        # -> (new_n_candidates, new_curr_len+1)

        unique_generated_tokens_log_probs = unique_generated_tokens_and_probs[:, -1]
        # -> (new_n_candidates)

        unique_generated_tokens = unique_generated_tokens_and_probs[:, :-1].long()
        # -> (new_n_candidates, new_curr_len)

        sorted_log_probs, sorted_inds = unique_generated_tokens_log_probs.sort(descending=descending)

        generated_tokens = unique_generated_tokens[sorted_inds]
        return generated_tokens, sorted_log_probs


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
