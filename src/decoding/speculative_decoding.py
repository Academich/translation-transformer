import torch

from torch.nn.utils.rnn import pad_sequence
from collections import defaultdict


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


def trim_left_pads(tensor_t, left_pad_id: int):
    """
    Remove columns from the left that contain only PAD tokens.
    tensor_t is supposed to have PAD tokens only on the left
    """
    rows_num, _ = tensor_t.size()
    # number of left columns filled with the pad id
    padded_columns_num = ((tensor_t == left_pad_id).sum(0) == rows_num).sum()
    return tensor_t[:, padded_columns_num:]


def trim_right_pads(tensor_t, pad_id: int):
    """
    Remove columns from the right that contain only PAD tokens.
    tensor_t is supposed to have PAD tokens only on the left
    """
    rows_num, t_length = tensor_t.size()
    # number of left columns filled with the pad id
    padded_columns_num = ((tensor_t == pad_id).sum(0) == rows_num).sum()
    return tensor_t[:, :t_length - padded_columns_num]


class TranslationInferenceBeamSearchSpeculativeBatchedWithoutLeftPads:
    def __init__(self,
                 model,  # TranslationModel
                 max_len: int,
                 n_speculative_tokens: int,
                 pad_token: int,
                 bos_token: int,
                 eos_token: int,
                 C_token: int,
                 n_best: int,
                 vocab_size,
                 max_num_of_drafts: int = 23
                 ) -> None:
        self.model = model
        self.max_len = max_len
        self.vocab_size = vocab_size

        self.pad_token_idx = pad_token
        self.bos_token_idx = bos_token
        self.eos_token_idx = eos_token
        self.C_token_idx = C_token

        self.n_best = n_best

        self.accepted_tokens_num = 0
        self.model_calls_num = 0

        self.max_drafts_num = max_num_of_drafts

        self.log_prob_pad = 7.  # should be more than 0.

        self.n_drafts = 0
        self.produced_non_pad_tokens = 0

        self.max_draft_len = 200
        self.min_draft_len = 5
        drafts_len = max(self.min_draft_len, self.n_speculative_tokens)
        drafts_len = min(drafts_len, self.max_draft_len)
        if drafts_len != n_speculative_tokens:
            print(f"The draft length should be in range [{self.min_draft_len}: {self.max_draft_len}], so it was changed to {drafts_len}")
        self.n_speculative_tokens = drafts_len

        self.acceptance_rate_pad_for_alredy_finished_seqs = -1  # should be negative
        self.acceptance_rate_pad_for_fake_seqs = -7  # should be negative

    def __str__(self):
        return f"SpeculativeSampling decoding (n_best={self.n_best}, max_len={self.max_len}, max_num_of_drafts={self.max_drafts_num}, n_speculative_tokens={self.n_speculative_tokens})"

    def sample(self, curr_lines, curr_log_probs_history, pred_logits, chosen_drafts, b_size, bool_idx, n_accepted):
        """
        This function samples all possible sequences within a selected draft. Each draft can
        produce (self.max_num_positions_for_sampling - 1) * num_of_approved_tokens + self.max_num_positions_for_sampling
        at most.

        :param curr_lines: tensor (n_candidates, drafted_len),
        :param curr_log_probs_history: tensor (n_candidates, max_len),
        :param pred_logits: tensor (n_candidates, draft_len + 1, vocab_size),
        :param chosen_drafts: tensor (n_candidates, draft_len),
        :param b_size: int,
        :param bool_idx: tensor (n_candidates, max_len), it contains true where the draft supposed to be in curr_lines,
            in each line there are draft_len trues
        :param n_accepted: tensor (n_candidates)
        :return:
          ->  new_lines: tensor (num_lines, max_len),
              new_log_probs_history: tensor (num_lines, max_len)
              num_of_new_seqs_for_each_in_batch: tensor (b_size)
              token_postn: tensor (num_lines), to calculate the number of accepted tokens in the next top n sequences
                later; self.acceptance_rate_pad_for_alredy_finished_seqs means that the given sequence had already the
                eos token and so didn't need subsequent tokens
        """
        drafted_len = curr_lines.shape[1]
        n_candidates, draft_len_plus_one, vocab_size = pred_logits.size()

        draft_len = draft_len_plus_one - 1

        masked_logits = mask_with_num_logits_according_nucleus(pred_logits, nucleus=20.,
                                                               max_num_of_unmasked_positions=self.n_best,
                                                               num=0.)
        #   -> (n_candidates, draft_len + 1, vocab_size)
        # any nucleus more than 1. fits well
        tmp_range = torch.arange(draft_len_plus_one).type_as(curr_lines).unsqueeze(0)
        #   -> (1, draft_len + 1)
        ####################################################################################

        mask_for_unaccepted_draft_tokens = tmp_range.repeat(n_candidates, 1) <= n_accepted.unsqueeze(-1)
        #   -> (n_candidates, draft_len + 1)
        masked_logits *= mask_for_unaccepted_draft_tokens.unsqueeze(-1)

        not_entirely_excepted_bool = n_accepted != draft_len
        #   -> (n_candidates)
        if not_entirely_excepted_bool.sum().item() > 0:
            # We need to change the first token in the drafts, following the last accepted token, to the bos token in
            # order to build the right top n tree of sequences
            chosen_drafts[not_entirely_excepted_bool] = chosen_drafts[not_entirely_excepted_bool].scatter(
                index=n_accepted[not_entirely_excepted_bool].unsqueeze(-1), dim=1, value=self.bos_token_idx)

        masked_logits[:, :-1, :].scatter_(index=chosen_drafts.unsqueeze(-1), dim=2, value=0.)  # the accepted tokens in
        # the drafts can not be leaves of the top n tree of sequences

        # Sampling the top n tree of sequences leaves:
        candts_inds, token_postn, token_inds = torch.nonzero(masked_logits, as_tuple=True)
        # -> (num)

        ################################################################################################################
        if n_candidates == b_size:
            beam_size = 1
        else:
            beam_size = self.n_best
        assert n_candidates / b_size == beam_size
        candts_inds_tmp = candts_inds.unsqueeze(-1).repeat(1, b_size)
        #  -> (b_size * beam_size, b_size)
        low_border = torch.arange(b_size).to(candts_inds.device) * beam_size
        high_border = low_border + beam_size
        num_of_new_seqs_for_each_in_batch = torch.logical_and(low_border <= candts_inds_tmp,
                                                              candts_inds_tmp < high_border).sum(0)
        # -> (b_size)
        ################################################################################################################

        num = token_inds.size()[0]
        previous_roots = curr_lines[candts_inds]  # (num, drafted_len)
        already_finished_given_seqs = (previous_roots == self.eos_token_idx).sum(-1).bool()  # -> (num)

        log_prob_history_of_roots = curr_log_probs_history[candts_inds]  # (num, max_len)
        bool_idx = bool_idx[candts_inds]  # (num, max_len)

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

        last_log_prob_from_roots = torch.min(log_prob_history_of_roots, dim=-1, keepdim=True).values
        # (num, 1)
        new_seqs_log_probs = last_log_prob_from_roots + new_seqs_log_probs
        #    -> (num, draft_len + 1)
        new_seqs.masked_fill_(mask_for_tokens_after_the_sampled, self.pad_token_idx)
        #    -> (num, draft_len + 1)
        new_seqs_log_probs.masked_fill_(mask_for_tokens_after_the_sampled, self.log_prob_pad)
        #    -> (num, draft_len + 1)

        tmp = torch.logical_or(bool_idx, torch.roll(bool_idx, 1, 1))
        # -> (num, max_len)
        previous_roots = torch.cat((previous_roots, tail), dim=-1)  # (num, drafted_len + 1)
        previous_roots[tmp[:, :drafted_len + 1]] = new_seqs.reshape(
            -1)  # it is new sequences sampled from the chosen drafts
        log_prob_history_of_roots[tmp] = new_seqs_log_probs.reshape(-1)

        token_postn[already_finished_given_seqs] = self.acceptance_rate_pad_for_alredy_finished_seqs
        # the given sequences with eos didn't need the draft tokens. We
        # don't take pads into account calculating the acceptance rate
        return previous_roots, log_prob_history_of_roots, num_of_new_seqs_for_each_in_batch, token_postn

    def generate(self, src: 'torch.LongTensor') -> list['torch.LongTensor']:
        draft_tokens = self.get_drafts_linspace(src)
        # -> (b_sz, n_drafts, draft_len)

        b_size, n_drafts, draft_len = draft_tokens.size()
        self.n_drafts += b_size * n_drafts

        src_pad_mask = (src == self.model.src_pad_token_i).bool()
        # -> (b_size, src_len)

        memory = self.model.encode_src(src, src_pad_mask)
        # -> (b_size, src_len, emb_dim)
        _, src_len, emb_dim = memory.size()
        memory = memory.unsqueeze(1).repeat(1, n_drafts, 1, 1).reshape(b_size * n_drafts, src_len, emb_dim)
        src_pad_mask = src_pad_mask.unsqueeze(1).repeat(1, n_drafts, 1).reshape(b_size * n_drafts, src_len)

        iters = -1

        generated_tokens = torch.full((1, 1), self.bos_token_idx).type_as(src).long().repeat(b_size, 1)
        # -> (b_size, 1)

        log_probs_history = torch.full((1, self.max_len), self.log_prob_pad).type_as(src).float().repeat(b_size, 1)
        # -> (b_size, max_len)
        log_probs_history[:, 0] = 0.

        possible_draft_len = self.max_len - 2

        logits_base = torch.full((b_size * n_drafts, draft_len + 1, self.vocab_size), 0., device=src.device)
        #   -> (b_s * n_drafts, draft_len + 1, vocab_size)

        while possible_draft_len > 1 and iters < self.max_len:
            iters += 1
            logits_base = logits_base * 0.
            # We use artificial logits to avoid calculation of obvious pad predicting after eos
            logits_base[:, :, self.pad_token_idx] = 35.
            # 35. will give about 100% probability for pad_token after softmax()

            if iters == 1:
                src_pad_mask = src_pad_mask.unsqueeze(1).repeat(1, self.n_best, 1).flatten(end_dim=1)
                # -> (b_size * n_drafts * bm_sz, src_len)
                memory = memory.unsqueeze(1).repeat(1, self.n_best, 1, 1).flatten(end_dim=1)
                # -> (b_size * n_drafts * bm_sz, src_len, emb_dim)
                draft_tokens = draft_tokens.unsqueeze(1).repeat(1, self.n_best, 1, 1).flatten(end_dim=1)
                # -> (b_s * bm_sz, n_drafts, draft_len)
                logits_base = logits_base.repeat(self.n_best, 1, 1)
                #   -> (b_s * n_drafts * bm_sz, draft_len + 1, vocab_size)

            bool_idx_of_unfinished = ~((generated_tokens == self.eos_token_idx).sum(-1).bool())
            # -> (n_candidates)

            draft_len = min(possible_draft_len, draft_len)
            draft_tokens = draft_tokens[:, :, :draft_len]
            n_candidates, curr_len = generated_tokens.size()

            pads_num = (generated_tokens == self.pad_token_idx).sum(-1)
            # -> (n_candidates)
            pad_base_len = draft_len - torch.min(pads_num).item()
            if pad_base_len > 0:
                draft_base = torch.full((n_candidates, pad_base_len), self.pad_token_idx, device=src.device)
                generated_tokens = torch.cat((generated_tokens, draft_base), dim=-1)
            # -> (n_candidates, drafted_len)

            logits_base = logits_base[:, :draft_len + 1, :]

            self.model_calls_num += 1
            log_prob_pad_t_bool = log_probs_history == self.log_prob_pad

            bool_idx = torch.logical_and(log_prob_pad_t_bool,
                                         log_prob_pad_t_bool.cumsum(-1) <= draft_len)
            # -> (b_s * bm_sz, max_len)
            bool_idx_input = bool_idx[:, :generated_tokens.shape[1]].unsqueeze(1).repeat(1, n_drafts, 1)
            # -> (b_s * bm_sz, n_drafts, drafted_len)
            generated_tokens_input = generated_tokens.unsqueeze(1).repeat(1, n_drafts, 1)
            # -> (b_s * bm_sz, n_drafts, drafted_len)

            generated_tokens_input[bool_idx_input] = draft_tokens.reshape(-1)
            bool_idx_input = bool_idx_input.flatten(end_dim=1)
            # -> (b_s * bm_sz * n_drafts, drafted_len)
            generated_tokens_input = generated_tokens_input.flatten(end_dim=1)
            # # -> (b_s * bm_sz * n_drafts, drafted_len, vocab_size)

            bool_idx_of_unfinished = bool_idx_of_unfinished.unsqueeze(-1).repeat(1, n_drafts).flatten(end_dim=1)
            # -> (b_s * bm_sz * n_drafts)
            bool_idx_input = bool_idx_input[bool_idx_of_unfinished]  #
            #   -> (num_of_unfinished, drafted_len)
            pred_logits = self.model.decode_tgt(generated_tokens_input[bool_idx_of_unfinished],
                                                memory[bool_idx_of_unfinished],
                                                memory_pad_mask=src_pad_mask[bool_idx_of_unfinished])
            #  -> (num_of_unfinished, drafted_len, vocab_size)

            vocab_size = pred_logits.shape[-1]

            pred_logits = pred_logits[torch.logical_or(bool_idx_input, torch.roll(bool_idx_input, -1, 1))].reshape(
                -1, draft_len + 1, vocab_size)
            #  -> (num_of_unfinished, draft_len + 1, vocab_size)

            logits_base[bool_idx_of_unfinished] = pred_logits
            pred_logits = logits_base
            #   -> (n_candidates * n_drafts, draft_len + 1, vocab_size)

            # Choosing the best draft for each candidate. The draft with the biggest number of
            # approved tokens is the best draft for the given candidate. #########################################

            # All unapproved tokens in masked_probs have zero probability
            # We use nucleus=0.9975 and max_num_of_unmasked_positions=5 to avoid sampling of low probable sequences
            # and reduce calculation
            masked_probs = mask_with_num_logits_according_nucleus(pred_logits, nucleus=0.9975,
                                                                  max_num_of_unmasked_positions=5,
                                                                  num="-inf").softmax(-1)
            #   -> (n_candidates * n_drafts, draft_len + 1, vocab_size)

            masked_probs = masked_probs.reshape(n_candidates, n_drafts, draft_len + 1, vocab_size)
            draft_tokens = draft_tokens.reshape(n_candidates, n_drafts, draft_len)  # each candidate has the same
            # collection of drafts

            n_accepted_in_drafts = self.calculate_n_accepted_in_drafts(draft_tokens, masked_probs)
            #   ->(n_candidates, n_drafts)

            # Each candidate needs its best draft. Choose the draft with the biggest number of approved tokens
            # for each candidate:
            n_accepted, draft_i = n_accepted_in_drafts.topk(1, dim=-1)
            # (n_candidates, n_drafts) -> (n_candidates, 1)

            self.accepted_tokens_num += n_accepted.sum().item()

            chosen_drafts = torch.gather(draft_tokens, dim=1,
                                         index=draft_i.unsqueeze(-1).expand(n_candidates, 1, draft_len)).squeeze(1)
            #   -> (n_candidates, draft_len)
            ########################################################################################################
            pred_logits = pred_logits.reshape(n_candidates, n_drafts, draft_len + 1, vocab_size)

            # Further we need information only about chosen drafts
            pred_logits = torch.gather(pred_logits, dim=1, index=draft_i.unsqueeze(-1).unsqueeze(-1).
                                       expand(n_candidates, 1, draft_len + 1, vocab_size)).squeeze(1)
            #   -> (n_candidates, draft_len + 1, vocab_size)

            # Sample all possible lines within the chosen drafts:
            # new_candidates have the initial tokens and the new ones

            new_candidates, new_log_probs_history, num_of_new_seqs_for_each_in_batch, accepted_tokens_num = \
                self.sample(generated_tokens, log_probs_history, pred_logits,
                            chosen_drafts, b_size, bool_idx, n_accepted.squeeze(-1))

            ###########################################################################################################
            max_num_of_new_seqs = torch.max(num_of_new_seqs_for_each_in_batch).item()
            max_num_of_new_seqs = max(max_num_of_new_seqs, self.n_best)
            if (num_of_new_seqs_for_each_in_batch == max_num_of_new_seqs).sum() != b_size:
                # We make fake sequences with an artificial probability -inf in case if a different number of sequences
                # were sampled on the basis of the chosen drafts
                start_inds = torch.roll(num_of_new_seqs_for_each_in_batch, 1, dims=-1)  # -> (b_size)
                start_inds[0] = 0
                start_inds = start_inds.cumsum(-1)

                inds = torch.arange(max_num_of_new_seqs).to(start_inds.device).unsqueeze(-1).repeat(1,
                                                                                                    b_size)
                # -> (max_num_of_new_seqs, b_size)

                mask_for_fake_seqs = inds >= num_of_new_seqs_for_each_in_batch.unsqueeze(0)
                inds = start_inds + (inds % num_of_new_seqs_for_each_in_batch)

                inds = inds.transpose(0, 1).reshape(-1)
                # -> (b_size * max_num_of_new_seqs)
                mask_for_fake_seqs = mask_for_fake_seqs.transpose(0, 1).reshape(
                    -1)  # -> (b_size * max_num_of_new_seqs)
                new_candidates = new_candidates[inds]
                # -> (b_size * max_num_of_new_seqs, drafted_len + 1)
                new_log_probs_history = new_log_probs_history[inds]
                # -> (b_size * max_num_of_new_seqs, max_len)
                accepted_tokens_num = accepted_tokens_num[inds]
                # -> (b_size * max_num_of_new_seqs)
                new_candidates[mask_for_fake_seqs, 1] = self.eos_token_idx  # fake sequences
                new_log_probs_history[mask_for_fake_seqs, 1] = -float("inf")  # fake probabilities
                accepted_tokens_num[mask_for_fake_seqs] = self.acceptance_rate_pad_for_fake_seqs  # fake
            #############################################################################################

            new_log_probs = torch.min(new_log_probs_history, dim=1).values
            # -> (b_size * max_num_of_new_seqs)
            new_log_probs = new_log_probs.reshape(b_size, max_num_of_new_seqs)
            # -> (b_size, max_num_of_new_seqs)
            v, top_inds = new_log_probs.topk(k=self.n_best, axis=-1, sorted=True)
            # -> (b_size, beam_size)

            new_candidates = new_candidates.reshape(b_size, max_num_of_new_seqs, -1)
            # -> (b_size, max_num_of_new_seqs, drafted_len + 1)
            new_log_probs_history = new_log_probs_history.reshape(b_size, max_num_of_new_seqs, -1)
            # -> (b_size, max_num_of_new_seqs, max_len)
            accepted_tokens_num = accepted_tokens_num.reshape(b_size, max_num_of_new_seqs)
            # -> (b_size, max_num_of_new_seqs)

            accepted_tokens_num = torch.gather(accepted_tokens_num, 1, top_inds)
            # -> (b_size, beam_size)

            not_fake_bool = accepted_tokens_num != self.acceptance_rate_pad_for_fake_seqs
            # -> (b_size, beam_size)
            accepted_tokens_num = accepted_tokens_num[accepted_tokens_num > 0]
            curr_accepted_tokens_num = accepted_tokens_num.sum().item()
            self.accepted_tokens_num += curr_accepted_tokens_num
            self.produced_non_pad_tokens += curr_accepted_tokens_num + accepted_tokens_num.size(0)

            top_inds = top_inds.unsqueeze(-1).repeat(1, 1, new_log_probs_history.shape[-1])
            # -> (b_size, beam_size, max_len)
            new_log_probs_history = torch.gather(new_log_probs_history, 1, top_inds)
            # -> (b_size, beam_size, max_len)
            new_candidates = torch.gather(new_candidates, 1, top_inds[:, :, :new_candidates.shape[-1]])
            # -> (b_size, beam_size, drafted_len + 1)

            if (new_candidates[not_fake_bool] == self.eos_token_idx).sum(-1).bool().sum() == b_size * self.n_best:
                break

            generated_tokens = new_candidates.reshape(b_size * self.n_best, -1)
            # -> (b_size * beam_size, drafted_len + 1)
            new_log_probs_history = new_log_probs_history.reshape(b_size * self.n_best, -1)
            # -> (b_size * beam_size, max_len)
            not_fake_bool = not_fake_bool.reshape(b_size * self.n_best)
            # -> (b_size * beam_size)
            log_probs_history = new_log_probs_history

            possible_draft_len = torch.min((new_log_probs_history[not_fake_bool] == self.log_prob_pad).sum(-1)).item() - 1
        return new_candidates

    def calculate_n_accepted_in_drafts(self, draft_tokens, masked_probs):
        """
        This function calculates the number of approved tokens in each draft for each candidate.

        :param draft_tokens: tensor of size (n_candidates, n_drafts, draft_len),
        :param masked_probs: (all unapproved tokens in masked_probs are supposed to be equal to 0.)
                             tensor of size (n_candidates, n_drafts, draft_len + 1, vocab_size),

        :return:
          ->  returns the number of approved tokens in each draft for each candidate:
                             tensor of size  (n_candidates, n_drafts)

        """
        draft_tokens_probs = torch.gather(masked_probs[:, :, :-1, :], dim=-1, index=draft_tokens.unsqueeze(-1)).squeeze(
            -1)
        #   -> (n_candidates, n_drafts, draft_len)
        verification = draft_tokens_probs != 0.

        _range = verification.cumsum(-1)  # (n_candidates, n_drafts, draft_len)
        accepted_in_drafts_bool = (torch.arange(1, verification.size(2) + 1).unsqueeze(0).unsqueeze(0).type_as(
            _range) == _range)  # (n_candidates, n_drafts, draft_len)

        return accepted_in_drafts_bool.sum(-1)  # (n_candidates, n_drafts, draft_len) -> (n_candidates, n_drafts)

    def get_drafts(self, src: 'torch.LongTensor'):
        """
        :param src: 'torch.LongTensor' of shape (b_sz, src_len),
        :return:
          ->  drafts: tensor (b_sz, n_drafts, draft_len),
        """

        src_for_drafts = src[:, 1:].clone()
        src_for_drafts.masked_fill_(src_for_drafts == self.pad_token_idx, self.C_token_idx)
        src_for_drafts.masked_fill_(src_for_drafts == self.eos_token_idx, self.C_token_idx)

        drafts = src_for_drafts.unfold(dimension=1, size=min(self.n_speculative_tokens, src_for_drafts.shape[-1]),
                                       step=1)[:, :self.max_drafts_num, :]
        # -> (b_sz, n_drafts, draft_len = min(self.draft_len, src_len))
        return drafts

    def get_drafts_linspace(self, src: torch.Tensor) -> torch.Tensor:
        s = src[:, 1:].clone().detach()  # we don't need the bos token
        end_inds = torch.nonzero(s == self.eos_token_idx, as_tuple=True)[1]  # (b_sz)
        s.masked_fill_(s == self.pad_token_idx, self.C_token_idx)
        s.masked_fill_(s == self.eos_token_idx, self.C_token_idx)
        b_sz, s_len = s.size()

        drafts_len = max(self.min_draft_len, self.n_speculative_tokens)
        drafts_len = min(drafts_len, self.max_draft_len)

        start_inds = torch.full((b_sz,), 0, device=s.device)  # (b_sz)

        # Ensure we don't try to create more drafts than possible
        last_draft_len = 10
        possible_drafts_num = min(end_inds) - last_draft_len + 1

        n_drafts = min(self.max_drafts_num, possible_drafts_num)

        if n_drafts <= 1:
            return s[:, :drafts_len].unsqueeze(1)

        base_range = torch.arange(n_drafts).unsqueeze(0).repeat(b_sz, 1).to(s.device)  # (b_sz, n_drafts)

        const_multiplier = (end_inds - last_draft_len - start_inds) / torch.full((b_sz,), n_drafts - 1,
                                                                                 device=s.device)  # (b_sz)
        start_inds_of_drafts = (
                start_inds.unsqueeze(-1) + const_multiplier.unsqueeze(-1) * base_range).long()  # (b_sz, n_drafts)
        the_most_right_ind = torch.max(start_inds_of_drafts).item() + drafts_len
        if the_most_right_ind > s_len - 1:
            pad_tensor = torch.full((b_sz, the_most_right_ind - s_len + 2), self.C_token_idx, device=s.device)
            s = torch.cat((s, pad_tensor), dim=-1)

        indices = start_inds_of_drafts.unsqueeze(-1) + torch.arange(drafts_len).unsqueeze(0).unsqueeze(
            0).to(s.device)  # (b_sz, n_drafts, drafts_len)

        # Use gather to create drafts
        drafts = torch.gather(s.unsqueeze(1).expand(-1, n_drafts, -1), 2, indices)  # (b_sz=1, n_drafts, drafts_len)

        return drafts  # Shape: (b_sz, n_drafts, drafts_len)


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
        self.c_token = 4
        if self.draft_mode:
            self.n_speculative_tokens = n_speculative_tokens
            self.max_drafts_num = max_num_of_drafts
        else:
            self.n_speculative_tokens = 1
            self.max_drafts_num = 1
        self.nucleus_for_sampling = nucleus
        self.n_best = n_best
        self.extra_pad = -1
        self.max_num_positions_for_sampling = 1 * n_best
        self.log_prob_pad = 1
        self.log_prob_extra_pad = 2

        self.given_tokens = 0
        self.accepted_tokens_num = 0
        self.model_calls_num = 0

        self.iter_to_draft_inds_to_accepted_tokens_num = defaultdict(lambda: defaultdict(int))

    def __str__(self):
        return f"BeamSearchSpeculativeUnbatched decoding (n_best={self.n_best}, draftlen={self.n_speculative_tokens}, max_len={self.max_len})"

    def sample(self, curr_lines, curr_log_probs_history, pred_logits, chosen_drafts, n_accepted=None):
        """
        This function samples all possible sequences within a selected draft. Each draft can
        produce (self.max_num_positions_for_sampling - 1) * num_of_approved_tokens + self.max_num_positions_for_sampling
        at most.

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
        draft_len = draft_len_plus_one - 1
        masked_logits = mask_with_num_logits_according_nucleus(pred_logits, nucleus=self.nucleus_for_sampling,
                                                               max_num_of_unmasked_positions=self.max_num_positions_for_sampling,
                                                               num=0.)
        #   -> (n_candidates, draft_len + 1, vocab_size)

        tmp_range = torch.arange(draft_len_plus_one).type_as(curr_lines).unsqueeze(0)
        #   -> (1, draft_len + 1)
        if n_accepted is None:
            n_accepted = self.calculate_n_accepted_in_drafts(chosen_drafts.unsqueeze(1),
                                                             masked_logits.unsqueeze(1)).squeeze(-1)
        else:
            not_fully_accepted_inds_bool = n_accepted != draft_len
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
        """
        Generate a batch of sequences using speculative beam search decoding.
        Input shape: B x L (batch size x source sequence length)
        Output shape: B x N x L (batch size x number of hypotheses x target sequence length)
        In speculative beam search decoding, N is equal to the top N best sequences generated.
        """
        b_size, src_len = src.size()

        src_pad_mask = (src == self.model.src_pad_token_i).bool()
        self.given_tokens += (b_size * src_len - src_pad_mask.sum().item())
        memory = self.model.encode_src(src, src_pad_mask)

        src_unbatched = src.unsqueeze(1)  # -> (b_size, 1, src_len)
        src_pad_mask_unbatched = src_pad_mask.unsqueeze(1)
        memory_unbatched = memory.unsqueeze(1)

        result = []

        for i in range(b_size):
            draft_mode = "linspace"
            if draft_mode == "linspace":
                drafts = self.get_drafts_linspace(src_unbatched[i])
            else:
                drafts = self.get_drafts(src_unbatched[i], self.draft_mode)
            # -> (min(self.max_drafts_num, max_possible_drafts), min(self.n_speculative_tokens, non_pad_length - 1))

            n_drafts, draft_len = drafts.size()

            iters = 0

            generated_tokens = torch.full((1, 1), self.bos_token).type_as(src).long()
            # -> (1, 1)
            memory_i = memory_unbatched[i].repeat(n_drafts, 1, 1)
            memory_pad_mask_i = src_pad_mask_unbatched[i].repeat(n_drafts, 1)

            finished_candidates_t = None

            log_probs_history = torch.full((1, 1), 0.).type_as(src).float()
            while (generated_tokens.size(1) + draft_len + 1) < self.max_len and iters < self.max_len:
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
                #  -> (n_candidates * n_drafts, curr_len + draft_len, vocab_size)
                self.model_calls_num += 1
                vocab_size = pred_logits.shape[-1]
                ###### Choosing the best draft for each candidate. The draft with the biggest number of
                # approved tokens is the best draft for the given candidate. #########################################
                pred_logits = pred_logits[:, -(draft_len + 1):, :]
                #   -> (n_candidates * n_drafts, draft_len + 1, vocab_size)

                # All unapproved tokens in masked_probs have zero probability
                # We use nucleus=0.9975 and max_num_of_unmasked_positions=5 to avoid sampling of low probable sequences
                # and reduce calculation
                masked_probs = mask_with_num_logits_according_nucleus(pred_logits, nucleus=0.9975,
                                                                      max_num_of_unmasked_positions=5,
                                                                      num="-inf").softmax(-1)
                #   -> (n_candidates * n_drafts, draft_len + 1, vocab_size)
                masked_probs = masked_probs.reshape(n_candidates, n_drafts, draft_len + 1, vocab_size)
                draft_tokens = draft_tokens.reshape(n_candidates, n_drafts, draft_len)  # each candidate has the same
                # collection of drafts

                n_accepted_in_drafts = self.calculate_n_accepted_in_drafts(draft_tokens, masked_probs)
                #   ->(n_candidates, n_drafts)

                # Each candidate needs its best draft. Choose the draft with the biggest number of approved tokens
                # for each candidate
                n_accepted, draft_i = n_accepted_in_drafts.topk(1, dim=-1)
                # (n_candidates, n_drafts) -> (n_candidates, 1)
                self.accepted_tokens_num += n_accepted.sum().item()
                for j in range(n_candidates):
                    if n_accepted[j, 0].item() >= 2:
                        self.iter_to_draft_inds_to_accepted_tokens_num[iters - 1][draft_i[j, 0].item()] += n_accepted[j, 0].item()

                chosen_drafts = torch.gather(draft_tokens, dim=1,
                                             index=draft_i.unsqueeze(-1).expand(n_candidates, 1, draft_len)).squeeze(1)
                #   -> (n_candidates, draft_len)
                ########################################################################################################
                pred_logits = pred_logits.reshape(n_candidates, n_drafts, draft_len + 1, vocab_size)

                # Further we need information only about chosen drafts
                pred_logits = torch.gather(pred_logits, dim=1, index=draft_i.unsqueeze(-1).unsqueeze(-1).
                                           expand(n_candidates, 1, draft_len + 1, vocab_size)).squeeze(1)
                #   -> (n_candidates, draft_len + 1, vocab_size)

                # Sample all possible lines within the chosen drafts.
                # new_candidates have the initial tokens and the new ones.
                new_candidates, new_log_probs_history = \
                    self.sample(generated_tokens, log_probs_history, pred_logits,
                                chosen_drafts, n_accepted.squeeze(-1))

                # We don't want to give the model lines with eos just to produce pads.
                # So we store the best sequences with eos.
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

        return torch.cat([r.unsqueeze(0) for r in result], dim=0)

    def calculate_n_accepted_in_drafts(self, draft_tokens, masked_probs):
        """
        This function calculates the number of approved tokens in each draft for each candidate.

        :param draft_tokens: tensor of size (n_candidates, n_drafts, draft_len),
        :param masked_probs: (all unapproved tokens in masked_probs are supposed to be equal to 0.)
                             tensor of size (n_candidates, n_drafts, draft_len + 1, vocab_size),

        :return:
          ->  returns the number of approved tokens in each draft for each candidate:
                             tensor of size  (n_candidates, n_drafts)

        """
        draft_tokens_probs = torch.gather(masked_probs[:, :, :-1, :], dim=-1, index=draft_tokens.unsqueeze(-1)).squeeze(
            -1)
        #   -> (n_candidates, n_drafts, draft_len)
        verification = draft_tokens_probs != 0.

        _range = verification.cumsum(-1)  # (n_candidates, n_drafts, draft_len)
        accepted_in_drafts_bool = (torch.arange(1, verification.size(2) + 1).unsqueeze(0).unsqueeze(0).type_as(
            _range) == _range)  # (n_candidates, n_drafts, draft_len)

        return accepted_in_drafts_bool.sum(-1)  # (n_candidates, n_drafts, draft_len) -> (n_candidates, n_drafts)

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

    def get_drafts(self, src_unbatched: 'torch.LongTensor', draft_mode: bool):
        # src_unbatched: 'torch.LongTensor' of shape (1, src_len)
        if draft_mode:
            src_unbatched_i = src_unbatched[:, 1:]
            src_unbatched_i_pads = (src_unbatched_i == self.pad_token).int().sum(-1)
            n_tokens_without_pads = src_unbatched_i.size(1) - src_unbatched_i_pads
            src_unbatched_i_unpadded = src_unbatched_i[:, :n_tokens_without_pads]
            drafts = src_unbatched_i_unpadded.unfold(-1, min(self.n_speculative_tokens,
                                                             src_unbatched_i_unpadded.shape[1] - 1),
                                                     1).squeeze(0)[:self.max_drafts_num, :]
            # -> (n_drafts, draft_len = min(self.draft_len, unpadded_src_len - 1))
        else:
            drafts = src_unbatched[:, 0].unsqueeze(0).expand(self.max_drafts_num, 1)  # just [bos]
            # -> (n_drafts, draft_len=1)
        return drafts

    def get_drafts_linspace(self, src: torch.Tensor) -> torch.Tensor:
        s = src[:, 1:].clone().detach()  # we don't need the bos token
        end_inds = torch.nonzero(s == self.eos_token, as_tuple=True)[1]  # (b_sz)
        s.masked_fill_(s == self.pad_token, self.c_token)
        s.masked_fill_(s == self.eos_token, self.c_token)
        b_sz, s_len = s.size()
        max_draft_len = 200
        min_draft_len = 10
        drafts_len = max(min_draft_len, self.n_speculative_tokens)
        drafts_len = min(drafts_len, max_draft_len)

        start_inds = torch.full((b_sz,), 0, device=s.device)  # (b_sz)

        # Ensure we don't try to create more drafts than possible
        last_draft_len = 10
        possible_drafts_num = min(end_inds) - last_draft_len + 1

        n_drafts = min(self.max_drafts_num, possible_drafts_num)

        if n_drafts <= 1:
            return s[:, :drafts_len].unsqueeze(1)

        base_range = torch.arange(n_drafts).unsqueeze(0).repeat(b_sz, 1).to(s.device)  # (b_sz, n_drafts)

        const_multiplier = (end_inds - last_draft_len - start_inds) / torch.full((b_sz,), n_drafts - 1,
                                                                                 device=s.device)  # (b_sz)
        start_inds_of_drafts = (
                start_inds.unsqueeze(-1) + const_multiplier.unsqueeze(-1) * base_range).long()  # (b_sz, n_drafts)
        the_most_right_ind = torch.max(start_inds_of_drafts).item() + drafts_len
        if the_most_right_ind > s_len - 1:
            pad_tensor = torch.full((b_sz, the_most_right_ind - s_len + 2), self.c_token, device=s.device)
            s = torch.cat((s, pad_tensor), dim=-1)

        indices = start_inds_of_drafts.unsqueeze(-1) + torch.arange(drafts_len).unsqueeze(0).unsqueeze(
            0).to(s.device)  # (b_sz, n_drafts, drafts_len)

        # Use gather to create drafts
        drafts = torch.gather(s.unsqueeze(1).expand(-1, n_drafts, -1), 2, indices)  # (b_sz=1, n_drafts, drafts_len)

        return drafts.squeeze(0)  # Shape: (n_drafts, drafts_len)


def mask_with_num_logits_according_nucleus(pred_logits, nucleus, max_num_of_unmasked_positions, num=0.):
    """
    This function fills all unapproved tokens' logits with float(num). It uses nucleus parameter to decide which logits
    are big enough. No more than max_num_of_unmasked_positions but at least the best logit will be left unmasked
    for each distribution.
    If nucleus < 0, then it works in greedy mode. It masks everything accept the best token in each distribution.
    If nucleus > 1, then it works in beam search mode. It masks nothing and chooses the top n tokens in each distribution,
        where n is equal to max_num_of_unmasked_positions.
    If 0 < nucleus < 1 (we recommend nucleus = 0.9975), it works in top k mode. It masks all tokens' logits with
    cumulative probability above or equal to the nucleus parameter. But no more than max_num_of_unmasked_positions will
    be left unmasked in each row.
    """
    n_candidates, curr_len, vocab_size = pred_logits.size()  # (n_candidates, draft_len + 1, vocab_size)
    pred_logits = pred_logits.reshape(n_candidates * curr_len, vocab_size)  # -> (n_candidates * curr_len, vocab_size)

    sorted_logits, sorted_indices = torch.sort(pred_logits,
                                               descending=True)  # -> (n_candidates * curr_len, vocab_size)
    cumulative_probs = torch.cumsum(sorted_logits.softmax(-1), dim=-1)  # -> (n_candidates * curr_len, vocab_size)

    cumulative_probs = torch.roll(cumulative_probs, 1, dims=-1)

    cumulative_probs[:, 0] = nucleus - 1  # this protects the best probability in each distribution
    # Remove tokens with cumulative probability above or equal to the threshold (nucleus parameter).
    # At least the best probability in each row will be left unmasked
    keep_candidates_mask = cumulative_probs < nucleus  # -> (n_candidates * curr_len, vocab_size)

    keep_candidates_mask[:, max_num_of_unmasked_positions:] = False  # no more than max_num_of_unmasked_positions

    sorted_logits.masked_fill_(~keep_candidates_mask, float(num))  # the all unapproved tokens logits
    # will be set equal to float(num)

    masked_logits_according_nucleus = torch.gather(sorted_logits, 1, sorted_indices.argsort(1))
    # -> (n_candidates * curr_len, vocab_size)
    return masked_logits_according_nucleus.reshape(n_candidates, curr_len, vocab_size)


def sort(candidates, candidates_log_probs, descending=True):
    sorted_log_probs, sorted_inds = candidates_log_probs.sort(descending=descending)
    return candidates[sorted_inds], sorted_log_probs


class TranslationInferenceGreedySpeculative:
    """
    Speculative greedy decoding that supports input batch sizes larger than 1.
    """

    def __init__(self,
                 model,  # TranslationModel
                 max_len: int,
                 n_speculative_tokens: int,
                 max_drafts_num: int,
                 pad_token: int,
                 bos_token: int,
                 eos_token: int) -> None:
        self.model = model
        self.max_len = max_len
        self.pad_token = pad_token
        self.bos_token = bos_token
        self.eos_token = eos_token

        self.n_speculative_tokens = n_speculative_tokens
        self.max_drafts_num = max_drafts_num
        self.left_pad_token = -1

        self.accepted_tokens_num = 0
        self.given_tokens = 0
        self.model_calls_num = 0

    def __str__(self):
        return f"Greedy speculative decoding (n_speculative_tokens={self.n_speculative_tokens}, max_len={self.max_len})"

    def get_drafts(self, s: torch.Tensor) -> torch.Tensor:
        _, length = s.size()
        return s.unfold(-1, min(self.n_speculative_tokens, length - 1), 1)

    def get_drafts_linspace_ver0(self, s: torch.Tensor) -> torch.Tensor:
        b_size, length = s.size()

        # Ensure we don't try to create more drafts than possible
        max_possible_drafts = length - self.n_speculative_tokens + 1
        n_drafts = min(self.max_drafts_num, max_possible_drafts)

        if n_drafts <= 1:
            return s[:, :self.n_speculative_tokens].unsqueeze(1)

        # Generate start indices for each draft
        start_indices = torch.linspace(0, length - self.n_speculative_tokens, n_drafts, device=s.device).long()
        indices = start_indices.unsqueeze(1) + torch.arange(self.n_speculative_tokens, device=s.device).unsqueeze(0)
        indices = indices.unsqueeze(0).expand(b_size, -1, -1)

        # Use gather to create drafts
        drafts = torch.gather(s.unsqueeze(1).expand(-1, n_drafts, -1), 2, indices)

        return drafts  # Shape: (batch_size, n_drafts, n_speculative_tokens)

    def get_drafts_linspace(self, s: torch.Tensor) -> torch.Tensor:
        b_sz, length = s.size()
        # return s.unfold(-1, min(self.n_speculative_tokens, length - 1), 1)
        drafts_len = min(self.n_speculative_tokens, length - 1)
        start_inds = torch.zeros(b_sz).long()  # (b_sz)
        end_inds = length - torch.cumsum(s == 0, dim=-1).bool().sum(-1).to(start_inds.device) - 1  # (b_sz)

        # Ensure we don't try to create more drafts than possible
        max_possible_drafts = min(end_inds) - drafts_len + 1
        n_drafts = min(self.max_drafts_num, max_possible_drafts)

        if n_drafts <= 1:
            return s[:, :drafts_len].unsqueeze(1)

        base_range = torch.arange(n_drafts).unsqueeze(0).repeat(b_sz, 1)  # (b_sz, n_drafts)

        const_multiplier = (end_inds - drafts_len - start_inds) / (torch.ones(b_sz) * n_drafts - 1)  # (b_sz)
        start_inds_of_drafts = (
                start_inds.unsqueeze(-1) + const_multiplier.unsqueeze(-1) * base_range).long()  # (b_sz, n_drafts)

        indices = start_inds_of_drafts.unsqueeze(-1) + torch.arange(drafts_len).unsqueeze(0).unsqueeze(
            0)  # (b_sz, n_drafts, drafts_len)

        indices = indices.to(s.device)

        # Use gather to create drafts
        drafts = torch.gather(s.unsqueeze(1).expand(-1, n_drafts, -1), 2, indices)  # (b_sz, n_drafts, drafts_len)

        return drafts  # Shape: (batch_size, n_drafts, drafts_len)

    def generate(self, src: 'torch.LongTensor') -> 'torch.LongTensor':
        src_b_size, src_len = src.size()

        src_pad_mask = (src == self.model.src_pad_token_i).bool()
        self.given_tokens += (src_b_size * src_len - src_pad_mask.sum().item())
        memory = self.model.encode_src(src, src_pad_mask)  # -> (given_b_size, src_len, emb_dim)
        emb_dim = memory.size(2)

        generated_tokens = torch.full((src_b_size, 1), self.bos_token).type_as(src).long()
        draft_tokens = self.get_drafts(src[:, 1:])  # Not including bos_token in drafts.
        # -> (given_b_size, n_drafts, n_spec_tok)
        _, n_drafts, n_spec_tok = draft_tokens.size()
        # (given_b_size, n_drafts, n_spec_tok = min(self.n_speculative_tokens, src_len - 2))

        iters = 0
        finished_predictions = torch.full((src_b_size, self.max_len), self.pad_token).type_as(src)
        batch_indices_of_unfinished_seqs = torch.arange(src_b_size).type_as(src)
        while generated_tokens.size(1) < self.max_len:  # while gnrtd_len < self.max_len
            iters += 1
            b_size = batch_indices_of_unfinished_seqs.size(0)  # this is the current b_size.
            # It can be less than the original src_b_size because some prediction seqs can be already finished

            # Attach every draft to every sequence decoded so far
            drafted_seqs = torch.cat([generated_tokens.unsqueeze(1).repeat(1, n_drafts, 1), draft_tokens],
                                     dim=-1).reshape(b_size * n_drafts, -1)
            # [generated_tokens: (b_size, gnrtd_len)] -unsqz-> (b_size, 1, gnrtd_len)
            # -rpt-> (b_size, n_drafts, gnrtd_len)
            # -cat wth (b_size, n_drafts, n_spec_tok)-> (b_size, n_drafts, gnrtd_len + n_spec_tok)
            # -rshp-> (b_size * n_drafts, gnrtd_len + n_spec_tok)

            # Handle the different length of every sequence
            pos_enc_offset = (drafted_seqs == self.left_pad_token).int().sum(-1).view(-1, 1)
            generated_tokens = generated_tokens.masked_fill(generated_tokens == self.left_pad_token, self.pad_token)
            drafted_seqs = drafted_seqs.masked_fill(drafted_seqs == self.left_pad_token, self.pad_token)
            # (b_size * n_drafts, gnrtd_len + n_spec_tok)

            # Copy memory and memory mask along the batch dimension to match the new effective batch size
            memory_inflated = memory.unsqueeze(1).repeat(1, n_drafts, 1, 1)  # (b_size, src_len, emb_dim) ->
            # -> (b_size, 1, src_len, emb_dim) -> (b_size, n_drafts, src_len, emb_dim)

            memory_inflated = memory_inflated.view(b_size * n_drafts, -1, emb_dim)
            # -> (b_size * n_drafts, src_len, emb_dim)

            src_pad_mask_inflated = src_pad_mask.unsqueeze(1).repeat(1, n_drafts, 1)
            src_pad_mask_inflated = src_pad_mask_inflated.view(b_size * n_drafts, -1)

            # Run the decoder and sample from the predicted distributions
            pred_logits = self.model.decode_tgt(drafted_seqs,
                                                memory_inflated,
                                                memory_pad_mask=src_pad_mask_inflated,
                                                pos_enc_offset=pos_enc_offset)
            # -> (b_size * n_drafts, gnrtd_len + n_spec_tok, vocab_sz)
            self.model_calls_num += 1

            pred_tokens = torch.argmax(pred_logits, dim=2)  # -> (b_size * n_drafts, gnrtd_len + n_spec_tok)

            # Select and keep the draft with the largest number of accepted tokens
            pred_tokens = pred_tokens[:, -(n_spec_tok + 1):]  # -> (b_size * n_drafts, n_spec_tok + 1)
            verification = draft_tokens.reshape(b_size * n_drafts, n_spec_tok) == pred_tokens[:, :-1]
            # -> (b_size * n_drafts, n_spec_tok)

            _range = verification.cumsum(-1)  # -> (b_size * n_drafts, n_spec_tok)
            accepted_in_drafts = (torch.arange(1, verification.size(1) + 1).type_as(_range) == _range)
            # -> (b_size * n_drafts, n_spec_tok)

            n_accepted_in_drafts = accepted_in_drafts.sum(-1)  # -> (b_size * n_drafts)
            n_accepted_in_drafts = n_accepted_in_drafts.reshape(b_size, n_drafts)  # -> (b_size, n_drafts)
            n_accepted_in_drafts = n_accepted_in_drafts.topk(1, -1)
            draft_i = n_accepted_in_drafts.indices  # -> (b_size, 1)
            n_accepted = n_accepted_in_drafts.values  # -> (b_size, 1)
            self.accepted_tokens_num += n_accepted.sum().item()  # int
            pred_tokens = pred_tokens.reshape(b_size, n_drafts, -1)  # -> (b_size, n_drafts, n_spec_tok + 1)

            chosen = torch.gather(pred_tokens, 1,
                                  draft_i.unsqueeze(-1).expand(b_size, 1, n_spec_tok + 1)).squeeze(1)
            # -> (b_size, n_spec_tok + 1)

            pred_tokens = chosen.masked_fill(torch.arange(n_spec_tok + 1).type_as(n_accepted) > n_accepted,
                                             self.left_pad_token)  # -> (b_sz, n_spec_tok + 1)

            generated_tokens = torch.cat(
                (generated_tokens,
                 pred_tokens),
                dim=1
            )  # ((b_size, gnrtd_len) , (b_size, n_spec_tok + 1)) -cat-> (b_size, gnrtd_len + n_spec_tok + 1)

            current_finished_mask = (generated_tokens == self.eos_token).sum(-1).bool()  # -> (b_size)
            ids_of_current_finished_seqs = current_finished_mask.nonzero().ravel()
            if ids_of_current_finished_seqs.nelement() > 0:
                current_continuing_mask = ~current_finished_mask  # (b_size)
                batch_finished_indices = batch_indices_of_unfinished_seqs[current_finished_mask]

                current_finished_seqs = generated_tokens[ids_of_current_finished_seqs]
                current_finished_seqs = move_pads_to_the_right(current_finished_seqs, self.pad_token)
                current_finished_seqs = current_finished_seqs.masked_fill(
                    current_finished_seqs == self.left_pad_token,
                    self.pad_token)
                current_finished_seqs = trim_right_pads(current_finished_seqs, self.pad_token)

                finished_predictions[batch_finished_indices, :current_finished_seqs.size(1)] = current_finished_seqs

                batch_indices_of_unfinished_seqs = batch_indices_of_unfinished_seqs[current_continuing_mask]

                generated_tokens = generated_tokens[current_continuing_mask]
                draft_tokens = draft_tokens[current_continuing_mask]
                memory = memory[current_continuing_mask]
                src_pad_mask = src_pad_mask[current_continuing_mask]
            if batch_indices_of_unfinished_seqs.nelement() == 0:
                break

            generated_tokens = move_pads_to_the_left(generated_tokens, self.left_pad_token)
            # -> (b_size, gnrtd_len + n_spec_tok + 1)
            generated_tokens = generated_tokens.masked_fill(generated_tokens == self.pad_token, self.left_pad_token)
            generated_tokens = trim_left_pads(generated_tokens, self.left_pad_token)
            # -> (b_size, gnrtd_len + 1 <= new gnrtd_len <= gnrtd_len + n_spec_tok + 1)

        return finished_predictions.unsqueeze(1)


def num_speculative_tokens_to_accept(arr: torch.BoolTensor):
    _range = arr.cumsum(-1)
    return (torch.arange(1, arr.size(1) + 1).type_as(_range) == _range).sum(-1)


class TranslationInferenceGreedySpeculativeUnbatched:
    """
    Speculative greedy decoding that iterates over the batch dimension.
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
        self.accepted_tokens_num = 0

    def __str__(self):
        return f"Greedy speculative decoding unbatched (n_speculative_tokens={self.n_speculative_tokens}, max_len={self.max_len})"

    def get_drafts(self, s: torch.Tensor) -> torch.Tensor:
        b_size, length = s.size()
        return s.unfold(-1, min(self.n_speculative_tokens, length - 1), 1)

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

            draft_tokens = self.get_drafts(src_unbatched[i, :, 1:]).squeeze(0)
            n_drafts, n_spec_tok = draft_tokens.size()
            iters = 0
            while generated_tokens.size(1) < self.max_len:
                iters += 1
                draft_sequence = torch.cat([generated_tokens.repeat(n_drafts, 1), draft_tokens], dim=-1)
                pred_logits = self.model.decode_tgt(draft_sequence,
                                                    memory_unbatched[i].repeat(n_drafts, 1, 1),
                                                    memory_pad_mask=src_pad_mask_unbatched[i].repeat(n_drafts, 1))
                pred_tokens = torch.argmax(pred_logits, dim=2)
                pred_tokens = pred_tokens[:, -(n_spec_tok + 1):]
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

