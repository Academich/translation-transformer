import torch

from torch.nn.functional import pad

from utils.drafting import make_drafts


class TranslationInferenceGreedySpeculative:
    """
    Speculative greedy decoding that supports various input batch sizes.
    """

    def __init__(self,
                 model,  # TranslationModel
                 max_len: int,
                 draft_len: int,
                 n_drafts: int,
                 pad_token: int,
                 bos_token: int,
                 eos_token: int,
                 replace_token: int) -> None:
        self.model = model
        self.max_len = max_len
        self.pad_token = pad_token
        self.bos_token = bos_token
        self.eos_token = eos_token
        self.replace_token = replace_token

        self.draft_len = draft_len
        self.n_drafts = n_drafts

        self.accepted_tokens_num = 0
        self.model_calls_num = 0


    def __str__(self):
        return f"Greedy speculative decoding (draft_len={self.draft_len}, n_drafts={self.n_drafts}, max_len={self.max_len})"

    def generate(self, src: 'torch.LongTensor') -> 'torch.LongTensor':
        """
        Generate the predictions for a batch of source sequences using speculative greedy decoding.
        As the sequences in the batch get finished, they are removed from the batch and the remaining sequences are processed.

        B - batch size
        Ls - source sequence length
        Lg - generated sequence length excluding padding tokens (increases with each iteration)
        N - number of drafts
        D - draft length
        E - embedding dimensionality
        V - vocabulary size
        Bc - current batch size (less or equal to B)
        Args:
            src: (B, Ls) - source sequences
        Returns:
            (B, 1, Lg) - predicted sequences
        """
        N, D = self.n_drafts, self.draft_len
        B, _ = src.size()
        # Encode the source and cache the source embeddings to be reused multiple times
        src_pad_mask = (src == self.model.src_pad_token_i).bool() # (B, Ls)
        memory = self.model.encode_src(src, src_pad_mask) # (B, Ls, E)

        # we don't need the bos token in drafts
        draft_tokens = make_drafts(
            src[:, 1:],
            draft_len=self.draft_len, 
            n_drafts=self.n_drafts, 
            min_draft_len=1, 
            max_draft_len=self.max_len, 
            eos_token_idx=self.eos_token, 
            pad_token_idx=self.pad_token, 
            replace_token_idx=self.replace_token
        ) # (B, N, D)
        assert draft_tokens.size(1) == N

        memory_inflated = torch.repeat_interleave(memory, N, dim=0) # (B*N, Ls, E)
        src_pad_mask_inflated = torch.repeat_interleave(src_pad_mask, N, dim=0) # (B*N, Ls)

        # Indices of the sequences in the batch. Needed to filter out the finished sequences
        unfinished_query_batch_indices = torch.arange(B).type_as(src) # (B,)

        generated_tokens = torch.full((B, 1), self.bos_token).type_as(src).long() # (B, 1)
        
        # Index of the last non-pad token in the generated sequence
        front_index = torch.zeros_like(generated_tokens) # (B, 1)

        finished_predictions = torch.full((B, self.max_len), self.pad_token).type_as(src) # (B, max_len)

        # A helper tensor
        draft_len_range = torch.arange(D + 1).type_as(src) # (D + 1,)

        iters = 0
        while generated_tokens.size(1) < self.max_len:  # while gnrtd_len < self.max_len
            iters += 1
            Bc = unfinished_query_batch_indices.nelement()

            generated_tokens_present_pad_num = ((generated_tokens == self.pad_token).sum(0) == Bc).sum()  # (1,)
            generated_tokens_padded = pad(generated_tokens, 
                (0, D + 1 - generated_tokens_present_pad_num),
                "constant", 
                value=self.pad_token
            ) # (Bc, Lg + D + 1)

            generated_tokens_padded_inflated = torch.repeat_interleave(generated_tokens_padded[:, :-1], N, dim=0) # (Bc*N, Lg + D)
            front_index_inflated = torch.repeat_interleave(front_index, N, dim=0) # (Bc*N, 1)

            insertion_indices = front_index_inflated + draft_len_range[:-1] + 1 # (Bc*N, D)

            # Concatenate the already generated sequences with the corresponding draft sequences
            draft_tokens_effective_batch_size = draft_tokens.view(draft_tokens.size(0) * draft_tokens.size(1), draft_tokens.size(2)) # (Bc*N, D)
            drafted_seqs = generated_tokens_padded_inflated.scatter(
                dim=1, 
                index=insertion_indices, 
                src=draft_tokens_effective_batch_size
            ) # (Bc*N, Lg + D)

            # Run the decoder and sample from the predicted distributions
            pred_logits = self.model.decode_tgt(drafted_seqs,
                                memory_inflated,
                                memory_pad_mask=src_pad_mask_inflated) # (Bc*N, Lg + D, V)
            self.model_calls_num += 1
            pred_tokens = torch.argmax(pred_logits, dim=2) # (Bc*N, Lg + D)

            # Consider only the tokens predicted for the draft
            retrieval_indices = front_index_inflated + draft_len_range # (Bc*N, D + 1)
            pred_tokens = pred_tokens.gather(dim=1, index=retrieval_indices) # (Bc*N, D + 1)
            
            # Find the drafts with the most accepted tokens
            verification = draft_tokens_effective_batch_size == pred_tokens[:, :-1]  # (Bc*N, D)
            accepted_in_drafts = (draft_len_range[1:] == verification.cumsum(-1)) # (Bc*N, D)
            n_accepted_in_drafts = accepted_in_drafts.sum(-1) # (Bc*N,)
            n_accepted_in_drafts = n_accepted_in_drafts.view(Bc, N) # (Bc, N)
            n_accepted, draft_i = n_accepted_in_drafts.topk(1, -1) # (Bc, 1), (Bc, 1)

            # Extract the best draft for every sequence in the batch
            pred_tokens = pred_tokens.view(Bc, N, -1) # (Bc, N, D + 1)
            chosen = torch.gather(pred_tokens, 1, draft_i.unsqueeze(-1).expand(Bc, 1, D + 1)).squeeze(1) # (Bc, D + 1)

            # Mask out the tokens that were not accepted
            decline_pred_tokens = draft_len_range > n_accepted # (Bc, D + 1)
            chosen_truncated_to_accepted = chosen.masked_fill(decline_pred_tokens, self.pad_token) # (Bc, D + 1)

            # Insert the newly generated tokens into the entire generated sequence at the correct positions
            insertion_indices = front_index + draft_len_range + 1 # (Bc, D + 1)
            generated_tokens = generated_tokens_padded.scatter(dim=1, index=insertion_indices, src=chosen_truncated_to_accepted) # (Bc, Lg + D + 1)
            front_index = front_index + n_accepted + 1 # (Bc, 1)

            # Put away the sequences that reached the EOS token. It speeds up the decoding
            sequence_finished = (generated_tokens == self.eos_token).sum(-1).bool() # (Bc,)
            sequence_indices_finised = sequence_finished.nonzero().ravel() # (Bc,)
            if sequence_indices_finised.nelement() > 0: # if there are finished sequences
                sequence_to_continue = ~sequence_finished # (Bc,)
                sequence_finished_inflated = torch.repeat_interleave(sequence_finished, N, dim=0) # (Bc*N,)
                sequence_to_continue_inflated = ~sequence_finished_inflated # (Bc*N,)
                query_batch_indices_finished = unfinished_query_batch_indices[sequence_finished] # (Bc,)

                # Save the finished sequences to the output tensor
                finished_predictions[query_batch_indices_finished, :generated_tokens.size(1)] = generated_tokens[sequence_indices_finised]

                # Here we remove the finished sequences from the batch and B becomes Bc
                unfinished_query_batch_indices = unfinished_query_batch_indices[sequence_to_continue]
                generated_tokens = generated_tokens[sequence_to_continue]
                front_index = front_index[sequence_to_continue]
                draft_tokens = draft_tokens[sequence_to_continue]
                memory = memory[sequence_to_continue]
                src_pad_mask = src_pad_mask[sequence_to_continue]
                memory_inflated = memory_inflated[sequence_to_continue_inflated]
                src_pad_mask_inflated = src_pad_mask_inflated[sequence_to_continue_inflated]

            # If all sequences are finished, stop the decoding
            if unfinished_query_batch_indices.nelement() == 0:
                break

        return finished_predictions.unsqueeze(1) # (B, 1, Lg)


def topk_in_each_group(score_1d, length_of_each_group, k, pad=None):
    """
    This function finds the biggest k values and the corresponding indices. It is needed when each group has different
    number of candidates.
    N - number of_groups.

    :param score_1d: tensor (shape_0 = sum(length_of_each_group), 1)
    :param length_of_each_group: tensor (N,), Each length should be >= k
    :param k: int
    :param pad: it's needed to fill fake score_1d positions to make reshape (N, max_len_of_group) possible. Pad should
    be less than each number in score_1d

    :return:
      ->  topk_score: tensor (N, k),
          topk_inds_1d: tensor (N * k,); score_1d[topk_inds_1d].reshape(N, k) is equal to topk_score.

    """
    b_size = length_of_each_group.shape[0]
    assert torch.min(length_of_each_group).item() >= k
    max_len_of_group = torch.max(length_of_each_group).item()

    # We make fake sequences with an artificial probability -inf in case if a different number of sequences
    # were sampled on the basis of the chosen drafts
    start_ind_of_each_group = torch.roll(length_of_each_group, 1, dims=-1)  # -> (b_size)
    start_ind_of_each_group[0] = 0
    start_ind_of_each_group = start_ind_of_each_group.cumsum(-1).unsqueeze(1)
    # -> (N, 1)

    different_num_of_candidates_in_groups = (length_of_each_group == max_len_of_group).sum() != b_size
    if different_num_of_candidates_in_groups:
        if pad is None:
            pad = torch.min(score_1d).item() - 1

        inds_for_2d = torch.arange(max_len_of_group).to(score_1d.device).unsqueeze(0).repeat(b_size, 1)
        # -> (N, max_len_of_group)

        mask_for_fake_seqs = inds_for_2d >= length_of_each_group.unsqueeze(1)
        inds_for_2d = start_ind_of_each_group + (inds_for_2d % length_of_each_group.unsqueeze(1))

        score_1d = score_1d[inds_for_2d.reshape(-1)]
        # -> (N * max_len_of_group, 1)

        score_1d[mask_for_fake_seqs.reshape(-1)] = pad  # pads
        score_2d = score_1d.reshape(b_size, max_len_of_group)
        # -> (N, max_len_of_group)

        topk_score, topk_inds = score_2d.topk(k=k, axis=-1, sorted=True)
        # -> (N, k)

        topk_inds_1d = torch.gather(inds_for_2d, dim=1, index=topk_inds)
        #  (N, max_len_of_group) -> (N, k)
    else:
        score_2d = score_1d.reshape(b_size, max_len_of_group)
        # -> (N, max_len_of_group)

        topk_score, topk_inds = score_2d.topk(k=k, axis=-1, sorted=True)
        # -> (N, k)

        topk_inds_1d = start_ind_of_each_group + topk_inds
    topk_inds_1d = topk_inds_1d.reshape(-1)
    #  -> (N * k,)
    return topk_score, topk_inds_1d


class TranslationInferenceBeamSearchSpeculative:
    def __init__(self,
                 model,  # TranslationModel
                 max_len: int,
                 n_best: int,
                 draft_len: int,
                 n_drafts: int,
                 vocab_size: int,
                 smart_drafts_mode: bool,
                 pad_token: int,
                 bos_token: int,
                 eos_token: int,
                 C_token: int,
                 ) -> None:
        self.smart_drafts_mode = smart_drafts_mode
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
        self.model_input_lines_num = 0

        self.max_drafts_num = n_drafts

        self.log_prob_pad = 7.  # should be more than 0.

        self.n_drafts = 0
        self.requested_drafts_num = n_drafts
        self.produced_non_pad_tokens = 0

        self.max_draft_len = 200
        self.min_draft_len = 5
        drafts_len = max(self.min_draft_len, draft_len)
        drafts_len = min(drafts_len, self.max_draft_len)
        if drafts_len != draft_len:
            print(f"The draft length should be in range [{self.min_draft_len}: {self.max_draft_len}], so it was changed to {drafts_len}")
        self.draft_len = drafts_len

        self.acceptance_rate_pad_for_alredy_finished_seqs = -1  # should be negative
        self.acceptance_rate_pad_for_fake_seqs = -7  # should be negative

        self.b_sz = 0

    def __str__(self):
        return f"SpeculativeSampling decoding (n_best={self.n_best}, max_len={self.max_len}, max_num_of_drafts={self.max_drafts_num}, draft_len={self.draft_len})"

    def sample(self, curr_lines, curr_log_probs, pred_logits, chosen_drafts, b_size, draft_place_bool, n_accepted):
        """
        This function samples all possible sequences within a selected draft. Each draft can
        produce (self.max_num_positions_for_sampling - 1) * num_of_approved_tokens + self.max_num_positions_for_sampling
        at most.

        :param curr_lines: tensor (n_candidates, drafted_len),
        :param curr_log_probs_history: tensor (n_candidates, 1),
        :param pred_logits: tensor (n_candidates, draft_len + 1, vocab_size),
        :param chosen_drafts: tensor (n_candidates, draft_len),
        :param b_size: int,
        :param draft_place_bool: tensor (n_candidates, drafted_len), it contains true where the draft supposed to be in curr_lines,
            in each line there are draft_len trues
        :param n_accepted: tensor (n_candidates)
        :return:
          ->  new_lines: tensor (num_lines, len),
              new_log_probs: tensor (num_lines, 1)
              num_of_new_seqs_for_each_in_batch: tensor (b_size)
              token_postn: tensor (num_lines), to calculate the number of accepted tokens in the next top n sequences
                later; self.acceptance_rate_pad_for_already_finished_seqs means that the given sequence had already the
                eos token and so didn't need subsequent tokens
        """
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
        # the drafts can not be leaves of the top n tree of the sequences

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

        log_prob_of_roots = curr_log_probs[candts_inds]  # (num, 1)
        draft_place_bool = draft_place_bool[candts_inds]  # (num, max_len)

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
        new_seqs_log_probs.masked_fill_(mask_for_tokens_after_the_sampled, 0.)
        #    -> (num, draft_len + 1)
        new_seqs_log_probs = new_seqs_log_probs.cumsum(dim=-1)  # -> (num, draft_len + 1)

        last_log_prob_from_roots = torch.min(log_prob_of_roots, dim=-1, keepdim=True).values
        # (num, 1)
        new_seqs_log_probs = last_log_prob_from_roots + new_seqs_log_probs[:, -1:]
        #    -> (num, 1)
        new_seqs.masked_fill_(mask_for_tokens_after_the_sampled, self.pad_token_idx)
        #    -> (num, draft_len + 1)

        new_seqs_place_bool = torch.logical_or(draft_place_bool, torch.roll(draft_place_bool, 1, 1))
        # -> (num, drafted_len) It contains draft_len+1 Trues in each line
        previous_roots[new_seqs_place_bool] = new_seqs.reshape(-1)

        token_postn[already_finished_given_seqs] = self.acceptance_rate_pad_for_alredy_finished_seqs
        # the given sequences with eos didn't need the draft tokens. We
        # don't take pads into account calculating the acceptance rate
        return previous_roots, new_seqs_log_probs, num_of_new_seqs_for_each_in_batch, token_postn

    def get_vocab_tokens_bool_lib(self, draft_lib):
        """
        :param draft_lib: tensor (b_size, n_drafts, draft_len),

        :return:
          ->  vocab_tokens_bool_lib: tensor (b_sz, vocab_size, n_drafts),
        """

        draft_start_tokens = draft_lib[:, :, 0]
        # -> (b_sz, n_drafts)
        b_sz, n_drafts = draft_start_tokens.size()
        vocab_tokens = torch.arange(self.vocab_size).unsqueeze(0).unsqueeze(-1).expand(b_sz, self.vocab_size, n_drafts)
        # -> (b_sz, vocab_size, n_drafts)
        vocab_tokens_bool = draft_start_tokens.unsqueeze(1).expand(b_sz, self.vocab_size, n_drafts) == vocab_tokens.type_as(draft_lib)
        # -> (b_sz, vocab_size, n_drafts)
        t = vocab_tokens_bool.view(-1, n_drafts)
        t[t.sum(-1) == 0, 0] = True   # Each line needs at least one draft
        t[t.cumsum(-1) > self.requested_drafts_num] = False
        return vocab_tokens_bool

    def generate(self, src: 'torch.LongTensor') -> list['torch.LongTensor']:
        if self.smart_drafts_mode:
            return self.generate_with_smart_drafts(src)
        else:
            return self.generate_trying_all_the_drafts(src)

    def generate_trying_all_the_drafts(self, src: 'torch.LongTensor') -> list['torch.LongTensor']:
        # we don't need the bos token in drafts
        draft_tokens = make_drafts(src[:, 1:], self.draft_len, self.requested_drafts_num, self.min_draft_len,
                                   self.max_draft_len, self.eos_token_idx, self.pad_token_idx, self.C_token_idx)

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

        generated_tokens = torch.full((b_size, 1), self.bos_token_idx, device=src.device)
        #   -> (b_size, 1)

        log_probs = torch.full((b_size, 1), 0., device=src.device)
        #   -> (b_size, 1)

        num_of_empty_columns = ((generated_tokens == self.pad_token_idx).sum(0) == b_size).sum().item()
        #   -> (1,)
        postn_after_the_last_meaning_token = generated_tokens.shape[1] - num_of_empty_columns
        #   -> (1,)
        possible_draft_len = self.max_len - postn_after_the_last_meaning_token - 1
        #   -> (b_size, 1)
        beam_size = 1

        logits_base = torch.full((b_size * n_drafts, draft_len + 1, self.vocab_size), 0., device=src.device)
        #   -> (b_s * n_drafts, draft_len + 1, vocab_size)

        while possible_draft_len >= 1 and postn_after_the_last_meaning_token <= self.max_len:
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

            draft_place_len = draft_len + 1 - num_of_empty_columns
            if draft_place_len > 0:
                draft_place = torch.full((n_candidates, draft_place_len), self.pad_token_idx, device=src.device)
                generated_tokens = torch.cat((generated_tokens, draft_place), dim=-1)
            # -> (n_candidates, drafted_len)

            logits_base = logits_base[:, :draft_len + 1, :]

            self.model_calls_num += 1
            pad_place_bool = generated_tokens == self.pad_token_idx
            # -> (n_candidates, drafted_len)
            draft_place_bool = torch.logical_and(pad_place_bool,
                                                 pad_place_bool.cumsum(-1) <= draft_len)
            # -> (n_candidates, drafted_len)

            draft_place_bool_idx_input = draft_place_bool.unsqueeze(1).repeat(1, n_drafts, 1)
            # -> (b_s * bm_sz, n_drafts, drafted_len)
            generated_tokens_input = generated_tokens.unsqueeze(1).repeat(1, n_drafts, 1)
            # -> (b_s * bm_sz, n_drafts, drafted_len)

            generated_tokens_input[draft_place_bool_idx_input] = draft_tokens.reshape(-1)
            draft_place_bool_idx_input = draft_place_bool_idx_input.flatten(end_dim=1)
            # -> (b_s * bm_sz * n_drafts, drafted_len)
            generated_tokens_input = generated_tokens_input.flatten(end_dim=1)
            # # -> (b_s * bm_sz * n_drafts, drafted_len)

            bool_idx_of_unfinished = bool_idx_of_unfinished.unsqueeze(-1).repeat(1, n_drafts).flatten(end_dim=1)
            # -> (b_s * bm_sz * n_drafts)
            draft_place_bool_idx_input = draft_place_bool_idx_input[bool_idx_of_unfinished]
            #   -> (num_of_unfinished, drafted_len)
            pred_logits = self.model.decode_tgt(generated_tokens_input[bool_idx_of_unfinished],
                                                memory[bool_idx_of_unfinished],
                                                memory_pad_mask=src_pad_mask[bool_idx_of_unfinished])
            #  -> (num_of_unfinished, drafted_len, vocab_size)
            vocab_size = pred_logits.shape[-1]

            pred_logits = pred_logits[
                torch.logical_or(draft_place_bool_idx_input, torch.roll(draft_place_bool_idx_input, -1, 1))].reshape(
                -1, draft_len + 1, vocab_size)
            #  -> (num_of_unfinished, draft_len + 1, vocab_size)

            logits_base[bool_idx_of_unfinished] = pred_logits
            pred_logits = logits_base
            #   -> (n_candidates * n_drafts, draft_len + 1, vocab_size)

            # Choosing the best draft for each candidate. The draft with the biggest number of
            # approved tokens is the best draft for the given candidate. #########################################

            # All unapproved tokens in masked_probs have zero probability
            # We use nucleus=0.9975 and max_num_of_unmasked_positions=n_best to avoid sampling of low probable sequences
            # and reduce calculation
            masked_probs = mask_with_num_logits_according_nucleus(pred_logits, nucleus=0.9975,
                                                                  max_num_of_unmasked_positions=self.n_best,
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
            new_candidates, new_log_probs, num_of_new_seqs_for_each_in_batch, accepted_tokens_num = \
                self.sample(generated_tokens, log_probs, pred_logits,
                            chosen_drafts, b_size, draft_place_bool, n_accepted.squeeze(-1))

            new_log_probs, top_inds_1d = topk_in_each_group(score_1d=new_log_probs,
                                                            length_of_each_group=num_of_new_seqs_for_each_in_batch,
                                                            k=self.n_best, pad=-float("inf"))
            new_candidates = new_candidates[top_inds_1d]
            # -> (b_size * beam_size, drafted_len)

            accepted_tokens_num = accepted_tokens_num[top_inds_1d]
            # -> (b_size * beam_size,)
            accepted_tokens_num = accepted_tokens_num[accepted_tokens_num >= 0]

            self.accepted_tokens_num += accepted_tokens_num.sum().item()
            self.produced_non_pad_tokens += accepted_tokens_num.sum().item() + accepted_tokens_num.size(0)

            if (new_candidates == self.eos_token_idx).sum(-1).bool().sum() == b_size * self.n_best:
                break
            generated_tokens = new_candidates
            log_probs = new_log_probs.reshape(b_size * self.n_best, 1)
            # -> (b_size * beam_size, 1)

            num_of_empty_columns = torch.min((generated_tokens == self.pad_token_idx).sum(-1)).item()
            #   -> (1,)
            postn_after_the_last_meaning_token = generated_tokens.shape[1] - num_of_empty_columns
            #   -> (1,)
            possible_draft_len = self.max_len - postn_after_the_last_meaning_token - 1
            #   -> (b_size, 1)
        return new_candidates.reshape(b_size, self.n_best, -1)

    def generate_with_smart_drafts(
        self, src: "torch.LongTensor"
    ) -> list["torch.LongTensor"]:
        preliminary_drafts_num = src.shape[1] - 5
        # the last draft will contain only 5 meaningfull tokens of the src molecule
        # We need the bos token in drafts
        draft_lib = make_drafts(
            src,
            self.draft_len + 1,
            preliminary_drafts_num,
            self.min_draft_len,
            self.max_draft_len,
            self.eos_token_idx,
            self.pad_token_idx,
            self.C_token_idx,
        )
        # -> (b_size, n_drafts, draft_len)

        vocab_tokens_bool_lib = self.get_vocab_tokens_bool_lib(draft_lib)
        # -> (b_sz, vocab_size, n_drafts)
        b_size, n_drafts, draft_len = draft_lib.size()

        draft_len -= 1
        src_pad_mask = (src == self.model.src_pad_token_i).bool()
        # -> (b_size, src_len)

        memory = self.model.encode_src(src, src_pad_mask)
        # -> (b_size, src_len, emb_dim)
        _, src_len, emb_dim = memory.size()

        iters = -1

        generated_tokens = torch.full(
            (b_size, 1), self.bos_token_idx, device=src.device
        )
        #   -> (b_size, 1)

        log_probs = torch.full((b_size, 1), 0.0, device=src.device)
        #   -> (b_size, 1)

        num_of_empty_columns = (
            ((generated_tokens == self.pad_token_idx).sum(0) == b_size).sum().item()
        )
        #   -> (1,)
        postn_after_the_last_meaning_token = (
            generated_tokens.shape[1] - num_of_empty_columns
        )
        #   -> (1,)
        possible_draft_len = self.max_len - postn_after_the_last_meaning_token - 1
        #   -> (b_size, 1)
        beam_size = 1

        while (
            possible_draft_len >= 1
            and postn_after_the_last_meaning_token <= self.max_len
        ):
            iters += 1

            if iters == 1:
                beam_size = self.n_best
                src_pad_mask = (
                    src_pad_mask.unsqueeze(1)
                    .repeat(1, self.n_best, 1)
                    .flatten(end_dim=1)
                )
                # (b_size, src_len) -> (b_size * bm_sz, src_len)
                memory = (
                    memory.unsqueeze(1).repeat(1, self.n_best, 1, 1).flatten(end_dim=1)
                )
                # (b_size, src_len, emb_dim) -> (b_size * bm_sz, src_len, emb_dim)

            draft_len = min(possible_draft_len, draft_len)
            draft_lib = draft_lib[
                :, :, : draft_len + 1
            ]  # we use the drafts without the first token
            n_candidates, curr_len = generated_tokens.size()

            pads_num = (generated_tokens == self.pad_token_idx).sum(-1)
            # -> (n_candidates)
            draft_place_len = draft_len + 1 - num_of_empty_columns
            if draft_place_len > 0:
                draft_place = torch.full(
                    (n_candidates, draft_place_len),
                    self.pad_token_idx,
                    device=src.device,
                )
                generated_tokens = torch.cat((generated_tokens, draft_place), dim=-1)
            # -> (n_candidates, drafted_len)
            ############################################################################################################
            # Choose drafts from the draft library starting from the last meaning tokens of the given sequence
            inds_of_last_tokens = (generated_tokens != self.pad_token_idx).sum(-1) - 1
            # -> (n_candidates)
            last_tokens = torch.gather(
                generated_tokens, dim=1, index=inds_of_last_tokens.unsqueeze(-1)
            )
            # -> (n_candidates, 1)
            last_tokens = (
                last_tokens.reshape(b_size, beam_size, 1)
                .unsqueeze(-1)
                .repeat(1, 1, 1, n_drafts)
            )
            # -> (b_size, bm_sz, 1, n_drafts)
            # vocab_tokens_bool: (b_sz, vocab_size, n_drafts)
            vocab_tokens_bool = vocab_tokens_bool_lib.unsqueeze(1).repeat(
                1, beam_size, 1, 1
            )
            # -> (b_sz, bm_sz, vocab_size, n_drafts)

            # library_bool marks suitable drafts from the draft library as True for the given sequence. The draft is
            # supposed to be suitable if it starts with the same token the sequence generated so far ends with.
            library_bool = torch.gather(vocab_tokens_bool, 2, last_tokens).squeeze(2)
            # -> (b_size, bm_sz, n_drafts)
            num_of_drafts_for_each_in_batch_and_beam = library_bool.sum(-1)
            # -> (b_size, bm_sz)
            num_of_drafts_for_each_candidate = (
                num_of_drafts_for_each_in_batch_and_beam.reshape(-1)
            )
            # -> (n_candidates)
            batch_idx, beam_idx, draft_idx = torch.nonzero(library_bool, as_tuple=True)
            # -> (n),(n),(n)
            candidate_idx = batch_idx * beam_size + beam_idx
            # -> (n)
            generated_tokens_n = generated_tokens[candidate_idx]
            # -> (n, drafted_len)
            # We don't need the first token of the draft. It only helps to choose the most suitable ones from the draft
            #  library
            drafts_n = draft_lib[(batch_idx, draft_idx)][:, 1:]
            # (b_sz, n_drafts, draft_len) -> (n, draft_len)
            pad_place_bool = generated_tokens == self.pad_token_idx
            # -> (n_candidates, drafted_len)
            draft_place_bool = torch.logical_and(
                pad_place_bool, pad_place_bool.cumsum(-1) <= draft_len
            )
            # -> (n_candidates, drafted_len)
            draft_place_bool_idx_n = draft_place_bool[candidate_idx][
                :, : generated_tokens.shape[1]
            ]
            # -> (n_candidates, drafted_len)
            generated_tokens_n[draft_place_bool_idx_n] = drafts_n.reshape(-1)
            # -> (n, drafted_len)
            ##########################################################################################################
            self.model_input_lines_num += generated_tokens_n.shape[0]
            self.model_calls_num += 1
            bool_idx_of_unfinished= ~(
                (generated_tokens_n == self.eos_token_idx).sum(-1).bool()
            )
            # -> (n)
            pred_logits_r = self.model.decode_tgt(
                generated_tokens_n[bool_idx_of_unfinished],
                memory[candidate_idx][bool_idx_of_unfinished],
                memory_pad_mask=src_pad_mask[candidate_idx][bool_idx_of_unfinished],
            )
            #  -> (num_of_unfinished, drafted_len, vocab_size)
            pred_logits_n = torch.full((generated_tokens_n.shape[0], draft_len + 1, self.vocab_size), 0., device=memory.device)
        #   -> (n, draft_len + 1, vocab_size)
            pred_logits_n[:,:,self.pad_token_idx] = 35.

            self.b_sz += pred_logits_r.shape[0]
            vocab_size = pred_logits_n.shape[-1]

            pred_logits_r = pred_logits_r[
                torch.logical_or(
                    draft_place_bool_idx_n[bool_idx_of_unfinished], torch.roll(draft_place_bool_idx_n[bool_idx_of_unfinished], -1, 1)
                )
            ].reshape(-1, draft_len + 1, vocab_size)
            #  -> (num_of_unfinished, draft_len + 1, vocab_size)
            pred_logits_n[bool_idx_of_unfinished] = pred_logits_r
            masked_probs_n = mask_with_num_logits_according_nucleus(
                pred_logits_n,
                nucleus=0.9975,
                max_num_of_unmasked_positions=self.n_best,
                num="-inf",
            ).softmax(-1)
            #   -> (n, draft_len + 1, vocab_size)

            num_of_accepted_in_drafts_n = self.calculate_n_accepted_in_drafts(
                drafts_n.unsqueeze(0), masked_probs_n.unsqueeze(0)
            ).squeeze(0)
            #   -> (n,) # n is equal to sum(num_of_drafts_for_each_candidate)
            n_accepted, top_inds_1d = topk_in_each_group(
                score_1d=num_of_accepted_in_drafts_n,
                length_of_each_group=num_of_drafts_for_each_candidate,
                k=1,
                pad=-1,
            )
            # -> (n_candidates, 1), (n_candidates,)
            pred_logits = pred_logits_n[top_inds_1d]
            # -> (n_candidates, draft_len + 1, vocab_size)
            chosen_drafts = drafts_n[top_inds_1d]
            # ->  (n_candidates, draft_len)

            # Sample all possible lines within the chosen drafts:
            # new_candidates have the initial tokens and the new ones
            (
                new_candidates,
                new_log_probs,
                num_of_new_seqs_for_each_in_batch,
                accepted_tokens_num,
            ) = self.sample(
                generated_tokens,
                log_probs,
                pred_logits,
                chosen_drafts,
                b_size,
                draft_place_bool,
                n_accepted.squeeze(-1),
            )

            ###########################################################################################################
            new_log_probs, top_inds_1d = topk_in_each_group(
                score_1d=new_log_probs,
                length_of_each_group=num_of_new_seqs_for_each_in_batch,
                k=self.n_best,
                pad=-float("inf"),
            )
            new_candidates = new_candidates[top_inds_1d]
            # -> (b_size * beam_size, drafted_len)

            accepted_tokens_num = accepted_tokens_num[top_inds_1d]
            # -> (b_size * beam_size,)
            accepted_tokens_num = accepted_tokens_num[accepted_tokens_num >= 0]
            self.accepted_tokens_num += accepted_tokens_num.sum().item()
            self.produced_non_pad_tokens += (
                accepted_tokens_num.sum().item() + accepted_tokens_num.size(0)
            )

            if (new_candidates == self.eos_token_idx).sum(
                -1
            ).bool().sum() == b_size * self.n_best:
                break
            generated_tokens = new_candidates
            log_probs = new_log_probs.reshape(b_size * self.n_best, 1)
            # -> (b_size * beam_size, 1)

            num_of_empty_columns = torch.min(
                (generated_tokens == self.pad_token_idx).sum(-1)
            ).item()
            #   -> (1,)
            postn_after_the_last_meaning_token = (
                generated_tokens.shape[1] - num_of_empty_columns
            )
            #   -> (1,)
            possible_draft_len = self.max_len - postn_after_the_last_meaning_token - 1
            #   -> (b_size, 1)

        return new_candidates.reshape(b_size, self.n_best, -1)

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


def num_speculative_tokens_to_accept(arr: torch.BoolTensor):
    _range = arr.cumsum(-1)
    return (torch.arange(1, arr.size(1) + 1).type_as(_range) == _range).sum(-1)

