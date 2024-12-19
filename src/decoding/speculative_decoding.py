import torch

from torch.nn.functional import pad
from collections import defaultdict

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



class TranslationInferenceBeamSearchSpeculativeBatchedWithoutLeftPads:
    def __init__(self,
                 model,  # TranslationModel
                 max_len: int,
                 n_best: int,
                 draft_len: int,
                 n_drafts: int,
                 pad_token: int,
                 bos_token: int,
                 eos_token: int,
                 C_token: int,
                 vocab_size,
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

    def __str__(self):
        return f"SpeculativeSampling decoding (n_best={self.n_best}, max_len={self.max_len}, max_num_of_drafts={self.max_drafts_num}, draft_len={self.draft_len})"

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
                later; self.acceptance_rate_pad_for_already_finished_seqs means that the given sequence had already the
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

