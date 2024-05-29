from heapq import heappush, heappop
import torch.nn.functional as F
import torch


# Beam size: K
# Batch size: B
# Current length: L

class TranslationInferenceNucleusSpeculativeUnbatchedNoCycles:

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
        self.temperature = temperature
        self.n_best = n_best

    def __str__(self):
        return f"NucleusSpeculativeUnbatched decoding (max_len={self.max_len}, nucleus={self.nucleus}, temperature={self.temperature})"

    def sample(self, pred_logits, num_samples):
        n, curr_len, vocab_size = pred_logits.size()  # (n_candidates * n_drafts, draft_len + 1, vocab_size)
        pred_logits = pred_logits.reshape(n * curr_len, vocab_size)  # -> (n * curr_len, vocab_size)
        predicted_log_probs = pred_logits.softmax(-1).log()

        # actually we don't need to generate the pad token
        logits_for_sampling = pred_logits
        logits_for_sampling[:, self.pad_token] = -float("inf")

        sorted_logits, sorted_indices = torch.sort(logits_for_sampling, descending=True)  # -> (n * curr_len, vocab_size)
        cumulative_probs = torch.cumsum(sorted_logits.softmax(-1), dim=-1)  # -> (n * curr_len, vocab_size)

        # Remove tokens with cumulative probability above the threshold
        cumulative_probs = torch.roll(cumulative_probs, 1, dims=-1)
        cumulative_probs[:, 0] = 0
        keep_candidates_mask = cumulative_probs < self.nucleus  # -> (n * curr_len, vocab_size)
        keep_candidates_mask[:, :2] = True   # this is temporary decision. we just take the top 2 for sampling.
        keep_candidates_mask[:, 2:] = False  # function "multinomial" supports no more than 4 positions
        sorted_logits.masked_fill_(~keep_candidates_mask, float("-inf"))

        best_candidates_logits_for_sampling = torch.gather(sorted_logits, 1, sorted_indices.argsort(1))
        # -> (n * curr_len, vocab_size)

        best_probs_for_sampling = (best_candidates_logits_for_sampling / self.temperature).softmax(-1)

        assert num_samples % 12 == 0
        sampled_tokens = multinomial(best_probs_for_sampling, num_samples // 12)  # -> (n * curr_len, num_samples)
        line_log_probs = torch.gather(predicted_log_probs, dim=1, index=sampled_tokens)  # -> (n * curr_len, num_samples)

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
            candidates_len_t = torch.ones(1).to(src.device)  # (n_candidates)
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
                # pred_logits = self.model.decode_tgt(draft_sequence, memory_i.repeat(n_candidates, 1, 1),
                #                                     memory_pad_mask=memory_pad_mask_i.repeat(n_candidates, 1),
                #                                     pos_enc_offset=pos_enc_offset)
                pred_logits = self.model.decode_tgt(draft_sequence, memory_i.repeat(n_candidates, 1, 1),
                                                    memory_pad_mask=memory_pad_mask_i.repeat(n_candidates, 1))
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
                accepted_in_drafts_bool = (torch.arange(1, verification.size(2) + 1).unsqueeze(0).unsqueeze(0).type_as(
                    _range) == _range)  # (num, n_drafts, draft_len)
                extra_pad = -1
                pred_tokens[:, :, 1:].masked_fill_(~accepted_in_drafts_bool, extra_pad)
                #   (num_samples * n_candidates, n_drafts, draft_len + 1)
                draft_log_probs[:, :, 1:].masked_fill_(~accepted_in_drafts_bool, 0.)
                #   (num_samples * n_candidates, n_drafts, draft_len + 1)
                accepted_tokens_len = accepted_in_drafts_bool.sum(-1)
                # -> (num, n_drafts, draft_len)

                tmp_ids = torch.arange(num).type_as(src) % n_candidates
                initial_tokens = generated_tokens[tmp_ids]
                #   -> (num_samples * n_candidates, curr_len)
                initial_log_probs = candidates_log_probs[tmp_ids]
                #   -> (num_samples * n_candidates)
                initial_len_t = candidates_len_t[tmp_ids]
                #   -> (num_samples * n_candidates)


                new_candidates = torch.cat((initial_tokens.unsqueeze(1).expand(num, n_drafts, curr_len),
                                            pred_tokens), dim=-1).reshape(num * n_drafts, -1)
                #   -> (num_samples * n_candidates * n_draft, curr_len + draft_len + 1)
                new_log_probs = (initial_log_probs.unsqueeze(1).expand(num, n_drafts) + draft_log_probs.sum(-1)).reshape(num * n_drafts)
                #   -> (num_samples * n_candidates * n_draft)
                new_len_t = (initial_len_t.unsqueeze(1) + accepted_tokens_len).reshape(num * n_drafts) + 1

                new_candidates, new_log_probs, new_len_t = self.unique_and_sort_with_len_adjusting(new_candidates, new_log_probs, new_len_t)

                new_candidates = new_candidates[:self.n_best]
                new_log_probs = new_log_probs[:self.n_best]
                new_len_t = new_len_t[:self.n_best]

                new_candidates = move_pads_to_the_left(new_candidates, extra_pad)
                #   -> (num_samples * n_candidates, curr_len + draft_len + 1)

                new_candidates.masked_fill_(new_candidates == extra_pad, self.pad_token)  # all pads will be at the left
                new_candidates = new_candidates[:, ((new_candidates == self.pad_token).sum(0) == num).sum():]

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
                    new_finished_log_probs_t = new_log_probs[finished_bool_ids] / new_len_t[finished_bool_ids]
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

                generated_tokens = new_candidates[~finished_bool_ids]
                candidates_log_probs = new_log_probs[~finished_bool_ids]
                candidates_len_t = new_len_t[~finished_bool_ids]

                if generated_tokens.size()[0] == 0:
                    finished_candidates_t, finished_candidates_log_probs_t = \
                        self.unique_and_sort(finished_candidates_t, finished_candidates_log_probs_t,
                                             descending=True)
                    finished_candidates_t = finished_candidates_t[:self.n_best]
                    finished_candidates_log_probs_t = finished_candidates_log_probs_t[:self.n_best]
                    break

                generated_tokens = generated_tokens[:,
                                   ((generated_tokens == self.pad_token).sum(0) == generated_tokens.size()[0]).sum():]

            result.append(finished_candidates_t)  # (n, max_len)
        return result

    def unique_and_sort_with_len_adjusting(self, generated_tokens, best_candidates_log_probs, len_t, descending=True):
        generated_tokens_and_probs = torch.cat(
            [generated_tokens, best_candidates_log_probs.unsqueeze(1)], dim=-1)
        #   -> (m, new_curr_len+1)

        generated_tokens_and_probs_and_len = torch.cat(
            [generated_tokens_and_probs, len_t.unsqueeze(1)], dim=-1)

        unique_generated_tokens_and_probs_and_len = torch.unique(generated_tokens_and_probs_and_len, dim=0)
        # -> (new_n_candidates, new_curr_len+1+1)

        unique_generated_tokens_log_probs = unique_generated_tokens_and_probs_and_len[:, -2]
        # -> (new_n_candidates)

        unique_generated_tokens_len = unique_generated_tokens_and_probs_and_len[:, -1]
        # -> (new_n_candidates)

        unique_generated_tokens = unique_generated_tokens_and_probs_and_len[:, :-2].long()
        # -> (new_n_candidates, new_curr_len)

        sorted_log_probs, sorted_inds = (unique_generated_tokens_log_probs / unique_generated_tokens_len).sort(descending=descending)

        return unique_generated_tokens[sorted_inds], unique_generated_tokens_log_probs[sorted_inds], unique_generated_tokens_len[sorted_inds]

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
        # vocab_len = 5
        vocab = [i for i in range(56)]
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

            y = torch.cat((y, drafts), axis=1)  # (b_sz * beam_width * n_drafts, curr_len), (b_sz * beam_width * n_drafts, 1) ->
            # -> (b_sz * beam_width * n_drafts, curr_len)   ###(beam_width * bs, 2+i))
            curr_len += 1
            outp = torch.log(torch.softmax(self.model(src_bw, y), dim=-1))  # -> (b_sz * beam_width * n_drafts, curr_len, vocab_size)
            drafts_probs = torch.gather(outp[:, :-1, :], dim=2, index=y[:, 1:].unsqueeze(2)).squeeze(2)  # -> (b_sz * beam_width * n_drafts, curr_len)
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
            y = y.reshape(b_sz * self.beam_width, curr_len)   #(b_sz, beam_width, curr_len) ->(b_sz * beam_width, curr_len)
            if curr_len == self.max_len:
                break

            probabilities = top_draft_probs  # (b_sz, beam_width)

            probabilities = probabilities.unsqueeze(-1) + next_probabilities
            # (examples,b_w,1) + (examples,b_w,vocab_size) ->(examples,b_w,vocab_size)

            probabilities = probabilities.flatten(start_dim=1)  # (examples,b_w * vocab_size)
            probabilities, idx = probabilities.topk(k=self.beam_width, axis=-1, sorted=True)  # (examples,b_w), (examples,b_w)
            next_chars = torch.remainder(idx, vocab_size).flatten().unsqueeze(-1)  # (examples * b_w,1)
            best_candidates = (idx / vocab_size).long()  # (examples,b_w)
            best_candidates += torch.arange(y.shape[0] // self.beam_width, device=src.device).unsqueeze(-1) * self.beam_width  # (beam_width * bs, 1)
            y = y[best_candidates].flatten(end_dim=-2)  # (beam_width * bs, 2+i)
            y = torch.cat((y, next_chars), axis=1)  # (beam_width * bs, 2+i)
            curr_len += 1
            if curr_len == self.max_len:
                break
        y = y.reshape(b_sz, self.beam_width, self.max_len)
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
        assert b_sz == 1   # Now we try this algorithm only for one reaction

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
            next_log_probs_for_bw = outp[tmp_inds, -2].unsqueeze(1).reshape(b_sz, self.beam_width, vocab_size) + curr_log_probs.unsqueeze(-1)
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
                 n_best: int = 2,
                 temperature: float = 20.,
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
                inp = generated_tokens.unsqueeze(1).expand(n_best, n_drafts, curr_len).reshape(n_best * n_drafts, curr_len)
                draft_sequence = torch.cat([inp, draft_tokens], dim=1)
                # (n_best * n_drafts, curr_len), (n_best * n_drafts, draft_len) -> (n_best * n_drafts, curr_len + draft_len)
                _, seq_len = draft_sequence.size()
                pred_logits = self.model.decode_tgt(draft_sequence,
                                                    memory_i,
                                                    memory_pad_mask=memory_pad_mask_i)
                #   -> (n_best * n_drafts, curr_len + draft_len, vocab_size)

                pred_tokens = self.sample(pred_logits)  # (n_best * n_drafts, curr_len + draft_len, vocab_size) -> (n_best * n_drafts, curr_len + draft_len)

                pred_tokens = pred_tokens.reshape(self.n_best, n_drafts, seq_len)
                pred_tokens = pred_tokens[:, :, -(draft_tokens.size(1) + 1):]  # (n_best, n_drafts, curr_len + draft_len) -> (n_best, n_drafts, draft_len + 1)
                verification = draft_tokens.reshape(n_best, n_drafts, -1) == pred_tokens[:, :, :-1]  # (n_best, n_drafts, draft_len + 1) -> (n_best, n_drafts, draft_len)
                _range = verification.cumsum(-1)  # (n_best, n_drafts, draft_len)
                accepted_in_drafts = (torch.arange(1, verification.size(2) + 1).unsqueeze(0).unsqueeze(0).type_as(_range) == _range)  # (n_best, n_drafts, draft_len)
                n_accepted_in_drafts = accepted_in_drafts.sum(2)  # (n_best, n_drafts, draft_len) -> (n_best, n_drafts)
                n_accepted_in_drafts = n_accepted_in_drafts.topk(1, dim=1)  # (n_best, n_drafts) -> (n_best, 1)
                draft_i = n_accepted_in_drafts.indices  # (n_best, 1)
                n_accepted = n_accepted_in_drafts.values  # (n_best, 1)

                min_num_accepted = torch.min(n_accepted).item()

                #pred_tokens = pred_tokens[draft_i, :n_accepted + 1]  # (n_drafts, curr_len + draft_len, n_best)
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
            result[i] =generated_tokens
        return result


def multinomial(probs, samples_num_coeff):
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
        assert self.beam_size >= self.n_best
        assert self.max_len > 1

    def sample(self, next_token_pred_logits, current_lines, current_log_probs):
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
        #logits_for_sampling[:, self.pad_token] = -float("inf")

        sorted_logits, sorted_indices = torch.sort(logits_for_sampling,
                                                   descending=True)  # -> (n, vocab_size)
        cumulative_probs = torch.cumsum(sorted_logits.softmax(-1), dim=-1)  # -> (n, vocab_size)

        # Remove tokens with cumulative probability above the threshold
        keep_candidates_mask = cumulative_probs < self.nucleus  # -> (n, vocab_size)
        # To sample at least one token
        keep_candidates_mask[:, 0] = True
        keep_candidates_mask[:, 1] = True
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
            if (curr_lines.flatten(end_dim=1) == self.eos_token).sum(-1).bool().sum().item() == bs * curr_lines.shape[1]:
                break
        return curr_lines  # (bs=1, 1 <= n_candidates <= beam_width, len)



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
