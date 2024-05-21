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

        assert self.beam_size >= self.n_best
        assert self.max_len > 1

    def __str__(self):
        return f"Beam search decoding (beam_size={self.beam_size}, n_best={self.n_best}, max_len={self.max_len})"

    def generate(self, src: 'torch.LongTensor') -> 'torch.LongTensor':
        bs, _ = src.size()

        # Prepare first tokens for decoder (bs, max_len)
        y = torch.tensor([self.bos_token]).repeat(bs, 1).long().type_as(src)  # (bs,1)

        # Decode for one step using decoder
        decoder_output = self.model(src, y)  # (bs, 1, dict_len)
        logprob_decoder_output = torch.log(torch.softmax(decoder_output, dim=-1))

        # check shape of the prediction
        vocab_size = logprob_decoder_output.shape[-1]

        probabilities, next_chars = torch.topk(logprob_decoder_output, self.beam_size,
                                               dim=-1, sorted=True)  # (bs, 1, beam_width), (bs, 1, beam_width)
        probabilities = probabilities.squeeze(1)  # (bs, beam_width)
        y = y.unsqueeze(1).repeat((1, 1, self.beam_size)).reshape(-1, 1)
        # (beam_width * bs, 1)

        next_chars = next_chars.reshape(-1, 1)  # (bs *beam_width, 1)
        y = torch.cat((y, next_chars), axis=-1)  # (beam_width * bs, 2)

        src_bw = src.repeat((self.beam_size, 1, 1)).transpose(0, 1).flatten(end_dim=1)  # (b_w, examples, length)
        # (examples, beam_width, length) # fin_X [[5,20],[5,20],[5,20],[2,31],[2,31],[2,31]]
        # (b_w * examples, length)

        predictions = self.max_len - 1

        for i in range(predictions - 1):
            next_probabilities = torch.log(torch.softmax(self.model(src_bw, y), dim=-1))[:, -1,
                                 :]  # (bs*b_w, vocab_size)

            next_probabilities = next_probabilities.reshape(
                (-1, self.beam_size, next_probabilities.shape[-1]))  # (examples, b_w, vocab_size)

            probabilities = probabilities.unsqueeze(-1) + next_probabilities
            # (examples,b_w,1) + (examples,b_w,vocab_size) ->(examples,b_w,vocab_size)

            probabilities = probabilities.flatten(start_dim=1)  # (examples,b_w * vocab_size)
            probabilities, idx = probabilities.topk(k=self.beam_size, axis=-1,
                                                    sorted=True)  # (examples,b_w), (examples,b_w)
            next_chars = torch.remainder(idx, vocab_size).flatten().unsqueeze(-1)  # (examples * b_w,1)
            best_candidates = (idx / vocab_size).long()  # (examples,b_w)
            best_candidates += torch.arange(y.shape[0] // self.beam_size, device=src.device).unsqueeze(
                -1) * self.beam_size  # (beam_width * bs, 1)
            y = y[best_candidates].flatten(end_dim=-2)  # (beam_width * bs, 2+i)
            y = torch.cat((y, next_chars), axis=1)  # (beam_width * bs, 2+i)
            if (y == self.eos_token).sum(-1).bool().sum().item() == y.size()[0]:
                break
        _, curr_len = y.size()
        y = y.reshape(bs, self.beam_size, curr_len)
        return y  # , probabilities  # (examples,b_w, max_len), (examples,b_w)


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

        generated_tokens = torch.full((b_size, 1), self.bos_token).type_as(src).long()
        draft_tokens = src[:, 1:].unfold(-1, self.n_speculative_tokens, 1)
        n_drafts = draft_tokens.size(1)
        iters = 0
        while generated_tokens.size(1) < self.max_len:
            iters += 1
            draft_sequence = torch.cat([generated_tokens.unsqueeze(1).repeat(1, n_drafts, 1), draft_tokens],
                                       dim=-1).reshape(b_size * n_drafts, -1)
            pos_enc_offset = (draft_sequence == -1).int().sum(-1).reshape(-1, 1)
            generated_tokens = generated_tokens.masked_fill(generated_tokens == -1, self.pad_token)
            draft_sequence = draft_sequence.masked_fill(draft_sequence == -1, self.pad_token)

            pred_logits = self.model.decode_tgt(draft_sequence,
                                                memory.unsqueeze(1).repeat(1, n_drafts, 1, 1).reshape(b_size * n_drafts,
                                                                                                      -1, 256),
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
            generated_tokens = move_pads_to_the_left(generated_tokens, -1)
            generated_tokens = generated_tokens.masked_fill(generated_tokens == self.pad_token, -1)
            generated_tokens = generated_tokens[:, ((generated_tokens == -1).sum(
                dim=0) == b_size).sum():]  # Strip columns on the left that are all pads
            if (generated_tokens == self.eos_token).sum(-1).bool().sum().item() == b_size:
                break
        generated_tokens = generated_tokens.masked_fill(generated_tokens == -1, self.pad_token)
        return generated_tokens.unsqueeze(1)


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


def move_pads_to_the_left(arr, pad_token=0):
    dim_indices = torch.arange(arr.shape[1]).type_as(arr).long().repeat(arr.shape[0]).reshape(arr.shape[0], -1)
    eos_index = (arr == pad_token).sum(1)
    indices = (dim_indices - eos_index.unsqueeze(1)) % arr.shape[1]
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
        line_log_probs = torch.gather(pred_logits.softmax(-1), dim=1, index=sampled_tokens).log()  # -> (n * curr_len, num_samples)

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
