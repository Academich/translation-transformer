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


def copy_sequence_draft(source: torch.LongTensor, prediction: torch.LongTensor, pad_token=0):
    # TODO Fails if a source sequence is copied to the target entirely
    B, L = source.size()
    first_divergent_index = (prediction != source).int().argmax(-1).unsqueeze(1)
    accept_mask = torch.arange(L).repeat(B).reshape(B, L).type_as(source) >= first_divergent_index
    draft_accepted = source.masked_fill(accept_mask, pad_token)
    draft_accepted_repadded = move_pads_to_the_left(draft_accepted, pad_token)
    position_offset = (draft_accepted_repadded == pad_token).int().sum(-1)
    return draft_accepted_repadded, position_offset.unsqueeze(1)


class TranslationInferenceGreedyWithCopyBatched:
    # TODO Slower than greedy with large batch sizes if a batch is bottlenecked by an unlucky sequence

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

        draft = src
        draft_verification = self.model.decode_tgt(draft, memory, memory_pad_mask=src_pad_mask).argmax(dim=2)[:, :-1]
        draft_verification = torch.cat([generated_tokens, draft_verification], dim=-1)
        generated_tokens, pos_emb_offset = copy_sequence_draft(draft, draft_verification, pad_token=self.pad_token)

        for _ in range(self.max_len - generated_tokens.shape[1]):
            pred_logits = self.model.decode_tgt(generated_tokens, memory, memory_pad_mask=src_pad_mask,
                                                pos_enc_offset=pos_emb_offset)
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

        sampled_tokens = torch.multinomial(best_probs, 1).squeeze(1)  # -> (n_drafts * curr_len, 1) ->
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
            n_drafts = draft_tokens.size(0)
            iters = 0
            for j in range(self.n_best):
                generated_tokens = torch.full((1, 1), self.bos_token).type_as(src).long()
                while generated_tokens.size(1) < self.max_len:
                    iters += 1
                    draft_sequence = torch.cat([generated_tokens.repeat(n_drafts, 1), draft_tokens], dim=-1)
                    #   -> (n_drafts, curr_len)

                    pred_logits = self.model.decode_tgt(draft_sequence,
                                                        memory_unbatched[i].repeat(n_drafts, 1, 1),
                                                        memory_pad_mask=src_pad_mask_unbatched[i].repeat(n_drafts, 1))
                    #   -> (n_drafts, curr_len, vocab_size)

                    pred_tokens = self.sample(pred_logits)  # (n_drafts, curr_len, vocab_size) -> (n_drafts, curr_len)

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
                    if (generated_tokens == self.eos_token).sum(-1).item() > 0:
                        break
                result[i].append(generated_tokens.squeeze(0))
            result[i] = pad_sequence(result[i], padding_value=0, batch_first=True)

        return result


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
