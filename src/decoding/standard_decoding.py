import torch


class TranslationInferenceGreedy:
    """
    Basic greedy decoding for a batch of sequences.
    """

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

        self.model_calls_num = 0
        self.given_tokens = 0

    def __str__(self):
        return f"Greedy decoding (max_len={self.max_len})"

    def sample(self, pred_logits):
        return torch.argmax(pred_logits, dim=2)[:, -1:]

    def generate(self, src: 'torch.LongTensor') -> 'torch.LongTensor':
        """
        Generate a batch of sequences using greedy decoding.
        Input shape: B x L (batch size x source sequence length)
        Output shape: B x N x L (batch size x number of hypotheses x target sequence length)
        In greedy decoding, N is always 1.
        """
        b_size = src.size()[0]
        generated_tokens = torch.full((b_size, self.max_len), self.pad_token).type_as(src)
        generated_tokens[:, 0] = self.bos_token

        src_pad_mask = (src == self.model.src_pad_token_i).bool()
        self.given_tokens += (b_size * src.shape[1] - src_pad_mask.sum().item())
        memory = self.model.encode_src(src, src_pad_mask)

        for i in range(1, self.max_len):
            pred_logits = self.model.decode_tgt(generated_tokens[:, :i], memory, memory_pad_mask=src_pad_mask)
            self.model_calls_num += 1
            pred_token = self.sample(pred_logits)
            generated_tokens[:, i] = pred_token.squeeze(-1)
            
            if (torch.logical_or(pred_token == self.eos_token,
                                 pred_token == self.pad_token)).sum().item() == b_size:
                break
        
        return generated_tokens.unsqueeze(1)


class TranslationInferenceBeamSearch:
    """
    Basic beam search decoding for a batch of sequences.
    Inspired by jarobyte91 src/pytorch_beam_search/seq2seq/search_algorithms.py
    """

    def __init__(self,
                 model,  # TranslationModel
                 beam_size: int,
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

        assert self.max_len > 1
        assert self.beam_size > 0

        self.model_calls_num = 0
        self.given_tokens = 0

        self.b_sz = 0

    def __str__(self):
        return f"Beam search decoding (beam_size={self.beam_size}, max_len={self.max_len})"

    def generate(self, src: 'torch.LongTensor') -> 'torch.LongTensor':
        """
        Generate a batch of sequences using beam search decoding.
        Input shape: B x L (batch size x source sequence length)
        Output shape: B x N x L (batch size x number of hypotheses x target sequence length)
        In beam search decoding, N is equal to beam_size.
        """
        bs, src_len = src.size()

        # Prepare first tokens for decoder (bs, max_len)
        y = torch.tensor([self.bos_token]).repeat(bs, 1).long().type_as(src)  # (bs,1)

        # Decode for one step using decoder
        decoder_output = self.model(src, y)  # (bs, 1, dict_len)
        self.b_sz += bs
        self.model_calls_num += 1
        self.given_tokens += (src != self.model.src_pad_token_i).bool().sum().item()
        logprob_decoder_output = torch.log(torch.softmax(decoder_output, dim=-1))

        # check shape of the prediction
        vocab_size = logprob_decoder_output.shape[-1]

        probabilities, next_chars = torch.topk(logprob_decoder_output, self.beam_size,
                                               dim=-1, sorted=True)  # (bs, 1, beam_width), (bs, 1, beam_width)
        probabilities = probabilities.squeeze(1)  # (bs, beam_width)
        y = y.unsqueeze(1).repeat((1, 1, self.beam_size)).reshape(-1, 1)
        # (bs * beam_width, 1)

        next_chars = next_chars.reshape(-1, 1)  # (bs * beam_width, 1)
        y = torch.cat((y, next_chars), axis=-1)  # (bs * beam_width, 2)

        src_bw = src.repeat((self.beam_size, 1, 1)).transpose(0, 1).flatten(end_dim=1)
        # -> (bs * beam_width, src_len)

        src_pad_mask = (src_bw == self.model.src_pad_token_i).bool()
        memory = self.model.encode_src(src_bw, src_pad_mask)    # -> (bs, src_len, emb_dim)

        predictions = self.max_len - 1

        logits_base = torch.full((bs * self.beam_size,  vocab_size), 0., device=src.device)
        #   -> (b_s * beam_size, vocab_size)

        for i in range(predictions - 1):

            logits_base = logits_base * 0.
            # We use artificial logits to avoid calculation of obvious pad predicting after eos
            logits_base[:, self.model.src_pad_token_i] = 35.
            # 35. will give about 100% probability for pad_token after softmax()

            bool_idx_of_unfinished = ~((y == self.eos_token).sum(-1).bool())
            # -> (bs * beam_width)

            self.b_sz += bool_idx_of_unfinished.sum().item()
            pred_logits = self.model.decode_tgt(y[bool_idx_of_unfinished],
                                    memory[bool_idx_of_unfinished],
                                    memory_pad_mask=src_pad_mask[bool_idx_of_unfinished])
            # -> (num_of_unfinished_candidates, 2 + i, vocab_size)
            pred_logits = pred_logits[:, -1, :]
            # -> (num_of_unfinished_candidates, vocab_size)

            logits_base[bool_idx_of_unfinished] = pred_logits
            #   -> (bs * b_w, vocab_size)
            next_probabilities = torch.log(torch.softmax(logits_base, dim=-1))
            # (bs * b_w, vocab_size)

            self.model_calls_num += 1
            next_probabilities = next_probabilities.reshape(
                (-1, self.beam_size, next_probabilities.shape[-1]))  # (bs, b_w, vocab_size)

            probabilities = probabilities.unsqueeze(-1) + next_probabilities
            # (bs, b_w, 1) + (bs, b_w, vocab_size) ->(bs, b_w, vocab_size)

            probabilities = probabilities.flatten(start_dim=1)  # (bs, b_w * vocab_size)
            probabilities, idx = probabilities.topk(k=self.beam_size, axis=-1,
                                                    sorted=True)  # (bs, b_w), (bs, b_w)
            next_chars = torch.remainder(idx, vocab_size).flatten().unsqueeze(-1)  # (bs * b_w, 1)
            best_candidates = (idx / vocab_size).long()  # (bs, b_w)
            best_candidates += torch.arange(y.shape[0] // self.beam_size, device=src.device).unsqueeze(
                -1) * self.beam_size  # (beam_width * bs, 1)
            y = y[best_candidates].flatten(end_dim=-2)  # (beam_width * bs, 2 + i)
            y = torch.cat((y, next_chars), axis=1)  # -> (beam_width * bs, 2 + i + 1)
            if (y == self.eos_token).sum(-1).bool().sum().item() == y.size()[0]:
                break
        _, curr_len = y.size()
        y = y.reshape(bs, self.beam_size, curr_len)
        return y  # , probabilities  # (examples,b_w, max_len), (examples,b_w)
