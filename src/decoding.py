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

    def __str__(self):
        return f"Greedy decoding (max_len={self.max_len})"

    def sample(self, pred_logits):
        return torch.argmax(pred_logits, dim=2)[:, -1:]

    def generate(self, src: 'torch.LongTensor') -> 'torch.LongTensor':
        b_size = src.size()[0]
        generated_tokens = torch.full((b_size, self.max_len), self.pad_token).type_as(src)
        generated_tokens[:, 0] = self.bos_token

        src_pad_mask = (src == self.model.src_pad_token_i).bool()
        memory = self.model.encode_src(src, src_pad_mask)

        for i in range(1, self.max_len):
            pred_logits = self.model.decode_tgt(generated_tokens[:, :i], memory, memory_pad_mask=src_pad_mask)
            pred_token = self.sample(pred_logits)
            generated_tokens[:, i] = pred_token.squeeze(-1)
            
            if (torch.logical_or(pred_token == self.eos_token,
                                 pred_token == self.pad_token)).sum().item() == b_size:
                break
        
        return torch.cat([i.unsqueeze(0) for i in generated_tokens.unsqueeze(1)], dim=0)


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

    def __str__(self):
        return f"Beam search decoding (beam_size={self.beam_size}, max_len={self.max_len})"

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
