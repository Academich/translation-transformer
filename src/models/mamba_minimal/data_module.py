import torch
from torch.nn.utils.rnn import pad_sequence

from data_wrappers import Seq2SeqDM


def reverse_tokens(i: int, length: int) -> list[int]:
    ids = range(length)
    return [ids[0], *reversed(ids[1:i]), *ids[i:]]


def get_token_permutation(tokens: torch.LongTensor, eos_token: int):
    """
    Prepares the permutation matrices to flip tokens in a sequence while preserving special tokens
    :param tokens: B x L
    :param eos_token: int
    :return:
    """
    _, le = tokens.size()
    ids = []
    for i in (tokens == eos_token).nonzero(as_tuple=True)[1]:  # Could not avoid a python cycle over the batch dim.
        ids.append(reverse_tokens(i, le))
    return torch.tensor(ids, dtype=torch.long)


class Seq2SeqWithEncDecMambaNaiveDM(Seq2SeqDM):

    def collate_fn(self, batch: list[tuple[str, str]]) -> dict[str, torch.Tensor]:
        src_strings, tgt_strings = zip(*batch)
        src_tokens, tgt_tokens = self._prepare_tokens(src_strings, tgt_strings)
        padded_src_tokens = pad_sequence(src_tokens,
                                         padding_value=self.src_tokenizer.pad_token_idx, batch_first=True)
        padded_tgt_tokens = pad_sequence(tgt_tokens,
                                         padding_value=self.tgt_tokenizer.pad_token_idx, batch_first=True)
        source_flip_indices = get_token_permutation(padded_src_tokens, self.src_tokenizer.eos_token_idx)
        return {"src_tokens": padded_src_tokens,
                "tgt_tokens": padded_tgt_tokens,
                "src_flip_ids": source_flip_indices}
