import torch
from torch.nn.functional import pad


def make_drafts(
        src: torch.Tensor,
        draft_len: int,
        n_drafts: int,
        min_draft_len: int,
        max_draft_len: int,
        eos_token_idx: int,
        pad_token_idx: int,
        replace_token_idx: int
) -> torch.Tensor:
    """
    This function makes drafts from the source tensor for the desired number of drafts and draft length.
    The drafts are first generated by the sliding window over the source tensor with the desired draft length.
    Then, the set of all possible drafts is filtered to obtain the desired number of drafts.
    Finally, all service tokens are replaced with the desired real token, e.g., the most common token in the dataset.

    B - batch size
    L - length of the source sequence
    D - draft length
    N - number of drafts

    Args:
        src (torch.Tensor): The source tensor of shape (B, L + 1).
        draft_len (int): The desired draft length.
        n_drafts (int): The desired number of drafts for every source sequence in the batch.
        min_draft_len (int): The minimum draft length. For example, draft length of zero makes little sense.
        max_draft_len (int): The maximum draft length.
        eos_token_idx (int): The index of the end of sentence token.
        pad_token_idx (int): The index of the padding token.
        replace_token_idx (int): The index of the token to replace the service tokens with.

    Returns:
        torch.Tensor: The final drafts of shape (B, N, D).
    """
    assert n_drafts > 0, "The number of drafts must be greater than 0"
    assert min_draft_len <= max_draft_len, "The minimum draft length must not be greater than the maximum draft length"
    assert pad_token_idx != replace_token_idx, "The pad token and the replace token must be different"
    assert eos_token_idx != replace_token_idx, "The eos token and the replace token must be different"
    assert eos_token_idx != pad_token_idx, "The eos token and the pad token must be different"

    s = src[:, 1:].clone()  # we don't need the bos token
    B, L = s.shape
    N = n_drafts
    D = min(max(min_draft_len, draft_len), max_draft_len)  # Adjusted draft length

    # If the requested length is greater than the sequence length, we will pad the source more to allow it
    additional_pads = N + D - L - 1
    if additional_pads > 0:
        s = pad(s, (0, additional_pads), "constant", pad_token_idx) # (B, L')
    # else L' is equal to L

    # print(f"{N} drafts of length {D} from {L} tokens")
    drafts = s.unfold(dimension=1, size=D, step=1)  # (B, L' - D + 1, D)
    service_tokens_amount_in_drafts = (drafts == eos_token_idx).logical_or(drafts == pad_token_idx).sum(
        -1)  # (B, L' - D + 1)
    n_drafts_without_service_tokens = (service_tokens_amount_in_drafts == 0).sum(-1)  # B,
    take_from = torch.maximum(n_drafts_without_service_tokens.view(B, 1), torch.tensor(N, device=s.device))  # (B, 1)
    steps = torch.arange(N, device=s.device)
    selected_draft_indices = (steps * ((take_from - 1) / max(N - 1, 1))).long()  # (B, N)
    final_drafts = drafts.gather(1, selected_draft_indices.unsqueeze(-1).expand(-1, -1, D))  # (B, N, D)
    final_drafts.masked_fill_(final_drafts == eos_token_idx, replace_token_idx)
    final_drafts.masked_fill_(final_drafts == pad_token_idx, replace_token_idx)
    return final_drafts  # (B, N, D)


if __name__ == "__main__":
    from torch.nn.utils.rnn import pad_sequence

    DRAFT_LEN = 4
    N_DRAFTS = 100
    MIN_DRAFT_LEN = 2
    MAX_DRAFT_LEN = 200
    EOS_TOKEN_IDX = 2
    PAD_TOKEN_IDX = 0
    REPLACE_TOKEN_IDX = 5

    SOURCE = [
        torch.cat([torch.tensor([1]), torch.arange(5, LENGTH := 9 + 5), torch.tensor([EOS_TOKEN_IDX])], dim=0),
        torch.cat([torch.tensor([1]), torch.arange(5, LENGTH := 7 + 5), torch.tensor([EOS_TOKEN_IDX])], dim=0),
        torch.cat([torch.tensor([1]), torch.arange(5, LENGTH := 6 + 5), torch.tensor([EOS_TOKEN_IDX])], dim=0),
    ]
    src = pad_sequence(SOURCE, batch_first=True)
    print(src)
    # src = torch.tensor([[1, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 2],
    #                     [1, 5, 6, 7, 8, 9, 10, 11, 2, 0, 0, 0, 0]])

    drafts = make_drafts(src,
                         DRAFT_LEN,
                         N_DRAFTS,
                         MIN_DRAFT_LEN,
                         MAX_DRAFT_LEN,
                         EOS_TOKEN_IDX,
                         PAD_TOKEN_IDX,
                         REPLACE_TOKEN_IDX)
    print(drafts)