def calc_token_acc(pred_ids, tgt_ids):
    single_tokens_predicted_right = (pred_ids == tgt_ids).float()  # TODO Beware of EOS != PAD
    return single_tokens_predicted_right.mean()


def calc_sequence_acc(pred_ids, tgt_ids, eos_token_idx):
    """
    Checks how many sequences in a batch are predicted perfectly.
    Considers only the tokens before the first end-of-sequence token.
    """
    hit = (pred_ids == tgt_ids).long()
    eos = tgt_ids == eos_token_idx
    return (hit.cumsum(dim=-1)[eos.roll(-1, dims=-1)] == eos.nonzero(as_tuple=True)[1]).float().mean()