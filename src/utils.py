# === Learning rate schedulers ===
class ConstantLRSchedule:
    def __init__(self, learning_rate: float, warmup_steps: int):
        self.lr = learning_rate
        self.ws = warmup_steps

    def __call__(self, i: int) -> float:
        """
        i = current step
        """
        if self.ws > 0 and i < self.ws:
            return (i + 1) / (self.ws + 1)
        return 1


class NoamLRSchedule:

    def __init__(self, emb_dim: int, learning_rate: float, warmup_steps: int):
        self.lr = learning_rate
        self.ws = warmup_steps
        self.mult = emb_dim ** -0.5

    def __call__(self, i: int) -> float:
        """
        i = current step
        """
        return self.mult * min((i + 1) ** (-0.5), (i + 1) * (self.ws + 1) ** (-1.5))


# === Metrics ====
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
