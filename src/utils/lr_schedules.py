class ConstantLRSchedule:
    def __init__(self, warmup_steps: int):
        self.ws = warmup_steps

    def __call__(self, i: int) -> float:
        """
        i = current step
        """
        if self.ws > 0 and i < self.ws:
            return (i + 1) / (self.ws + 1)
        return 1


class NoamLRSchedule:

    def __init__(self, emb_dim: int, warmup_steps: int):
        self.ws = warmup_steps
        self.mult = emb_dim ** -0.5

    def __call__(self, i: int) -> float:
        """
        i = current step
        """
        return self.mult * min((i + 1) ** (-0.5), (i + 1) * (self.ws + 1) ** (-1.5))