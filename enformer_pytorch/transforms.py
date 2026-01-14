import random
import torch


class RandomShift:
    def __init__(self, max_shift=3, pad_value=-1):
        # shifts = [-max_shift, …, 0, …, +max_shift]
        self.shifts = list(range(-max_shift, max_shift + 1))
        self.pad_value = pad_value

    def __call__(self, seq: torch.Tensor) -> torch.Tensor:
        shift = random.choice(self.shifts)
        if shift == 0:
            return seq
        # pad with pad_value using zeros_like
        pad = torch.ones_like(seq[: abs(shift), :]) * self.pad_value
        if shift > 0:
            # shift right
            sliced = seq[:-shift, :]
            return torch.cat([pad, sliced], dim=0)
        else:
            # shift left
            sliced = seq[-shift:, :]
            return torch.cat([sliced, pad], dim=0)


class RandomReverseComplement:
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, seq: torch.Tensor, tgt: torch.Tensor):
        if random.random() < self.prob:
            rev_ids = seq.flip(dims=[1])
            rc_ids = 3 - rev_ids
            tgt_r = tgt.flip(0)
            return rc_ids, tgt_r
        return seq, tgt
