import random
import torch


class RandomShift:
    def __init__(self, max_shift=3, pad_value=0.0):
        # shifts = [-max_shift, …, 0, …, +max_shift]
        self.shifts = list(range(-max_shift, max_shift + 1))
        self.pad_value = pad_value

    def __call__(self, seq: torch.Tensor) -> torch.Tensor:
        shift = random.choice(self.shifts)
        if shift == 0:
            return seq
        L, C = seq.shape
        # pad with pad_value using zeros_like
        pad = torch.zeros_like(seq[: abs(shift), :]) * self.pad_value
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
            # [A,C,G,T] → [T,G,C,A], then reverse along length
            seq_rc = seq[..., [3, 2, 1, 0]].flip(0)
            # target is only reversing, not complement here
            tgt_r = tgt.flip(0)
            return seq_rc, tgt_r
        return seq, tgt
