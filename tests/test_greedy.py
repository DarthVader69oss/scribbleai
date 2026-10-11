import numpy as np
import torch
from src.decoding.greedy import collapse, greedy_decode, greedy_decode_batch

B = 9


def test_collapse_then_remove_blanks_keeps_double_letters():
    # h e l l(blank)l o  -> "hello" needs blank between the two l's
    assert collapse([1, 1, B, 2, 3, 3, B, 3, 4], B) == [1, 2, 3, 3, 4]
    assert collapse([3, 3, 3], B) == [3]


def test_greedy_from_logprobs():
    path = [0, B, 1, 1, B]
    lp = np.full((5, 10), -10.0)
    lp[np.arange(5), path] = 0
    assert greedy_decode(lp, B) == [0, 1]


def test_batch_respects_lengths():
    lp = torch.full((4, 2, 10), -10.0)
    lp[:, 0, 2] = 0; lp[:, 1, 5] = 0
    lp[3, 1, 6] = 1            # beyond length 3 for item 1 -> ignored
    assert greedy_decode_batch(lp, torch.tensor([4, 3]), B) == [[2], [5]]
