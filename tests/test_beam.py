import numpy as np
from src.decoding.beam import prefix_beam_search
from src.decoding.greedy import greedy_decode
from src.decoding.lm import CharNgramLM

CHARS = list("ab ")
BLANK = 3


def peaky(path, C=4, p=0.97):
    lp = np.full((len(path), C), np.log((1 - p) / (C - 1)))
    lp[np.arange(len(path)), path] = np.log(p)
    return lp


def test_matches_greedy_on_confident_output():
    lp = peaky([0, BLANK, 0, 1, 1, BLANK, 2, 0])
    g = "".join(CHARS[i] for i in greedy_decode(lp, BLANK))
    assert prefix_beam_search(lp, CHARS, BLANK, beam_width=8, blank_skip=1.1)[0] == g == "aab a"


def test_beam_beats_greedy_when_paths_sum_up():
    # classic example: greedy picks blank,blank but "a" has more total mass
    lp = np.log(np.array([[0.4, 0.0001, 0.0001, 0.5998], [0.4, 0.0001, 0.0001, 0.5998]]))
    assert greedy_decode(lp, BLANK) == []
    assert prefix_beam_search(lp, CHARS, BLANK, beam_width=5, blank_skip=1.1)[0] == "a"


def test_lm_changes_ambiguous_choice():
    lm = CharNgramLM(3).train(["ab ab ab ab"] * 50)
    # second char ambiguous between a and b
    lp = np.log(np.array([[0.98, 0.005, 0.005, 0.01], [0.49, 0.49, 0.01, 0.01]]))
    no_lm = prefix_beam_search(lp, CHARS, BLANK, 5, blank_skip=1.1)[0]
    with_lm = prefix_beam_search(lp, CHARS, BLANK, 5, lm=lm, alpha=1.0, beta=0.0, blank_skip=1.1)[0]
    assert with_lm == "ab" and no_lm in ("aa", "ab", "a")


def test_returns_ranked_beam():
    best, beam = prefix_beam_search(peaky([0, 1]), CHARS, BLANK, 4)
    assert beam[0][0] == best and all(beam[i][1] >= beam[i + 1][1] for i in range(len(beam) - 1))
