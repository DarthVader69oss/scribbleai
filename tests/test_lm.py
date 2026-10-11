import math
from src.decoding.lm import CharNgramLM, write_word_arpa, EOS


def test_probabilities_normalised():
    lm = CharNgramLM(4).train(["the cat", "the hat", "a cat sat"])
    for ctx in ["", "th", "the c", "zzz"]:
        total = sum(lm.prob(ctx, c) for c in lm.vocab)
        assert abs(total - 1.0) < 0.05          # uniform floor leaves a little mass for unseen chars


def test_likely_beats_unlikely():
    lm = CharNgramLM(4).train(["the cat sat on the mat"] * 20)
    assert lm.score("the cat") > lm.score("tqe xat")


def test_save_load(tmp_path):
    lm = CharNgramLM(3).train(["hello", "help"])
    lm.save(tmp_path / "lm.json")
    lm2 = CharNgramLM.load(tmp_path / "lm.json")
    assert math.isclose(lm.score("help"), lm2.score("help"))


def test_perplexity_finite():
    lm = CharNgramLM(3).train(["abc"] * 5)
    assert 1 <= lm.perplexity(["abc", "abd"]) < 1e6


def test_arpa_format(tmp_path):
    p = tmp_path / "w.arpa"
    write_word_arpa(["the cat sat", "the dog sat", "a cat"], p, order=3)
    text = p.read_text()
    assert text.startswith("\\data\\") and "\\3-grams:" in text and text.strip().endswith("\\end\\")
    n1 = int(text.split("ngram 1=")[1].split()[0])
    block = text.split("\\1-grams:\n")[1].split("\n\n")[0].strip().splitlines()
    assert len(block) == n1
    probs = [10 ** float(l.split("\t")[0]) for l in block if "<s>" not in l]
    assert abs(sum(probs) - 1) < 1e-6
