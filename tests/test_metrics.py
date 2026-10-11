from src.evaluation.metrics import edit_distance, cer, wer, corpus_scores, Latency


def test_edit_distance():
    assert edit_distance("kitten", "sitting") == 3
    assert edit_distance(["a", "b"], ["a", "c", "b"]) == 1
    assert edit_distance("", "abc") == 3


def test_cer_wer():
    assert cer("abcd", "abxd") == 0.25
    assert wer("the cat sat", "the bat sat") == 1 / 3


def test_corpus_is_total_edits_over_total_length():
    s = corpus_scores(["ab", "abcdefgh"], ["xb", "abcdefgh"])
    assert abs(s["cer"] - 1 / 10) < 1e-9 and s["line_acc"] == 0.5


def test_latency():
    l = Latency()
    with l.time():
        sum(range(1000))
    assert l.summary()["mean_ms"] >= 0
