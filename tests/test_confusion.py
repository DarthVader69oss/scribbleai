from src.evaluation.confusion import align, confusion_counts, top_confusions, per_char_error, plot_confusion, EPS


def test_align_ops():
    ops = align("cat", "cut")
    assert ops == [("=", "c", "c"), ("S", "a", "u"), ("=", "t", "t")]
    kinds = [o[0] for o in align("abc", "ac")]
    assert kinds.count("D") == 1 and kinds.count("=") == 2


def test_counts_and_top():
    c = confusion_counts(["rn", "rn"], ["m", "m"])
    top = dict(top_confusions(c))
    assert sum(top.values()) == 4
    assert per_char_error(c)["r"]["err_rate"] == 1.0


def test_plot(tmp_path):
    plot_confusion(confusion_counts(["abc"], ["abd"]), tmp_path / "c.png")
    assert (tmp_path / "c.png").stat().st_size > 1000
