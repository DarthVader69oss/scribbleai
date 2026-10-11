from src.evaluation.analysis import cer_by_bins, worst_samples, quantile_bins, plot_bins


ROWS = [{"ref": "ab", "hyp": "ab", "n": 2}, {"ref": "abcdefghij", "hyp": "abcdefghiX", "n": 10},
        {"ref": "xyz", "hyp": "", "n": 3}]


def test_bins():
    t = cer_by_bins(ROWS, "n", [0, 5, 100])
    assert t[0]["bin"] == "0-5" and t[0]["n"] == 2 and abs(t[0]["cer"] - 3 / 5) < 1e-9
    assert abs(t[1]["cer"] - 0.1) < 1e-9


def test_worst():
    assert worst_samples(ROWS, 1)[0]["ref"] == "xyz"


def test_quantile_bins_monotonic():
    b = quantile_bins([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 4)
    assert b == sorted(b) and b[0] == 1 and b[-1] > 10


def test_plot(tmp_path):
    plot_bins(cer_by_bins(ROWS, "n", [0, 5, 100]), tmp_path / "p.png", "len", "t")
    assert (tmp_path / "p.png").exists()
