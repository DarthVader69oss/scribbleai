import numpy as np
import pytest
from src.preprocessing.resample import resample_stroke, resample_ink, ink_length


def test_equal_spacing():
    s = np.array([[0, 0], [10, 0]], float)
    r = resample_stroke(s, 1.0)
    assert len(r) == 11
    assert np.allclose(np.diff(r[:, 0]), 1.0)


def test_spacing_independent_of_input_density():
    sparse = np.array([[0, 0], [5, 0], [5, 5]], float)
    dense = np.concatenate([np.linspace([0, 0], [5, 0], 100), np.linspace([5, 0], [5, 5], 100)[1:]])
    assert len(resample_stroke(sparse, 0.5)) == len(resample_stroke(dense, 0.5))


def test_endpoints_preserved():
    s = np.array([[0, 0], [3, 4], [6, 0]], float)
    r = resample_stroke(s, 0.7)
    assert np.allclose(r[0], s[0]) and np.allclose(r[-1], s[-1])


def test_zero_length_stroke_becomes_one_point():
    assert resample_stroke(np.zeros((5, 2)), 0.1).shape == (1, 2)


def test_vectorised_matches_per_stroke():
    rng = np.random.default_rng(0)
    for _ in range(50):
        ink = [np.cumsum(rng.normal(size=(rng.integers(1, 20), 2)), 0) for _ in range(rng.integers(1, 5))]
        a = resample_ink(ink, 0.4)
        b = [resample_stroke(s, 0.4) for s in ink]
        assert all(x.shape == y.shape and np.allclose(x, y) for x, y in zip(a, b))


def test_bad_step():
    with pytest.raises(ValueError):
        resample_stroke(np.zeros((2, 2)), 0)


def test_ink_length():
    assert ink_length([np.array([[0, 0], [3, 4]], float), np.array([[0, 0]], float)]) == 5.0
